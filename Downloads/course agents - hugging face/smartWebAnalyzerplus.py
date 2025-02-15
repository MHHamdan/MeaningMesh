import os
import json
import time
import requests
import gradio as gr
from typing import List, Dict, Union
from bs4 import BeautifulSoup
from pathlib import Path
from transformers import pipeline, Pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import torch
from functools import lru_cache

# Configuration & Constants
THEMES = ["light", "dark"]
ANALYSIS_MODES = ["analyze", "summarize", "sentiment", "topics"]

MODEL_MAP = {
    "summarize": "facebook/bart-large-cnn",
    "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
    "topics": "facebook/bart-large-mnli"
}
DEVICE = 0 if torch.cuda.is_available() else -1
CANDIDATE_TOPICS = ["politics", "sports", "technology", "health", "entertainment"]

@lru_cache(maxsize=32)
def load_analysis_model(task: str) -> Pipeline:
    if task not in MODEL_MAP:
        raise ValueError("Invalid task for model loading.")
    return pipeline(task, model=MODEL_MAP[task], device=DEVICE)

def fetch_web_content(url: str) -> str:
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'meta', 'nav', 'footer']):
            element.decompose()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch content: {str(e)}")

def generate_summary(text: str, max_length: int = 130) -> str:
    summarizer = load_analysis_model("summarize")
    return summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']

def analyze_sentiment(text: str, detailed: bool = False) -> Dict[str, Union[str, List[Dict]]]:
    sentiment_model = load_analysis_model("sentiment")
    if detailed:
        paragraphs = [para for para in text.split("\n") if para.strip()]
        sentiments = []
        for para in paragraphs:
            result = sentiment_model(para[:512])
            sentiments.append({"paragraph": para[:50] + "...", "label": result[0]["label"], "score": round(result[0]["score"], 2)})
        return {"sentiments": sentiments, "sentiment_chart": generate_sentiment_chart(sentiments)}
    else:
        result = sentiment_model(text[:512])
        return {"overall_sentiment": result[0]}

def analyze_topics(text: str) -> Dict[str, List]:
    topic_model = load_analysis_model("topics")
    result = topic_model(text, candidate_labels=CANDIDATE_TOPICS)
    return {"topics": result}

def generate_sentiment_chart(sentiments: List[Dict]) -> str:
    labels = [s['paragraph'] for s in sentiments]
    scores = [s['score'] for s in sentiments]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(scores)), scores, tick_label=[f"Sec {i+1}" for i in range(len(scores))])
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment per Section")
    chart_path = "sentiment_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def analyze_content(input_data: str, mode: str, is_url: bool, summary_length: int, progress=gr.Progress()) -> Dict:
    try:
        progress(0.1, desc="Initializing analysis...")
        text = fetch_web_content(input_data) if is_url else Path(input_data).read_text(encoding="utf-8")
        if len(text) < 100:
            raise ValueError("Insufficient content for analysis.")
        results = {}
        if mode == "analyze":
            results.update({"summary": generate_summary(text, max_length=summary_length)})
            results.update({"sentiment": analyze_sentiment(text, detailed=True)})
            results.update({"topics": analyze_topics(text)})
        elif mode == "summarize":
            results["summary"] = generate_summary(text, max_length=summary_length)
        elif mode == "sentiment":
            results["sentiment"] = analyze_sentiment(text, detailed=True)
        elif mode == "topics":
            results["topics"] = analyze_topics(text)
        return {"raw_text": text, "analysis": results, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
def create_interface():
    with gr.Blocks(title="Smart Web Analyzer Plus") as iface:
        gr.Markdown("# 🚀 Smart Web Analyzer Plus")
        gr.Markdown("""
        Advanced content analysis with AI-powered insights:
        * 📊 Comprehensive Analysis
        * 😊 Detailed Sentiment Analysis
        * 📝 Smart Summarization
        * 🎯 Topic Detection
        """)

        # Theme toggle
        theme = gr.Radio(choices=["light", "dark"], value="light", label="Theme", interactive=True)

        with gr.Tabs():
            with gr.Tab("Analysis"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="URL or Text to Analyze",
                            placeholder="Enter URL or paste text",
                            lines=5
                        )
                        mode = gr.Radio(
                            choices=["analyze", "summarize", "sentiment", "topics"],
                            value="analyze",
                            label="Analysis Mode"
                        )
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                        status = gr.Markdown("Status: Ready")

                    with gr.Column():
                        results = gr.JSON(label="Analysis Results")
                        chart = gr.Plot(label="Visualization", visible=False)

                # Ensure correct UI state handling
                mode.change(
                    lambda m: gr.update(visible=(m == "sentiment")),
                    inputs=[mode],
                    outputs=[chart]
                )

            with gr.Tab("Preview"):
                preview = gr.Textbox(label="Content Preview", lines=10, interactive=False)

            with gr.Tab("Report"):
                download_btn = gr.Button("📥 Download PDF Report")
                pdf_output = gr.File(label="Generated Report")

        # Wire up the analysis button
        analyze_btn.click(
            fn=process_content,
            inputs=[input_text, mode, theme],
            outputs=[results, preview, status, chart]
        )

        # Wire up PDF download
        download_btn.click(
            fn=lambda: generate_pdf_report(json.loads(results.value) if results.value else {}),
            inputs=[],
            outputs=[pdf_output]
        )

    return iface

demo = create_interface()
demo.launch()

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), show_error=True)
