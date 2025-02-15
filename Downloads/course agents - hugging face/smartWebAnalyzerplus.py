# app.py
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

# --------------------------
# Configuration & Constants
# --------------------------
THEMES = ["light", "dark"]
ANALYSIS_MODES = ["analyze", "summarize", "sentiment", "topics"]

# Define which model to use for each task
MODEL_MAP = {
    "summarize": "facebook/bart-large-cnn",
    "sentiment": "nlptown/bert-base-multilingual-uncased-sentiment",
    "topics": "facebook/bart-large-mnli"  # used for zero-shot classification
}
DEVICE = 0 if torch.cuda.is_available() else -1

# Candidate labels for topic detection (can be customized)
CANDIDATE_TOPICS = ["politics", "sports", "technology", "health", "entertainment"]

# --------------------------
# Utility & Caching Functions
# --------------------------
@lru_cache(maxsize=32)
def load_analysis_model(task: str) -> Pipeline:
    """
    Load and cache NLP models using Hugging Face transformers.
    Uses different pipelines based on the task.
    """
    if task == "summarize":
        model_name = MODEL_MAP["summarize"]
        return pipeline("summarization", model=model_name, device=DEVICE)
    elif task == "sentiment":
        model_name = MODEL_MAP["sentiment"]
        return pipeline("sentiment-analysis", model=model_name, device=DEVICE)
    elif task == "topics":
        model_name = MODEL_MAP["topics"]
        # For zero-shot classification, use the 'zero-shot-classification' pipeline.
        return pipeline("zero-shot-classification", model=model_name, device=DEVICE)
    else:
        raise ValueError("Invalid task for model loading.")

def fetch_web_content(url: str) -> str:
    """
    Fetches web content from a URL and cleans the HTML using BeautifulSoup.
    Removes non-content elements.
    """
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove script, style, meta, nav, and footer elements.
        for element in soup(['script', 'style', 'meta', 'nav', 'footer']):
            element.decompose()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        raise RuntimeError(f"Failed to fetch content: {str(e)}")

# --------------------------
# Analysis Functions Using LLMs
# --------------------------
def generate_summary(text: str, max_length: int = 130) -> str:
    """
    Generate a summary for the given text using a summarization model.
    """
    summarizer = load_analysis_model("summarize")
    # The summarizer might have input length limits; here we assume text is short enough.
    summary_list = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    return summary_list[0]['summary_text']

def analyze_sentiment(text: str, detailed: bool = False) -> Dict[str, Union[str, List[Dict]]]:
    """
    Analyze sentiment of the text. In detailed mode, split text into paragraphs and analyze each.
    Returns a dictionary with sentiment results and a generated sentiment chart (if detailed).
    """
    sentiment_model = load_analysis_model("sentiment")
    if detailed:
        paragraphs = [para for para in text.split("\n") if para.strip()]
        sentiments = []
        for para in paragraphs:
            # Get sentiment result for the paragraph
            result = sentiment_model(para[:512])  # limit to first 512 characters per paragraph
            # Assuming the model returns a list with one dict result
            sentiments.append({
                "paragraph": para if len(para) < 50 else para[:50] + "...",
                "label": result[0]["label"],
                "score": round(result[0]["score"], 2)
            })
        chart = generate_sentiment_chart(sentiments)
        return {"sentiments": sentiments, "sentiment_chart": chart}
    else:
        # Overall sentiment for the entire text
        result = sentiment_model(text[:512])
        return {"overall_sentiment": result[0]}

def analyze_topics(text: str) -> Dict[str, List]:
    """
    Analyze topics in the text using a zero-shot classification pipeline.
    Returns a dictionary with topic scores.
    """
    topic_model = load_analysis_model("topics")
    result = topic_model(text, candidate_labels=CANDIDATE_TOPICS)
    return {"topics": result}

def generate_sentiment_chart(sentiments: List[Dict]) -> str:
    """
    Generates and saves a bar chart for sentiment scores.
    Returns the path to the saved image file.
    """
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
    """
    Main analysis pipeline.
    - Fetches content from URL or file.
    - Depending on the mode, runs summarization, sentiment analysis, or topic detection.
    - In "analyze" mode, runs all analyses.
    Returns a dictionary with raw text and analysis results.
    """
    try:
        progress(0.1, desc="Initializing analysis...")
        # Fetch or read text
        if is_url:
            progress(0.2, desc="Fetching web content...")
            text = fetch_web_content(input_data)
        else:
            progress(0.2, desc="Reading uploaded file...")
            # Assume input_data is a temporary file path
            with open(input_data, 'r', encoding="utf-8") as f:
                text = f.read()

        if len(text) < 100:
            raise ValueError("Insufficient content for analysis.")

        results = {}

        if mode == "analyze":
            progress(0.3, desc="Generating summary...")
            results["summary"] = generate_summary(text, max_length=summary_length)
            progress(0.5, desc="Analyzing sentiment (detailed)...")
            results["sentiment"] = analyze_sentiment(text, detailed=True)
            progress(0.7, desc="Detecting topics...")
            results["topics"] = analyze_topics(text)
        elif mode == "summarize":
            progress(0.4, desc="Generating summary...")
            results["summary"] = generate_summary(text, max_length=summary_length)
        elif mode == "sentiment":
            progress(0.4, desc="Analyzing sentiment (detailed)...")
            results["sentiment"] = analyze_sentiment(text, detailed=True)
        elif mode == "topics":
            progress(0.4, desc="Detecting topics...")
            results["topics"] = analyze_topics(text)
        else:
            raise ValueError("Invalid analysis mode selected.")

        progress(0.9, desc="Formatting results...")
        return {"raw_text": text, "analysis": results, "status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --------------------------
# PDF Export Functionality
# --------------------------
def generate_pdf_report(report_text: str) -> str:
    """
    Exports the given report text to a PDF file using ReportLab.
    Returns the filename of the generated PDF.
    """
    filename = "analysis_report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("AI Analysis Report", styles['Title']), Spacer(1, 12)]
    for line in report_text.split('\n'):
        story.append(Paragraph(line, styles['BodyText']))
        story.append(Spacer(1, 12))
    doc.build(story)
    return filename

# --------------------------
# Gradio UI Construction
# --------------------------
def create_analysis_tab():
    """
    Creates the Analysis tab UI components.
    """
    with gr.Tab("Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                input_type = gr.Radio(choices=["URL", "File"], label="Input Type", value="URL")
                url_input = gr.Textbox(label="Enter Web URL", visible=True)
                file_input = gr.File(label="Upload File", file_types=[".txt"], visible=False)
                mode = gr.Dropdown(choices=ANALYSIS_MODES, label="Analysis Mode", value="analyze")
                theme = gr.Radio(choices=THEMES, label="Interface Theme", value="light")
                summary_length = gr.Slider(minimum=50, maximum=300, step=10, value=130,
                                           label="Summary Max Length (words)")
                # Switch visibility based on input type
                input_type.change(fn=lambda x: (gr.update(visible=x=="URL"), gr.update(visible=x!="URL")),
                                  inputs=input_type, outputs=[url_input, file_input])
            with gr.Column(scale=1):
                output_json = gr.JSON(label="Analysis Results")
                progress_box = gr.Textbox(label="Status", interactive=False)
        analyze_btn = gr.Button("Analyze", variant="primary")
        # When analyze is clicked, run the analysis pipeline.
        analyze_btn.click(
            fn=analyze_content,
            inputs=[url_input, mode, input_type, summary_length],
            outputs=output_json,
            show_progress="full"
        )

def create_report_tab():
    """
    Creates the Report tab UI components.
    """
    with gr.Tab("Report"):
        report_view = gr.Textbox(label="Formatted Report", lines=20)
        pdf_btn = gr.Button("Export PDF")
        pdf_output = gr.File(label="Download PDF")
        pdf_btn.click(
            fn=generate_pdf_report,
            inputs=report_view,
            outputs=pdf_output
        )

def create_interface():
    """
    Constructs the complete Gradio Blocks interface.
    """
    with gr.Blocks(title="Smart Web Analyzer Plus") as demo:
        gr.Markdown("# Smart Web Analyzer Plus")
        with gr.Tabs():
            create_analysis_tab()
            create_report_tab()
    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True
    )
