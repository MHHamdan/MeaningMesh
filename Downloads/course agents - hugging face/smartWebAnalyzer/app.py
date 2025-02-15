# app.py
"""
Gradio App for Smart Web Analyzer Plus - Human-Readable Outputs

Key Features:
- Accepts a URL
- Lets users select analysis modes (Clean Text, Summarization, Sentiment, Topic)
- Fetches and processes content (smart_web_analyzer.py)
- Displays each result in its own tab for readability
- Includes example URLs
"""

import gradio as gr
from smart_web_analyzer import (
    fetch_web_content,
    clean_text,
    summarize_text,
    analyze_sentiment,
    detect_topic,
    preview_clean_text,
)

def analyze_url(url, modes):
    """
    Fetches web content and performs selected analyses (modes).
    
    Parameters:
        url (str): URL to analyze
        modes (list): list of selected modes
    
    Returns:
        tuple of str: (clean_text_result, summarization_result, sentiment_result, topics_result)
    """
    # Default messages if a mode is not selected
    clean_text_result = "Mode not selected."
    summarization_result = "Mode not selected."
    sentiment_result = "Mode not selected."
    topics_result = "Mode not selected."
    
    # 1) Fetch/clean the web content
    try:
        html_content = fetch_web_content(url)
    except Exception as e:
        # Return the error in each field for clarity
        error_msg = f"**Error fetching URL**: {e}"
        return (error_msg, error_msg, error_msg, error_msg)
    
    # Clean the text (keeping <script> and <style>)
    cleaned = clean_text(html_content)
    
    # 2) If the user requested a text preview
    if "Clean Text Preview" in modes:
        clean_text_result = preview_clean_text(cleaned, max_chars=500)
    
    # 3) Summarization
    if "Summarization" in modes:
        result = summarize_text(cleaned)
        # If the result starts with "Error", we can highlight it
        if isinstance(result, str) and "Error" in result:
            summarization_result = f"**Error during summarization**: {result}"
        else:
            summarization_result = result
    
    # 4) Sentiment Analysis
    if "Sentiment Analysis" in modes:
        result = analyze_sentiment(cleaned)
        if isinstance(result, str) and "Error" in result:
            sentiment_result = f"**Error during sentiment analysis**: {result}"
        else:
            sentiment_result = f"**Predicted Sentiment**: {result}"
    
    # 5) Topic Detection
    if "Topic Detection" in modes:
        topics = detect_topic(cleaned)
        # Check if there's an error
        if isinstance(topics, dict) and "error" in topics:
            topics_result = f"**Error during topic detection**: {topics['error']}"
        else:
            # Format the topics into a readable string
            formatted = ""
            for t, score in topics.items():
                formatted += f"- **{t}**: {score:.2f}\n"
            topics_result = formatted if formatted else "No topics detected."
    
    return (clean_text_result, summarization_result, sentiment_result, topics_result)

def build_app():
    with gr.Blocks(title="Smart Web Analyzer Plus") as demo:
        gr.Markdown("## Smart Web Analyzer Plus\n"
                    "Analyze web content for **summarization**, **sentiment**, and **topics**. "
                    "Choose your analysis modes and enter a URL below.")

        with gr.Row():
            url_input = gr.Textbox(
                label="Enter URL",
                placeholder="https://example.com",
                lines=1
            )
            mode_selector = gr.CheckboxGroup(
                label="Select Analysis Modes",
                choices=["Clean Text Preview", "Summarization", "Sentiment Analysis", "Topic Detection"],
                value=["Clean Text Preview", "Summarization", "Sentiment Analysis", "Topic Detection"]
            )

        # We'll display results in separate tabs for clarity
        with gr.Tabs():
            with gr.Tab("Clean Text Preview"):
                preview_output = gr.Markdown()
            with gr.Tab("Summarization"):
                summary_output = gr.Markdown()
            with gr.Tab("Sentiment Analysis"):
                sentiment_output = gr.Markdown()
            with gr.Tab("Topic Detection"):
                topic_output = gr.Markdown()
        
        analyze_button = gr.Button("Analyze")
        
        # The "analyze_url" function returns a tuple of four strings
        analyze_button.click(
            fn=analyze_url,
            inputs=[url_input, mode_selector],
            outputs=[preview_output, summary_output, sentiment_output, topic_output]
        )
        
        # Example URLs
        gr.Markdown("### Example URLs")
        gr.Examples(
            examples=[
                ["https://www.artificialintelligence-news.com/2024/02/14/openai-anthropic-google-white-house-red-teaming/"],
                ["https://www.artificialintelligence-news.com/2024/02/13/ai-21-labs-wordtune-chatgpt-plugin/"]
            ],
            inputs=url_input,
            label="Click an example to analyze"
        )
    
    return demo

if __name__ == "__main__":
    demo = build_app()
    demo.launch()
