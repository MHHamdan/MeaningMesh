import gradio as gr
import json
from smartWebAnalyzer import WebAnalyzer

# Initialize analyzer
analyzer = WebAnalyzer()

def process_content(text: str, mode: str) -> str:
    """Process content and return results."""
    try:
        return analyzer.analyze(text, mode)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })

# Create interface
def create_interface():
    with gr.Blocks(title="Web Content Analyzer") as iface:
        gr.Markdown("# 🌐 Web Content Analyzer")
        gr.Markdown("""
        Analyze web content or text with AI:
        * 📝 Smart Summarization
        * 😊 Sentiment Analysis
        * 📊 Content Statistics
        """)
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="URL or Text",
                    placeholder="Enter URL or paste text",
                    lines=5
                )
                mode = gr.Radio(
                    choices=["analyze", "summarize", "sentiment"],
                    value="analyze",
                    label="Analysis Mode"
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column():
                output = gr.JSON(label="Results")
        
        # Examples
        gr.Examples(
            examples=[
                ["https://www.bbc.com/news", "analyze"],
                ["This is a test message to analyze sentiment.", "sentiment"]
            ],
            inputs=[input_text, mode],
            outputs=output,
            fn=process_content,
            cache_examples=True
        )
        
        analyze_btn.click(
            fn=process_content,
            inputs=[input_text, mode],
            outputs=output
        )
    
    return iface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()