import gradio as gr
from analyzer import WebAnalyzer
import json

class WebAnalyzerApp:
    def __init__(self):
        self.analyzer = WebAnalyzer()
        self.cached_results = {}

    def process_content(self, text: str, mode: str) -> tuple:
        """Process content and return formatted results."""
        try:
            # Check cache
            cache_key = f"{text}_{mode}"
            if cache_key in self.cached_results:
                return (
                    self.cached_results[cache_key],  # Results
                    "Using cached results"  # Status
                )

            # Process new request
            results = self.analyzer.analyze(text, mode)
            
            # Cache results
            self.cached_results[cache_key] = results
            
            return results, "Analysis complete!"
            
        except Exception as e:
            return None, f"Error: {str(e)}"

    def create_interface(self):
        with gr.Blocks(title="Smart Web Analyzer") as iface:
            gr.Markdown("# 🌐 Smart Web Analyzer")
            
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
                    analyze_btn = gr.Button("Analyze")
                
                with gr.Column():
                    results = gr.JSON(label="Results")
                    status = gr.Textbox(label="Status", value="Ready")

            # Examples
            examples = [
                ["https://www.artificialintelligence-news.com/2024/02/14/openai-anthropic-google-white-house-red-teaming/", "analyze"],
                ["https://www.artificialintelligence-news.com/2024/02/13/ai-21-labs-wordtune-chatgpt-plugin/", "sentiment"]
            ]
            
            gr.Examples(
                examples=examples,
                inputs=[input_text, mode],
                outputs=[results, status],
                fn=self.process_content,
                cache_examples=True
            )

            # Handle analysis
            analyze_btn.click(
                fn=self.process_content,
                inputs=[input_text, mode],
                outputs=[results, status]
            )

        return iface

# Create and launch app
app = WebAnalyzerApp()
demo = app.create_interface()
demo.queue()  # Enable queuing
demo.launch(server_name="0.0.0.0", 
           server_port=7860,
           show_error=True)