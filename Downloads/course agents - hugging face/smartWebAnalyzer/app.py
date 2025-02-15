import gradio as gr
from analyzer import WebAnalyzer
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile
from fpdf import FPDF
import os

class WebAnalyzerUI:
    def __init__(self):
        self.analyzer = WebAnalyzer()
        self.cache = {}

    def format_analysis_results(self, results, theme):
        """Convert JSON results to visual HTML output"""
        if isinstance(results, str):
            results = json.loads(results)
            
        dark_mode = "background-color: #1a1a1a; color: #ffffff;" if theme == "dark" else ""
        
        html = f"""
        <div style="{dark_mode} padding: 20px; border-radius: 10px;">
            <h2 style="color: {'#fff' if theme == 'dark' else '#000'};">Analysis Results</h2>
        """
        
        if results.get("stats"):
            html += f"""
            <div style="margin: 20px 0;">
                <h3>📊 Content Statistics</h3>
                <p>Words: {results['stats']['words']}</p>
                <p>Characters: {results['stats']['characters']}</p>
                <p>Reading Time: {results['stats']['reading_time']}</p>
            </div>
            """
            
        if results.get("sentiment_analysis"):
            sentiment_data = results["sentiment_analysis"]
            fig = go.Figure()
            sections = []
            scores = []
            
            for section in sentiment_data.get("sections", []):
                sections.append(f"Section {section['section']}")
                scores.append(section['score'])
                
            fig.add_trace(go.Bar(
                x=sections,
                y=scores,
                marker_color='rgb(55, 83, 109)',
                text=scores,
                textposition='auto',
            ))
            
            fig.update_layout(
                title='Sentiment Analysis by Section',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#fff' if theme == "dark" else '#000'
            )
            
            html += f"""
            <div style="margin: 20px 0;">
                <h3>😊 Sentiment Analysis</h3>
                {fig.to_html(full_html=False)}
            </div>
            """
            
        if results.get("summary"):
            html += f"""
            <div style="margin: 20px 0;">
                <h3>📝 Summary</h3>
                <p>{results['summary']}</p>
            </div>
            """
            
        html += "</div>"
        return html

    def generate_pdf_report(self, results):
        """Generate PDF report from analysis results"""
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Web Content Analysis Report', 0, 1, 'C')
        pdf.line(10, 30, 200, 30)
        
        # Date
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        
        if isinstance(results, str):
            results = json.loads(results)
        
        # Content
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Statistics:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        if results.get("stats"):
            for key, value in results["stats"].items():
                pdf.cell(0, 10, f'{key.replace("_", " ").title()}: {value}', 0, 1)
        
        if results.get("summary"):
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, 'Summary:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 10, results["summary"])
        
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, 'analysis_report.pdf')
        pdf.output(pdf_path)
        
        return pdf_path

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Default()) as interface:
            gr.Markdown("# 🚀 Smart Web Analyzer Plus")
            
            with gr.Row():
                theme = gr.Radio(
                    choices=["light", "dark"],
                    value="light",
                    label="Theme",
                    interactive=True
                )
            
            with gr.Tabs():
                with gr.Tab("Analysis"):
                    with gr.Row():
                        with gr.Column():
                            url_input = gr.Textbox(
                                label="URL or Text to Analyze",
                                placeholder="Enter URL or paste text",
                                lines=5
                            )
                            mode = gr.Radio(
                                choices=["analyze", "sentiment", "summarize", "topics"],
                                value="analyze",
                                label="Analysis Mode"
                            )
                            with gr.Row():
                                analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                                clear_btn = gr.Button("🗑️ Clear")

                        with gr.Column():
                            results = gr.HTML(label="Analysis Results")
                            progress = gr.Textbox(label="Status", value="Ready")

                with gr.Tab("Preview"):
                    preview = gr.Textbox(
                        label="Content Preview",
                        lines=10,
                        interactive=False
                    )

                with gr.Tab("Report"):
                    report_btn = gr.Button("📥 Generate PDF Report")
                    pdf_output = gr.File(label="Download Report")

            # Examples
            gr.Examples(
                examples=[
                    ["https://www.artificialintelligence-news.com/2024/02/14/openai-anthropic-google-white-house-red-teaming/", "analyze", "light"],
                    ["https://www.artificialintelligence-news.com/2024/02/13/ai-21-labs-wordtune-chatgpt-plugin/", "sentiment", "light"]
                ],
                inputs=[url_input, mode, theme],
                outputs=[results, preview, progress],
                fn=self.process_content,
                cache_examples=True
            )

            # Handle theme changes
            theme.change(
                fn=lambda t: gr.update(theme=gr.themes.Default() if t == "light" else gr.themes.Soft()),
                inputs=[theme],
                outputs=[interface]
            )

            # Wire up buttons
            analyze_btn.click(
                fn=self.process_content,
                inputs=[url_input, mode, theme],
                outputs=[results, preview, progress]
            )

            clear_btn.click(
                fn=lambda: (None, "", "Ready"),
                inputs=[],
                outputs=[results, preview, progress]
            )

            report_btn.click(
                fn=self.generate_pdf_report,
                inputs=[results],
                outputs=[pdf_output]
            )

        return interface

    def process_content(self, text, mode, theme):
        """Process content with progress updates"""
        try:
            # Use cached results if available
            cache_key = f"{text}_{mode}"
            if cache_key in self.cache:
                return (
                    self.format_analysis_results(self.cache[cache_key], theme),
                    "Content preview unavailable for cached results",
                    "Using cached results"
                )

            # Analyze content
            results = self.analyzer.analyze(text, mode)
            results_dict = json.loads(results)
            
            # Cache results
            self.cache[cache_key] = results_dict
            
            # Get preview
            preview = results_dict.get("content", "")[:1000] + "..."
            
            return (
                self.format_analysis_results(results_dict, theme),
                preview,
                "Analysis complete!"
            )
            
        except Exception as e:
            return None, "", f"Error: {str(e)}"

# Create and launch interface
if __name__ == "__main__":
    analyzer_ui = WebAnalyzerUI()
    demo = analyzer_ui.create_interface()
    demo.launch()