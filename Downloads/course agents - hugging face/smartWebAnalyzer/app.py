import gradio as gr
from analyzer import WebAnalyzer

def analyze_content(url: str, mode: str) -> str:
    """Process content and return formatted results."""
    try:
        analyzer = WebAnalyzer()
        results = analyzer.analyze(url, mode)
        
        if "error" in results:
            return f"Error: {results['error']}"
            
        # Format output
        output = f"Title: {results.get('title', 'N/A')}\n\n"
        
        if "summary" in results:
            output += f"Summary:\n{results['summary']}\n\n"
            
        if "sentiment" in results:
            output += f"Sentiment: {results['sentiment']}\n\n"
            
        if "stats" in results:
            stats = results["stats"]
            output += f"Statistics:\n"
            output += f"- Words: {stats['words']}\n"
            output += f"- Characters: {stats['chars']}\n"
            
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# Create interface
with gr.Blocks(title="Web Content Analyzer") as demo:
    gr.Markdown("# Web Content Analyzer")
    with gr.Row():
        url_input = gr.Textbox(
            label="URL or Text",
            placeholder="Enter URL or paste text"
        )
        mode = gr.Radio(
            choices=["analyze", "summarize"],
            label="Mode",
            value="analyze"
        )
    analyze_btn = gr.Button("Analyze")
    output = gr.Textbox(label="Results", lines=10)
    
    # Handle analysis
    analyze_btn.click(
        fn=analyze_content,
        inputs=[url_input, mode],
        outputs=output
    )

demo.launch()