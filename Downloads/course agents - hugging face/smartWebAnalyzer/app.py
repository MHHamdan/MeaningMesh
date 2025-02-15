import gradio as gr
import os
from smartWebAnalyzer.smart_web_analyzer import WebAnalyzer
from huggingface_hub import upload_file

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
with gr.Blocks(title="Smart Web Analyzer Plus") as demo:
    gr.Markdown("# 🚀 Smart Web Analyzer Plus")
    gr.Markdown("""
    Advanced content analysis with AI-powered insights:
    * 📊 Comprehensive Analysis
    * 😊 Detailed Sentiment Analysis
    * 📝 Smart Summarization
    * 🎯 Topic Detection
    """)
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


# When pushing the tool
WebAnalyzer.push_to_hub(
    "MHamdan/smart-web-analyzer-plus",  # New repo name
    token=os.environ["HF_Repo_API"],
    private=False
)

# When uploading files
for filename in ["app.py", "requirements.txt"]:
    upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id="MHamdan/smart-web-analyzer-plus",  # New repo name
        repo_type="space",
        token=os.environ["HF_Repo_API"]
    )
 
#web_analyzer = load_tool("MHamdan/smart-web-analyzer-plus", trust_remote_code=True)   
demo.launch()