from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
import json

class WebAnalyzer:
    def __init__(self):
        # Initialize models
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.classifier = pipeline("text-classification", 
                                 model="nlptown/bert-base-multilingual-uncased-sentiment")
        
    def clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def fetch_content(self, url: str) -> Dict[str, Any]:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            for tag in soup(['script', 'style', 'meta']):
                tag.decompose()
                
            return {
                'title': soup.title.string if soup.title else "No title found",
                'content': self.clean_text(soup.get_text())
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze(self, text: str, mode: str = "analyze") -> str:
        try:
            if len(text) < 100:
                return json.dumps({
                    "status": "error",
                    "message": "Text too short for analysis"
                })
            
            # Get content
            content = text
            if text.startswith(('http://', 'https://')):
                result = self.fetch_content(text)
                if 'error' in result:
                    return json.dumps(result)
                content = result['content']
                
            # Process based on mode
            if mode == "analyze":
                summary = self.summarizer(content[:1024], max_length=100, min_length=30)[0]['summary_text']
                sentiment = self.classifier(content[:512])[0]
                
                return json.dumps({
                    "status": "success",
                    "summary": summary,
                    "sentiment": sentiment['label'],
                    "stats": {
                        "length": len(content),
                        "words": len(content.split())
                    }
                })
                
            elif mode == "summarize":
                summary = self.summarizer(content[:1024], max_length=100, min_length=30)[0]['summary_text']
                return json.dumps({
                    "status": "success",
                    "summary": summary
                })
                
            elif mode == "sentiment":
                sentiment = self.classifier(content[:512])[0]
                return json.dumps({
                    "status": "success",
                    "sentiment": sentiment['label']
                })
                
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Unknown mode: {mode}"
                })
                
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": str(e)
            })