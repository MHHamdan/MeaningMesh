import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import torch
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class WebAnalyzer:
    def __init__(self):
        """Initialize analyzers and download required NLTK data."""
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
        except:
            pass

        # Set device
        self.device = 0 if torch.cuda.is_available() else -1
        
        # Initialize models
        self.summarizer = pipeline("summarization", 
                                 model="facebook/bart-large-cnn", 
                                 device=self.device)
        self.classifier = pipeline("text-classification", 
                                 model="nlptown/bert-base-multilingual-uncased-sentiment",
                                 device=self.device)
        self.zero_shot = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=self.device)

        # Topic categories
        self.topic_categories = [
            "Technology", "Business", "Science", "Politics",
            "Health", "Entertainment", "Education", "Environment"
        ]

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove special characters and extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """Generate comprehensive text statistics."""
        # Tokenize text
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        words_no_stop = [w for w in words if w.isalnum() and w not in stop_words]
        
        # Calculate statistics
        stats = {
            "characters": len(text),
            "words": len(words),
            "sentences": len(sentences),
            "paragraphs": len(text.split('\n\n')),
            "avg_word_length": np.mean([len(w) for w in words_no_stop]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "reading_time": f"{len(words) // 200} minutes",  # Assuming 200 WPM reading speed
            "vocabulary_size": len(set(words_no_stop))
        }
        
        # Get most common words
        word_freq = Counter(words_no_stop).most_common(10)
        stats["common_words"] = dict(word_freq)
        
        return stats

    def fetch_web_content(self, url: str) -> Dict[str, str]:
        """Fetch and parse web content with advanced error handling."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'ad']):
                element.decompose()
                
            # Extract metadata
            metadata = {
                'title': soup.title.string if soup.title else "No title found",
                'description': next((meta.get('content', '') for meta in soup.find_all('meta', {'name': 'description'})), ''),
                'author': next((meta.get('content', '') for meta in soup.find_all('meta', {'name': 'author'})), 'Unknown'),
                'published_date': next((meta.get('content', '') for meta in soup.find_all('meta', {'name': ['date', 'published_date']})), 'Unknown')
            }
            
            # Get main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            content = self.clean_text(main_content.get_text()) if main_content else ""
            
            return {
                'metadata': metadata,
                'content': content,
                'status': 'success'
            }
            
        except requests.RequestException as e:
            return {
                'status': 'error',
                'message': f"Failed to fetch content: {str(e)}"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Error processing content: {str(e)}"
            }

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Perform detailed sentiment analysis."""
        # Split into paragraphs for section analysis
        paragraphs = [p for p in text.split('\n') if len(p.strip()) > 50]
        sentiment_results = {
            "sections": [],
            "total_sections": len(paragraphs)
        }
        
        # Analyze each section
        for idx, para in enumerate(paragraphs[:5]):  # Limit to 5 sections
            result = self.classifier(para[:512])[0]
            score = int(result['label'][0])
            sentiment_results["sections"].append({
                "section": idx + 1,
                "text": para[:100] + "...",
                "sentiment": ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"][score-1],
                "score": score,
                "confidence": round(result['score'], 3)
            })
            
        # Calculate overall sentiment
        avg_score = np.mean([section['score'] for section in sentiment_results['sections']])
        sentiment_results["overall"] = {
            "score": round(avg_score, 2),
            "label": ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"][int(round(avg_score))-1]
        }
        
        return sentiment_results

    def analyze_topics(self, text: str) -> Dict[str, Any]:
        """Perform topic analysis and keyword extraction."""
        # Zero-shot classification
        topic_results = self.zero_shot(text[:1024], self.topic_categories)
        
        # Extract keywords using TF-IDF
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            keywords = [{"word": word, "score": round(score, 3)} 
                       for word, score in zip(feature_names, scores)]
        except:
            keywords = []
        
        return {
            "main_topics": [
                {"topic": label, "confidence": round(score, 3)}
                for label, score in zip(topic_results['labels'], topic_results['scores'])
                if score > 0.1
            ],
            "keywords": keywords
        }

    def analyze(self, input_text: str, mode: str = "analyze") -> str:
        """Main analysis function with comprehensive results."""
        try:
            # Determine if input is URL or text
            is_url = input_text.startswith(('http://', 'https://'))
            content_data = self.fetch_web_content(input_text) if is_url else {'content': input_text, 'status': 'success'}
            
            if content_data['status'] == 'error':
                return json.dumps(content_data)
                
            text = content_data.get('content', '')
            if len(text) < 100:
                return json.dumps({
                    "status": "error",
                    "message": "Insufficient content for analysis (minimum 100 characters required)"
                })
            
            # Base results with text statistics
            results = {
                "status": "success",
                "stats": self.get_text_stats(text),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add metadata if available
            if is_url and 'metadata' in content_data:
                results['metadata'] = content_data['metadata']
            
            # Mode-specific analysis
            if mode in ["analyze", "summarize"]:
                summary = self.summarizer(text[:1024], 
                                       max_length=150,
                                       min_length=30,
                                       do_sample=False)[0]['summary_text']
                results['summary'] = summary
                
            if mode in ["analyze", "sentiment"]:
                results['sentiment_analysis'] = self.analyze_sentiment(text)
                
            if mode in ["analyze", "topics"]:
                results['topic_analysis'] = self.analyze_topics(text)
            
            return json.dumps(results, indent=2)
            
        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            })