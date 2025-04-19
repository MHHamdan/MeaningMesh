"""Mock vectorizer for testing purposes."""

import random
import hashlib
from typing import List, Optional, Dict, Any
import re

from .base import Vectorizer


class MockVectorizer(Vectorizer):
    """
    Mock vectorizer for testing that produces deterministic embeddings.
    
    This vectorizer creates fake embeddings based on semantic categories,
    making it useful for testing without external dependencies.
    """
    
    # Define semantic categories for keyword-based similarity
    SEMANTIC_CATEGORIES = {
        "weather": ["weather", "forecast", "rain", "sunny", "temperature", "umbrella", "cold", "hot", "climate"],
        "greeting": ["hello", "hi", "hey", "morning", "greetings", "howdy", "welcome", "sup"],
        "support": ["problem", "issue", "help", "support", "order", "package", "return", "account", "missing", "defective"]
    }
    
    def __init__(
        self, 
        dimensions: int = 384,
        seed: int = 42,
        semantic_boost: float = 0.7
    ):
        """
        Initialize a MockVectorizer.
        
        Args:
            dimensions: Number of dimensions for the mock embeddings
            seed: Random seed for reproducibility
            semantic_boost: How much to boost semantic category matches (0-1)
        """
        self.dimensions = dimensions
        self.random = random.Random(seed)
        self.semantic_boost = semantic_boost
        self._embedding_cache: Dict[str, List[float]] = {}
        
        # Pre-generate category base vectors
        self.category_vectors = {}
        for category in self.SEMANTIC_CATEGORIES:
            # Create a base vector for each category
            self.category_vectors[category] = [
                self.random.uniform(-1.0, 1.0) for _ in range(self.dimensions)
            ]
            
            # Normalize to unit length
            magnitude = sum(x * x for x in self.category_vectors[category]) ** 0.5
            if magnitude > 0:
                self.category_vectors[category] = [
                    x / magnitude for x in self.category_vectors[category]
                ]
    
    async def vectorize(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Create mock embeddings for a list of texts.
        
        Args:
            texts: List of text strings to vectorize
            **kwargs: Additional parameters (ignored)
            
        Returns:
            List of mock vector embeddings
        """
        return [await self.vectorize_single(text) for text in texts]
    
    async def vectorize_single(self, text: str, **kwargs) -> List[float]:
        """
        Create a mock embedding for a single text.
        
        Args:
            text: Text string to vectorize
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock vector embedding
        """
        # Check cache
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        # Determine semantic categories in the text
        text_lower = text.lower()
        
        # Calculate category scores based on word presence
        category_scores = {}
        for category, keywords in self.SEMANTIC_CATEGORIES.items():
            # Count matching keywords
            word_matches = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower))
            if word_matches > 0:
                category_scores[category] = min(1.0, word_matches / 3)  # Cap at 1.0
        
        # Create embedding based on categories
        if category_scores:
            # Blend category vectors based on scores
            total_score = sum(category_scores.values())
            embedding = [0.0] * self.dimensions
            
            # Mix category vectors
            for category, score in category_scores.items():
                weight = score / total_score
                for i in range(self.dimensions):
                    embedding[i] += self.category_vectors[category][i] * weight
                    
            # Add randomness for uniqueness
            for i in range(self.dimensions):
                # Add small random noise
                embedding[i] += (self.random.uniform(-0.1, 0.1) * (1 - self.semantic_boost))
        else:
            # No category match, generate random embedding
            embedding = [self.random.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        
        # Normalize to unit length
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
        # Cache the result
        self._embedding_cache[text] = embedding
        return embedding
    
    @property
    def embedding_dimensions(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Number of dimensions in the embedding
        """
        return self.dimensions
    
    @property
    def provider_name(self) -> str:
        """
        Get the name of the embedding provider.
        
        Returns:
            Name of the embedding provider
        """
        return "mock"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this vectorizer.
        
        Returns:
            Dictionary with the vectorizer configuration
        """
        return {
            "provider": self.provider_name,
            "dimensions": self.dimensions,
            "semantic_boost": self.semantic_boost
        }
