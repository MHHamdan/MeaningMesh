"""Mock vectorizer for testing purposes."""

import random
import hashlib
from typing import List, Optional, Dict, Any

from .base import Vectorizer


class MockVectorizer(Vectorizer):
    """
    Mock vectorizer for testing that produces deterministic embeddings.
    
    This vectorizer creates fake embeddings based on the hash of the text,
    making it useful for testing without external dependencies.
    """
    
    def __init__(
        self, 
        dimensions: int = 384,
        seed: int = 42,
        deterministic: bool = True
    ):
        """
        Initialize a MockVectorizer.
        
        Args:
            dimensions: Number of dimensions for the mock embeddings
            seed: Random seed for reproducibility
            deterministic: Whether to use deterministic embeddings based on text hash
        """
        self.dimensions = dimensions
        self.random = random.Random(seed)
        self.deterministic = deterministic
        self._embedding_cache: Dict[str, List[float]] = {}
    
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
        
        if self.deterministic:
            # Create a deterministic embedding based on the text hash
            embedding = self._create_deterministic_embedding(text)
        else:
            # Create a random embedding
            embedding = [self.random.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
            
            # Normalize to unit length
            magnitude = sum(x * x for x in embedding) ** 0.5
            if magnitude > 0:
                embedding = [x / magnitude for x in embedding]
        
        # Cache the result
        self._embedding_cache[text] = embedding
        return embedding
    
    def _create_deterministic_embedding(self, text: str) -> List[float]:
        """
        Create a deterministic embedding based on text hash.
        
        Args:
            text: Input text
            
        Returns:
            Deterministic embedding vector
        """
        # Get a hash of the text
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Use the hash to seed a random generator
        seed = int.from_bytes(hash_bytes[:4], byteorder='big')
        rand = random.Random(seed)
        
        # Generate deterministic vector
        embedding = [rand.uniform(-1.0, 1.0) for _ in range(self.dimensions)]
        
        # Normalize to unit length
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        
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
            "deterministic": self.deterministic
        }
