"""Vectorizer implementation using HuggingFace models."""

import asyncio
from typing import List, Optional, Dict, Any
import numpy as np

from .base import Vectorizer


class HuggingFaceVectorizer(Vectorizer):
    """Vectorizer that uses HuggingFace's embedding models."""
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the HuggingFace vectorizer.
        
        Args:
            model_name: Name of the HuggingFace model to use
            device: Device to run the model on ("cpu" or "cuda")
            batch_size: Batch size for processing
            **kwargs: Additional parameters passed to the SentenceTransformer
        """
        # Import here to avoid dependency if not used
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "HuggingFaceVectorizer requires the sentence-transformers package. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.batch_size = batch_size
        self.model_name = model_name
        self._dimensions = None
    
    async def vectorize(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Convert a list of texts to their vector embeddings using HuggingFace.
        
        Args:
            texts: List of text strings to vectorize
            **kwargs: Additional parameters passed to the encode method
            
        Returns:
            List of vector embeddings
        """
        # Run in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: self.model.encode(
                texts, 
                batch_size=self.batch_size, 
                show_progress_bar=False,
                **kwargs
            )
        )
        
        # Store dimensions if not already set
        if self._dimensions is None and len(embeddings) > 0:
            self._dimensions = embeddings.shape[1]
        
        # Convert numpy array to list of lists
        return embeddings.tolist()
    
    async def vectorize_single(self, text: str, **kwargs) -> List[float]:
        """
        Convert a single text to its vector embedding using HuggingFace.
        
        Args:
            text: Text string to vectorize
            **kwargs: Additional parameters passed to the encode method
            
        Returns:
            Vector embedding
        """
        embeddings = await self.vectorize([text], **kwargs)
        return embeddings[0]
    
    @property
    def embedding_dimensions(self) -> Optional[int]:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Number of dimensions in the embedding, or None if not yet determined
        """
        if self._dimensions is None:
            # Try to determine dimensions
            try:
                sample_embedding = self.model.encode(["sample text"])
                self._dimensions = sample_embedding.shape[1]
            except Exception:
                pass
        
        return self._dimensions
    
    @property
    def provider_name(self) -> str:
        """
        Get the name of the embedding provider.
        
        Returns:
            Name of the embedding provider
        """
        return "huggingface"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this vectorizer.
        
        Returns:
            Dictionary with the vectorizer configuration
        """
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "dimensions": self.embedding_dimensions,
            "batch_size": self.batch_size
        }
