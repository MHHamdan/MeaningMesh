"""Vectorizer implementation using OpenAI embeddings."""

import openai
from typing import List, Optional, Dict, Any

from .base import Vectorizer


class OpenAIVectorizer(Vectorizer):
    """Vectorizer that uses OpenAI's embedding models."""
    
    # Model dimension mapping (model_name -> dimension)
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI vectorizer.
        
        Args:
            api_key: OpenAI API key (optional if already set in env)
            model: Name of the embedding model to use
            dimensions: Optional override for embedding dimensions
            **kwargs: Additional parameters passed to the OpenAI client
        """
        self.client = openai.OpenAI(api_key=api_key, **kwargs) if api_key else openai.OpenAI(**kwargs)
        self.model = model
        self._dimensions = dimensions or self.MODEL_DIMENSIONS.get(model)
    
    async def vectorize(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Convert a list of texts to their vector embeddings using OpenAI.
        
        Args:
            texts: List of text strings to vectorize
            **kwargs: Additional parameters passed to the embedding request
            
        Returns:
            List of vector embeddings
        """
        # Process in batches to avoid API limits
        max_batch_size = 20
        results = []
        
        # Process in batches
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
                **kwargs
            )
            batch_embeddings = [item.embedding for item in response.data]
            results.extend(batch_embeddings)
            
        return results
    
    async def vectorize_single(self, text: str, **kwargs) -> List[float]:
        """
        Convert a single text to its vector embedding using OpenAI.
        
        Args:
            text: Text string to vectorize
            **kwargs: Additional parameters passed to the embedding request
            
        Returns:
            Vector embedding
        """
        response = await self.client.embeddings.create(
            input=[text],
            model=self.model,
            **kwargs
        )
        return response.data[0].embedding
    
    @property
    def embedding_dimensions(self) -> Optional[int]:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Number of dimensions in the embedding
        """
        return self._dimensions
    
    @property
    def provider_name(self) -> str:
        """
        Get the name of the embedding provider.
        
        Returns:
            Name of the embedding provider
        """
        return "openai"
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this vectorizer.
        
        Returns:
            Dictionary with the vectorizer configuration
        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "dimensions": self.embedding_dimensions
        }
