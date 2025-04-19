"""Vectorizer implementation using Cohere embeddings."""

from typing import List, Optional, Dict, Any

from .base import Vectorizer


class CohereVectorizer(Vectorizer):
    """Vectorizer that uses Cohere's embedding models."""
    
    # Model dimension mapping (model_name -> dimension)
    MODEL_DIMENSIONS = {
        "embed-english-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-v3.0": 1024,
        "embed-multilingual-light-v3.0": 384,
    }
    
    def __init__(
        self, 
        api_key: str,
        model: str = "embed-english-v3.0",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the Cohere vectorizer.
        
        Args:
            api_key: Cohere API key
            model: Name of the embedding model to use
            dimensions: Optional override for embedding dimensions
            **kwargs: Additional parameters passed to the Cohere client
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "CohereVectorizer requires the cohere package. "
                "Install it with: pip install cohere"
            )
        
        self.client = cohere.Client(api_key)
        self.model = model
        self._dimensions = dimensions or self.MODEL_DIMENSIONS.get(model)
        self.kwargs = kwargs
    
    async def vectorize(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Convert a list of texts to their vector embeddings using Cohere.
        
        Args:
            texts: List of text strings to vectorize
            **kwargs: Additional parameters passed to the embedding request
            
        Returns:
            List of vector embeddings
        """
        # Process in batches to avoid API limits
        max_batch_size = 96  # Cohere's limit
        results = []
        
        # Merge kwargs
        request_kwargs = self.kwargs.copy()
        request_kwargs.update(kwargs)
        
        # Process in batches
        for i in range(0, len(texts), max_batch_size):
            batch = texts[i:i + max_batch_size]
            
            # Use the asyncio event loop to run the blocking API call
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embed(
                    texts=batch,
                    model=self.model,
                    **request_kwargs
                )
            )
            
            batch_embeddings = response.embeddings
            results.extend(batch_embeddings)
            
        return results
    
    async def vectorize_single(self, text: str, **kwargs) -> List[float]:
        """
        Convert a single text to its vector embedding using Cohere.
        
        Args:
            text: Text string to vectorize
            **kwargs: Additional parameters passed to the embedding request
            
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
        return "cohere"
    
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
