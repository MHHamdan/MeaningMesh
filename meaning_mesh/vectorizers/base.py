"""Base interface for text vectorizers."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union


class Vectorizer(ABC):
    """
    Base class for text vectorizers that convert text to embeddings.
    
    This abstract class defines the interface for all vectorizer implementations.
    Implement this class to add support for new embedding providers.
    """
    
    @abstractmethod
    async def vectorize(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Convert a list of texts to their vector embeddings.
        
        Args:
            texts: List of text strings to vectorize
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of vector embeddings (each as a list of floats)
        """
        pass
    
    @abstractmethod
    async def vectorize_single(self, text: str, **kwargs) -> List[float]:
        """
        Convert a single text to its vector embedding.
        
        Args:
            text: Text string to vectorize
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Vector embedding as a list of floats
        """
        pass
    
    @property
    def embedding_dimensions(self) -> Optional[int]:
        """
        Get the dimensionality of the embeddings produced by this vectorizer.
        
        Returns:
            Number of dimensions in the embedding, or None if unknown/variable
        """
        return None
    
    @property
    def provider_name(self) -> str:
        """
        Get the name of the embedding provider.
        
        Returns:
            Name of the embedding provider (e.g., "openai", "huggingface")
        """
        return self.__class__.__name__.replace("Vectorizer", "").lower()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the configuration of this vectorizer.
        
        Returns:
            Dictionary with the vectorizer configuration
        """
        return {"provider": self.provider_name}


# Factory function to create vectorizers
def create_vectorizer(
    provider: str = "openai",
    api_key: Optional[str] = None,
    **kwargs
) -> Vectorizer:
    """
    Create a vectorizer instance based on the provider name.
    
    Args:
        provider: Name of the embedding provider 
                 ("openai", "huggingface", "cohere", or "mock")
        api_key: API key for the provider (if needed)
        **kwargs: Additional provider-specific parameters
        
    Returns:
        Vectorizer instance
    
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "openai":
        from .openai import OpenAIVectorizer
        return OpenAIVectorizer(api_key=api_key, **kwargs)
    
    elif provider == "huggingface":
        from .huggingface import HuggingFaceVectorizer
        return HuggingFaceVectorizer(**kwargs)
    
    elif provider == "cohere":
        from .cohere import CohereVectorizer
        return CohereVectorizer(api_key=api_key, **kwargs)
    
    elif provider == "mock":
        from .mock import MockVectorizer
        return MockVectorizer(**kwargs)
    
    else:
        raise ValueError(f"Unsupported vectorizer provider: {provider}")
