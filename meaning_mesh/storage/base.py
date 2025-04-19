"""Base classes for embedding storage."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Import Path class here to avoid circular imports
from ..paths.path import Path


class EmbeddingStore(ABC):
    """
    Base class for storing and retrieving path embeddings.
    
    This abstract class defines the interface for all storage implementations.
    Implement this class to add support for new storage backends.
    """
    
    @abstractmethod
    async def store_path(self, path: Path, embeddings: List[List[float]]) -> None:
        """
        Store embeddings for a path.
        
        Args:
            path: The path object
            embeddings: List of embeddings for the path's examples
        """
        pass
    
    @abstractmethod
    async def get_path(self, path_id: str) -> Optional[Path]:
        """
        Get a path by its ID.
        
        Args:
            path_id: ID of the path to retrieve
            
        Returns:
            Path object if found, None otherwise
        """
        pass
    
    @abstractmethod
    async def get_all_paths(self) -> List[Path]:
        """
        Get all stored paths.
        
        Returns:
            List of Path objects
        """
        pass
    
    @abstractmethod
    async def get_path_embeddings(self, path_id: str) -> List[List[float]]:
        """
        Get embeddings for a specific path.
        
        Args:
            path_id: ID of the path
            
        Returns:
            List of embeddings for the path
        """
        pass
    
    @abstractmethod
    async def get_all_embeddings(self) -> Dict[str, List[List[float]]]:
        """
        Get all stored embeddings, keyed by path ID.
        
        Returns:
            Dictionary mapping path IDs to lists of embeddings
        """
        pass
    
    @abstractmethod
    async def delete_path(self, path_id: str) -> bool:
        """
        Delete a path and its embeddings.
        
        Args:
            path_id: ID of the path to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored paths and embeddings."""
        pass
    
    @abstractmethod
    async def save(self, file_path: Optional[str] = None) -> None:
        """
        Save the store to a file or persistence layer.
        
        Args:
            file_path: Optional path to save to (implementation-specific)
        """
        pass
    
    @abstractmethod
    async def load(self, file_path: Optional[str] = None) -> None:
        """
        Load the store from a file or persistence layer.
        
        Args:
            file_path: Optional path to load from (implementation-specific)
        """
        pass


def create_store(store_type: str = "memory", **kwargs) -> EmbeddingStore:
    """
    Create a store instance based on the store type.
    
    Args:
        store_type: Type of store to create ("memory", "json", etc.)
        **kwargs: Additional store-specific parameters
        
    Returns:
        EmbeddingStore instance
        
    Raises:
        ValueError: If the store type is not supported
    """
    store_type = store_type.lower()
    
    if store_type == "memory":
        from .memory import InMemoryEmbeddingStore
        return InMemoryEmbeddingStore(**kwargs)
    else:
        raise ValueError(f"Unsupported store type: {store_type}")
