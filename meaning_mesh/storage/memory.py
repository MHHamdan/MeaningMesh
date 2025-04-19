"""In-memory implementation of the embedding store."""

import json
import os
from typing import List, Dict, Any, Optional
import copy

from ..paths.path import Path
from .base import EmbeddingStore


class InMemoryEmbeddingStore(EmbeddingStore):
    """
    In-memory implementation of EmbeddingStore.
    
    This class stores paths and embeddings in memory. It's suitable for
    development, testing, and small-scale deployments.
    """
    
    def __init__(self):
        """Initialize an empty in-memory store."""
        self.paths: Dict[str, Path] = {}
        self.embeddings: Dict[str, List[List[float]]] = {}
    
    async def store_path(self, path: Path, embeddings: List[List[float]]) -> None:
        """
        Store embeddings for a path in memory.
        
        Args:
            path: The path object
            embeddings: List of embeddings for the path's examples
        """
        self.paths[path.id] = path
        self.embeddings[path.id] = embeddings
        # Also store the embeddings in the path object for convenience
        path.embeddings = embeddings
    
    async def get_path(self, path_id: str) -> Optional[Path]:
        """
        Get a path by its ID.
        
        Args:
            path_id: ID of the path to retrieve
            
        Returns:
            Path object if found, None otherwise
        """
        return self.paths.get(path_id)
    
    async def get_all_paths(self) -> List[Path]:
        """
        Get all stored paths.
        
        Returns:
            List of Path objects
        """
        return list(self.paths.values())
    
    async def get_path_embeddings(self, path_id: str) -> List[List[float]]:
        """
        Get embeddings for a specific path.
        
        Args:
            path_id: ID of the path
            
        Returns:
            List of embeddings for the path
        """
        return self.embeddings.get(path_id, [])
    
    async def get_all_embeddings(self) -> Dict[str, List[List[float]]]:
        """
        Get all stored embeddings, keyed by path ID.
        
        Returns:
            Dictionary mapping path IDs to lists of embeddings
        """
        return copy.deepcopy(self.embeddings)
    
    async def delete_path(self, path_id: str) -> bool:
        """
        Delete a path and its embeddings.
        
        Args:
            path_id: ID of the path to delete
            
        Returns:
            True if deleted, False if not found
        """
        if path_id in self.paths:
            del self.paths[path_id]
            if path_id in self.embeddings:
                del self.embeddings[path_id]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all stored paths and embeddings."""
        self.paths.clear()
        self.embeddings.clear()
    
    async def save(self, file_path: Optional[str] = None) -> None:
        """
        Save the store to a JSON file.
        
        Args:
            file_path: Path to save to (default: "meaningmesh_store.json")
        """
        file_path = file_path or "meaningmesh_store.json"
        
        data = {
            "paths": {
                path_id: path.to_dict() 
                for path_id, path in self.paths.items()
            },
            "embeddings": self.embeddings
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
    
    async def load(self, file_path: Optional[str] = None) -> None:
        """
        Load the store from a JSON file.
        
        Args:
            file_path: Path to load from (default: "meaningmesh_store.json")
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        file_path = file_path or "meaningmesh_store.json"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Store file not found: {file_path}")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        self.clear()
        
        # Load paths
        for path_id, path_data in data["paths"].items():
            path = Path.from_dict(path_data)
            self.paths[path_id] = path
        
        # Load embeddings
        self.embeddings = data["embeddings"]
        
        # Update path embeddings
        for path_id, embeddings in self.embeddings.items():
            if path_id in self.paths:
                self.paths[path_id].embeddings = embeddings
