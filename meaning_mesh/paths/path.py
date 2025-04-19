"""Path definitions for semantic routing."""

import asyncio
from typing import List, Optional, Dict, Any, Callable, Union, Awaitable
from uuid import uuid4


class Path:
    """
    Represents a destination with example phrases that define its semantic domain.
    
    A Path is a core concept in MeaningMesh that defines where text can be routed
    based on semantic similarity to example phrases.
    """
    
    def __init__(
        self,
        name: str,
        examples: List[str],
        handler: Optional[Callable[[str, Dict[str, Any]], Union[Any, Awaitable[Any]]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None
    ):
        """
        Initialize a Path.
        
        Args:
            name: Human-readable name for the path
            examples: List of example phrases that represent this path's domain
            handler: Optional function to call when text is routed to this path
            metadata: Optional additional information about the path
            id: Unique identifier (generated if not provided)
        """
        self.name = name
        self.examples = examples
        self.handler = handler
        self.metadata = metadata or {}
        self.id = id or str(uuid4())
        self.embeddings: List[List[float]] = []
        
    def __repr__(self) -> str:
        return f"Path(name='{self.name}', examples={len(self.examples)})"
    
    def __str__(self) -> str:
        return f"{self.name} (id: {self.id})"
    
    async def handle(self, text: str, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Process text that has been routed to this path.
        
        Args:
            text: The input text
            context: Optional context information
            
        Returns:
            Result of the handler function, or None if no handler is set
        """
        if not self.handler:
            return None
            
        context = context or {}
        result = self.handler(text, context)
        
        # If handler returns a coroutine, await it
        if asyncio.iscoroutine(result):
            return await result
            
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the path to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the path
        """
        return {
            "id": self.id,
            "name": self.name,
            "examples": self.examples,
            "metadata": self.metadata,
            # Note: handler and embeddings are not serialized
        }
    
    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any],
        handler: Optional[Callable[[str, Dict[str, Any]], Union[Any, Awaitable[Any]]]] = None
    ) -> "Path":
        """
        Create a Path instance from a dictionary.
        
        Args:
            data: Dictionary with path data
            handler: Optional handler function
            
        Returns:
            Path instance
        """
        return cls(
            name=data["name"],
            examples=data["examples"],
            handler=handler,
            metadata=data.get("metadata", {}),
            id=data.get("id")
        )
