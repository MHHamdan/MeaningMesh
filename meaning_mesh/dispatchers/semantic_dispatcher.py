"""Main dispatcher component for semantic routing."""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import asyncio

from ..paths.path import Path
from ..vectorizers.base import Vectorizer
from ..storage.base import EmbeddingStore
from ..utils.similarity import find_best_match, find_top_matches, SIMILARITY_FUNCTIONS


class DispatchResult:
    """
    Result of a dispatch operation.
    
    This class contains information about the result of dispatching a text
    to a path, including the matched path, confidence score, etc.
    """
    
    def __init__(
        self,
        path: Optional[Path],
        confidence: float,
        text: str,
        fallback_used: bool = False,
        matches: Optional[List[Tuple[Path, float]]] = None
    ):
        """
        Initialize a dispatch result.
        
        Args:
            path: The matched path, or None if no match
            confidence: Confidence score for the match
            text: The original input text
            fallback_used: Whether a fallback path was used
            matches: List of (path, score) tuples for all considered matches
        """
        self.path = path
        self.confidence = confidence
        self.text = text
        self.fallback_used = fallback_used
        self.matches = matches or []
    
    def __repr__(self) -> str:
        if self.path:
            return f"DispatchResult(path='{self.path.name}', confidence={self.confidence:.4f})"
        return f"DispatchResult(path=None, confidence={self.confidence:.4f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result to a dictionary.
        
        Returns:
            Dictionary representation of the dispatch result
        """
        return {
            "path": self.path.to_dict() if self.path else None,
            "confidence": self.confidence,
            "text": self.text,
            "fallback_used": self.fallback_used,
            "matches": [
                {"path": path.to_dict(), "score": score}
                for path, score in self.matches
            ]
        }


class SemanticDispatcher:
    """
    Main component for routing text to semantically similar paths.
    
    This class handles the process of matching input text to the most
    semantically similar path based on example phrases.
    """
    
    def __init__(
        self,
        vectorizer: Vectorizer,
        store: EmbeddingStore,
        similarity_fn: str = "cosine",
        confidence_threshold: float = 0.7,
        fallback_path: Optional[Path] = None
    ):
        """
        Initialize a SemanticDispatcher.
        
        Args:
            vectorizer: Vectorizer for converting text to embeddings
            store: Storage for paths and embeddings
            similarity_fn: Name of the similarity function to use
            confidence_threshold: Minimum confidence required for a match
            fallback_path: Optional path to use when confidence is below threshold
        """
        self.vectorizer = vectorizer
        self.store = store
        self.similarity_fn = similarity_fn
        self.confidence_threshold = confidence_threshold
        self.fallback_path = fallback_path
    
    async def register_path(self, path: Path) -> None:
        """
        Register a path with the dispatcher.
        
        Args:
            path: The path to register
        """
        # Vectorize the path's example phrases
        embeddings = await self.vectorizer.vectorize(path.examples)
        # Store the path and its embeddings
        await self.store.store_path(path, embeddings)
    
    async def dispatch(
        self, 
        text: str,
        context: Optional[Dict[str, Any]] = None,
        return_all_matches: bool = False,
        top_k: int = 3
    ) -> DispatchResult:
        """
        Dispatch text to the most semantically similar path.
        
        Args:
            text: The input text to dispatch
            context: Optional context information
            return_all_matches: Whether to return all considered matches
            top_k: Number of top matches to return if return_all_matches is True
            
        Returns:
            DispatchResult with the matched path and confidence
        """
        # Vectorize the input text
        query_embedding = await self.vectorizer.vectorize_single(text)
        
        # Get all stored paths
        paths = await self.store.get_all_paths()
        
        if not paths:
            # No paths registered
            return DispatchResult(None, 0.0, text)
        
        # Find the best matching path
        best_path = None
        best_score = -float('inf')
        all_matches = []
        
        for path in paths:
            path_embeddings = await self.store.get_path_embeddings(path.id)
            if not path_embeddings:
                continue
            
            # Find the best matching example within this path
            best_idx, score = find_best_match(
                query_embedding, 
                path_embeddings, 
                self.similarity_fn
            )
            
            if return_all_matches:
                all_matches.append((path, score))
                
            if score > best_score:
                best_score = score
                best_path = path
        
        # Sort matches by score if needed
        if return_all_matches:
            all_matches.sort(key=lambda x: x[1], reverse=True)
            all_matches = all_matches[:top_k]
        
        # Check confidence threshold
        if best_score >= self.confidence_threshold:
            return DispatchResult(best_path, best_score, text, matches=all_matches)
        elif self.fallback_path:
            # Use fallback path with the original confidence score
            return DispatchResult(
                self.fallback_path, 
                best_score, 
                text, 
                fallback_used=True,
                matches=all_matches
            )
        else:
            # No fallback, return the best path but indicate low confidence
            return DispatchResult(best_path, best_score, text, matches=all_matches)
    
    async def dispatch_and_handle(
        self, 
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[DispatchResult, Any]:
        """
        Dispatch text and invoke the handler of the matched path.
        
        Args:
            text: The input text to dispatch
            context: Optional context information
            
        Returns:
            Tuple of (dispatch_result, handler_result)
        """
        result = await self.dispatch(text, context)
        
        if result.path:
            # Call the path's handler
            handler_result = await result.path.handle(text, context)
            return result, handler_result
        
        return result, None
