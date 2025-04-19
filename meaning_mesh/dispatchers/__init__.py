"""
Dispatcher components for semantic routing.

This package provides the dispatchers that route text to appropriate paths
based on semantic similarity.
"""

from .semantic_dispatcher import SemanticDispatcher, DispatchResult

__all__ = [
    "SemanticDispatcher",
    "DispatchResult"
]
