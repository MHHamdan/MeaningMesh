"""
Storage components for embeddings and paths.

This package provides interfaces for storing and retrieving embeddings and paths
used by the MeaningMesh framework.
"""

from .base import EmbeddingStore, create_store
from .memory import InMemoryEmbeddingStore

__all__ = [
    "EmbeddingStore",
    "create_store",
    "InMemoryEmbeddingStore",
]
