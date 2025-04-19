"""
MeaningMesh: A semantic text dispatching framework.

This framework allows routing text to appropriate handlers based
on semantic meaning rather than keywords or patterns.
"""

from .paths.path import Path
from .dispatchers.semantic_dispatcher import SemanticDispatcher, DispatchResult
from .vectorizers.base import Vectorizer, create_vectorizer
from .storage.base import EmbeddingStore, create_store
from .storage.memory import InMemoryEmbeddingStore
from .utils.similarity import (
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    euclidean_similarity,
    find_best_match,
    find_top_matches,
    SIMILARITY_FUNCTIONS
)

# We don't import the vectorizer implementations directly here
# since they're already handled in the vectorizers/__init__.py file
# and will be accessible through that namespace

__version__ = "0.1.0"
