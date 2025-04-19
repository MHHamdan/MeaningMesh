"""
Vectorizer implementations for various embedding providers.

This package provides interfaces to different embedding models that can be used
with MeaningMesh for semantic text routing.
"""

from .base import Vectorizer, create_vectorizer
from .openai import OpenAIVectorizer
from .huggingface import HuggingFaceVectorizer
from .mock import MockVectorizer

# Import optional providers that might not be installed
try:
    from .cohere import CohereVectorizer
except ImportError:
    pass

__all__ = [
    "Vectorizer",
    "create_vectorizer",
    "OpenAIVectorizer",
    "HuggingFaceVectorizer",
    "MockVectorizer",
    "CohereVectorizer",  # Will be undefined if import failed, but that's okay for __all__
]
