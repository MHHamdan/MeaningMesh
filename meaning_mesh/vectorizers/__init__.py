"""
Vectorizer implementations for various embedding providers.

This package provides interfaces to different embedding models that can be used
with MeaningMesh for semantic text routing.
"""

from .base import Vectorizer, create_vectorizer
from .mock import MockVectorizer

# Define the list of exported symbols
__all__ = [
    "Vectorizer",
    "create_vectorizer",
    "MockVectorizer",
]

# Try to import OpenAI vectorizer
try:
    from .openai import OpenAIVectorizer
    __all__.append("OpenAIVectorizer")
except ImportError:
    # OpenAI is not installed, which is fine
    pass

# Try to import HuggingFace vectorizer
try:
    from .huggingface import HuggingFaceVectorizer
    __all__.append("HuggingFaceVectorizer")
except ImportError:
    # HuggingFace dependencies are not installed, which is fine
    pass

# Try to import Cohere vectorizer
try:
    from .cohere import CohereVectorizer
    __all__.append("CohereVectorizer")
except ImportError:
    # Cohere is not installed, which is fine
    pass
