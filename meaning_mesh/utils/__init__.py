"""
Utility functions for the MeaningMesh framework.

This package provides utility functions used by the MeaningMesh framework
for calculating similarities, processing text, etc.
"""

from .similarity import (
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    euclidean_similarity,
    find_best_match,
    find_top_matches,
    SIMILARITY_FUNCTIONS
)

__all__ = [
    "cosine_similarity",
    "dot_product_similarity",
    "euclidean_distance",
    "euclidean_similarity",
    "find_best_match",
    "find_top_matches",
    "SIMILARITY_FUNCTIONS"
]
