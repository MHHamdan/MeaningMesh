"""Utility functions for calculating vector similarities."""

from typing import List, Callable, Tuple, Dict, Any, Optional
import numpy as np


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Cosine similarity score (between -1 and 1)
    """
    v1_array = np.array(v1)
    v2_array = np.array(v2)
    
    dot_product = np.dot(v1_array, v2_array)
    norm_v1 = np.linalg.norm(v1_array)
    norm_v2 = np.linalg.norm(v2_array)
    
    # Avoid division by zero
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    
    return dot_product / (norm_v1 * norm_v2)


def dot_product_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Calculate dot product similarity between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Dot product similarity score
    """
    return np.dot(np.array(v1), np.array(v2))


def euclidean_distance(v1: List[float], v2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Euclidean distance (lower is more similar)
    """
    return np.linalg.norm(np.array(v1) - np.array(v2))


def euclidean_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Convert Euclidean distance to a similarity score.
    
    Args:
        v1: First vector
        v2: Second vector
        
    Returns:
        Similarity score (higher is more similar)
    """
    distance = euclidean_distance(v1, v2)
    # Convert distance to similarity (1 when identical, approaches 0 as distance increases)
    return 1.0 / (1.0 + distance)


# Dictionary of available similarity functions
SIMILARITY_FUNCTIONS: Dict[str, Callable[[List[float], List[float]], float]] = {
    "cosine": cosine_similarity,
    "dot_product": dot_product_similarity,
    "euclidean": euclidean_similarity
}


def find_best_match(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    similarity_fn: str = "cosine"
) -> Tuple[int, float]:
    """
    Find the best matching embedding from a list of candidates.
    
    Args:
        query_embedding: The query vector
        candidate_embeddings: List of candidate vectors to compare against
        similarity_fn: Name of the similarity function to use
        
    Returns:
        Tuple of (best_match_index, similarity_score)
    """
    if not candidate_embeddings:
        return -1, 0.0
    
    sim_fn = SIMILARITY_FUNCTIONS.get(similarity_fn, cosine_similarity)
    
    best_score = -float('inf')
    best_index = -1
    
    for i, candidate in enumerate(candidate_embeddings):
        score = sim_fn(query_embedding, candidate)
        if score > best_score:
            best_score = score
            best_index = i
    
    return best_index, best_score


def find_top_matches(
    query_embedding: List[float],
    candidate_embeddings: List[List[float]],
    similarity_fn: str = "cosine",
    top_k: int = 3
) -> List[Tuple[int, float]]:
    """
    Find the top-k matching embeddings from a list of candidates.
    
    Args:
        query_embedding: The query vector
        candidate_embeddings: List of candidate vectors to compare against
        similarity_fn: Name of the similarity function to use
        top_k: Number of top matches to return
        
    Returns:
        List of tuples (index, score) for the top k matches
    """
    if not candidate_embeddings:
        return []
    
    sim_fn = SIMILARITY_FUNCTIONS.get(similarity_fn, cosine_similarity)
    
    # Calculate similarities for all candidates
    similarities = [(i, sim_fn(query_embedding, emb)) for i, emb in enumerate(candidate_embeddings)]
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:top_k]
