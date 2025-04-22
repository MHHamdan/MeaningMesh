/**
 * Vector utility functions for MeaningMesh
 * These utilities perform various vector operations needed for semantic routing
 */

/**
 * Calculate cosine distance between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} - Cosine distance (0-2, where 0 is identical)
 */
export function cosineDistance(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  if (normA === 0 || normB === 0) {
    return 1; // Maximum distance for zero vectors
  }
  
  const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  
  // Cosine distance is 1 - similarity
  // Clamp to [0, 2] range to handle floating point errors
  return Math.min(Math.max(1 - similarity, 0), 2);
}

/**
 * Calculate cosine similarity between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} - Cosine similarity (-1 to 1, where 1 is identical)
 */
export function cosineSimilarity(a, b) {
  return 1 - cosineDistance(a, b);
}

/**
 * Calculate Euclidean distance between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} - Euclidean distance
 */
export function euclideanDistance(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  
  return Math.sqrt(sum);
}

/**
 * Calculate Manhattan (L1) distance between two vectors
 * @param {number[]} a - First vector
 * @param {number[]} b - Second vector
 * @returns {number} - Manhattan distance
 */
export function manhattanDistance(a, b) {
  if (a.length !== b.length) {
    throw new Error('Vectors must have the same dimensions');
  }
  
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += Math.abs(a[i] - b[i]);
  }
  
  return sum;
}

/**
 * Normalize a vector to unit length
 * @param {number[]} vector - Vector to normalize
 * @returns {number[]} - Normalized vector
 */
export function normalize(vector) {
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  
  if (magnitude === 0) {
    return vector.slice(); // Return copy of zero vector
  }
  
  return vector.map(v => v / magnitude);
}

/**
 * Calculate the mean vector from an array of vectors
 * @param {number[][]} vectors - Array of vectors
 * @returns {number[]} - Mean vector
 */
export function meanVector(vectors) {
  if (vectors.length === 0) {
    return [];
  }
  
  const dimensions = vectors[0].length;
  const result = new Array(dimensions).fill(0);
  
  for (const vector of vectors) {
    if (vector.length !== dimensions) {
      throw new Error('All vectors must have the same dimensions');
    }
    
    for (let i = 0; i < dimensions; i++) {
      result[i] += vector[i];
    }
  }
  
  return result.map(val => val / vectors.length);
}

/**
 * Compute pairwise distances between all vectors in two sets
 * @param {number[][]} setA - First set of vectors
 * @param {number[][]} setB - Second set of vectors
 * @param {Function} distanceFn - Distance function to use
 * @returns {number[][]} - Matrix of distances
 */
export function pairwiseDistances(setA, setB, distanceFn = cosineDistance) {
  return setA.map(a => setB.map(b => distanceFn(a, b)));
}