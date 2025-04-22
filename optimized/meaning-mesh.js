/**
 * MeaningMesh - Enhanced Semantic Router Implementation
 * 
 * A contribution to the semantic-router project (https://github.com/aurelio-labs/semantic-router)
 * with additional features and optimizations.
 */

import { cosineDistance } from './utils/vector';

/**
 * Supported embedding models and their configurations
 */
const EMBEDDING_MODELS = {
  'openai': {
    modelName: 'text-embedding-3-small',
    dimensions: 1536,
    apiEndpoint: 'https://api.openai.com/v1/embeddings',
    maxBatchSize: 16,
  },
  'cohere': {
    modelName: 'embed-english-v3.0',
    dimensions: 1024,
    apiEndpoint: 'https://api.cohere.ai/v1/embed',
    maxBatchSize: 96,
  },
  'huggingface': {
    modelName: 'sentence-transformers/all-MiniLM-L6-v2',
    dimensions: 384,
    apiEndpoint: 'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2',
    maxBatchSize: 32,
  },
  'local': {
    modelName: 'local-embeddings',
    dimensions: 384,
    maxBatchSize: 64,
  }
};

/**
 * Route class - Represents a semantic route with example phrases
 */
class Route {
  /**
   * Create a new semantic route
   * @param {Object} config - Route configuration
   * @param {string} config.name - Unique name for the route
   * @param {string[]} config.examples - Example phrases that represent this route
   * @param {Function|null} config.handler - Function to execute when route matches
   * @param {Object} config.metadata - Additional metadata for the route
   * @param {number} config.threshold - Minimum similarity threshold (0-1)
   */
  constructor(config) {
    this.name = config.name;
    this.examples = config.examples || [];
    this.handler = config.handler || null;
    this.metadata = config.metadata || {};
    this.threshold = config.threshold || 0.75;
    this.embeddings = null; // Will be populated by the router
  }

  /**
   * Add more examples to the route
   * @param {string[]} examples - New examples to add
   */
  addExamples(examples) {
    this.examples.push(...examples);
    this.embeddings = null; // Clear cached embeddings
    return this;
  }

  /**
   * Set the handler function for this route
   * @param {Function} handler - Handler function
   */
  setHandler(handler) {
    this.handler = handler;
    return this;
  }

  /**
   * Execute the route handler
   * @param {string} input - Original input text
   * @param {Object} context - Additional context
   * @returns {Promise<any>} - Result from handler
   */
  async execute(input, context = {}) {
    if (!this.handler) {
      throw new Error(`No handler defined for route "${this.name}"`);
    }
    return await this.handler(input, context);
  }
}

/**
 * EmbeddingProvider - Interface for embedding generation
 */
class EmbeddingProvider {
  /**
   * Create a new embedding provider
   * @param {Object} config - Provider configuration
   */
  constructor(config) {
    this.config = config;
  }

  /**
   * Generate embeddings for a list of texts
   * @param {string[]} texts - Texts to embed
   * @returns {Promise<number[][]>} - Array of embedding vectors
   */
  async getEmbeddings(texts) {
    throw new Error('Method must be implemented by subclass');
  }
}

/**
 * OpenAIEmbeddingProvider - Uses OpenAI's embedding API
 */
class OpenAIEmbeddingProvider extends EmbeddingProvider {
  /**
   * Create a new OpenAI embedding provider
   * @param {Object} config - Provider configuration
   * @param {string} config.apiKey - OpenAI API key
   */
  constructor(config) {
    super({
      ...EMBEDDING_MODELS.openai,
      ...config
    });
    
    if (!config.apiKey) {
      throw new Error('OpenAI API key is required');
    }
    
    this.apiKey = config.apiKey;
  }

  /**
   * Generate embeddings using OpenAI's API
   * @param {string[]} texts - Texts to embed
   * @returns {Promise<number[][]>} - Array of embedding vectors
   */
  async getEmbeddings(texts) {
    const batchSize = this.config.maxBatchSize;
    const results = [];

    // Process in batches to avoid API limits
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchResults = await this._processBatch(batch);
      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Process a batch of texts
   * @private
   * @param {string[]} batch - Batch of texts
   * @returns {Promise<number[][]>} - Embeddings for the batch
   */
  async _processBatch(batch) {
    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: this.config.modelName,
          input: batch
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(`OpenAI API error: ${error.error.message}`);
      }

      const data = await response.json();
      return data.data.map(item => item.embedding);
    } catch (error) {
      console.error('Error generating embeddings:', error);
      throw error;
    }
  }
}

/**
 * CohereEmbeddingProvider - Uses Cohere's embedding API
 */
class CohereEmbeddingProvider extends EmbeddingProvider {
  /**
   * Create a new Cohere embedding provider
   * @param {Object} config - Provider configuration
   * @param {string} config.apiKey - Cohere API key
   */
  constructor(config) {
    super({
      ...EMBEDDING_MODELS.cohere,
      ...config
    });
    
    if (!config.apiKey) {
      throw new Error('Cohere API key is required');
    }
    
    this.apiKey = config.apiKey;
  }

  /**
   * Generate embeddings using Cohere's API
   * @param {string[]} texts - Texts to embed
   * @returns {Promise<number[][]>} - Array of embedding vectors
   */
  async getEmbeddings(texts) {
    const batchSize = this.config.maxBatchSize;
    const results = [];

    // Process in batches to avoid API limits
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchResults = await this._processBatch(batch);
      results.push(...batchResults);
    }

    return results;
  }

  /**
   * Process a batch of texts
   * @private
   * @param {string[]} batch - Batch of texts
   * @returns {Promise<number[][]>} - Embeddings for the batch
   */
  async _processBatch(batch) {
    try {
      const response = await fetch(this.config.apiEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`
        },
        body: JSON.stringify({
          model: this.config.modelName,
          texts: batch,
          truncate: 'END'
        })
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(`Cohere API error: ${error.message}`);
      }

      const data = await response.json();
      return data.embeddings;
    } catch (error) {
      console.error('Error generating embeddings:', error);
      throw error;
    }
  }
}

/**
 * LocalEmbeddingProvider - Uses a local embedding model
 */
class LocalEmbeddingProvider extends EmbeddingProvider {
  /**
   * Create a new local embedding provider
   * @param {Object} config - Provider configuration
   * @param {Function} config.embedFn - Function that generates embeddings
   */
  constructor(config) {
    super({
      ...EMBEDDING_MODELS.local,
      ...config
    });
    
    if (!config.embedFn) {
      throw new Error('Embedding function is required for local provider');
    }
    
    this.embedFn = config.embedFn;
  }

  /**
   * Generate embeddings using the provided local function
   * @param {string[]} texts - Texts to embed
   * @returns {Promise<number[][]>} - Array of embedding vectors
   */
  async getEmbeddings(texts) {
    try {
      return await this.embedFn(texts);
    } catch (error) {
      console.error('Error generating local embeddings:', error);
      throw error;
    }
  }
}

/**
 * In-memory cache for embeddings
 */
class EmbeddingCache {
  constructor() {
    this.cache = new Map();
  }

  /**
   * Get embedding from cache
   * @param {string} key - Cache key
   * @returns {number[]|null} - Cached embedding or null
   */
  get(key) {
    return this.cache.get(key) || null;
  }

  /**
   * Store embedding in cache
   * @param {string} key - Cache key
   * @param {number[]} embedding - Embedding vector
   */
  set(key, embedding) {
    this.cache.set(key, embedding);
  }

  /**
   * Check if key exists in cache
   * @param {string} key - Cache key
   * @returns {boolean} - Whether key exists
   */
  has(key) {
    return this.cache.has(key);
  }

  /**
   * Clear the cache
   */
  clear() {
    this.cache.clear();
  }
}

/**
 * MeaningMesh Router - Main class for semantic routing
 */
class Router {
  /**
   * Create a new MeaningMesh router
   * @param {Object} config - Router configuration
   * @param {Route[]} config.routes - Array of routes
   * @param {EmbeddingProvider} config.embeddingProvider - Provider for generating embeddings
   * @param {boolean} config.useCache - Whether to cache embeddings
   * @param {Object} config.options - Additional options
   */
  constructor(config) {
    this.routes = config.routes || [];
    this.embeddingProvider = config.embeddingProvider;
    this.useCache = config.useCache !== false;
    this.options = {
      defaultThreshold: 0.75,
      fallbackRoute: null,
      precomputeEmbeddings: true,
      ...config.options
    };
    
    this.cache = this.useCache ? new EmbeddingCache() : null;
    
    // Initialize route index for faster lookups
    this._routeIndex = new Map();
    this.routes.forEach(route => {
      this._routeIndex.set(route.name, route);
    });
    
    // Precompute embeddings for all examples if enabled
    if (this.options.precomputeEmbeddings) {
      this._precomputeEmbeddings();
    }
  }

  /**
   * Add a new route to the router
   * @param {Route} route - Route to add
   * @returns {Router} - Router instance for chaining
   */
  addRoute(route) {
    this.routes.push(route);
    this._routeIndex.set(route.name, route);
    
    if (this.options.precomputeEmbeddings) {
      this._computeRouteEmbeddings(route);
    }
    
    return this;
  }

  /**
   * Set a fallback route for when no routes match
   * @param {Route} route - Fallback route
   * @returns {Router} - Router instance for chaining
   */
  setFallbackRoute(route) {
    this.options.fallbackRoute = route;
    this._routeIndex.set(route.name, route);
    return this;
  }

  /**
   * Precompute embeddings for all routes
   * @private
   */
  async _precomputeEmbeddings() {
    const promises = this.routes.map(route => this._computeRouteEmbeddings(route));
    await Promise.all(promises);
  }

  /**
   * Compute embeddings for a specific route
   * @private
   * @param {Route} route - Route to compute embeddings for
   */
  async _computeRouteEmbeddings(route) {
    if (!route.examples || route.examples.length === 0) {
      route.embeddings = [];
      return;
    }
    
    // Check cache first if enabled
    if (this.useCache) {
      const cachedEmbeddings = route.examples.map(example => {
        const cacheKey = this._getCacheKey(example);
        return this.cache.get(cacheKey);
      });
      
      // If all embeddings are in cache, use them
      if (cachedEmbeddings.every(emb => emb !== null)) {
        route.embeddings = cachedEmbeddings;
        return;
      }
    }
    
    try {
      const embeddings = await this.embeddingProvider.getEmbeddings(route.examples);
      route.embeddings = embeddings;
      
      // Cache embeddings if enabled
      if (this.useCache) {
        route.examples.forEach((example, i) => {
          const cacheKey = this._getCacheKey(example);
          this.cache.set(cacheKey, embeddings[i]);
        });
      }
    } catch (error) {
      console.error(`Error computing embeddings for route "${route.name}":`, error);
      route.embeddings = [];
    }
  }

  /**
   * Get cache key for a text
   * @private
   * @param {string} text - Text to generate key for
   * @returns {string} - Cache key
   */
  _getCacheKey(text) {
    return `${this.embeddingProvider.config.modelName}:${text}`;
  }

  /**
   * Route an input text to the best matching route
   * @param {string} input - Input text to route
   * @param {Object} context - Additional context
   * @param {Object} options - Routing options
   * @returns {Promise<Object>} - Routing result
   */
  async route(input, context = {}, options = {}) {
    const routeOptions = {
      threshold: this.options.defaultThreshold,
      executeHandler: true,
      includeConfidenceScores: false,
      ...options
    };

    // Get embedding for input
    let inputEmbedding;
    const cacheKey = this._getCacheKey(input);
    
    if (this.useCache && this.cache.has(cacheKey)) {
      inputEmbedding = this.cache.get(cacheKey);
    } else {
      try {
        const embeddings = await this.embeddingProvider.getEmbeddings([input]);
        inputEmbedding = embeddings[0];
        
        if (this.useCache) {
          this.cache.set(cacheKey, inputEmbedding);
        }
      } catch (error) {
        throw new Error(`Failed to generate embedding for input: ${error.message}`);
      }
    }

    // Find best matching route
    const matches = await this._findMatches(inputEmbedding, routeOptions.threshold);
    
    // No matches found
    if (matches.length === 0) {
      if (this.options.fallbackRoute) {
        const result = {
          route: this.options.fallbackRoute,
          confidence: 0,
          input,
          matched: false
        };
        
        if (routeOptions.executeHandler) {
          result.output = await this.options.fallbackRoute.execute(input, context);
        }
        
        return result;
      }
      
      return {
        route: null,
        confidence: 0,
        input,
        matched: false
      };
    }

    // Return best match
    const bestMatch = matches[0];
    const result = {
      route: bestMatch.route,
      confidence: bestMatch.confidence,
      input,
      matched: true
    };
    
    if (routeOptions.includeConfidenceScores) {
      result.allScores = matches.reduce((acc, match) => {
        acc[match.route.name] = match.confidence;
        return acc;
      }, {});
    }
    
    if (routeOptions.executeHandler) {
      result.output = await bestMatch.route.execute(input, context);
    }
    
    return result;
  }

  /**
   * Find matching routes for an input embedding
   * @private
   * @param {number[]} inputEmbedding - Input embedding vector
   * @param {number} threshold - Minimum confidence threshold
   * @returns {Promise<Array<{route: Route, confidence: number}>>} - Matching routes with confidence scores
   */
  async _findMatches(inputEmbedding, threshold) {
    const matches = [];

    // Make sure all routes have computed embeddings
    for (const route of this.routes) {
      if (!route.embeddings) {
        await this._computeRouteEmbeddings(route);
      }
    }

    // Find best match for each route
    for (const route of this.routes) {
      if (!route.embeddings || route.embeddings.length === 0) {
        continue;
      }
      
      // Find best similarity among all examples
      let bestSimilarity = -Infinity;
      
      for (const exampleEmbedding of route.embeddings) {
        const similarity = 1 - cosineDistance(inputEmbedding, exampleEmbedding);
        if (similarity > bestSimilarity) {
          bestSimilarity = similarity;
        }
      }
      
      // Apply route-specific threshold if available, otherwise use global threshold
      const routeThreshold = route.threshold || threshold;
      
      if (bestSimilarity >= routeThreshold) {
        matches.push({
          route,
          confidence: bestSimilarity
        });
      }
    }

    // Sort by confidence (descending)
    return matches.sort((a, b) => b.confidence - a.confidence);
  }

  /**
   * Clear the embedding cache
   */
  clearCache() {
    if (this.cache) {
      this.cache.clear();
    }
  }
}

/**
 * Utility functions for MeaningMesh
 */
const utils = {
  /**
   * Create a new route
   * @param {Object} config - Route configuration
   * @returns {Route} - New route instance
   */
  createRoute(config) {
    return new Route(config);
  },

  /**
   * Create an OpenAI embedding provider
   * @param {Object} config - Provider configuration
   * @returns {OpenAIEmbeddingProvider} - New provider instance
   */
  createOpenAIProvider(config) {
    return new OpenAIEmbeddingProvider(config);
  },

  /**
   * Create a Cohere embedding provider
   * @param {Object} config - Provider configuration
   * @returns {CohereEmbeddingProvider} - New provider instance
   */
  createCohereProvider(config) {
    return new CohereEmbeddingProvider(config);
  },

  /**
   * Create a local embedding provider
   * @param {Object} config - Provider configuration
   * @returns {LocalEmbeddingProvider} - New provider instance
   */
  createLocalProvider(config) {
    return new LocalEmbeddingProvider(config);
  },

  /**
   * Vectorize a text string for experimental use
   * Very simplified embedding function for demo purposes
   * @param {string} text - Text to vectorize
   * @returns {number[]} - Vector representation
   */
  experimentalVectorize(text) {
    // This is just a simple hashing-based vectorization
    // NOT for production use, just for demos without API access
    const hash = (str) => {
      let hash = 0;
      for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash;
      }
      return hash;
    };
    
    // Generate a 384-dimensional vector (matching dimensions of some models)
    const normalized = text.toLowerCase().trim();
    const tokens = normalized.split(/\s+/);
    
    // Create a sparse vector
    const vec = new Array(384).fill(0);
    
    tokens.forEach(token => {
      const h = Math.abs(hash(token)) % 384;
      vec[h] += 1;
    });
    
    // Normalize the vector
    const magnitude = Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    return magnitude === 0 ? vec : vec.map(v => v / magnitude);
  }
};

// Vector utility functions
const vectorUtils = {
  /**
   * Calculate cosine distance between two vectors
   * @param {number[]} a - First vector
   * @param {number[]} b - Second vector
   * @returns {number} - Cosine distance (0-2, where 0 is identical)
   */
  cosineDistance(a, b) {
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
  },
  
  /**
   * Euclidean distance between two vectors
   * @param {number[]} a - First vector
   * @param {number[]} b - Second vector
   * @returns {number} - Euclidean distance
   */
  euclideanDistance(a, b) {
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
};

// Export all components
export {
  Router,
  Route,
  EmbeddingProvider,
  OpenAIEmbeddingProvider,
  CohereEmbeddingProvider,
  LocalEmbeddingProvider,
  EmbeddingCache,
  utils,
  vectorUtils,
  EMBEDDING_MODELS
};
