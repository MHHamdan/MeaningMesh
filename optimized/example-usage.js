/**
 * MeaningMesh Advanced Features
 * 
 * This file demonstrates additional advanced capabilities that enhance
 * the core functionality of MeaningMesh.
 */

import {
    Router,
    Route,
    utils,
    vectorUtils
  } from './meaning-mesh.js';
  
  /**
   * Dynamic route weighting
   * 
   * Adjusts the importance of routes based on context or user preferences
   */
  class WeightedRouter extends Router {
    constructor(config) {
      super(config);
      this.weights = new Map();
      
      // Initialize weights to 1.0 for all routes
      this.routes.forEach(route => {
        this.weights.set(route.name, 1.0);
      });
    }
    
    /**
     * Set weight for a specific route
     * @param {string} routeName - Name of the route
     * @param {number} weight - Weight factor (0-2, where 1 is neutral)
     */
    setRouteWeight(routeName, weight) {
      if (weight < 0 || weight > 2) {
        throw new Error('Weight must be between 0 and 2');
      }
      
      if (!this._routeIndex.has(routeName)) {
        throw new Error(`Route "${routeName}" not found`);
      }
      
      this.weights.set(routeName, weight);
      return this;
    }
    
    /**
     * Override the _findMatches method to apply weights
     */
    async _findMatches(inputEmbedding, threshold) {
      // Call the parent method to get base matches
      const baseMatches = await super._findMatches(inputEmbedding, threshold);
      
      // Apply weights to the confidence scores
      const weightedMatches = baseMatches.map(match => {
        const weight = this.weights.get(match.route.name) || 1.0;
        return {
          route: match.route,
          confidence: Math.min(match.confidence * weight, 1.0), // Cap at 1.0
          originalConfidence: match.confidence,
          appliedWeight: weight
        };
      });
      
      // Re-sort by weighted confidence
      return weightedMatches.sort((a, b) => b.confidence - a.confidence);
    }
  }
  
  /**
   * Contextual router that maintains conversation context
   */
  class ContextualRouter {
    constructor(config) {
      this.router = new Router(config);
      this.contextWindow = config.contextWindow || 3; // Number of previous exchanges to remember
      this.conversationHistory = [];
      this.contextWeight = config.contextWeight || 0.3; // Weight of context in routing decisions
    }
    
    /**
     * Add a message to the conversation history
     * @param {string} message - User message
     * @param {string} routeName - Matched route name
     */
    addToHistory(message, routeName) {
      this.conversationHistory.push({ message, routeName });
      
      // Keep only the most recent messages based on contextWindow
      if (this.conversationHistory.length > this.contextWindow) {
        this.conversationHistory.shift();
      }
    }
    
    /**
     * Route a message with consideration of conversation context
     * @param {string} input - Input message
     * @param {Object} context - Additional context
     * @param {Object} options - Routing options
     * @returns {Promise<Object>} - Routing result
     */
    async route(input, context = {}, options = {}) {
      // First, get standard routing result
      const baseResult = await this.router.route(input, context, {
        ...options,
        includeConfidenceScores: true,
        executeHandler: false
      });
      
      // If no conversation history, use the base result
      if (this.conversationHistory.length === 0) {
        // Add this exchange to history
        this.addToHistory(input, baseResult.route.name);
        
        // Execute handler if needed
        if (options.executeHandler !== false) {
          baseResult.output = await baseResult.route.execute(input, context);
        }
        
        return baseResult;
      }
      
      // Consider conversation context
      const routeScores = baseResult.allScores || {};
      const contextualScores = { ...routeScores };
      
      // Adjust scores based on conversation history
      for (const { routeName } of this.conversationHistory) {
        if (routeName in contextualScores) {
          contextualScores[routeName] += this.contextWeight / this.conversationHistory.length;
        }
      }
      
      // Find best route based on adjusted scores
      let bestRouteName = baseResult.route.name;
      let highestScore = 0;
      
      for (const [routeName, score] of Object.entries(contextualScores)) {
        if (score > highestScore) {
          highestScore = score;
          bestRouteName = routeName;
        }
      }
      
      // Get the selected route
      const selectedRoute = this.router._routeIndex.get(bestRouteName);
      
      // Create result object
      const result = {
        route: selectedRoute,
        confidence: contextualScores[bestRouteName],
        input,
        matched: true,
        baseRoute: baseResult.route.name,
        baseConfidence: baseResult.confidence,
        contextualScores
      };
      
      // Execute handler if needed
      if (options.executeHandler !== false) {
        result.output = await selectedRoute.execute(input, context);
      }
      
      // Add to conversation history
      this.addToHistory(input, bestRouteName);
      
      return result;
    }
    
    /**
     * Clear conversation history
     */
    clearHistory() {
      this.conversationHistory = [];
    }
  }
  
  /**
   * Enhanced route with fuzzy matching capability
   */
  class FuzzyRoute extends Route {
    constructor(config) {
      super(config);
      this.fuzzyThreshold = config.fuzzyThreshold || 0.6;
      this.keywordImportance = config.keywordImportance || 0.25;
    }
    
    /**
     * Extract keywords from examples
     * @returns {Set<string>} - Set of keywords
     */
    extractKeywords() {
      const stopWords = new Set([
        'a', 'an', 'the', 'and', 'or', 'but', 'for', 'nor', 'on', 'at', 'to', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could',
        'may', 'might', 'must', 'in', 'of', 'with', 'about', 'from', 'as', 'that',
        'this', 'these', 'those'
      ]);
      
      const keywords = new Set();
      
      for (const example of this.examples) {
        const words = example.toLowerCase().match(/\b\w+\b/g) || [];
        
        for (const word of words) {
          if (!stopWords.has(word) && word.length > 2) {
            keywords.add(word);
          }
        }
      }
      
      return keywords;
    }
    
    /**
     * Calculate keyword match score
     * @param {string} input - Input text
     * @returns {number} - Match score (0-1)
     */
    keywordMatchScore(input) {
      const keywords = this.extractKeywords();
      if (keywords.size === 0) return 0;
      
      const inputWords = new Set(
        (input.toLowerCase().match(/\b\w+\b/g) || [])
          .filter(word => word.length > 2)
      );
      
      let matches = 0;
      for (const keyword of keywords) {
        if (inputWords.has(keyword)) {
          matches++;
        }
      }
      
      return matches / keywords.size;
    }
  }
  
  /**
   * Adaptive router that learns from user feedback
   */
  class AdaptiveRouter {
    constructor(config) {
      this.router = new Router(config);
      this.feedbackWeight = config.feedbackWeight || 0.1;
      this.positiveExamples = new Map(); // routeName -> [examples]
      this.negativeExamples = new Map(); // routeName -> [examples]
      this.adaptationInterval = config.adaptationInterval || 10;
      this.feedbackCount = 0;
    }
    
    /**
     * Route a message
     * @param {string} input - Input message
     * @param {Object} context - Additional context
     * @param {Object} options - Routing options
     * @returns {Promise<Object>} - Routing result
     */
    async route(input, context = {}, options = {}) {
      return await this.router.route(input, context, options);
    }
    
    /**
     * Add positive feedback for a routing decision
     * @param {string} input - The input that was routed
     * @param {string} routeName - The route it should have matched
     */
    addPositiveFeedback(input, routeName) {
      if (!this.router._routeIndex.has(routeName)) {
        throw new Error(`Route "${routeName}" not found`);
      }
      
      if (!this.positiveExamples.has(routeName)) {
        this.positiveExamples.set(routeName, []);
      }
      
      this.positiveExamples.get(routeName).push(input);
      this.feedbackCount++;
      
      if (this.feedbackCount >= this.adaptationInterval) {
        this._adaptRoutes();
      }
    }
    
    /**
     * Add negative feedback for a routing decision
     * @param {string} input - The input that was misrouted
     * @param {string} incorrectRouteName - The route it incorrectly matched
     * @param {string} correctRouteName - The route it should have matched
     */
    addNegativeFeedback(input, incorrectRouteName, correctRouteName) {
      if (!this.router._routeIndex.has(incorrectRouteName)) {
        throw new Error(`Route "${incorrectRouteName}" not found`);
      }
      
      if (correctRouteName && !this.router._routeIndex.has(correctRouteName)) {
        throw new Error(`Route "${correctRouteName}" not found`);
      }
      
      if (!this.negativeExamples.has(incorrectRouteName)) {
        this.negativeExamples.set(incorrectRouteName, []);
      }
      
      this.negativeExamples.get(incorrectRouteName).push(input);
      
      if (correctRouteName) {
        this.addPositiveFeedback(input, correctRouteName);
      } else {
        this.feedbackCount++;
      }
      
      if (this.feedbackCount >= this.adaptationInterval) {
        this._adaptRoutes();
      }
    }
    
    /**
     * Adapt routes based on collected feedback
     * @private
     */
    _adaptRoutes() {
      // Add positive examples to routes
      for (const [routeName, examples] of this.positiveExamples.entries()) {
        if (examples.length > 0) {
          const route = this.router._routeIndex.get(routeName);
          route.addExamples(examples);
        }
      }
      
      // Clear feedback collections
      this.positiveExamples.clear();
      this.negativeExamples.clear();
      this.feedbackCount = 0;
      
      // Clear the cache to force re-computation of embeddings
      this.router.clearCache();
    }
  }
  
  /**
   * Route group for organizing related routes
   */
  class RouteGroup {
    constructor(name, routes = []) {
      this.name = name;
      this.routes = routes;
    }
    
    /**
     * Add a route to the group
     * @param {Route} route - Route to add
     */
    addRoute(route) {
      this.routes.push(route);
      return this;
    }
    
    /**
     * Get all routes in the group
     * @returns {Route[]} - Array of routes
     */
    getRoutes() {
      return this.routes;
    }
    
    /**
     * Set handler for all routes in the group
     * @param {Function} handlerFactory - Function that returns a handler
     * @returns {RouteGroup} - This group for chaining
     */
    setHandlers(handlerFactory) {
      for (const route of this.routes) {
        route.setHandler(handlerFactory(route.name));
      }
      return this;
    }
  }
  
  /**
   * Fine-tuning utility for optimizing route examples
   */
  class RouteTuner {
    /**
     * Analyze route performance
     * @param {Router} router - Router instance
     * @param {Object[]} testCases - Test cases with expected routes
     * @returns {Object} - Analysis results
     */
    static async analyzePerformance(router, testCases) {
      const results = {
        totalCases: testCases.length,
        correctMatches: 0,
        incorrectMatches: 0,
        confusionMatrix: {},
        routeStats: {}
      };
      
      // Initialize confusion matrix and route stats
      for (const route of router.routes) {
        results.confusionMatrix[route.name] = {};
        results.routeStats[route.name] = {
          expectedCount: 0,
          correctMatches: 0,
          falsePositives: 0,
          falseNegatives: 0,
          averageConfidence: 0,
          examples: route.examples.length
        };
        
        for (const otherRoute of router.routes) {
          results.confusionMatrix[route.name][otherRoute.name] = 0;
        }
      }
      
      // Process each test case
      for (const testCase of testCases) {
        const { input, expectedRoute } = testCase;
        
        results.routeStats[expectedRoute].expectedCount++;
        
        const result = await router.route(input, {}, {
          executeHandler: false,
          includeConfidenceScores: true
        });
        
        const matchedRoute = result.route.name;
        results.confusionMatrix[expectedRoute][matchedRoute]++;
        
        if (matchedRoute === expectedRoute) {
          results.correctMatches++;
          results.routeStats[expectedRoute].correctMatches++;
          results.routeStats[expectedRoute].averageConfidence += result.confidence;
        } else {
          results.incorrectMatches++;
          results.routeStats[expectedRoute].falseNegatives++;
          results.routeStats[matchedRoute].falsePositives++;
        }
      }
      
      // Calculate final stats
      results.accuracy = results.correctMatches / results.totalCases;
      
      for (const routeName in results.routeStats) {
        const stats = results.routeStats[routeName];
        if (stats.correctMatches > 0) {
          stats.averageConfidence /= stats.correctMatches;
        }
        
        const totalPredicted = stats.correctMatches + stats.falsePositives;
        stats.precision = totalPredicted > 0 ? stats.correctMatches / totalPredicted : 0;
        
        const totalActual = stats.correctMatches + stats.falseNegatives;
        stats.recall = totalActual > 0 ? stats.correctMatches / totalActual : 0;
        
        stats.f1Score = (stats.precision + stats.recall) > 0 
          ? 2 * (stats.precision * stats.recall) / (stats.precision + stats.recall)
          : 0;
      }
      
      return results;
    }
    
    /**
     * Suggest improvements based on analysis
     * @param {Object} analysis - Analysis results
     * @returns {Object} - Improvement suggestions
     */
    static suggestImprovements(analysis) {
      const suggestions = {
        routeSuggestions: {},
        generalSuggestions: []
      };
      
      // Check overall accuracy
      if (analysis.accuracy < 0.8) {
        suggestions.generalSuggestions.push(
          "Overall accuracy is below 80%. Consider adding more diverse examples to all routes."
        );
      }
      
      // Analyze each route
      for (const routeName in analysis.routeStats) {
        const stats = analysis.routeStats[routeName];
        const routeSuggestions = [];
        
        // Low example count
        if (stats.examples < 5) {
          routeSuggestions.push(
            `Add more examples (current: ${stats.examples}). Aim for at least 5-10 diverse examples.`
          );
        }
        
        // Low recall (missing matches)
        if (stats.recall < 0.7 && stats.expectedCount > 3) {
          routeSuggestions.push(
            `Improve recall (${(stats.recall * 100).toFixed(1)}%). This route is missing ${stats.falseNegatives} expected matches.`
          );
        }
        
        // Low precision (false positives)
        if (stats.precision < 0.7 && (stats.correctMatches + stats.falsePositives) > 3) {
          routeSuggestions.push(
            `Improve precision (${(stats.precision * 100).toFixed(1)}%). This route has ${stats.falsePositives} false positives.`
          );
        }
        
        // Confused with other routes
        const confusions = [];
        for (const otherRoute in analysis.confusionMatrix[routeName]) {
          if (otherRoute !== routeName && analysis.confusionMatrix[routeName][otherRoute] > 0) {
            confusions.push({
              route: otherRoute,
              count: analysis.confusionMatrix[routeName][otherRoute]
            });
          }
        }
        
        if (confusions.length > 0) {
          confusions.sort((a, b) => b.count - a.count);
          const topConfusion = confusions[0];
          
          routeSuggestions.push(
            `Reduce confusion with '${topConfusion.route}' (${topConfusion.count} cases). Make examples more distinct.`
          );
        }
        
        if (routeSuggestions.length > 0) {
          suggestions.routeSuggestions[routeName] = routeSuggestions;
        }
      }
      
      return suggestions;
    }
  }
  
  // Export all components
  export {
    WeightedRouter,
    ContextualRouter,
    FuzzyRoute,
    AdaptiveRouter,
    RouteGroup,
    RouteTuner
  };