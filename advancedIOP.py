"""
MeaningMesh Advanced Features

This module extends the core MeaningMesh functionality with advanced routing capabilities,
including contextual routing, weighted routes, and adaptive learning.
"""

import numpy as np
import time
from typing import List, Dict, Any, Callable, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
import logging
from collections import defaultdict, deque

from meaning_mesh import (
    Router,
    Route,
    EmbeddingProvider,
    cosine_similarity,
    create_route
)

logger = logging.getLogger("meaning_mesh.advanced")

# ===============================
# Weighted Router
# ===============================

class WeightedRouter(Router):
    """Router that applies weights to routes based on context or preferences."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = {route.name: 1.0 for route in self.routes}
    
    def set_route_weight(self, route_name: str, weight: float) -> 'WeightedRouter':
        """
        Set weight for a specific route.
        
        Args:
            route_name: Name of the route
            weight: Weight factor (0-2, where 1 is neutral)
            
        Returns:
            Self for chaining
        """
        if weight < 0 or weight > 2:
            raise ValueError("Weight must be between 0 and 2")
        
        if route_name not in self.route_index:
            raise ValueError(f"Route '{route_name}' not found")
        
        self.weights[route_name] = weight
        return self
    
    async def _find_matches(self, input_embedding: np.ndarray, threshold: float) -> List[Dict[str, Any]]:
        """Override to apply weights to confidence scores."""
        # Get base matches from parent method
        base_matches = await super()._find_matches(input_embedding, threshold)
        
        # Apply weights to the confidence scores
        weighted_matches = []
        for match in base_matches:
            route = match["route"]
            weight = self.weights.get(route.name, 1.0)
            
            weighted_matches.append({
                "route": route,
                "confidence": min(match["confidence"] * weight, 1.0),  # Cap at 1.0
                "original_confidence": match["confidence"],
                "applied_weight": weight
            })
        
        # Re-sort by weighted confidence
        return sorted(weighted_matches, key=lambda x: x["confidence"], reverse=True)

# ===============================
# Contextual Router
# ===============================

class ContextualRouter:
    """Router that maintains conversation context for improved routing accuracy."""
    
    def __init__(self, router: Router, context_window: int = 3, context_weight: float = 0.3):
        """
        Initialize a contextual router.
        
        Args:
            router: Base router instance
            context_window: Number of previous exchanges to remember
            context_weight: Weight of context in routing decisions (0-1)
        """
        self.router = router
        self.context_window = context_window
        self.context_weight = context_weight
        self.conversation_history = deque(maxlen=context_window)
    
    def add_to_history(self, message: str, route_name: str) -> None:
        """
        Add a message to the conversation history.
        
        Args:
            message: User message
            route_name: Matched route name
        """
        self.conversation_history.append({"message": message, "route_name": route_name})
    
    async def route(self, 
                    input_text: str, 
                    context: Dict[str, Any] = None,
                    execute_handler: bool = True,
                    include_confidence_scores: bool = False) -> Dict[str, Any]:
        """
        Route a message with consideration of conversation context.
        
        Args:
            input_text: Input message
            context: Additional context
            execute_handler: Whether to execute the handler
            include_confidence_scores: Whether to include all confidence scores
            
        Returns:
            Routing result
        """
        # First, get standard routing result
        base_result = await self.router.route(
            input_text, 
            context, 
            execute_handler=False,
            include_confidence_scores=True
        )
        
        # If no conversation history, use the base result
        if not self.conversation_history:
            # Add this exchange to history
            self.add_to_history(input_text, base_result["route"].name)
            
            # Execute handler if needed
            if execute_handler:
                base_result["output"] = await base_result["route"].execute(input_text, context or {})
            
            return base_result
        
        # Consider conversation context
        route_scores = base_result.get("all_scores", {})
        contextual_scores = dict(route_scores)
        
        # Adjust scores based on conversation history
        history_weight = self.context_weight / len(self.conversation_history)
        for entry in self.conversation_history:
            route_name = entry["route_name"]
            if route_name in contextual_scores:
                contextual_scores[route_name] += history_weight
        
        # Find best route based on adjusted scores
        best_route_name = base_result["route"].name
        highest_score = 0
        
        for route_name, score in contextual_scores.items():
            if score > highest_score:
                highest_score = score
                best_route_name = route_name
        
        # Get the selected route
        selected_route = self.router.route_index.get(best_route_name)
        
        # Create result object
        result = {
            "route": selected_route,
            "confidence": contextual_scores[best_route_name],
            "input": input_text,
            "matched": True,
            "base_route": base_result["route"].name,
            "base_confidence": base_result["confidence"],
            "contextual_scores": contextual_scores
        }
        
        # Include all scores if requested
        if include_confidence_scores:
            result["all_scores"] = contextual_scores
        
        # Execute handler if needed
        if execute_handler:
            result["output"] = await selected_route.execute(input_text, context or {})
        
        # Add to conversation history
        self.add_to_history(input_text, best_route_name)
        
        return result
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()

# ===============================
# Multi-Stage Router
# ===============================

class MultiStageRouter:
    """Implements hierarchical routing through multiple stages."""
    
    def __init__(self, primary_router: Router):
        """
        Initialize a multi-stage router.
        
        Args:
            primary_router: The top-level router for initial classification
        """
        self.primary_router = primary_router
        self.secondary_routers = {}
    
    def add_secondary_router(self, primary_route_name: str, router: Router) -> 'MultiStageRouter':
        """
        Add a secondary router for a primary route.
        
        Args:
            primary_route_name: Name of the primary route
            router: Secondary router to use when primary route matches
            
        Returns:
            Self for chaining
        """
        if primary_route_name not in self.primary_router.route_index:
            raise ValueError(f"Primary route '{primary_route_name}' not found")
        
        self.secondary_routers[primary_route_name] = router
        return self
    
    async def route(self, 
                   input_text: str, 
                   context: Dict[str, Any] = None,
                   execute_handler: bool = True,
                   include_confidence_scores: bool = False) -> Dict[str, Any]:
        """
        Route through multiple stages.
        
        Args:
            input_text: Input message
            context: Additional context
            execute_handler: Whether to execute the handler
            include_confidence_scores: Whether to include all confidence scores
            
        Returns:
            Routing result with routing path
        """
        if context is None:
            context = {}
            
        # First stage: route with primary router
        primary_result = await self.primary_router.route(
            input_text, 
            context,
            execute_handler=False,  # Don't execute handler at primary stage
            include_confidence_scores=include_confidence_scores
        )
        
        primary_route_name = primary_result["route"].name if primary_result["route"] else None
        
        # If no match at primary level or no secondary router, return primary result
        if not primary_route_name or primary_route_name not in self.secondary_routers:
            # Execute handler if needed and available
            if execute_handler and primary_result["route"] and primary_result["route"].handler:
                primary_result["output"] = await primary_result["route"].execute(input_text, context)
                
            primary_result["routing_path"] = [primary_route_name] if primary_route_name else []
            return primary_result
        
        # Second stage: route with secondary router
        secondary_router = self.secondary_routers[primary_route_name]
        secondary_result = await secondary_router.route(
            input_text,
            context,
            execute_handler=execute_handler,
            include_confidence_scores=include_confidence_scores
        )
        
        # Combine results
        result = secondary_result.copy()
        result["primary_route"] = primary_result["route"]
        result["primary_confidence"] = primary_result["confidence"]
        result["routing_path"] = [primary_route_name, secondary_result["route"].name]
        
        if include_confidence_scores:
            result["primary_all_scores"] = primary_result.get("all_scores", {})
            
        return result

# ===============================
# Adaptive Router
# ===============================

class AdaptiveRouter:
    """Router that learns from user feedback to improve routing accuracy."""
    
    def __init__(self, router: Router, adaptation_interval: int = 10, feedback_weight: float = 0.1):
        """
        Initialize an adaptive router.
        
        Args:
            router: Base router instance
            adaptation_interval: Number of feedback items before adaptation
            feedback_weight: Weight of feedback in adaptation
        """
        self.router = router
        self.feedback_weight = feedback_weight
        self.positive_examples = defaultdict(list)  # route_name -> [examples]
        self.negative_examples = defaultdict(list)  # route_name -> [examples]
        self.adaptation_interval = adaptation_interval
        self.feedback_count = 0
    
    async def route(self, 
                   input_text: str, 
                   context: Dict[str, Any] = None,
                   execute_handler: bool = True,
                   include_confidence_scores: bool = False) -> Dict[str, Any]:
        """Delegate to the base router."""
        return await self.router.route(
            input_text,
            context,
            execute_handler=execute_handler,
            include_confidence_scores=include_confidence_scores
        )
    
    def add_positive_feedback(self, input_text: str, route_name: str) -> None:
        """
        Add positive feedback for a routing decision.
        
        Args:
            input_text: The input that was routed
            route_name: The route it should have matched
        """
        if route_name not in self.router.route_index:
            raise ValueError(f"Route '{route_name}' not found")
        
        self.positive_examples[route_name].append(input_text)
        self.feedback_count += 1
        
        if self.feedback_count >= self.adaptation_interval:
            self._adapt_routes()
    
    def add_negative_feedback(self, 
                             input_text: str, 
                             incorrect_route_name: str, 
                             correct_route_name: Optional[str] = None) -> None:
        """
        Add negative feedback for a routing decision.
        
        Args:
            input_text: The input that was misrouted
            incorrect_route_name: The route it incorrectly matched
            correct_route_name: The route it should have matched (optional)
        """
        if incorrect_route_name not in self.router.route_index:
            raise ValueError(f"Route '{incorrect_route_name}' not found")
        
        if correct_route_name and correct_route_name not in self.router.route_index:
            raise ValueError(f"Route '{correct_route_name}' not found")
        
        self.negative_examples[incorrect_route_name].append(input_text)
        
        if correct_route_name:
            self.add_positive_feedback(input_text, correct_route_name)
        else:
            self.feedback_count += 1
        
        if self.feedback_count >= self.adaptation_interval:
            self._adapt_routes()
    
    def _adapt_routes(self) -> None:
        """Adapt routes based on collected feedback."""
        # Add positive examples to routes
        for route_name, examples in self.positive_examples.items():
            if examples:
                route = self.router.route_index[route_name]
                route.add_examples(examples)
                logger.info(f"Added {len(examples)} positive examples to route '{route_name}'")
        
        # TODO: Handle negative examples (more complex, might require retraining or adjusting thresholds)
        
        # Clear feedback collections
        self.positive_examples.clear()
        self.negative_examples.clear()
        self.feedback_count = 0
        
        # Clear the cache to force re-computation of embeddings
        self.router.clear_cache()
        logger.info("Adapted routes based on feedback")

# ===============================
# Route Group
# ===============================

class RouteGroup:
    """Group of related routes for organization and shared handling."""
    
    def __init__(self, name: str, routes: List[Route] = None):
        """
        Initialize a route group.
        
        Args:
            name: Name of the group
            routes: Initial routes in the group
        """
        self.name = name
        self.routes = routes or []
    
    def add_route(self, route: Route) -> 'RouteGroup':
        """
        Add a route to the group.
        
        Args:
            route: Route to add
            
        Returns:
            Self for chaining
        """
        self.routes.append(route)
        return self
    
    def get_routes(self) -> List[Route]:
        """Get all routes in the group."""
        return self.routes
    
    def set_handlers(self, handler_factory: Callable[[str], Callable]) -> 'RouteGroup':
        """
        Set handler for all routes in the group.
        
        Args:
            handler_factory: Function that returns a handler based on route name
            
        Returns:
            Self for chaining
        """
        for route in self.routes:
            route.set_handler(handler_factory(route.name))
        return self
    
    def add_to_router(self, router: Router) -> Router:
        """
        Add all routes in the group to a router.
        
        Args:
            router: Router to add routes to
            
        Returns:
            The router with routes added
        """
        for route in self.routes:
            router.add_route(route)
        return router

# ===============================
# Route Tuner
# ===============================

class RouteTuner:
    """Utility for analyzing and improving route performance."""
    
    @staticmethod
    async def analyze_performance(router: Router, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze route performance on test cases.
        
        Args:
            router: Router to test
            test_cases: List of test cases with input and expected_route
            
        Returns:
            Analysis results with metrics
        """
        results = {
            "total_cases": len(test_cases),
            "correct_matches": 0,
            "incorrect_matches": 0,
            "confusion_matrix": defaultdict(lambda: defaultdict(int)),
            "route_stats": {}
        }
        
        # Initialize route stats
        for route in router.routes:
            results["route_stats"][route.name] = {
                "expected_count": 0,
                "correct_matches": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "average_confidence": 0,
                "examples": len(route.examples)
            }
        
        # Process each test case
        for test_case in test_cases:
            input_text = test_case["input"]
            expected_route = test_case["expected_route"]
            
            results["route_stats"][expected_route]["expected_count"] += 1
            
            result = await router.route(
                input_text,
                execute_handler=False,
                include_confidence_scores=True
            )
            
            matched_route = result["route"].name if result["route"] else "no_match"
            results["confusion_matrix"][expected_route][matched_route] += 1
            
            if matched_route == expected_route:
                results["correct_matches"] += 1
                results["route_stats"][expected_route]["correct_matches"] += 1
                results["route_stats"][expected_route]["average_confidence"] += result["confidence"]
            else:
                results["incorrect_matches"] += 1
                results["route_stats"][expected_route]["false_negatives"] += 1
                if matched_route != "no_match":
                    results["route_stats"][matched_route]["false_positives"] += 1
        
        # Calculate final stats
        results["accuracy"] = results["correct_matches"] / results["total_cases"]
        
        for route_name, stats in results["route_stats"].items():
            if stats["correct_matches"] > 0:
                stats["average_confidence"] /= stats["correct_matches"]
            
            total_predicted = stats["correct_matches"] + stats["false_positives"]
            stats["precision"] = stats["correct_matches"] / total_predicted if total_predicted > 0 else 0
            
            total_actual = stats["correct_matches"] + stats["false_negatives"]
            stats["recall"] = stats["correct_matches"] / total_actual if total_actual > 0 else 0
            
            if stats["precision"] + stats["recall"] > 0:
                stats["f1_score"] = 2 * (stats["precision"] * stats["recall"]) / (stats["precision"] + stats["recall"])
            else:
                stats["f1_score"] = 0
        
        return results
    
    @staticmethod
    def suggest_improvements(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest improvements based on performance analysis.
        
        Args:
            analysis: Analysis results from analyze_performance
            
        Returns:
            Dictionary with improvement suggestions
        """
        suggestions = {
            "route_suggestions": {},
            "general_suggestions": []
        }
        
        # Check overall accuracy
        if analysis["accuracy"] < 0.8:
            suggestions["general_suggestions"].append(
                "Overall accuracy is below 80%. Consider adding more diverse examples to all routes."
            )
        
        # Analyze each route
        for route_name, stats in analysis["route_stats"].items():
            route_suggestions = []
            
            # Low example count
            if stats["examples"] < 5:
                route_suggestions.append(
                    f"Add more examples (current: {stats['examples']}). Aim for at least 5-10 diverse examples."
                )
            
            # Low recall (missing matches)
            if stats["recall"] < 0.7 and stats["expected_count"] > 3:
                route_suggestions.append(
                    f"Improve recall ({stats['recall']*100:.1f}%). This route is missing {stats['false_negatives']} expected matches."
                )
            
            # Low precision (false positives)
            if stats["precision"] < 0.7 and (stats["correct_matches"] + stats["false_positives"]) > 3:
                route_suggestions.append(
                    f"Improve precision ({stats['precision']*100:.1f}%). This route has {stats['false_positives']} false positives."
                )
            
            # Confused with other routes
            confusions = []
            for other_route, count in analysis["confusion_matrix"][route_name].items():
                if other_route != route_name and count > 0:
                    confusions.append({"route": other_route, "count": count})
            
            if confusions:
                confusions.sort(key=lambda x: x["count"], reverse=True)
                top_confusion = confusions[0]
                
                route_suggestions.append(
                    f"Reduce confusion with '{top_confusion['route']}' ({top_confusion['count']} cases). Make examples more distinct."
                )
            
            if route_suggestions:
                suggestions["route_suggestions"][route_name] = route_suggestions
        
        return suggestions

# ===============================
# Enhanced Embedding Providers
# ===============================

class CachedEmbeddingProvider:
    """Wrapper for embedding providers with persistent caching."""
    
    def __init__(self, provider: EmbeddingProvider, cache_file: str = None):
        """
        Initialize a cached provider.
        
        Args:
            provider: Base embedding provider
            cache_file: File to persist cache (optional)
        """
        self.provider = provider
        self.cache_file = cache_file
        self.cache = {}
        
        # Load cache from file if provided
        if cache_file:
            try:
                import json
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)
                    for key, value in cache_data.items():
                        self.cache[key] = np.array(value)
                logger.info(f"Loaded {len(self.cache)} cached embeddings from {cache_file}")
            except (IOError, json.JSONDecodeError):
                logger.warning(f"Could not load cache from {cache_file}, starting with empty cache")
    
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings with caching."""
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = f"{self.provider.model_name}:{text}"
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # Get embeddings for texts not in cache
        if texts_to_embed:
            new_embeddings = await self.provider.get_embeddings(texts_to_embed)
            
            # Store in cache and insert at correct positions
            for j, embedding in enumerate(new_embeddings):
                text = texts_to_embed[j]
                cache_key = f"{self.provider.model_name}:{text}"
                self.cache[cache_key] = embedding
            
            # Insert at correct positions
            final_results = [None] * len(texts)
            for i, emb in enumerate(results):
                final_results[i] = emb
            
            for j, embedding in enumerate(new_embeddings):
                original_idx = indices_to_embed[j]
                final_results[original_idx] = embedding
            
            results = [r for r in final_results if r is not None]
        
        # Save cache to file if provided
        if self.cache_file and texts_to_embed:
            try:
                import json
                with open(self.cache_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_cache = {k: v.tolist() for k, v in self.cache.items()}
                    json.dump(serializable_cache, f)
                logger.info(f"Saved {len(self.cache)} embeddings to {self.cache_file}")
            except IOError:
                logger.warning(f"Could not save cache to {self.cache_file}")
        
        return results

# ===============================
# Example Usage
# ===============================

async def advanced_example():
    """Example usage of advanced MeaningMesh features."""
    from meaning_mesh import LocalEmbeddingProvider, experimental_vectorize, create_route, Router
    
    # Create embedding provider
    embedding_provider = LocalEmbeddingProvider(
        embed_fn=lambda texts: [experimental_vectorize(text) for text in texts]
    )
    
    # Create routes for first level
    domain_routes = [
        create_route(
            name="technical",
            examples=[
                "How do I fix my computer?",
                "What's wrong with my internet connection?",
                "My printer isn't working properly",
                "How to install this software?",
                "My device keeps crashing"
            ]
        ),
        create_route(
            name="account",
            examples=[
                "I can't log into my account",
                "How do I change my password?",
                "I need to update my billing information",
                "I want to cancel my subscription",
                "How do I update my email address?"
            ]
        )
    ]
    
    # Create technical support routes
    tech_routes = [
        create_route(
            name="connectivity",
            examples=[
                "My internet keeps disconnecting",
                "WiFi connection issues",
                "Cannot connect to the network",
                "Slow internet speed",
                "Router configuration problems"
            ],
            handler=lambda input_text, context: f"Connectivity troubleshooting for: '{input_text}'"
        ),
        create_route(
            name="hardware",
            examples=[
                "My computer won't turn on",
                "Blue screen error",
                "Printer not printing",
                "Device overheating",
                "Strange noises from my laptop"
            ],
            handler=lambda input_text, context: f"Hardware troubleshooting for: '{input_text}'"
        )
    ]
    
    # Create account routes
    account_routes = [
        create_route(
            name="login",
            examples=[
                "Forgot my password",
                "Cannot sign in",
                "Two-factor authentication issues",
                "Account locked out",
                "Username not recognized"
            ],
            handler=lambda input_text, context: f"Login assistance for: '{input_text}'"
        ),
        create_route(
            name="profile",
            examples=[
                "Update my personal information",
                "Change email address",
                "Update profile picture",
                "Privacy settings",
                "Account preferences"
            ],
            handler=lambda input_text, context: f"Profile management for: '{input_text}'"
        )
    ]
    
    # Create routers
    primary_router = Router(
        routes=domain_routes,
        embedding_provider=embedding_provider,
        default_threshold=0.6
    )
    
    technical_router = Router(
        routes=tech_routes,
        embedding_provider=embedding_provider,
        default_threshold=0.65
    )
    
    account_router = Router(
        routes=account_routes,
        embedding_provider=embedding_provider,
        default_threshold=0.65
    )
    
    # Create multi-stage router
    multi_router = MultiStageRouter(primary_router)
    multi_router.add_secondary_router("technical", technical_router)
    multi_router.add_secondary_router("account", account_router)
    
    # Test the multi-stage router
    queries = [
        "My internet connection keeps dropping every few minutes",
        "I forgot my password and can't log in to my account",
        "My laptop is making strange noises and overheating"
    ]
    
    print("=== Multi-Stage Routing Example ===")
    for query in queries:
        print(f"\nQuery: '{query}'")
        result = await multi_router.route(query)
        print(f"Routing path: {' → '.join(result['routing_path'])}")
        print(f"Response: {result['output']}")
        print("-" * 50)
    
    # Create contextual router example
    print("\n=== Contextual Router Example ===")
    context_router = ContextualRouter(technical_router, context_window=2, context_weight=0.4)
    
    # Simulate a conversation
    conversation = [
        "My internet is not working",
        "I've tried restarting the router",
        "Now my computer won't connect to WiFi"
    ]
    
    for i, message in enumerate(conversation):
        print(f"\nUser: '{message}'")
        result = await context_router.route(message)
        print(f"Matched route: {result['route'].name}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Response: {result['output']}")
        
        if i > 0 and "contextual_scores" in result:
            print("Context affected routing:")
            base_route = result["base_route"]
            actual_route = result["route"].name
            if base_route != actual_route:
                print(f"  Without context would have matched: {base_route}")
                print(f"  With context matched: {actual_route}")
    
    print("\n=== Route Tuner Example ===")
    # Create test cases
    test_cases = [
        {"input": "My WiFi connection keeps dropping", "expected_route": "connectivity"},
        {"input": "I can't connect to the internet", "expected_route": "connectivity"},
        {"input": "My router isn't working properly", "expected_route": "connectivity"},
        {"input": "My laptop won't turn on", "expected_route": "hardware"},
        {"input": "Computer making strange noises", "expected_route": "hardware"},
        {"input": "Blue screen when starting Windows", "expected_route": "hardware"}
    ]
    
    # Analyze performance
    analysis = await RouteTuner.analyze_performance(technical_router, test_cases)
    print(f"Accuracy: {analysis['accuracy']:.2f}")
    
    for route_name, stats in analysis["route_stats"].items():
        print(f"\nRoute: {route_name}")
        print(f"  Precision: {stats['precision']:.2f}")
        print(f"  Recall: {stats['recall']:.2f}")
        print(f"  F1 Score: {stats['f1_score']:.2f}")
    
    # Get suggestions
    suggestions = RouteTuner.suggest_improvements(analysis)
    
    print("\nImprovement Suggestions:")
    for suggestion in suggestions["general_suggestions"]:
        print(f"- {suggestion}")
    
    for route_name, route_suggestions in suggestions["route_suggestions"].items():
        print(f"\nFor route '{route_name}':")
        for suggestion in route_suggestions:
            print(f"- {suggestion}")

    
    def add_secondary_router(self, primary_route_name: str, router: Router) -> 'MultiStageRouter':
        """
        Add a secondary router for a primary route.
        
        Args:
            primary_route_name: Name of the primary route
            router: Secondary router to use when primary route matches
            
        Returns:
            Self for chaining
        """
        if primary_route_name not in self.primary_router.route_index:
            raise ValueError(f"Primary route '{primary_route_name}' not found")
        
        self.secondary_routers