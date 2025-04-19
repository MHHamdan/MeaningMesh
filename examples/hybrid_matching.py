"""
Advanced example demonstrating hybrid matching in MeaningMesh.

This example combines semantic matching with keyword/regex patterns
to improve dispatch accuracy, especially for edge cases.
"""

import asyncio
import re
from typing import Dict, Any, List, Tuple, Optional

# Import MeaningMesh components
from meaning_mesh import (
    Path,
    SemanticDispatcher,
    DispatchResult,
    create_vectorizer,
    InMemoryEmbeddingStore
)


class HybridDispatcher:
    """
    Dispatcher that combines semantic matching with keyword/regex matching.
    """
    
    def __init__(
        self,
        semantic_dispatcher: SemanticDispatcher,
        keyword_patterns: Dict[str, List[Tuple[str, float]]] = None,
        semantic_weight: float = 0.7
    ):
        """
        Initialize a HybridDispatcher.
        
        Args:
            semantic_dispatcher: The semantic dispatcher to use
            keyword_patterns: Dict mapping path IDs to lists of (regex, weight) tuples
            semantic_weight: Weight to give semantic matching vs keyword matching
        """
        self.semantic_dispatcher = semantic_dispatcher
        self.keyword_patterns = keyword_patterns or {}
        self.semantic_weight = semantic_weight
    
    def add_keyword_pattern(self, path_id: str, pattern: str, weight: float = 1.0):
        """
        Add a keyword/regex pattern for a path.
        
        Args:
            path_id: ID of the path
            pattern: Regex pattern to match
            weight: Weight to give this pattern when matched
        """
        if path_id not in self.keyword_patterns:
            self.keyword_patterns[path_id] = []
        self.keyword_patterns[path_id].append((pattern, weight))
    
    async def get_keyword_confidence(
        self, 
        text: str, 
        path_id: str
    ) -> float:
        """
        Calculate confidence based on keyword matching.
        
        Args:
            text: Input text
            path_id: Path ID to check patterns for
            
        Returns:
            Confidence score based on keyword matches
        """
        patterns = self.keyword_patterns.get(path_id, [])
        if not patterns:
            return 0.0
        
        max_confidence = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                max_confidence = max(max_confidence, weight)
        
        return max_confidence
    
    async def dispatch(
        self, 
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> DispatchResult:
        """
        Dispatch text using both semantic and keyword matching.
        
        Args:
            text: The input text to dispatch
            context: Optional context information
            
        Returns:
            DispatchResult with the matched path and confidence
        """
        # Get semantic dispatch result
        semantic_result = await self.semantic_dispatcher.dispatch(
            text, context, return_all_matches=True
        )
        
        if not semantic_result.matches:
            return semantic_result
        
        # Initialize combined matches
        combined_matches = []
        
        # Process each match with keyword boosting
        for path, semantic_score in semantic_result.matches:
            # Get keyword confidence
            keyword_confidence = await self.get_keyword_confidence(text, path.id)
            
            # Combine confidences
            combined_confidence = (
                self.semantic_weight * semantic_score +
                (1 - self.semantic_weight) * keyword_confidence
            )
            
            combined_matches.append((path, combined_confidence))
        
        # Sort combined matches by confidence
        combined_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get best match
        best_path, best_score = combined_matches[0] if combined_matches else (None, 0.0)
        
        # Check against confidence threshold
        if best_score >= self.semantic_dispatcher.confidence_threshold:
            return DispatchResult(
                path=best_path,
                confidence=best_score,
                text=text,
                matches=combined_matches
            )
        elif self.semantic_dispatcher.fallback_path:
            return DispatchResult(
                path=self.semantic_dispatcher.fallback_path,
                confidence=best_score,
                text=text,
                fallback_used=True,
                matches=combined_matches
            )
        else:
            return DispatchResult(
                path=best_path,
                confidence=best_score,
                text=text,
                matches=combined_matches
            )
    
    async def dispatch_and_handle(
        self, 
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[DispatchResult, Any]:
        """
        Dispatch text and invoke the handler of the matched path.
        
        Args:
            text: The input text to dispatch
            context: Optional context information
            
        Returns:
            Tuple of (dispatch_result, handler_result)
        """
        result = await self.dispatch(text, context)
        
        if result.path:
            # Call the path's handler
            handler_result = await result.path.handle(text, context)
            return result, handler_result
        
        return result, None


# Define handlers for each path
async def weather_handler(text: str, context: Dict[str, Any]) -> str:
    return f"Weather service: You asked about weather with: '{text}'"


async def greeting_handler(text: str, context: Dict[str, Any]) -> str:
    return f"Greeting service: Hello there! You said: '{text}'"


async def support_handler(text: str, context: Dict[str, Any]) -> str:
    return f"Support service: I'll help with your issue: '{text}'"


async def fallback_handler(text: str, context: Dict[str, Any]) -> str:
    return f"I'm not sure how to handle: '{text}'. Can you try rephrasing?"


async def main():
    print("Initializing MeaningMesh components...")
    
    # Initialize components
    vectorizer = create_vectorizer(
        provider="mock",
        dimensions=384,
        semantic_boost=0.7
    )
    store = InMemoryEmbeddingStore()
    
    # Create paths with example phrases
    weather_path = Path(
        name="Weather Inquiries",
        examples=[
            "What's the weather like today?",
            "Will it rain tomorrow?",
            "Is it sunny outside?",
            "What's the temperature right now?",
            "Should I bring an umbrella?"
        ],
        handler=weather_handler
    )
    
    greeting_path = Path(
        name="Greetings",
        examples=[
            "Hello there!",
            "Hi, how are you?",
            "Good morning",
            "Hey, what's up?",
            "Greetings!"
        ],
        handler=greeting_handler
    )
    
    support_path = Path(
        name="Customer Support",
        examples=[
            "I have a problem with my order",
            "My package hasn't arrived yet",
            "How do I return this item?",
            "The product is defective",
            "I need help with my account"
        ],
        handler=support_handler
    )
    
    fallback_path = Path(
        name="Fallback",
        examples=[],
        handler=fallback_handler
    )
    
    # Create semantic dispatcher with lower threshold
    semantic_dispatcher = SemanticDispatcher(
        vectorizer=vectorizer,
        store=store,
        confidence_threshold=0.3,  # Lower threshold for the mock vectorizer
        fallback_path=fallback_path
    )
    
    print("Registering paths...")
    # Register paths
    await semantic_dispatcher.register_path(weather_path)
    await semantic_dispatcher.register_path(greeting_path)
    await semantic_dispatcher.register_path(support_path)
    
    # Create hybrid dispatcher
    hybrid_dispatcher = HybridDispatcher(
        semantic_dispatcher=semantic_dispatcher,
        semantic_weight=0.7  # 70% semantic, 30% keyword
    )
    
    # Add keyword patterns for each path
    print("Adding keyword patterns...")
    hybrid_dispatcher.add_keyword_pattern(
        weather_path.id, 
        r'\b(weather|temperature|rain|sunny|forecast|cold|hot|warm|umbrella)\b', 
        weight=0.9
    )
    
    hybrid_dispatcher.add_keyword_pattern(
        greeting_path.id, 
        r'\b(hello|hi|hey|greetings|morning|afternoon|evening|howdy)\b', 
        weight=0.8
    )
    
    hybrid_dispatcher.add_keyword_pattern(
        support_path.id, 
        r'\b(order|package|return|defective|broken|account|help|support|issue|problem)\b', 
        weight=0.9
    )
    
    # Test inputs
    test_inputs = [
        "What's the forecast for tomorrow?",
        "Hey there, how's it going?",
        "My order #12345 is missing an item",
        "I need to check on my package status",  # Contains keyword "package"
        "Will I need an umbrella this weekend?",  # Weather-related but phrased differently
        "Tell me a joke"  # Should trigger fallback
    ]
    
    print("\nComparing semantic vs hybrid dispatching:")
    print("-" * 70)
    
    for text in test_inputs:
        print(f"\nInput: \"{text}\"")
        
        # Compare semantic vs hybrid
        semantic_result, semantic_response = await semantic_dispatcher.dispatch_and_handle(text)
        hybrid_result, hybrid_response = await hybrid_dispatcher.dispatch_and_handle(text)
        
        print("Semantic match:")
        print(f"  Path: {semantic_result.path.name if semantic_result.path else 'None'}")
        print(f"  Confidence: {semantic_result.confidence:.4f}")
        print(f"  Fallback used: {semantic_result.fallback_used}")
        
        print("Hybrid match:")
        print(f"  Path: {hybrid_result.path.name if hybrid_result.path else 'None'}")
        print(f"  Confidence: {hybrid_result.confidence:.4f}")
        print(f"  Fallback used: {hybrid_result.fallback_used}")
        print(f"  Response: {hybrid_response}")
        print("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
