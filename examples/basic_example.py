"""
Basic example demonstrating the usage of MeaningMesh.

This example sets up a simple dispatcher with three paths:
1. Weather inquiries
2. Greetings
3. Customer support

It then shows how to dispatch different inputs to these paths.
"""

import asyncio
from typing import Dict, Any

# Import MeaningMesh components
from meaning_mesh import (
    Path,
    SemanticDispatcher,
    create_vectorizer,
    InMemoryEmbeddingStore
)


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
    
    # Initialize components with the mock vectorizer
    # The mock vectorizer simulates semantic similarity for testing
    vectorizer = create_vectorizer(
        provider="mock",       # Options: "openai", "huggingface", "cohere", "mock"
        dimensions=384,        # Mock embedding dimensions
        semantic_boost=0.7     # How much to boost semantic matching
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
    
    # For the mock vectorizer, use a lower confidence threshold
    # Real embedding models typically produce higher similarity scores
    confidence_threshold = 0.3  # Lower threshold for the mock vectorizer
    
    # Create dispatcher
    dispatcher = SemanticDispatcher(
        vectorizer=vectorizer,
        store=store,
        confidence_threshold=confidence_threshold,
        fallback_path=fallback_path
    )
    
    print("Registering paths...")
    # Register paths
    await dispatcher.register_path(weather_path)
    await dispatcher.register_path(greeting_path)
    await dispatcher.register_path(support_path)
    
    # Test inputs
    test_inputs = [
        "What's the forecast for tomorrow?",
        "Hey there, how's it going?",
        "My order #12345 is missing an item",
        "Tell me a joke"  # Should trigger fallback
    ]
    
    print("\nTesting dispatcher with example inputs:")
    print("-" * 50)
    
    for text in test_inputs:
        print(f"\nInput: \"{text}\"")
        result, response = await dispatcher.dispatch_and_handle(text)
        print(f"Matched path: {result.path.name if result.path else 'None'}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Fallback used: {result.fallback_used}")
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
