# MeaningMesh: Semantic Text Dispatching Framework

MeaningMesh is a Python framework for routing text to appropriate handlers based on semantic meaning rather than keywords or regex patterns. It enables developers to create more natural and robust text processing systems by leveraging modern embedding models from any provider.

## Overview

Traditional text routing relies on keywords, regex patterns, or exact matches. MeaningMesh takes a different approach, using semantic embeddings to understand the meaning behind text and route it to the most appropriate handler. This creates more natural interactions and better handles edge cases and variations in language.

## Key Features

- **Provider-Agnostic Embedding**: Easily switch between OpenAI, HuggingFace, Cohere, or any custom embedding provider
- **Pluggable Architecture**: Add new embedding providers with minimal code
- **Semantic Routing**: Routes text based on meaning, not just keywords
- **Flexible Paths**: Define destinations with example phrases that represent their domain
- **Confidence Thresholds**: Configure minimum confidence levels for matches
- **Fallback Handlers**: Define default behavior for low-confidence matches
- **Hybrid Matching**: Combine semantic and keyword approaches for optimal results
- **Testing Support**: Mock vectorizer for development without external dependencies

## Installation

```bash
# Clone the repository
git clone https://github.com/MHHamdan/MeaningMesh.git
cd MeaningMesh

# Install the package in development mode with minimal dependencies
pip install -e .

# Or install with specific provider support:
pip install -e ".[openai]"     # For OpenAI support
pip install -e ".[huggingface]"  # For HuggingFace support
pip install -e ".[cohere]"     # For Cohere support
pip install -e ".[all]"        # For all providers
pip install -e ".[dev]"        # For development tools
```

## Quick Start

Here's a simple example of using MeaningMesh:

```python
import asyncio
from meaning_mesh import Path, SemanticDispatcher, create_vectorizer, InMemoryEmbeddingStore

# Define handlers
async def weather_handler(text, context):
    return f"Weather service: {text}"

async def greeting_handler(text, context):
    return f"Greeting service: {text}"

# Create paths with example phrases
weather_path = Path(
    name="Weather",
    examples=[
        "What's the weather like today?",
        "Will it rain tomorrow?",
        "Is it sunny outside?"
    ],
    handler=weather_handler
)

greeting_path = Path(
    name="Greetings",
    examples=[
        "Hello there!",
        "Hi, how are you?",
        "Good morning"
    ],
    handler=greeting_handler
)

async def main():
    # Initialize components
    # For development, use the mock vectorizer
    vectorizer = create_vectorizer(provider="mock")
    
    # For production, use OpenAI, HuggingFace or Cohere
    # vectorizer = create_vectorizer(provider="openai", api_key="your-api-key")
    # vectorizer = create_vectorizer(provider="huggingface")
    # vectorizer = create_vectorizer(provider="cohere", api_key="your-api-key")
    
    store = InMemoryEmbeddingStore()
    
    # Create dispatcher
    dispatcher = SemanticDispatcher(
        vectorizer=vectorizer,
        store=store,
        confidence_threshold=0.3  # Use 0.3 for mock, 0.7 for real providers
    )
    
    # Register paths
    await dispatcher.register_path(weather_path)
    await dispatcher.register_path(greeting_path)
    
    # Test dispatch
    result, response = await dispatcher.dispatch_and_handle(
        "What's the forecast for tomorrow?"
    )
    
    print(f"Matched path: {result.path.name}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Components

### Path

A `Path` represents a destination for routing with example phrases that define its semantic domain.

```python
from meaning_mesh import Path

support_path = Path(
    name="Customer Support",
    examples=[
        "I have a problem with my order",
        "My package hasn't arrived yet",
        "How do I return this item?"
    ],
    handler=lambda text, context: f"Support: {text}",
    metadata={"department": "customer_service"}
)
```

### Vectorizers

Vectorizers convert text to embeddings. MeaningMesh includes:

- `OpenAIVectorizer`: Uses OpenAI's embedding models
- `HuggingFaceVectorizer`: Uses HuggingFace's sentence-transformers
- `CohereVectorizer`: Uses Cohere's embedding models
- `MockVectorizer`: Creates simulated embeddings for testing

You can use the `create_vectorizer` factory function to easily create any type:

```python
from meaning_mesh import create_vectorizer

# Create a vectorizer based on provider name
vectorizer = create_vectorizer(
    provider="openai",  # Options: "openai", "huggingface", "cohere", "mock"
    api_key="your-api-key",  # Required for OpenAI and Cohere
    model="text-embedding-3-small"  # Optional model specification
)
```

### Dispatcher

The `SemanticDispatcher` routes text to paths based on semantic similarity:

```python
from meaning_mesh import SemanticDispatcher

dispatcher = SemanticDispatcher(
    vectorizer=vectorizer,
    store=store,
    similarity_fn="cosine",  # Similarity function: "cosine", "dot_product", or "euclidean"
    confidence_threshold=0.7,  # Minimum confidence for a match
    fallback_path=fallback_path  # Optional path for low-confidence matches
)

# Register paths
await dispatcher.register_path(weather_path)
await dispatcher.register_path(greeting_path)

# Dispatch text
result = await dispatcher.dispatch("What's the weather like?")
print(f"Matched: {result.path.name}, Confidence: {result.confidence}")

# Dispatch and handle
result, response = await dispatcher.dispatch_and_handle("Hi there!")
print(f"Response: {response}")
```

## Advanced Features

### Hybrid Matching

Combine semantic and keyword-based approaches:

```python
from examples.hybrid_matching import HybridDispatcher

hybrid_dispatcher = HybridDispatcher(
    semantic_dispatcher=semantic_dispatcher,
    semantic_weight=0.7  # 70% semantic, 30% keyword
)

# Add keyword patterns
hybrid_dispatcher.add_keyword_pattern(
    weather_path.id, 
    r'\b(weather|temperature|rain|sunny|forecast)\b', 
    weight=0.9
)

# Dispatch using hybrid matching
result, response = await hybrid_dispatcher.dispatch_and_handle(
    "What's the temperature going to be?"
)
```

## Running Tests

To run the unit tests:

```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests
python -m unittest discover tests
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.