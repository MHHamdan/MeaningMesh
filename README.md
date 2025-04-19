# MeaningMesh: Semantic Text Dispatching Framework

MeaningMesh is a Python framework for routing text to appropriate handlers based on semantic meaning rather than keywords or regex patterns. It enables developers to create more natural and robust text processing systems by leveraging modern embedding models.

## Overview

Traditional text routing relies on keywords, regex patterns, or exact matches. MeaningMesh takes a different approach, using semantic embeddings to understand the meaning behind text and route it to the most appropriate handler. This creates more natural interactions and better handles edge cases and variations in language.

## Key Features

- **Semantic Routing**: Routes text based on meaning, not just keywords
- **Multiple Embedding Providers**: Supports OpenAI, HuggingFace, and custom vectorizers
- **Flexible Paths**: Define destinations with example phrases that represent their domain
- **Confidence Thresholds**: Configure minimum confidence levels for matches
- **Fallback Handlers**: Define default behavior for low-confidence matches
- **Context Awareness**: Take conversation history into account for better routing
- **Hybrid Matching**: Combine semantic and keyword approaches for optimal results
- **Asynchronous API**: Built with asyncio for efficient processing
- **Caching**: Performance optimization through caching mechanisms
- **Type Annotations**: Comprehensive typing throughout the codebase

## Project Structure

```
meaning_mesh/
├── __init__.py                 # Package exports
├── vectorizers/                # Embedding model interfaces
│   ├── __init__.py
│   ├── base.py                 # Base vectorizer interface
│   ├── openai.py               # OpenAI embeddings implementation
│   └── huggingface.py          # HuggingFace embeddings implementation
├── paths/                      # Destination definitions
│   ├── __init__.py
│   └── path.py                 # Path class definition
├── dispatchers/                # Routing logic
│   ├── __init__.py
│   └── semantic_dispatcher.py  # Main dispatcher implementation
├── storage/                    # Embedding storage and retrieval
│   ├── __init__.py
│   ├── base.py                 # Base storage interface
│   └── memory.py               # In-memory implementation
└── utils/                      # Helper functions
    ├── __init__.py
    └── similarity.py           # Vector similarity functions

examples/
├── basic_example.py            # Simple dispatcher usage
├── hybrid_matching.py          # Combining semantic with keywords
└── context_awareness.py        # Contextual dispatching example

tests/
└── test_dispatcher.py          # Unit tests
```

## Quick Start

Here's a simple example of using MeaningMesh:

```python
import asyncio
from meaning_mesh import (
    Path,
    SemanticDispatcher,
    HuggingFaceVectorizer,
    InMemoryEmbeddingStore
)

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
    vectorizer = HuggingFaceVectorizer()
    store = InMemoryEmbeddingStore()
    
    # Create dispatcher
    dispatcher = SemanticDispatcher(
        vectorizer=vectorizer,
        store=store,
        confidence_threshold=0.7
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

## Installation

```bash
# Clone the repository
git clone https://github.com/MHHamdan/LLM_Reasoning.git
cd LLM_Reasoning/MeaningMesh

# Install the package in development mode
pip install -e .

# With HuggingFace support
pip install -e ".[huggingface]"

# With development tools
pip install -e ".[dev]"
```

## Use Cases

MeaningMesh is ideal for:

- **Chatbots**: Route user messages to appropriate handlers
- **Customer Support**: Direct inquiries to specialized agents
- **Content Classification**: Categorize texts by semantic meaning
- **Intent Recognition**: Identify user intents in natural language
- **Command Routing**: Direct textual commands to appropriate services

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
```

### Context-Aware Dispatching

Maintain conversation context for better dispatching:

```python
from examples.context_awareness import ContextAwareDispatcher

context_dispatcher = ContextAwareDispatcher(
    semantic_dispatcher=semantic_dispatcher,
    context_weight=0.3  # 30% context, 70% semantic
)

# Dispatch with conversation context
result, response, conversation_id = await context_dispatcher.dispatch_and_handle(
    "What about tomorrow?",
    conversation_id="conversation-123"
)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.