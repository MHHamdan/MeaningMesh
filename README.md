# MeaningMesh: Semantic Text Dispatching Framework

**MeaningMesh** is a Python framework for routing text to appropriate handlers based on **semantic meaning** rather than keywords or regex patterns. It enables developers to create more natural, robust text-processing systems by leveraging modern embedding models from various providers.

---

## 🚀 Overview

Traditional text routing often relies on keywords, regex patterns, or exact matches. MeaningMesh takes a different approach by using **semantic embeddings** to understand text meaning, creating more natural interactions and effectively handling linguistic variations and edge cases.

---

## ✨ Key Features

- **Provider-Agnostic Embedding:** Supports OpenAI, HuggingFace, Cohere, and custom embedding providers.
- **Pluggable Architecture:** Easily add new embedding providers.
- **Semantic Routing:** Routes based on meaning, not keywords.
- **Flexible Paths:** Define handlers with example phrases.
- **Confidence Thresholds:** Control routing precision.
- **Fallback Handlers:** Default handling for uncertain matches.
- **Context Awareness:** Leverages conversation history.
- **Hybrid Matching:** Combines semantic and keyword-based routing.
- **Asynchronous API:** Built with `asyncio`.
- **Type Annotations:** Fully typed codebase.

---

## 📂 Project Structure

```
meaning_mesh/
├── __init__.py                  # Package exports
├── vectorizers/                 # Embedding model interfaces
│   ├── __init__.py
│   ├── base.py                  # Base vectorizer interface
│   ├── openai.py                # OpenAI embeddings
│   ├── huggingface.py           # HuggingFace embeddings
│   ├── cohere.py                # Cohere embeddings
│   └── mock.py                  # Mock vectorizer (testing)
├── paths/                       # Destination definitions
│   ├── __init__.py
│   └── path.py                  # Path class definition
├── dispatchers/                 # Routing logic
│   ├── __init__.py
│   └── semantic_dispatcher.py   # Main dispatcher
├── storage/                     # Embedding storage
│   ├── __init__.py
│   ├── base.py                  # Base storage interface
│   └── memory.py                # In-memory storage
└── utils/                       # Helper functions
    ├── __init__.py
    └── similarity.py            # Similarity computations

examples/
├── basic_example.py             # Basic usage
├── embedding_providers.py       # Different embeddings
├── hybrid_matching.py           # Semantic + keyword routing
└── context_awareness.py         # Contextual dispatching

tests/
└── test_dispatcher.py           # Unit tests
```

---

## ⚡ Quick Start

### Example Usage

```python
import asyncio
from meaning_mesh import Path, SemanticDispatcher, InMemoryEmbeddingStore
from meaning_mesh.vectorizers import HuggingFaceVectorizer

async def weather_handler(text, context):
    return f"Weather service: {text}"

async def greeting_handler(text, context):
    return f"Greeting service: {text}"

weather_path = Path(
    name="Weather",
    examples=["What's the weather today?", "Rain tomorrow?"],
    handler=weather_handler
)

greeting_path = Path(
    name="Greetings",
    examples=["Hello!", "Good morning!"],
    handler=greeting_handler
)

async def main():
    vectorizer = HuggingFaceVectorizer(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = InMemoryEmbeddingStore()

    dispatcher = SemanticDispatcher(vectorizer, store, confidence_threshold=0.7)

    await dispatcher.register_path(weather_path)
    await dispatcher.register_path(greeting_path)

    result, response = await dispatcher.dispatch_and_handle("What's tomorrow's forecast?")

    print(f"Matched path: {result.path.name}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/MHHamdan/MeaningMesh.git
cd MeaningMesh
```

Install in development mode:

```bash
pip install -e ".[all]"
```

Or with specific embedding providers:

```bash
pip install -e ".[openai]"
pip install -e ".[huggingface]"
pip install -e ".[cohere]"
pip install -e ".[dev]"
```

---

## 🌟 Use Cases

MeaningMesh is ideal for:

- **Chatbots:** Route user messages effectively.
- **Customer Support:** Specialized inquiry handling.
- **Content Classification:** Semantic categorization.
- **Intent Recognition:** Natural language intent detection.
- **Command Routing:** Direct commands to services.

---

## 🔧 Advanced Features

### Custom Embedding Providers

Extend with your own embeddings:

```python
from meaning_mesh.vectorizers import Vectorizer
from typing import List

class CustomVectorizer(Vectorizer):
    def __init__(self, **kwargs):
        self.model = YourEmbeddingModel(**kwargs)

    async def vectorize(self, texts: List[str]) -> List[List[float]]:
        return [await self.vectorize_single(text) for text in texts]

    async def vectorize_single(self, text: str) -> List[float]:
        return self.model.embed(text)
```

### Hybrid Matching

Combine semantic and keyword-based routing:

```python
from examples.hybrid_matching import HybridDispatcher

hybrid_dispatcher = HybridDispatcher(semantic_dispatcher, semantic_weight=0.7)
hybrid_dispatcher.add_keyword_pattern(weather_path.id, r'\b(weather|forecast)\b', weight=0.9)
```

### Context-Aware Dispatching

Maintain context:

```python
from examples.context_awareness import ContextAwareDispatcher

context_dispatcher = ContextAwareDispatcher(semantic_dispatcher, context_weight=0.3)
result, response, conversation_id = await context_dispatcher.dispatch_and_handle(
    "Tomorrow?", conversation_id="conversation-123"
)
```

---

## 🤝 Contributing

Contributions are welcome! Submit a pull request to propose changes.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
