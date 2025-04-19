"""
Unit tests for the semantic dispatcher.
"""

import asyncio
import unittest
from typing import List, Dict, Any

from meaning_mesh import (
    Path,
    SemanticDispatcher,
    create_vectorizer,
    InMemoryEmbeddingStore
)


class TestSemanticDispatcher(unittest.IsolatedAsyncioTestCase):
    """Test cases for the SemanticDispatcher."""
    
    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.vectorizer = create_vectorizer(provider="mock", dimensions=384)
        self.store = InMemoryEmbeddingStore()
        
        # Create test paths
        self.weather_path = Path(
            name="Weather",
            examples=["What's the weather like?", "Will it rain?", "Is it sunny?"],
            handler=lambda text, ctx: f"Weather: {text}"
        )
        
        self.greeting_path = Path(
            name="Greeting",
            examples=["Hello there", "Hi, how are you?", "Good morning"],
            handler=lambda text, ctx: f"Greeting: {text}"
        )
        
        self.support_path = Path(
            name="Support",
            examples=["I need help", "Support please", "Having an issue"],
            handler=lambda text, ctx: f"Support: {text}"
        )
        
        self.fallback_path = Path(
            name="Fallback",
            examples=[],
            handler=lambda text, ctx: f"Fallback: {text}"
        )
        
        # Create dispatcher with 0.3 threshold (suitable for mock vectorizer)
        self.dispatcher = SemanticDispatcher(
            vectorizer=self.vectorizer,
            store=self.store,
            confidence_threshold=0.3,
            fallback_path=self.fallback_path
        )
        
        # Register paths
        await self.dispatcher.register_path(self.weather_path)
        await self.dispatcher.register_path(self.greeting_path)
        await self.dispatcher.register_path(self.support_path)
    
    async def test_dispatch_exact_match(self):
        """Test dispatching with exact phrase matches."""
        # Test weather path
        result, response = await self.dispatcher.dispatch_and_handle("What's the weather like?")
        self.assertEqual(result.path.id, self.weather_path.id)
        self.assertGreaterEqual(result.confidence, 0.7)
        self.assertEqual(response, "Weather: What's the weather like?")
        
        # Test greeting path
        result, response = await self.dispatcher.dispatch_and_handle("Hello there")
        self.assertEqual(result.path.id, self.greeting_path.id)
        self.assertGreaterEqual(result.confidence, 0.7)
        self.assertEqual(response, "Greeting: Hello there")
        
        # Test support path
        result, response = await self.dispatcher.dispatch_and_handle("I need help")
        self.assertEqual(result.path.id, self.support_path.id)
        self.assertGreaterEqual(result.confidence, 0.7)
        self.assertEqual(response, "Support: I need help")
    
    async def test_dispatch_similar_match(self):
        """Test dispatching with semantically similar phrases."""
        # Test weather path
        result, response = await self.dispatcher.dispatch_and_handle("Is it going to rain today?")
        self.assertEqual(result.path.id, self.weather_path.id)
        self.assertGreaterEqual(result.confidence, 0.3)
        
        # Test greeting path
        result, response = await self.dispatcher.dispatch_and_handle("Hey there, how's it going?")
        self.assertEqual(result.path.id, self.greeting_path.id)
        self.assertGreaterEqual(result.confidence, 0.3)
        
        # Test support path
        result, response = await self.dispatcher.dispatch_and_handle("I'm having an issue with my account")
        self.assertEqual(result.path.id, self.support_path.id)
        self.assertGreaterEqual(result.confidence, 0.3)
    
    async def test_dispatch_fallback(self):
        """Test fallback for low-confidence matches."""
        # This should trigger fallback
        result, response = await self.dispatcher.dispatch_and_handle("Tell me a joke")
        self.assertEqual(result.path.id, self.fallback_path.id)
        self.assertTrue(result.fallback_used)
        self.assertEqual(response, "Fallback: Tell me a joke")
    
    async def test_all_matches(self):
        """Test retrieving all matches."""
        result = await self.dispatcher.dispatch(
            "Is it going to rain?",
            return_all_matches=True
        )
        
        # Ensure we have match data
        self.assertGreater(len(result.matches), 0)
        
        # First match should be weather path
        first_match_path, first_match_score = result.matches[0]
        self.assertEqual(first_match_path.id, self.weather_path.id)


if __name__ == "__main__":
    unittest.main()
