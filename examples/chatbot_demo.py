"""
Enhanced chatbot demo using MeaningMesh for intent recognition.
"""

import asyncio
import sys
import os
import re
from typing import Dict, Any, List, Optional

# Add the parent directory to Python's path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meaning_mesh import (
    Path,
    SemanticDispatcher,
    create_vectorizer,
    InMemoryEmbeddingStore
)
from examples.context_awareness import ContextAwareDispatcher


# Define handlers for the chatbot
async def greeting_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle greeting intents."""
    conversation = context.get("conversation", {})
    history = conversation.get("history", [])
    
    # Check if asking about chatbot's well-being
    if re.search(r'how are you|how\'s it going|how do you feel|how are things', text.lower()):
        return {
            "message": "I'm doing great, thanks for asking! How can I help you today?",
            "suggestions": ["Tell me about the weather", "Help with my order", "What can you do?"]
        }
    
    if len(history) <= 1:
        return {
            "message": "Hello! I'm MeaningMesh Bot. How can I help you today?",
            "suggestions": ["Tell me about the weather", "I need help with my order", "What can you do?"]
        }
    else:
        return {
            "message": "I'm here! What can I help you with?",
            "suggestions": ["Tell me about the weather", "I need help with my order"]
        }


async def weather_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle weather inquiries."""
    # Check for specific weather queries
    if 'rain' in text.lower() or 'umbrella' in text.lower():
        return {
            "message": "There's a 30% chance of rain today, so bringing an umbrella might be a good idea!",
            "suggestions": ["What about tomorrow?", "Will it be cold?"]
        }
    elif 'temperature' in text.lower() or 'hot' in text.lower() or 'cold' in text.lower():
        return {
            "message": "The current temperature is 75°F (24°C). It should stay warm throughout the day.",
            "suggestions": ["Will it rain?", "Weekend forecast?"]
        }
    else:
        return {
            "message": "It looks like it will be sunny today with a high of 75°F!",
            "suggestions": ["Will it rain tomorrow?", "What's the temperature?"]
        }


async def order_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle order inquiries."""
    return {
        "message": "I'd be happy to help with your order. What's your order number?",
        "suggestions": ["My order number is #12345", "I don't know my order number"]
    }


async def order_status_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle order status inquiries."""
    # Check if there's an order number in the text
    order_match = re.search(r'#?(\d{5,})', text)
    
    if order_match:
        order_number = order_match.group(1)
        return {
            "message": f"Your order #{order_number} is currently in transit and should arrive by Friday!",
            "suggestions": ["Where is it right now?", "Can I change the delivery address?"]
        }
    else:
        return {
            "message": "I'll need your order number to check the status. It should be in your confirmation email.",
            "suggestions": ["My order number is #12345", "I lost my order number"]
        }


async def smalltalk_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle small talk and casual conversation."""
    if re.search(r'thank|thanks', text.lower()):
        return {
            "message": "You're welcome! Is there anything else I can help you with?",
            "suggestions": ["Check weather", "Order status", "No thanks"]
        }
    elif re.search(r'how (do you work|does this work)', text.lower()):
        return {
            "message": "I use semantic understanding to route your questions to the right handler based on meaning, not just keywords!",
            "suggestions": ["Tell me more", "Try weather info", "Try order help"]
        }
    else:
        return {
            "message": "I enjoy our conversations! Is there something specific I can help you with?",
            "suggestions": ["Weather", "Orders", "Help"]
        }


async def help_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle help requests."""
    return {
        "message": "I can help with the following:\n- Weather information\n- Order status updates\n- General questions\n\nJust ask me anything in natural language!",
        "suggestions": ["Check weather", "Order status", "Tell me a joke"]
    }


async def fallback_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle unknown intents."""
    return {
        "message": "I'm not sure I understand. Can you try rephrasing that?",
        "suggestions": ["Help", "Check weather", "Order status"]
    }


class ChatBot:
    """Enhanced chatbot using MeaningMesh for intent recognition."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.conversation_id = None
        self.dispatcher = None
    
    async def initialize(self):
        """Initialize the dispatcher and paths."""
        # Create vectorizer and store
        vectorizer = create_vectorizer("mock", semantic_boost=0.8)
        store = InMemoryEmbeddingStore()
        
        # Create paths with more comprehensive examples
        greeting_path = Path(
            name="Greeting",
            examples=[
                "Hello there!",
                "Hi, how are you?",
                "Good morning",
                "Hey, what's up?",
                "Greetings!",
                "How are you doing?",
                "How's it going?",
                "Nice to meet you",
                "Hello, I have a question"
            ],
            handler=greeting_handler
        )
        
        weather_path = Path(
            name="Weather",
            examples=[
                "What's the weather like today?",
                "Will it rain tomorrow?",
                "Is it sunny outside?",
                "What's the temperature right now?",
                "Should I bring an umbrella?",
                "How hot will it be?",
                "Is it going to be cold?",
                "Weekend forecast",
                "Weather for tomorrow"
            ],
            handler=weather_handler
        )
        
        order_path = Path(
            name="Order",
            examples=[
                "I want to check my order",
                "Help with my purchase",
                "Where is my order?",
                "I have a question about my order",
                "Order information",
                "I bought something",
                "My purchase",
                "I need help with something I bought"
            ],
            handler=order_handler
        )
        
        order_status_path = Path(
            name="OrderStatus",
            examples=[
                "What's the status of my order?",
                "My order number is #12345",
                "When will my order arrive?",
                "Has my order shipped?",
                "Track my package",
                "Delivery status",
                "Is my order on the way?",
                "Order tracking"
            ],
            handler=order_status_handler
        )
        
        smalltalk_path = Path(
            name="SmallTalk",
            examples=[
                "Thanks for your help",
                "Thank you",
                "You're doing great",
                "How does this work?",
                "That's interesting",
                "Nice talking to you",
                "You're helpful",
                "I appreciate it"
            ],
            handler=smalltalk_handler
        )
        
        help_path = Path(
            name="Help",
            examples=[
                "What can you do?",
                "Help me",
                "I need assistance",
                "Show me what you can do",
                "What are your features?",
                "How can you help me?",
                "What are you capable of?",
                "Give me some options"
            ],
            handler=help_handler
        )
        
        fallback_path = Path(
            name="Fallback",
            examples=[],
            handler=fallback_handler
        )
        
        # Create semantic dispatcher with lower threshold for better matching
        semantic_dispatcher = SemanticDispatcher(
            vectorizer=vectorizer,
            store=store,
            confidence_threshold=0.25,  # Lower threshold for more matches
            fallback_path=fallback_path
        )
        
        # Register paths
        await semantic_dispatcher.register_path(greeting_path)
        await semantic_dispatcher.register_path(weather_path)
        await semantic_dispatcher.register_path(order_path)
        await semantic_dispatcher.register_path(order_status_path)
        await semantic_dispatcher.register_path(smalltalk_path)
        await semantic_dispatcher.register_path(help_path)
        
        # Create context-aware dispatcher
        self.dispatcher = ContextAwareDispatcher(
            semantic_dispatcher=semantic_dispatcher,
            context_weight=0.3
        )
    
    async def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            message: User message
            
        Returns:
            Response with message and suggestions
        """
        if not self.dispatcher:
            await self.initialize()
        
        # Dispatch message
        result, response, conversation_id = await self.dispatcher.dispatch_and_handle(
            message, self.conversation_id
        )
        
        # Update conversation ID
        self.conversation_id = conversation_id
        
        # Return response
        return {
            "intent": result.path.name if result.path else "Unknown",
            "confidence": result.confidence,
            "fallback_used": result.fallback_used,
            "response": response
        }


async def interactive_chat():
    """Run an interactive chat session."""
    chatbot = ChatBot()
    
    print("Starting MeaningMesh ChatBot...")
    print("Type 'exit' to quit.")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nChatbot: Goodbye!")
            break
        
        # Process the message
        response = await chatbot.process_message(user_input)
        
        # Display debug info (intent and confidence)
        print(f"[Intent: {response['intent']}, Confidence: {response['confidence']:.4f}]")
        
        # Display the chatbot's response
        print(f"\nChatbot: {response['response']['message']}")
        
        # Display suggestions
        if "suggestions" in response["response"]:
            print("\nSuggestions:")
            for suggestion in response["response"]["suggestions"]:
                print(f"- {suggestion}")


if __name__ == "__main__":
    asyncio.run(interactive_chat())
