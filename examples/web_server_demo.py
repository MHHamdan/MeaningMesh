"""
Self-contained web server demo using MeaningMesh for intent recognition.

This version includes all necessary code without external dependencies.
"""

import asyncio
import json
import sys
import os
import re
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union, Callable, Awaitable

# Add the parent directory to Python's path to ensure imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First, install aiohttp if not already installed
try:
    import aiohttp
    from aiohttp import web
except ImportError:
    import subprocess
    import sys
    print("Installing aiohttp...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "aiohttp"])
    import aiohttp
    from aiohttp import web

# Import core MeaningMesh components
from meaning_mesh import (
    Path,
    SemanticDispatcher,
    DispatchResult,
    create_vectorizer,
    InMemoryEmbeddingStore
)


###########################################
# Context-Aware Dispatcher Implementation #
###########################################

class ConversationContext:
    """Class to manage conversation context."""
    
    def __init__(
        self,
        max_history: int = 5,
        path_stickiness: float = 0.8,
        max_stickiness_turns: int = 3
    ):
        """
        Initialize a ConversationContext.
        
        Args:
            max_history: Maximum number of turns to remember
            path_stickiness: Factor to boost the previous path (0-1)
            max_stickiness_turns: How many turns path stickiness persists
        """
        self.conversation_id = str(uuid.uuid4())
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.path_stickiness = path_stickiness
        self.max_stickiness_turns = max_stickiness_turns
        self.current_path_id: Optional[str] = None
        self.turns_in_current_path = 0
    
    def add_turn(
        self,
        text: str,
        path_id: Optional[str],
        confidence: float,
        response: Optional[Any] = None
    ) -> None:
        """
        Add a turn to the conversation history.
        
        Args:
            text: User input text
            path_id: ID of the matched path
            confidence: Confidence score
            response: Optional system response
        """
        turn = {
            "text": text,
            "path_id": path_id,
            "confidence": confidence,
            "response": response,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        self.history.append(turn)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Update path stickiness tracking
        if path_id != self.current_path_id:
            self.current_path_id = path_id
            self.turns_in_current_path = 1
        else:
            self.turns_in_current_path += 1
    
    def get_path_boost(self, path_id: str) -> float:
        """
        Get confidence boost for a specific path based on conversation context.
        
        Args:
            path_id: Path ID to check
            
        Returns:
            Confidence boost factor (0-1)
        """
        # If we've been in the same path for too many turns, reduce stickiness
        if (
            path_id == self.current_path_id and 
            self.turns_in_current_path <= self.max_stickiness_turns
        ):
            # Linear decay of stickiness over turns
            decay_factor = 1 - (self.turns_in_current_path / self.max_stickiness_turns)
            return self.path_stickiness * decay_factor
        
        # Check if this path appeared in recent history
        recency_factor = 0.0
        for i, turn in enumerate(reversed(self.history)):
            if turn["path_id"] == path_id:
                # More recent turns get higher weight
                position_weight = 1.0 - (i / len(self.history))
                recency_factor = max(recency_factor, position_weight * 0.3)
                
        return recency_factor


class ContextAwareDispatcher:
    """
    Dispatcher that uses conversation context to improve matching.
    """
    
    def __init__(
        self,
        semantic_dispatcher: SemanticDispatcher,
        context_weight: float = 0.3
    ):
        """
        Initialize a ContextAwareDispatcher.
        
        Args:
            semantic_dispatcher: The semantic dispatcher to use
            context_weight: Weight to give context vs semantic matching
        """
        self.semantic_dispatcher = semantic_dispatcher
        self.context_weight = context_weight
        self.conversations: Dict[str, ConversationContext] = {}
    
    def get_or_create_context(
        self,
        conversation_id: Optional[str] = None
    ) -> ConversationContext:
        """
        Get an existing conversation context or create a new one.
        
        Args:
            conversation_id: Optional ID of existing conversation
            
        Returns:
            ConversationContext object
        """
        if conversation_id and conversation_id in self.conversations:
            return self.conversations[conversation_id]
        
        # Create new context
        context = ConversationContext()
        self.conversations[context.conversation_id] = context
        return context
    
    async def dispatch(
        self, 
        text: str,
        conversation_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[DispatchResult, str]:
        """
        Dispatch text using semantic matching and conversation context.
        
        Args:
            text: The input text to dispatch
            conversation_id: Optional conversation ID
            context_data: Additional context data
            
        Returns:
            Tuple of (dispatch_result, conversation_id)
        """
        # Get or create conversation context
        conversation = self.get_or_create_context(conversation_id)
        
        # Combine context_data with conversation context
        dispatch_context = context_data or {}
        dispatch_context["conversation"] = {
            "id": conversation.conversation_id,
            "history": conversation.history,
            "current_path_id": conversation.current_path_id,
            "turns_in_current_path": conversation.turns_in_current_path
        }
        
        # Get semantic dispatch result with all matches
        semantic_result = await self.semantic_dispatcher.dispatch(
            text, dispatch_context, return_all_matches=True
        )
        
        if not semantic_result.matches:
            # No matches, just return semantic result
            if semantic_result.path:
                conversation.add_turn(
                    text, 
                    semantic_result.path.id, 
                    semantic_result.confidence
                )
            return semantic_result, conversation.conversation_id
        
        # Process each match with context boosting
        context_boosted_matches = []
        
        for path, semantic_score in semantic_result.matches:
            # Get context boost for this path
            context_boost = conversation.get_path_boost(path.id)
            
            # Combine semantic score with context boost
            combined_score = (
                (1 - self.context_weight) * semantic_score +
                self.context_weight * context_boost
            )
            
            context_boosted_matches.append((path, combined_score))
        
        # Sort boosted matches by score
        context_boosted_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get best match
        best_path, best_score = context_boosted_matches[0]
        
        # Create result
        result = DispatchResult(
            path=best_path,
            confidence=best_score,
            text=text,
            fallback_used=best_score < self.semantic_dispatcher.confidence_threshold,
            matches=context_boosted_matches
        )
        
        # Check against confidence threshold and use fallback if needed
        if (
            result.fallback_used and 
            self.semantic_dispatcher.fallback_path
        ):
            result.path = self.semantic_dispatcher.fallback_path
        
        # Update conversation context
        conversation.add_turn(
            text, 
            result.path.id if result.path else None, 
            best_score
        )
        
        return result, conversation.conversation_id
    
    async def dispatch_and_handle(
        self, 
        text: str,
        conversation_id: Optional[str] = None,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[DispatchResult, Any, str]:
        """
        Dispatch text and invoke the handler of the matched path.
        
        Args:
            text: The input text to dispatch
            conversation_id: Optional conversation ID
            context_data: Additional context data
            
        Returns:
            Tuple of (dispatch_result, handler_result, conversation_id)
        """
        result, conversation_id = await self.dispatch(text, conversation_id, context_data)
        
        if result.path:
            # Create handler context with conversation data
            conversation = self.conversations[conversation_id]
            handler_context = context_data or {}
            handler_context["conversation"] = {
                "id": conversation.conversation_id,
                "history": conversation.history,
                "current_path_id": conversation.current_path_id,
                "turns_in_current_path": conversation.turns_in_current_path
            }
            
            # Call the path's handler
            handler_result = await result.path.handle(text, handler_context)
            
            # Update conversation with response
            if handler_result and conversation.history:
                conversation.history[-1]["response"] = handler_result
            
            return result, handler_result, conversation_id
        
        return result, None, conversation_id


##################################
# Chat Handlers and Bot Implementation
##################################

async def greeting_handler(text: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Handle greeting intents."""
    # Check if asking about chatbot's well-being
    if "how are you" in text.lower() or "how's it going" in text.lower():
        return {
            "message": "I'm doing great, thanks for asking! How can I help you today?",
            "suggestions": ["Tell me about the weather", "Help with my order", "What can you do?"]
        }
    
    conversation = context.get("conversation", {})
    history = conversation.get("history", [])
    
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
    """Web-based chatbot using MeaningMesh for intent recognition."""
    
    def __init__(self):
        """Initialize the chatbot."""
        self.conversation_id = None
        self.dispatcher = None
    
    async def initialize(self):
        """Initialize the dispatcher and paths."""
        # Create vectorizer and store
        vectorizer = create_vectorizer("mock", semantic_boost=0.8)
        store = InMemoryEmbeddingStore()
        
        # Create paths with example phrases
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
        
        # Create semantic dispatcher
        semantic_dispatcher = SemanticDispatcher(
            vectorizer=vectorizer,
            store=store,
            confidence_threshold=0.25,
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
    
    async def process_message(self, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a user message and return a response.
        
        Args:
            message: User message
            conversation_id: Optional conversation ID
            
        Returns:
            Response with message and suggestions
        """
        if not self.dispatcher:
            await self.initialize()
        
        # Dispatch message
        result, response, new_conversation_id = await self.dispatcher.dispatch_and_handle(
            message, conversation_id
        )
        
        # Return response with conversation ID
        return {
            "conversation_id": new_conversation_id,
            "intent": result.path.name if result.path else "Unknown",
            "confidence": result.confidence,
            "fallback_used": result.fallback_used,
            "response": response
        }


###########################################
# Web Server Implementation
###########################################

# Create a chatbot instance
chatbot = ChatBot()

# Define route handlers
async def chat_endpoint(request):
    """API endpoint for chat messages."""
    try:
        # Parse JSON body
        data = await request.json()
        
        # Extract message and conversation ID
        message = data.get('message', '')
        conversation_id = data.get('conversation_id')
        
        if not message:
            return web.json_response({"error": "Message is required"}, status=400)
        
        # Process message
        response = await chatbot.process_message(message, conversation_id)
        
        # Return response
        return web.json_response(response)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


async def index(request):
    """Serve the chatbot interface."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MeaningMesh Chat Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            #chat-container {
                border: 1px solid #ccc;
                border-radius: 5px;
                height: 400px;
                overflow-y: auto;
                padding: 10px;
                margin-bottom: 10px;
            }
            .message {
                margin-bottom: 10px;
                padding: 8px 12px;
                border-radius: 5px;
                max-width: 80%;
                word-wrap: break-word;
            }
            .user-message {
                background-color: #e1f5fe;
                margin-left: auto;
                margin-right: 0;
            }
            .bot-message {
                background-color: #f1f1f1;
                margin-right: auto;
                margin-left: 0;
            }
            #message-form {
                display: flex;
            }
            #message-input {
                flex-grow: 1;
                padding: 8px;
                border: 1px solid #ccc;
                border-radius: 5px;
            }
            #send-button {
                padding: 8px 12px;
                background-color: #4caf50;
                color: white;
                border: none;
                border-radius: 5px;
                margin-left: 5px;
                cursor: pointer;
            }
            .suggestions {
                margin-top: 5px;
            }
            .suggestion {
                display: inline-block;
                margin-right: 5px;
                margin-bottom: 5px;
                padding: 5px 10px;
                background-color: #e8f5e9;
                border-radius: 15px;
                font-size: 0.85em;
                cursor: pointer;
            }
            .intent-info {
                font-size: 0.8em;
                color: #666;
                margin-top: 3px;
            }
        </style>
    </head>
    <body>
        <h1>MeaningMesh Chat Demo</h1>
        <div id="chat-container"></div>
        <form id="message-form">
            <input type="text" id="message-input" placeholder="Type your message...">
            <button type="submit" id="send-button">Send</button>
        </form>

        <script>
            let conversationId = null;
            const chatContainer = document.getElementById('chat-container');
            const messageForm = document.getElementById('message-form');
            const messageInput = document.getElementById('message-input');

            // Add welcome message
            addBotMessage({
                message: "Hello! I'm MeaningMesh Bot. How can I help you today?",
                suggestions: ["Tell me about the weather", "Help with my order", "What can you do?"]
            });

            messageForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const message = messageInput.value.trim();
                if (!message) return;

                // Add user message to chat
                addUserMessage(message);
                messageInput.value = '';

                try {
                    // Send message to API
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            message: message,
                            conversation_id: conversationId
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to send message');
                    }

                    const data = await response.json();
                    
                    // Update conversation ID
                    conversationId = data.conversation_id;
                    
                    // Add bot message to chat
                    addBotMessage(data.response, data.intent, data.confidence);
                } catch (error) {
                    console.error('Error:', error);
                    addBotMessage({
                        message: "Sorry, I encountered an error processing your request."
                    });
                }
            });

            function addUserMessage(text) {
                const messageElement = document.createElement('div');
                messageElement.className = 'message user-message';
                messageElement.textContent = text;
                chatContainer.appendChild(messageElement);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function addBotMessage(response, intent = null, confidence = null) {
                const messageContainer = document.createElement('div');
                
                // Main message
                const messageElement = document.createElement('div');
                messageElement.className = 'message bot-message';
                messageElement.textContent = response.message;
                messageContainer.appendChild(messageElement);
                
                // Intent info if available
                if (intent) {
                    const intentElement = document.createElement('div');
                    intentElement.className = 'intent-info';
                    intentElement.textContent = `Intent: ${intent}, Confidence: ${confidence.toFixed(4)}`;
                    messageContainer.appendChild(intentElement);
                }
                
                // Suggestions if available
                if (response.suggestions && response.suggestions.length > 0) {
                    const suggestionsContainer = document.createElement('div');
                    suggestionsContainer.className = 'suggestions';
                    
                    response.suggestions.forEach(suggestion => {
                        const suggestionElement = document.createElement('span');
                        suggestionElement.className = 'suggestion';
                        suggestionElement.textContent = suggestion;
                        suggestionElement.onclick = () => {
                            messageInput.value = suggestion;
                            messageInput.focus();
                        };
                        suggestionsContainer.appendChild(suggestionElement);
                    });
                    
                    messageContainer.appendChild(suggestionsContainer);
                }
                
                chatContainer.appendChild(messageContainer);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return web.Response(text=html, content_type='text/html')


async def init_chatbot():
    """Initialize the chatbot."""
    await chatbot.initialize()
    print("Chatbot initialized")


def main():
    """Run the web application."""
    # Create the web application
    app = web.Application()
    
    # Set up routes
    app.add_routes([
        web.get('/', index),
        web.post('/api/chat', chat_endpoint)
    ])
    
    # Initialize the chatbot
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_chatbot())
    
    # Start the server
    port = 8080
    print(f"Starting MeaningMesh Web Demo on http://localhost:{port}")
    web.run_app(app, port=port)


if __name__ == "__main__":
    main()
