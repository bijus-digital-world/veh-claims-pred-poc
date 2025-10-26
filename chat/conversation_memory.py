"""
Conversation Memory Module for Chat Functionality

Provides conversation context and memory management for maintaining
conversation state across multiple exchanges.
"""

from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
import json
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ChatExchange:
    """Represents a single exchange in the conversation"""
    query: str
    response: str
    timestamp: str
    exchange_id: str
    handler_used: Optional[str] = None
    processing_time_ms: Optional[float] = None
    context_used: Optional[Dict[str, Any]] = None


class ConversationContext:
    """
    Manages conversation memory and context for chat functionality.
    
    Features:
    - Maintains conversation history with configurable window size
    - Provides context for follow-up questions
    - Supports context-aware query processing
    - Handles memory persistence and retrieval
    """
    
    def __init__(self, context_window: int = 10, max_memory_size: int = 100):
        """
        Initialize conversation context.
        
        Args:
            context_window: Number of recent exchanges to keep in active context
            max_memory_size: Maximum total exchanges to keep in memory
        """
        self.memory: List[ChatExchange] = []
        self.context_window = context_window
        self.max_memory_size = max_memory_size
        self.current_session_id: Optional[str] = None
        
    def add_exchange(self, 
                    query: str, 
                    response: str, 
                    exchange_id: str,
                    handler_used: Optional[str] = None,
                    processing_time_ms: Optional[float] = None,
                    context_used: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a new exchange to the conversation memory.
        
        Args:
            query: User's query
            response: Assistant's response
            exchange_id: Unique identifier for this exchange
            handler_used: Name of the handler that processed the query
            processing_time_ms: Time taken to process the query
            context_used: Additional context used in processing
        """
        exchange = ChatExchange(
            query=query,
            response=response,
            timestamp=datetime.now(timezone.utc).isoformat(),
            exchange_id=exchange_id,
            handler_used=handler_used,
            processing_time_ms=processing_time_ms,
            context_used=context_used
        )
        
        self.memory.append(exchange)
        
        # Maintain memory size limits
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size:]
        
        logger.debug(f"Added exchange {exchange_id} to conversation memory. Total exchanges: {len(self.memory)}")
    
    def get_recent_context(self, window_size: Optional[int] = None) -> List[ChatExchange]:
        """
        Get recent exchanges for context.
        
        Args:
            window_size: Number of recent exchanges to return (defaults to context_window)
            
        Returns:
            List of recent ChatExchange objects
        """
        window = window_size or self.context_window
        return self.memory[-window:] if self.memory else []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation.
        
        Returns:
            Dictionary containing conversation statistics and recent topics
        """
        if not self.memory:
            return {
                "total_exchanges": 0,
                "session_duration": 0,
                "recent_topics": [],
                "handlers_used": {},
                "average_response_time": 0
            }
        
        # Calculate session duration
        if len(self.memory) >= 2:
            start_time = datetime.fromisoformat(self.memory[0].timestamp.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(self.memory[-1].timestamp.replace('Z', '+00:00'))
            session_duration = (end_time - start_time).total_seconds()
        else:
            session_duration = 0
        
        # Extract recent topics (simplified - just first few words of queries)
        recent_topics = [exchange.query.split()[:3] for exchange in self.memory[-5:]]
        
        # Count handlers used
        handlers_used = {}
        for exchange in self.memory:
            if exchange.handler_used:
                handlers_used[exchange.handler_used] = handlers_used.get(exchange.handler_used, 0) + 1
        
        # Calculate average response time
        response_times = [ex.processing_time_ms for ex in self.memory if ex.processing_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_exchanges": len(self.memory),
            "session_duration": session_duration,
            "recent_topics": recent_topics,
            "handlers_used": handlers_used,
            "average_response_time": avg_response_time
        }
    
    def find_related_exchanges(self, query: str, max_results: int = 3) -> List[ChatExchange]:
        """
        Find exchanges related to the current query.
        
        Args:
            query: Current query to find related exchanges for
            max_results: Maximum number of related exchanges to return
            
        Returns:
            List of related ChatExchange objects
        """
        if not self.memory:
            return []
        
        query_lower = query.lower()
        related = []
        
        for exchange in self.memory:
            # Simple keyword matching (could be enhanced with semantic similarity)
            if any(word in exchange.query.lower() for word in query_lower.split() if len(word) > 3):
                related.append(exchange)
        
        return related[-max_results:]
    
    def get_context_for_query(self, query: str) -> Dict[str, Any]:
        """
        Get relevant context for processing a new query.
        
        Args:
            query: The new query being processed
            
        Returns:
            Dictionary containing relevant context information
        """
        recent_context = self.get_recent_context()
        related_exchanges = self.find_related_exchanges(query)
        conversation_summary = self.get_conversation_summary()
        
        return {
            "recent_exchanges": [asdict(ex) for ex in recent_context],
            "related_exchanges": [asdict(ex) for ex in related_exchanges],
            "conversation_summary": conversation_summary,
            "context_window_size": self.context_window,
            "total_memory_size": len(self.memory)
        }
    
    def clear_memory(self) -> None:
        """Clear all conversation memory."""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def save_to_session_state(self, session_state: Dict[str, Any]) -> None:
        """
        Save conversation memory to Streamlit session state.
        
        Args:
            session_state: Streamlit session state dictionary
        """
        session_state["conversation_memory"] = {
            "memory": [asdict(exchange) for exchange in self.memory],
            "context_window": self.context_window,
            "max_memory_size": self.max_memory_size,
            "session_id": self.current_session_id
        }
    
    @classmethod
    def load_from_session_state(cls, session_state: Dict[str, Any]) -> 'ConversationContext':
        """
        Load conversation memory from Streamlit session state.
        
        Args:
            session_state: Streamlit session state dictionary
            
        Returns:
            ConversationContext instance loaded from session state
        """
        if "conversation_memory" not in session_state:
            return cls()
        
        memory_data = session_state["conversation_memory"]
        
        # Handle case where memory_data might be a ConversationContext object
        if isinstance(memory_data, ConversationContext):
            return memory_data
        
        # Handle case where memory_data is a dictionary
        if isinstance(memory_data, dict):
            context = cls(
                context_window=memory_data.get("context_window", 10),
                max_memory_size=memory_data.get("max_memory_size", 100)
            )
            
            # Reconstruct ChatExchange objects
            for exchange_data in memory_data.get("memory", []):
                try:
                    exchange = ChatExchange(**exchange_data)
                    context.memory.append(exchange)
                except Exception as e:
                    logger.warning(f"Failed to reconstruct exchange: {e}")
                    continue
            
            context.current_session_id = memory_data.get("session_id")
            return context
        
        # Fallback: return new instance
        logger.warning("Unexpected memory_data type, returning new ConversationContext")
        return cls()


class ContextAwareQueryProcessor:
    """
    Enhanced query processor that uses conversation context.
    """
    
    def __init__(self, conversation_context: ConversationContext):
        self.context = conversation_context
    
    def enhance_query_with_context(self, query: str) -> str:
        """
        Enhance the query with conversation context.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with context information
        """
        context_info = self.context.get_context_for_query(query)
        
        # Add context hints to the query
        enhanced_parts = [query]
        
        # Add recent topic context
        if context_info["recent_exchanges"]:
            recent_topics = [ex["query"] for ex in context_info["recent_exchanges"][-2:]]
            if recent_topics:
                enhanced_parts.append(f"[Recent context: {'; '.join(recent_topics)}]")
        
        # Add related exchange context
        if context_info["related_exchanges"]:
            related_queries = [ex["query"] for ex in context_info["related_exchanges"]]
            if related_queries:
                enhanced_parts.append(f"[Related: {'; '.join(related_queries)}]")
        
        return " ".join(enhanced_parts)
    
    def should_use_context(self, query: str) -> bool:
        """
        Determine if the query should use conversation context.
        
        Args:
            query: User query
            
        Returns:
            True if context should be used
        """
        # Use context for follow-up questions, clarifications, and references
        context_indicators = [
            "what about", "how about", "also", "additionally", "furthermore",
            "in addition", "moreover", "besides", "on the other hand",
            "however", "but", "although", "despite", "regarding", "concerning",
            "about that", "regarding that", "as for", "when it comes to"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in context_indicators) or len(self.context.memory) > 0
