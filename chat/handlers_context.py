"""
Context-aware handlers for conversation memory integration.

These handlers can use conversation context to provide more intelligent responses.
"""

import re
import html as _html
import pandas as pd
from typing import Optional

from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger


class ContextAwareHandler(QueryHandler):
    """
    Handler that can use conversation context to provide better responses.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        # This handler can handle any query, but only uses context when available
        return context.conversation_context is not None and len(context.conversation_context.memory) > 0
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ContextAwareHandler processing with conversation context")
        
        # Get conversation context
        conv_context = context.conversation_context
        recent_exchanges = conv_context.get_recent_context(window_size=3)
        related_exchanges = conv_context.find_related_exchanges(context.query, max_results=2)
        
        # Check if this is a follow-up question
        if self._is_follow_up_question(context.query, recent_exchanges):
            return self._handle_follow_up(context)
        
        # Check if this is asking for clarification
        if self._is_clarification_request(context.query):
            return self._handle_clarification(context, recent_exchanges)
        
        # Check if this is asking for comparison
        if self._is_comparison_request(context.query):
            return self._handle_comparison(context, recent_exchanges)
        
        # Default: provide context-aware response
        return self._handle_context_aware_response(context, recent_exchanges)
    
    def _is_follow_up_question(self, query: str, recent_exchanges: list) -> bool:
        """Check if the query is a follow-up to recent exchanges."""
        follow_up_indicators = [
            "what about", "how about", "also", "additionally", "furthermore",
            "in addition", "moreover", "besides", "on the other hand",
            "however", "but", "although", "despite", "regarding", "concerning",
            "about that", "regarding that", "as for", "when it comes to",
            "speaking of", "by the way", "incidentally"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in follow_up_indicators)
    
    def _is_clarification_request(self, query: str) -> bool:
        """Check if the query is asking for clarification."""
        clarification_indicators = [
            "what do you mean", "can you explain", "clarify", "elaborate",
            "more details", "tell me more", "what does that mean",
            "i don't understand", "confused", "unclear"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in clarification_indicators)
    
    def _is_comparison_request(self, query: str) -> bool:
        """Check if the query is asking for comparison."""
        comparison_indicators = [
            "compare", "comparison", "versus", "vs", "difference between",
            "better than", "worse than", "similar to", "unlike", "contrast"
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in comparison_indicators)
    
    def _handle_follow_up(self, context: QueryContext) -> str:
        """Handle follow-up questions using conversation context."""
        recent_exchanges = context.conversation_context.get_recent_context(window_size=2)
        
        if not recent_exchanges:
            return "<p>I don't have enough context to answer your follow-up question. Could you provide more details?</p>"
        
        # Extract the most recent query to understand what we're following up on
        last_query = recent_exchanges[-1].query
        current_query = context.query
        
        # Enhance the current query with context from the previous query
        enhanced_query = self._enhance_query_with_context(current_query, last_query)
        logger.debug(f"Enhanced query: '{current_query}' -> '{enhanced_query}'")
        
        # Create a new context with the enhanced query
        from chat.handlers import QueryContext, QueryRouter
        enhanced_context = QueryContext(
            query=enhanced_query,
            df_history=context.df_history,
            faiss_res=context.faiss_res,
            tfidf_vect=context.tfidf_vect,
            tfidf_X=context.tfidf_X,
            tfidf_rows=context.tfidf_rows,
            get_bedrock_summary_callable=context.get_bedrock_summary,
            top_k=context.top_k,
            conversation_context=context.conversation_context
        )
        
        # Route to the appropriate handler with enhanced context
        router = QueryRouter()
        response = router.route(enhanced_context)
        logger.debug(f"Context-aware response generated for enhanced query: '{enhanced_query}'")
        
        # Return the response directly without unnecessary context line
        return response
    
    def _enhance_query_with_context(self, current_query: str, previous_query: str) -> str:
        """Enhance the current query with context from the previous query."""
        current_lower = current_query.lower()
        previous_lower = previous_query.lower()
        
        # Extract key terms from previous query
        models = ["sentra", "leaf", "ariya"]
        metrics = ["failure rate", "claim rate", "repair rate", "recall rate", "failures", "claims", "repairs", "recalls"]
        
        # Find model mentioned in previous query
        mentioned_model = None
        for model in models:
            if model in previous_lower:
                mentioned_model = model
                break
        
        # Find metric mentioned in previous query
        mentioned_metric = None
        for metric in metrics:
            if metric in previous_lower:
                mentioned_metric = metric
                break
        
        # If asking "what about" or "how about", use the same metric with the mentioned model
        if any(phrase in current_lower for phrase in ["what about", "how about", "also", "additionally"]):
            # Extract model from current query if mentioned
            current_model = None
            for model in models:
                if model in current_lower:
                    current_model = model
                    break
            
            if current_model and mentioned_metric:
                # Use the model from current query with metric from previous query
                return f"{mentioned_metric} for {current_model.title()}"
            elif mentioned_model and mentioned_metric:
                # Use the model from previous query with metric from previous query
                return f"{mentioned_metric} for {mentioned_model.title()}"
            elif current_model:
                # Just add the metric to the current query
                return f"{mentioned_metric} for {current_model.title()}"
            elif mentioned_model:
                # Just add the model to the current query
                return f"{current_query} for {mentioned_model.title()}"
            elif mentioned_metric:
                # Just add the metric to the current query
                return f"{mentioned_metric} {current_query}"
        
        # If asking for comparison, include both current and previous topics
        if any(word in current_lower for word in ["compare", "versus", "vs", "difference", "comparison"]):
            if mentioned_model and mentioned_metric:
                return f"Compare {mentioned_metric} for {mentioned_model.title()} with other models"
            elif mentioned_model:
                return f"Compare {mentioned_model.title()} with other models"
        
        # If asking for more details about the same topic
        if any(phrase in current_lower for phrase in ["more details", "tell me more", "elaborate", "explain more"]):
            if mentioned_model and mentioned_metric:
                return f"Detailed analysis of {mentioned_metric} for {mentioned_model.title()}"
            elif mentioned_model:
                return f"Detailed analysis of {mentioned_model.title()}"
        
        # Default: return current query with context hint
        return f"{current_query} (context: {previous_query})"
    
    def _handle_clarification(self, context: QueryContext, recent_exchanges: list) -> str:
        """Handle clarification requests using conversation context."""
        if not recent_exchanges:
            return "<p>I don't have enough context to clarify. Could you provide more details about what you'd like me to explain?</p>"
        
        # Find the most recent exchange that might need clarification
        last_exchange = recent_exchanges[-1]
        
        return (f"<p>I provided information about {self._extract_topic(last_exchange.response)}. "
                f"Would you like me to go into more detail about any specific aspect?</p>")
    
    def _handle_comparison(self, context: QueryContext, recent_exchanges: list) -> str:
        """Handle comparison requests using conversation context."""
        if len(recent_exchanges) < 2:
            return "<p>I need more context to make a comparison. Could you provide more details about what you'd like to compare?</p>"
        
        # Extract topics from recent exchanges for comparison
        topics = [self._extract_topic(exchange.query) for exchange in recent_exchanges[-2:]]
        
        return f"<p>Let me provide a detailed comparison of {topics[0]} vs {topics[1]}.</p>"
    
    def _handle_context_aware_response(self, context: QueryContext, recent_exchanges: list) -> str:
        """Handle general queries with conversation context."""
        if not recent_exchanges:
            return "<p>I don't have enough context to provide a context-aware response.</p>"
        
        # Provide a response that acknowledges the conversation context
        context_summary = self._summarize_conversation_context(recent_exchanges)
        
        return "<p>Let me provide additional insights based on the context of our discussion.</p>"
    
    def _extract_topic(self, text: str) -> str:
        """Extract a simple topic from text."""
        # Remove HTML tags
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Take first few words
        words = clean_text.split()[:3]
        return " ".join(words)
    
    def _summarize_conversation_context(self, recent_exchanges: list) -> str:
        """Summarize the conversation context."""
        if not recent_exchanges:
            return "various topics"
        
        topics = [self._extract_topic(exchange.query) for exchange in recent_exchanges[-2:]]
        return ", ".join(topics)


class ConversationSummaryHandler(QueryHandler):
    """
    Handler for conversation summary requests.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        return any(phrase in query_lower for phrase in [
            "conversation summary", "what have we discussed", "summary of our chat",
            "conversation overview", "what did we talk about", "chat summary"
        ])
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ConversationSummaryHandler processing")
        
        if not context.conversation_context or not context.conversation_context.memory:
            return "<p>We haven't had much of a conversation yet. Ask me something about your data!</p>"
        
        conv_context = context.conversation_context
        summary = conv_context.get_conversation_summary()
        
        html_parts = [
            f"<p><strong>Conversation Summary:</strong></p>",
            f"<ul style='margin-top:6px;'>",
            f"<li><strong>Total exchanges:</strong> {summary['total_exchanges']}</li>",
            f"<li><strong>Session duration:</strong> {summary['session_duration']:.1f} seconds</li>",
            f"<li><strong>Average response time:</strong> {summary['average_response_time']:.1f}ms</li>",
            f"</ul>"
        ]
        
        if summary['handlers_used']:
            html_parts.append("<p><strong>Handlers used:</strong></p><ul>")
            for handler, count in summary['handlers_used'].items():
                html_parts.append(f"<li>{handler}: {count} times</li>")
            html_parts.append("</ul>")
        
        if summary['recent_topics']:
            topics_text = ", ".join([" ".join(topic) for topic in summary['recent_topics']])
            html_parts.append(f"<p><strong>Recent topics:</strong> {topics_text}</p>")
        
        return "".join(html_parts)
