"""Query handler pattern for chat functionality."""

from abc import ABC, abstractmethod
from typing import Optional
import re
import html as _html
import pandas as pd

from config import config
from utils.logger import chat_logger as logger
from chat.intent_classifier import classify_intent


class QueryContext:
    """Context object passed to all handlers."""
    def __init__(
        self,
        query: str,
        df_history: pd.DataFrame,
        get_bedrock_summary_callable,
        conversation_context=None,
        intent: Optional[str] = None,
        df_inference: Optional[pd.DataFrame] = None
    ):
        self.query = query
        self.query_lower = query.lower()
        self.df_history = df_history
        self.get_bedrock_summary = get_bedrock_summary_callable
        self.conversation_context = conversation_context
        self.df_inference = df_inference
        
        self.requested_metric = None
        self.metric_col = None
        self.sample_df = None
        self.intent = intent


class QueryHandler(ABC):
    """Base class for query handlers."""
    
    @abstractmethod
    def can_handle(self, context: QueryContext) -> bool:
        """Check if this handler can process the query."""
        pass
    
    @abstractmethod
    def handle(self, context: QueryContext) -> str:
        """Process the query and return HTML response."""
        pass
    
    def get_name(self) -> str:
        return self.__class__.__name__


class EmptyQueryHandler(QueryHandler):
    """Handle empty or whitespace-only queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        return not context.query.strip()
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("EmptyQueryHandler triggered")
        return "<p>Please ask a question about the historical data (e.g. 'failure rate for model Sentra').</p>"


class GreetingHandler(QueryHandler):
    """Handle greetings like 'hi', 'hello', 'hey'"""
    
    def can_handle(self, context: QueryContext) -> bool:
        q = context.query_lower.strip()
        if any(g in q for g in [" hi ", " hello ", " hey ", "hiya", "yo "]):
            return True
        if re.match(r"^(hi|hello|hey|good\s+(morning|afternoon|evening))\b", q):
            return True
        if re.search(r"\bhow are you\b|\bhow's it going\b|\bwhat's up\b", q):
            return True
        return (len(context.query.split()) <= 3 and any(g in q for g in ["hi", "hello", "hey"]))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("GreetingHandler triggered")
        
        # grab some stats for the greeting
        try:
            total_records = len(context.df_history)
            models = context.df_history['model'].nunique() if 'model' in context.df_history.columns else 0
            failures = context.df_history['failures_count'].sum() if 'failures_count' in context.df_history.columns else 0
            
            html_parts = [
                "<p><strong>Hello! I'm your Nissan Telematics Analytics Assistant.</strong></p>",
                f"<p>I can help you analyze your vehicle data with <strong>{total_records:,} records</strong> covering <strong>{models} models</strong> and <strong>{failures:,} total failures</strong>.</p>",
                "<p><strong>Here's what I can help you with:</strong></p>",
                "<ul style='margin-top:8px;'>",
                "<li><strong>Failure Analysis:</strong> \"What's the failure rate for Sentra?\" or \"Show me failure trends\"</li>",
                "<li><strong>Model Comparisons:</strong> \"Compare Leaf vs Ariya failure rates\" or \"Which model has the most failures?\"</li>",
                "<li><strong>Trends & Patterns:</strong> \"Show monthly failure trends\" or \"What's the failure rate by age?\"</li>",
                "<li><strong>Prescriptive Insights:</strong> \"Prescribe for model Leaf part Battery\" or \"What should I do about high failure rates?\"</li>",
                "<li><strong>Location Analysis:</strong> \"Which dealers have the most issues?\" or \"Show failures by region\"</li>",
                "</ul>",
                "<p><em>Just ask me anything about your vehicle data - I'm here to help!</em></p>"
            ]
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.warning(f"Failed to get dataset stats for greeting: {e}")
            # fallback greeting
            return ("<p><strong>Hello! I'm your Nissan Telematics Analytics Assistant.</strong></p>"
                    "<p>I can help you analyze vehicle failure data, compare models, track trends, and provide prescriptive insights.</p>"
                    "<p><em>Try asking: \"What's the failure rate for Sentra?\" or \"Show me failure trends\"</em></p>")


class PrescriptiveHandler(QueryHandler):
    """Handle prescriptive/recommendation requests using Bedrock"""
    
    def can_handle(self, context: QueryContext) -> bool:
        """
        Only handle true prescriptive/recommendation queries.
        
        Comparison queries ("compare", "vs", "versus") should go to TextToSQLHandler
        for data analysis, not prescriptive recommendations.
        """
        query_lower = context.query_lower
        
        # Don't handle comparison queries - these are data queries, not prescriptive
        if any(word in query_lower for word in ["compare", "comparison", "vs", "versus", "versus"]):
            return False
        
        prescriptive_keywords = [
            "prescribe", "recommend", "prescriptive", "advice", "action",
            "what should i do", "what should we do", "how to fix", "how to improve",
            "what to do about", "suggestions", "recommendations", "guidance",
            "help with", "solution", "remedy", "intervention", "improving"
        ]
        return any(w in query_lower for w in prescriptive_keywords)
    
    def handle(self, context: QueryContext) -> str:
        from chat_helper import parse_model_part_from_text, ensure_failures_column
        
        logger.debug("PrescriptiveHandler triggered")
        
        try:
            model, part = parse_model_part_from_text(context.df_history, context.query)
        except Exception as e:
            logger.warning(f"Failed to parse model/part from query: {e}")
            model, part = None, None
        
        if not (model and part):
            # handle general failure queries
            if any(phrase in context.query_lower for phrase in ["high failure", "failure rate", "failures", "problems", "issues"]):
                return self._handle_general_prescriptive_guidance(context)
            else:
                # show available models/parts
                available_models = context.df_history.get("model", pd.Series(dtype=str)).unique()[:5]
                available_parts = context.df_history.get("primary_failed_part", pd.Series(dtype=str)).unique()[:5]
                
                models_text = ", ".join([str(m) for m in available_models if pd.notna(m)])
                parts_text = ", ".join([str(p) for p in available_parts if pd.notna(p)])
                
                return (f"<p>I can generate a prescriptive summary if you include a model and part. "
                        f"<br><br><strong>Available models:</strong> {models_text}"
                        f"<br><strong>Available parts:</strong> {parts_text}"
                        f"<br><br><strong>Example:</strong> 'prescribe for model Leaf part Battery'</p>")
        
        slice_df = context.df_history[
            (context.df_history.get("model") == model) & 
            (context.df_history.get("primary_failed_part") == part)
        ]
        
        if slice_df.empty:
            # Generate helpful error message with suggestions
            available_models = context.df_history.get("model", pd.Series(dtype=str)).dropna().unique()[:5]
            available_parts = context.df_history.get("primary_failed_part", pd.Series(dtype=str)).dropna().unique()[:5]
            
            models_text = ", ".join([str(m) for m in available_models if pd.notna(m)])
            parts_text = ", ".join([str(p) for p in available_parts if pd.notna(p)])
            
            return (
                f"<p>No data found for model '<strong>{_html.escape(str(model))}</strong>' and part '<strong>{_html.escape(str(part))}</strong>'.</p>"
                f"<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                f"This combination might not exist in the dataset, or the model/part names might be spelled differently.</p>"
                f"<p style='font-size: 0.9em; color: #cfe9ff; margin-top: 8px;'>"
                f"<strong>Available models:</strong> {models_text}</p>"
                f"<p style='font-size: 0.9em; color: #cfe9ff;'>"
                f"<strong>Available parts:</strong> {parts_text}</p>"
                f"<p style='font-size: 0.85em; color: #64748b; margin-top: 8px;'>"
                f"<strong>Tip:</strong> Try one of these combinations, or ask 'What columns are available?' to explore the data structure.</p>"
            )
        
        mileage_bucket = slice_df.iloc[0].get("mileage_bucket", "")
        age_bucket = slice_df.iloc[0].get("age_bucket", "")
        total_inc = slice_df.shape[0]
        
        # make sure failures_count exists
        if "failures_count" not in slice_df.columns:
            slice_df = ensure_failures_column(slice_df, out_col="failures_count")
        
        total_failures = int(slice_df["failures_count"].sum())
        pct = (total_failures / total_inc * 100.0) if total_inc > 0 else 0.0
        
        try:
            if context.get_bedrock_summary is None:
                return ("<p>Prescriptive analysis is not available at the moment. "
                        "The Bedrock service is not configured or accessible.</p>")
            
            logger.info(f"Generating prescriptive summary via Bedrock for {model}/{part}")
            summary_html = context.get_bedrock_summary(model, part, mileage_bucket, age_bucket, pct)
            plain = re.sub(r"<[^>]+>", "", summary_html).strip()
            return f"<p>{_html.escape(plain)}</p>"
        except Exception as e:
            logger.error(f"Bedrock prescriptive summary failed: {e}", exc_info=True)
            # Generate user-friendly error message
            error_msg = str(e).lower()
            
            error_parts = [
                "<p>I couldn't generate prescriptive recommendations for your request.</p>"
            ]
            
            # Check error type and provide specific guidance
            if "bedrock" in error_msg or "aws" in error_msg or "credentials" in error_msg:
                error_parts.append(
                    "<p style='font-size: 0.9em; color: #fca5a5; margin-top: 8px;'>"
                    "<strong>Issue:</strong> The AI recommendation service is not available right now.</p>"
                    "<p style='font-size: 0.85em; color: #94a3b8; margin-top: 8px;'>"
                    "This might be a temporary issue. Please try again in a moment, or try a different query.</p>"
                )
            elif "timeout" in error_msg or "timed out" in error_msg:
                error_parts.append(
                    "<p style='font-size: 0.9em; color: #fca5a5; margin-top: 8px;'>"
                    "<strong>Issue:</strong> The request took too long to process.</p>"
                    "<p style='font-size: 0.85em; color: #94a3b8; margin-top: 8px;'>"
                    "Please try again with a more specific query, or try again in a moment.</p>"
                )
            else:
                error_parts.append(
                    "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                    "This might be a temporary issue. Please try again, or rephrase your question.</p>"
                )
            
            # Add helpful suggestions
            error_parts.append(
                "<p style='font-size: 0.85em; color: #64748b; margin-top: 8px;'>"
                "<strong>You can try:</strong></p>"
                "<ul style='font-size: 0.85em; color: #64748b; margin-left: 20px;'>"
                "<li>Asking again in a moment</li>"
                "<li>Rephrasing your question more specifically</li>"
                "<li>Trying a different model/part combination</li>"
                "<li>Asking a data query instead: 'What's the failure rate for Leaf Battery?'</li>"
                "</ul>"
            )
            
            return "".join(error_parts)
    
    def _handle_general_prescriptive_guidance(self, context: QueryContext) -> str:
        """Handle general prescriptive queries about high failure rates."""
        try:
            # get overall stats
            df = context.df_history
            total_records = len(df)
            total_failures = df.get('failures_count', pd.Series(dtype=int)).sum() if 'failures_count' in df.columns else 0
            overall_rate = (total_failures / total_records * 100) if total_records > 0 else 0
            
            # model failure rates
            model_rates = []
            if 'model' in df.columns and 'failures_count' in df.columns:
                model_stats = df.groupby('model').agg({
                    'failures_count': 'sum',
                    'model': 'count'
                }).rename(columns={'model': 'count'})
                model_stats['rate'] = (model_stats['failures_count'] / model_stats['count'] * 100)
                model_rates = model_stats.sort_values('rate', ascending=False)
            
            # top failing parts
            top_parts = []
            if 'primary_failed_part' in df.columns and 'failures_count' in df.columns:
                part_stats = df.groupby('primary_failed_part')['failures_count'].sum().sort_values(ascending=False)
                top_parts = part_stats.head(3)
            
            html_parts = [
                "<p><strong>Prescriptive Guidance for High Failure Rates</strong></p>",
                f"<p>Based on your data analysis showing an overall failure rate of <strong>{overall_rate:.1f}%</strong> across {total_records:,} records, here are my recommendations:</p>"
            ]
            
            # model-specific recs
            if model_rates is not None and not model_rates.empty:
                worst_model = model_rates.index[0]
                worst_rate = model_rates.iloc[0]['rate']
                best_model = model_rates.index[-1]
                best_rate = model_rates.iloc[-1]['rate']
                
                html_parts.extend([
                    "<p><strong>Model-Specific Actions:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Priority Focus:</strong> {worst_model} has the highest failure rate at {worst_rate:.1f}% - investigate root causes immediately</li>",
                    f"<li><strong>Best Practice:</strong> Study {best_model}'s success factors (rate: {best_rate:.1f}%) and apply learnings to other models</li>",
                    "<li><strong>Cross-Model Analysis:</strong> Compare manufacturing processes, supplier quality, and design differences</li>",
                    "</ul>"
                ])
            
            # part-specific recs
            if top_parts is not None and not top_parts.empty:
                worst_part = top_parts.index[0]
                worst_part_failures = top_parts.iloc[0]
                
                html_parts.extend([
                    "<p><strong>Part-Specific Actions:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Critical Component:</strong> {worst_part} is the most problematic part with {worst_part_failures} failures</li>",
                    "<li><strong>Supplier Review:</strong> Evaluate supplier quality and consider alternative sources</li>",
                    "<li><strong>Design Review:</strong> Assess if design improvements or material changes are needed</li>",
                    "<li><strong>Testing Protocol:</strong> Implement enhanced testing procedures for high-failure components</li>",
                    "</ul>"
                ])
            
            # general recs
            html_parts.extend([
                "<p><strong>General Improvement Strategies:</strong></p>",
                "<ul style='margin-top:6px;'>",
                "<li><strong>Root Cause Analysis:</strong> Conduct detailed RCA on the top 3 failure modes</li>",
                "<li><strong>Quality Gates:</strong> Implement additional quality checkpoints in manufacturing</li>",
                "<li><strong>Supplier Management:</strong> Review and potentially replace underperforming suppliers</li>",
                "<li><strong>Predictive Maintenance:</strong> Implement early warning systems for failure-prone components</li>",
                "<li><strong>Training Programs:</strong> Enhance technician training on common failure patterns</li>",
                "<li><strong>Monitoring:</strong> Set up real-time monitoring dashboards for key failure indicators</li>",
                "</ul>",
                "<p><strong>Next Steps:</strong> For specific model-part combinations, ask me to 'prescribe for model [X] part [Y]' for detailed recommendations.</p>"
            ])
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"General prescriptive guidance failed: {e}", exc_info=True)
            # Generate user-friendly error message
            error_parts = [
                "<p>I couldn't generate prescriptive guidance for your query about failure rates.</p>",
                "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                "This might be because the dataset doesn't have the required data structure.</p>"
            ]
            
            # Check if basic columns exist
            if context.df_history is not None:
                has_model = 'model' in context.df_history.columns
                has_failures = 'failures_count' in context.df_history.columns or 'failures' in context.df_history.columns.lower()
                
                if not has_model:
                    error_parts.append(
                        "<p style='font-size: 0.85em; color: #fca5a5; margin-top: 8px;'>"
                        "<strong>Issue:</strong> The dataset doesn't contain a 'model' column.</p>"
                    )
                if not has_failures:
                    error_parts.append(
                        "<p style='font-size: 0.85em; color: #fca5a5; margin-top: 8px;'>"
                        "<strong>Issue:</strong> The dataset doesn't contain failure count data.</p>"
                    )
                
                # Add suggestion
                error_parts.append(
                    "<p style='font-size: 0.85em; color: #64748b; margin-top: 8px;'>"
                    "<strong>Suggestion:</strong> Try asking for a specific model-part combination instead, "
                    "like 'Prescribe for model Leaf part Battery'.</p>"
                )
            
            return "".join(error_parts)


# Simplified imports - only essential handlers
from chat.handlers_text_to_sql import TextToSQLHandler
from chat.handlers_intent import GenericIntentHandler
from chat.handlers_supplier import VehicleLocationHandler, LocationAnalysisHandler


class QueryRouter:
    """
    Simplified router using TextToSQL for all data queries.
    
    Only essential meta handlers are kept for basic queries (empty, greetings).
    Schema queries, date range queries, and all other data queries are handled by TextToSQLHandler 
    which uses LLM to generate SQL (or direct handling for schema queries).
    """
    
    def __init__(self):
        self.handlers = [
            # Essential meta queries only
            EmptyQueryHandler(),
            GreetingHandler(),
            
            # Prescriptive queries (needs special handling, not SQL)
            PrescriptiveHandler(),
            
            # Specialized handlers for vehicle tracking and location analysis (must come before TextToSQLHandler)
            VehicleLocationHandler(),  # Tracks WHERE A SPECIFIC VEHICLE IS (vehicle-centric)
            LocationAnalysisHandler(),  # Analyzes WHERE failures/issues occur (geographic-centric)
            
            # TextToSQL handles all data queries including schema and date range queries - catch-all for everything else
            TextToSQLHandler(),
        ]
        
        logger.info(f"QueryRouter initialized with {len(self.handlers)} handlers (Simplified mode - SchemaHandler and DateRangeHandler removed)")
    
    def route(self, context: QueryContext) -> str:
        """Route query to appropriate handler."""
        intent_result = classify_intent(context.query)
        context.intent = intent_result.label
        logger.debug(f"Intent classified as {intent_result.label} ({intent_result.reason})")

        if context.intent == "empty":
            return EmptyQueryHandler().handle(context)

        if context.intent in {"small_talk", "off_domain", "safety"}:
            return GenericIntentHandler().handle(context)

        logger.debug(f"Routing query: '{context.query[:50]}...'")
        
        for handler in self.handlers:
            if handler.can_handle(context):
                handler_name = handler.get_name()
                logger.info(f"Query routed to {handler_name}")
                
                try:
                    response = handler.handle(context)
                    if response is None:
                        logger.debug(f"{handler_name} returned None, continuing to next handler")
                        continue
                    logger.debug(f"{handler_name} completed successfully")
                    return response
                except Exception as e:
                    logger.error(f"{handler_name} failed: {e}", exc_info=True)
                    # Generate user-friendly error message
                    from chat.handlers_errors import generate_user_friendly_error
                    return generate_user_friendly_error(
                        error=e,
                        query=context.query,
                        context=context,
                        error_type=None,
                        handler_name=handler_name
                    )
        
        # shouldn't happen but just in case
        logger.error("No handler matched query - this should not happen!")
        # Generate helpful error message
        error_parts = [
            "<p>I couldn't process your query. This is unexpected, but here's how I can help:</p>"
        ]
        
        # Add suggestions based on query type
        query_lower = context.query_lower
        if any(kw in query_lower for kw in ['prescribe', 'recommend', 'advice']):
            error_parts.append(
                "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                "<strong>For prescriptive queries:</strong> Try 'Prescribe for model [ModelName] part [PartName]'</p>"
            )
        elif any(kw in query_lower for kw in ['rate', 'percentage', 'failure rate']):
            error_parts.append(
                "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                "<strong>For rate queries:</strong> Try 'What's the failure rate for [Model]?'</p>"
            )
        else:
            error_parts.append(
                "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 8px;'>"
                "<strong>Example queries:</strong></p>"
                "<ul style='font-size: 0.9em; color: #94a3b8; margin-left: 20px;'>"
                "<li>'What's the failure rate for Sentra?'</li>"
                "<li>'Show me all vehicles with mileage > 50000'</li>"
                "<li>'Top 5 failing parts'</li>"
                "<li>'What columns are available?'</li>"
                "</ul>"
            )
        
        return "".join(error_parts)

