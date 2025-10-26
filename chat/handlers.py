"""
chat/handlers.py

Query handler pattern for chat functionality.
Each handler is responsible for one type of query.

This modular approach replaces the 499-line generate_reply() god function
with focused, testable, and maintainable handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import re
import html as _html
import pandas as pd
import numpy as np

from config import config
from utils.logger import chat_logger as logger


class QueryContext:
    """
    Context object passed to all handlers containing necessary data and state.
    """
    def __init__(
        self,
        query: str,
        df_history: pd.DataFrame,
        faiss_res: dict,
        tfidf_vect,
        tfidf_X,
        tfidf_rows,
        get_bedrock_summary_callable,
        top_k: int,
        conversation_context=None
    ):
        self.query = query
        self.query_lower = query.lower()
        self.df_history = df_history
        self.faiss_res = faiss_res
        self.tfidf_vect = tfidf_vect
        self.tfidf_X = tfidf_X
        self.tfidf_rows = tfidf_rows
        self.get_bedrock_summary = get_bedrock_summary_callable
        self.top_k = top_k
        self.conversation_context = conversation_context
        
        # Computed properties
        self.requested_metric = None
        self.metric_col = None
        self.sample_df = None


class QueryHandler(ABC):
    """
    Abstract base class for query handlers.
    
    Each handler should:
    - Detect if it can handle a query (can_handle)
    - Process the query and return HTML (handle)
    - Be completely independent and testable
    """
    
    @abstractmethod
    def can_handle(self, context: QueryContext) -> bool:
        """
        Determine if this handler can process the query.
        
        Args:
            context: Query context with query text and data
        
        Returns:
            True if this handler should process the query
        """
        pass
    
    @abstractmethod
    def handle(self, context: QueryContext) -> str:
        """
        Process the query and return HTML response.
        
        Args:
            context: Query context with query text and data
        
        Returns:
            HTML string response
        """
        pass
    
    def get_name(self) -> str:
        """Get handler name for logging"""
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
        return (len(context.query.split()) <= 3 and 
                any(g in context.query_lower for g in ["hi", "hello", "hey"]))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("GreetingHandler triggered")
        
        # Get some basic stats about the dataset to make the greeting more informative
        try:
            total_records = len(context.df_history)
            models = context.df_history['model'].nunique() if 'model' in context.df_history.columns else 0
            failures = context.df_history['failures_count'].sum() if 'failures_count' in context.df_history.columns else 0
            
            # Create a more engaging and informative greeting
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
            # Fallback to simpler greeting
            return ("<p><strong>Hello! I'm your Nissan Telematics Analytics Assistant.</strong></p>"
                    "<p>I can help you analyze vehicle failure data, compare models, track trends, and provide prescriptive insights.</p>"
                    "<p><em>Try asking: \"What's the failure rate for Sentra?\" or \"Show me failure trends\"</em></p>")


class SchemaHandler(QueryHandler):
    """Handle schema/column listing requests"""
    
    def can_handle(self, context: QueryContext) -> bool:
        return any(p in context.query_lower for p in 
                   ["column names", "columns", "fields", "schema", "features", "headers"])
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("SchemaHandler triggered")
        cols = list(context.df_history.columns)
        formatted = ", ".join(cols)
        return (f"<p>Your dataset contains the following columns:</p>"
                f"<p style='color:#cfe9ff'>{formatted}</p>")


class DateRangeHandler(QueryHandler):
    """Handle date range / timespan queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        pattern = r"\b(how many months|months (?:of )?data|how many years|years (?:of )?data|date range|earliest date|latest date|first date|last date|data from)\b"
        return bool(re.search(pattern, context.query_lower))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("DateRangeHandler triggered")
        
        if "date" not in context.df_history.columns:
            return "<p>Your dataset does not contain a <strong>'date'</strong> column, so I can't compute the time span.</p>"
        
        try:
            df_dates = context.df_history.copy()
            df_dates["date_parsed"] = pd.to_datetime(df_dates["date"], errors="coerce")
            df_dates = df_dates.dropna(subset=["date_parsed"])
            
            if df_dates.empty:
                return "<p>Could not find any parsable dates in the 'date' column.</p>"
            
            min_d = df_dates["date_parsed"].min()
            max_d = df_dates["date_parsed"].max()
            months_span = (max_d.year - min_d.year) * 12 + (max_d.month - min_d.month) + 1
            distinct_months = int(df_dates["date_parsed"].dt.to_period("M").nunique())
            rows_with_dates = int(df_dates.shape[0])
            min_s = min_d.strftime("%b %Y")
            max_s = max_d.strftime("%b %Y")
            
            return (f"<p>Dataset date range: <strong>{min_s}</strong> → <strong>{max_s}</strong>.</p>"
                    f"<p>This spans <strong>{months_span}</strong> calendar months, "
                    f"with data present in <strong>{distinct_months}</strong> distinct months. "
                    f"There are <strong>{rows_with_dates}</strong> records with parsable dates.</p>")
        except Exception as e:
            logger.error(f"DateRangeHandler failed: {e}", exc_info=True)
            return f"<p>Couldn't compute date range (error: {_html.escape(str(e))}).</p>"


class PrescriptiveHandler(QueryHandler):
    """Handle prescriptive/recommendation requests using Bedrock"""
    
    def can_handle(self, context: QueryContext) -> bool:
        prescriptive_keywords = [
            "prescribe", "recommend", "prescriptive", "advice", "action",
            "what should i do", "what should we do", "how to fix", "how to improve",
            "what to do about", "suggestions", "recommendations", "guidance",
            "help with", "solution", "remedy", "intervention", "improving"
        ]
        return any(w in context.query_lower for w in prescriptive_keywords)
    
    def handle(self, context: QueryContext) -> str:
        from chat_helper import parse_model_part_from_text, ensure_failures_column
        
        logger.debug("PrescriptiveHandler triggered")
        
        try:
            model, part = parse_model_part_from_text(context.df_history, context.query)
        except Exception as e:
            logger.warning(f"Failed to parse model/part from query: {e}")
            model, part = None, None
        
        if not (model and part):
            # Provide general prescriptive guidance for high-level queries
            if any(phrase in context.query_lower for phrase in ["high failure", "failure rate", "failures", "problems", "issues"]):
                return self._handle_general_prescriptive_guidance(context)
            else:
                # Get available models and parts for better guidance
                available_models = context.df_history.get("model", pd.Series(dtype=str)).unique()[:5]  # First 5 models
                available_parts = context.df_history.get("primary_failed_part", pd.Series(dtype=str)).unique()[:5]  # First 5 parts
                
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
            return f"<p>No data found for model '{model}' and part '{part}'.</p>"
        
        mileage_bucket = slice_df.iloc[0].get("mileage_bucket", "")
        age_bucket = slice_df.iloc[0].get("age_bucket", "")
        total_inc = slice_df.shape[0]
        
        # Ensure failures_count present
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
            return f"<p>Could not generate prescriptive summary via Bedrock: {_html.escape(str(e))}</p>"
    
    def _handle_general_prescriptive_guidance(self, context: QueryContext) -> str:
        """Handle general prescriptive queries about high failure rates."""
        try:
            # Calculate overall failure statistics
            df = context.df_history
            total_records = len(df)
            total_failures = df.get('failures_count', pd.Series(dtype=int)).sum() if 'failures_count' in df.columns else 0
            overall_rate = (total_failures / total_records * 100) if total_records > 0 else 0
            
            # Get model-specific failure rates
            model_rates = []
            if 'model' in df.columns and 'failures_count' in df.columns:
                model_stats = df.groupby('model').agg({
                    'failures_count': 'sum',
                    'model': 'count'
                }).rename(columns={'model': 'count'})
                model_stats['rate'] = (model_stats['failures_count'] / model_stats['count'] * 100)
                model_rates = model_stats.sort_values('rate', ascending=False)
            
            # Get top failing parts
            top_parts = []
            if 'primary_failed_part' in df.columns and 'failures_count' in df.columns:
                part_stats = df.groupby('primary_failed_part')['failures_count'].sum().sort_values(ascending=False)
                top_parts = part_stats.head(3)
            
            # Build prescriptive response
            html_parts = [
                "<p><strong>Prescriptive Guidance for High Failure Rates</strong></p>",
                f"<p>Based on your data analysis showing an overall failure rate of <strong>{overall_rate:.1f}%</strong> across {total_records:,} records, here are my recommendations:</p>"
            ]
            
            # Model-specific recommendations
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
            
            # Part-specific recommendations
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
            
            # General recommendations
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
            return f"<p>Could not generate prescriptive guidance: {_html.escape(str(e))}</p>"


# Import additional handlers (we'll create these next)
from chat.handlers_metrics import (
    TotalMetricHandler,
    CountAndAverageHandler,
    TimeToResolutionHandler
)

from chat.handlers_analysis import (
    MonthlyAggregateHandler,
    RateHandler,
    TrendHandler,
    TopFailedPartsHandler,
    IncidentDetailsHandler
)

from chat.handlers_default import DefaultHandler

from chat.handlers_supplier import (
    SupplierListHandler,
    DefectiveSupplierHandler,
    VINQueryHandler,
    FailureReasonHandler,
    LocationQueryHandler
)

from chat.handlers_distribution import (
    MileageDistributionHandler,
    AgeDistributionHandler
)

from chat.handlers_context import (
    ContextAwareHandler,
    ConversationSummaryHandler
)

from chat.handlers_model_specific import (
    ModelSpecificRateHandler,
    ModelSpecificCountHandler
)

from chat.handlers_comparison import (
    ModelComparisonHandler
)

from chat.handlers_ranking import (
    ModelRankingHandler,
    PartRankingHandler
)


class QueryRouter:
    """
    Routes queries to appropriate handlers.
    
    Handlers are tried in order until one can handle the query.
    The DefaultHandler should always be last as a catch-all.
    """
    
    def __init__(self):
        self.handlers = [
            # Meta queries (should be first)
            EmptyQueryHandler(),
            GreetingHandler(),
            SchemaHandler(),
            DateRangeHandler(),
            
            # Conversation context handlers
            ConversationSummaryHandler(),
            
            # Model comparison handlers (highest priority for comparison queries)
            ModelComparisonHandler(),
            
            # Ranking handlers (high priority for ranking queries)
            ModelRankingHandler(),
            PartRankingHandler(),
            
            # Model-specific handlers (high priority for specific model queries)
            ModelSpecificRateHandler(),
            ModelSpecificCountHandler(),
            
            # NEW: Supplier, VIN, and Location queries
            SupplierListHandler(),
            DefectiveSupplierHandler(),
            VINQueryHandler(),
            FailureReasonHandler(),
            LocationQueryHandler(),
            
            # NEW: Mileage and Age distribution queries
            MileageDistributionHandler(),
            AgeDistributionHandler(),
            
            # Specific analysis queries
            PrescriptiveHandler(),
            TimeToResolutionHandler(),
            TotalMetricHandler(),
            CountAndAverageHandler(),
            MonthlyAggregateHandler(),
            RateHandler(),
            TrendHandler(),
            TopFailedPartsHandler(),
            IncidentDetailsHandler(),
            
            # Context-aware handler (before default)
            ContextAwareHandler(),
            
            # Catch-all (must be last)
            DefaultHandler(),
        ]
        
        logger.info(f"QueryRouter initialized with {len(self.handlers)} handlers")
    
    def route(self, context: QueryContext) -> str:
        """
        Route query to appropriate handler.
        
        Args:
            context: Query context
        
        Returns:
            HTML response string
        """
        logger.debug(f"Routing query: '{context.query[:50]}...'")
        
        for handler in self.handlers:
            if handler.can_handle(context):
                handler_name = handler.get_name()
                logger.info(f"Query routed to {handler_name}")
                
                try:
                    response = handler.handle(context)
                    logger.debug(f"{handler_name} completed successfully")
                    return response
                except Exception as e:
                    logger.error(f"{handler_name} failed: {e}", exc_info=True)
                    return f"<p>Error processing query with {handler_name}: {_html.escape(str(e))}</p>"
        
        # This should never happen (DefaultHandler should catch everything)
        logger.error("No handler matched query - this should not happen!")
        return "<p>I couldn't process your query. Please try rephrasing.</p>"

