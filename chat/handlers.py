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
        top_k: int
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
        return ("<p>Hello, I'm the Nissan chat assistant. Ask for monthly failure counts, "
                "failure rates by model/age/mileage, or prescriptive guidance "
                "(e.g. 'prescribe for model Leaf part Battery').</p>")


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
        return any(w in context.query_lower for w in 
                   ["prescribe", "recommend", "prescriptive", "advice", "action"])
    
    def handle(self, context: QueryContext) -> str:
        from chat_helper import parse_model_part_from_text, ensure_failures_column
        
        logger.debug("PrescriptiveHandler triggered")
        
        try:
            model, part = parse_model_part_from_text(context.df_history, context.query)
        except Exception as e:
            logger.warning(f"Failed to parse model/part from query: {e}")
            model, part = None, None
        
        if not (model and part):
            return ("<p>I can generate a prescriptive summary if you include a model and part "
                    "(e.g. 'prescribe for model Sentra part Battery').</p>")
        
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
            logger.info(f"Generating prescriptive summary via Bedrock for {model}/{part}")
            summary_html = context.get_bedrock_summary(model, part, mileage_bucket, age_bucket, pct)
            plain = re.sub(r"<[^>]+>", "", summary_html).strip()
            return f"<p>{_html.escape(plain)}</p>"
        except Exception as e:
            logger.error(f"Bedrock prescriptive summary failed: {e}", exc_info=True)
            return f"<p>Could not generate prescriptive summary via Bedrock: {_html.escape(str(e))}</p>"


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

