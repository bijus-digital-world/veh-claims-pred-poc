"""
chat/handlers_default.py

Default handler for general queries.
This is the catch-all handler when no specific handler matches.
"""

from typing import Optional
import html as _html
import pandas as pd

from chat.handlers import QueryHandler, QueryContext
from chat_helper import (
    _detect_metric_from_text,
    _metric_or_fallback_column
)
from utils.logger import chat_logger as logger


class DefaultHandler(QueryHandler):
    """
    Default handler for general queries.
    
    Provides a dataset overview using direct data queries.
    
    This should always be the last handler in the router.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        # Default handler always returns True (catch-all)
        return True
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("DefaultHandler processing (catch-all)")
        
        # Check if this is a completely unrelated query
        if self._is_unrelated_query(context.query):
            return self._handle_unrelated_query(context)
        
        # Generate overview from full dataset
        return self._generate_overview(context, None)
    
    def _is_unrelated_query(self, query: str) -> bool:
        """Check if the query is completely unrelated to vehicle analytics."""
        query_lower = query.lower()
        
        # Unrelated topics
        unrelated_topics = [
            "weather", "cooking", "recipe", "sports", "football", "basketball", "soccer",
            "movies", "music", "entertainment", "celebrity", "news", "politics",
            "travel", "vacation", "hotel", "restaurant", "shopping", "fashion",
            "health", "medical", "doctor", "medicine", "fitness", "exercise",
            "education", "school", "university", "homework", "study",
            "technology", "computer", "software", "programming", "coding",
            "finance", "banking", "investment", "stock", "money", "salary",
            "personal", "family", "relationship", "dating", "marriage"
        ]
        
        # Check if query contains unrelated topics
        for topic in unrelated_topics:
            if topic in query_lower:
                return True
        
        # Check for very short, vague queries
        if len(query.split()) <= 2 and not any(word in query_lower for word in 
            ["data", "analysis", "failure", "model", "vehicle", "car", "nissan", "rate", "trend"]):
            return True
        
        return False
    
    def _handle_unrelated_query(self, context: QueryContext) -> str:
        """Handle queries that are unrelated to vehicle analytics."""
        query = context.query.strip()
        
        # Get some basic stats about the dataset
        try:
            total_records = len(context.df_history)
            models = context.df_history['model'].nunique() if 'model' in context.df_history.columns else 0
            failures = context.df_history['failures_count'].sum() if 'failures_count' in context.df_history.columns else 0
        except (KeyError, AttributeError, TypeError):
            total_records = 0
            models = 0
            failures = 0
        
        html_parts = [
            "<p><strong>I'm not sure how to help with that question.</strong></p>",
            "<p>I'm specialized in analyzing Nissan vehicle telematics data. I can help you with:</p>",
            "<ul style='margin-top:8px;'>",
            "<li><strong>Vehicle Analytics:</strong> Failure rates, trends, and patterns</li>",
            "<li><strong>Model Analysis:</strong> Compare Sentra, Leaf, Ariya, and other models</li>",
            "<li><strong>Data Insights:</strong> Monthly trends, age/mileage analysis</li>",
            "<li><strong>Prescriptive Guidance:</strong> Recommendations for specific parts/models</li>",
            "</ul>"
        ]
        
        if total_records > 0:
            html_parts.append(f"<p><em>I have access to <strong>{total_records:,} records</strong> covering <strong>{models} models</strong> with <strong>{failures:,} total failures</strong>.</em></p>")
        
        html_parts.extend([
            "<p><strong>Try asking something like:</strong></p>",
            "<ul style='margin-top:8px;'>",
            "<li>\"What's the failure rate for Sentra?\"</li>",
            "<li>\"Show me failure trends by month\"</li>",
            "<li>\"Which model has the most failures?\"</li>",
            "<li>\"Prescribe for model Leaf part Battery\"</li>",
            "</ul>",
            "<p><em>I'm here to help with your vehicle data analysis!</em></p>"
        ])
        
        return "".join(html_parts)
    
    def _generate_overview(self, context: QueryContext, sample_df: Optional[pd.DataFrame]) -> str:
        """Generate dataset overview"""
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        try:
            # Full dataset overview
            df = df_with_metric if df_with_metric is not None else context.df_history
            
            if metric_col is None:
                return "<p>Couldn't determine the metric to summarize. Try asking about failures, claims, repairs, or recalls.</p>"
            
            total_rows = len(df)
            total_metric = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            rate_sample = (total_metric / total_rows * 100.0) if total_rows > 0 else 0.0
            
            # Top model
            top_model = None
            if "model" in df.columns and not df["model"].isna().all():
                modes = df["model"].mode()
                if len(modes) > 0:
                    top_model = modes.iloc[0]
            
            # Top part
            top_part_val = None
            from chat_helper import _safe_column
            top_part_col = _safe_column(df, ["primary_failed_part", "failed_part", "part"])
            if top_part_col:
                try:
                    grp = df.groupby(top_part_col)[metric_col].sum()
                    if not grp.empty:
                        top_part_val = grp.idxmax()
                except Exception:
                    top_part_val = None
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            parts = []
            if top_model:
                parts.append(f"Top model overall: {_html.escape(str(top_model))}.")
            if top_part_val:
                parts.append(f"Top failed part overall: {_html.escape(str(top_part_val))}.")
            parts_txt = " ".join(parts)
            
            return (f"<p>Dataset overview: total records = {total_rows}, total {label} = {int(total_metric)} "
                   f"(approx. {rate_sample:.1f}% per-record). {parts_txt}</p>")
        
        except Exception as e:
            logger.error(f"Overview generation failed: {e}", exc_info=True)
            return f"<p>Could not summarize dataset (error: {_html.escape(str(e))}).</p>"
    
    def _handle_which_models_rising(self, df, metric_col):
        """Handle which models show rising trends"""
        try:
            model_trends = _compute_model_trends(df, metric_col, min_months=6, slope_threshold=0.0, top_n=6)
            
            if not model_trends:
                return ("<p>No models show a clearly rising trend (or there is insufficient monthly "
                       "history per model to determine trend).</p>")
            
            lines = [f"{_html.escape(str(m))} → slope ≈ {s:.2f} per month (months: {mon}, last: {int(last)})" 
                    for (m, s, mon, last) in model_trends]
            
            return ("<p>Models showing the strongest rising trends (sorted by slope):<br>"
                   + "<br>".join(lines)
                   + "<br><em>Note:</em> slopes are fitted over each model's monthly series.</p>")
        
        except Exception as e:
            logger.error(f"Model trends calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't determine per-model trends: {_html.escape(str(e))}</p>"
    
    def _handle_overall_trend(self, df, metric_col):
        """Handle overall trend calculation"""
        try:
            df = df.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            
            if len(df_month) < 3:
                return "<p>Not enough months to determine a reliable overall trend.</p>"
            
            y = df_month.values
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            return f"<p>Overall {label} trend is {trend} (slope={slope:.1f} per month).</p>"
        
        except Exception as e:
            logger.error(f"Overall trend calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't determine trend: {_html.escape(str(e))}</p>"

