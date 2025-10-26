"""
Handlers for model-specific queries that should prioritize the specific model.
"""

import re
import html as _html
import pandas as pd
from typing import Optional

from chat.handlers import QueryHandler, QueryContext
from chat_helper import _detect_metric_from_text, _metric_or_fallback_column, _safe_column
from utils.logger import chat_logger as logger


class ModelSpecificRateHandler(QueryHandler):
    """
    Handle rate queries for specific models (e.g., "failure rate for Sentra").
    Prioritizes the specific model over overall statistics.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        # Check if query mentions a specific model
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        query_lower = context.query_lower
        
        # Look for patterns like "failure rate for Sentra", "Sentra failure rate", etc.
        has_model = any(model in query_lower for model in models)
        has_rate = "rate" in query_lower or "per 100" in query_lower
        
        return has_model and has_rate
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ModelSpecificRateHandler processing")
        
        # Extract the specific model from the query
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        mentioned_model = None
        for model in models:
            if model in context.query_lower:
                mentioned_model = model
                break
        
        if not mentioned_model:
            return "<p>Could not identify the specific model in your query.</p>"
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            return "<p>Requested metric not available to compute rates.</p>"
        
        # Find the model column
        model_col = _safe_column(df, ["model"])
        if not model_col:
            return "<p>Model information not available in the dataset.</p>"
        
        try:
            # Filter data for the specific model
            model_data = df[df[model_col].str.lower() == mentioned_model.lower()]
            
            if model_data.empty:
                return f"<p>No data found for {mentioned_model.title()} in the dataset.</p>"
            
            # Calculate rates for the specific model
            model_total = float(pd.to_numeric(model_data[metric_col], errors="coerce").fillna(0).sum())
            model_rows = len(model_data)
            model_rate = (model_total / model_rows * 100.0) if model_rows > 0 else 0.0
            
            # Calculate overall rates for comparison
            overall_total = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            overall_rows = len(df)
            overall_rate = (overall_total / overall_rows * 100.0) if overall_rows > 0 else 0.0
            
            # Calculate rates for all models for comparison
            grp = df.groupby(model_col).agg(total_metric=(metric_col, "sum"), rows=(metric_col, "count"))
            grp = grp[grp["rows"] > 0]
            grp["rate_per_100"] = (grp["total_metric"] / grp["rows"]) * 100.0
            all_models = grp.sort_values("rate_per_100", ascending=False)
            
            # Find the specific model's rank
            model_rank = None
            if mentioned_model.lower() in all_models.index.str.lower().values:
                model_rank = (all_models.index.str.lower() == mentioned_model.lower()).argmax() + 1
            
            label = "failure rate" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build the response with specific model first
            html_parts = []
            
            # Main answer - specific model
            html_parts.append(f"<p><strong>{mentioned_model.title()} {label.title()}: {model_rate:.1f}%</strong></p>")
            
            # Context about the specific model
            if model_total > 0:
                html_parts.append(f"<p>This represents {int(model_total)} {label.replace(' rate', 's')} across {model_rows} {mentioned_model.title()} records.</p>")
            else:
                html_parts.append(f"<p>No {label.replace(' rate', 's')} recorded for {mentioned_model.title()} in the dataset.</p>")
            
            # Comparison with overall rate
            if overall_rate > 0:
                if model_rate > overall_rate:
                    diff = model_rate - overall_rate
                    html_parts.append(f"<p><em>This is {diff:.1f} percentage points higher than the overall {label} of {overall_rate:.1f}%.</em></p>")
                elif model_rate < overall_rate:
                    diff = overall_rate - model_rate
                    html_parts.append(f"<p><em>This is {diff:.1f} percentage points lower than the overall {label} of {overall_rate:.1f}%.</em></p>")
                else:
                    html_parts.append(f"<p><em>This matches the overall {label} of {overall_rate:.1f}%.</em></p>")
            
            # Ranking information
            if model_rank is not None:
                total_models = len(all_models)
                if model_rank == 1:
                    html_parts.append(f"<p><em>{mentioned_model.title()} has the highest {label} among all models.</em></p>")
                elif model_rank == total_models:
                    html_parts.append(f"<p><em>{mentioned_model.title()} has the lowest {label} among all models.</em></p>")
                else:
                    html_parts.append(f"<p><em>{mentioned_model.title()} ranks #{model_rank} out of {total_models} models by {label}.</em></p>")
            
            # Show comparison with other models
            if len(all_models) > 1:
                html_parts.append(f"<p><strong>For comparison, here are all model {label}s:</strong></p>")
                html_parts.append("<ul style='margin-top:6px;'>")
                
                for idx, row in all_models.head(6).iterrows():
                    model_name = str(idx)
                    rate = row['rate_per_100']
                    # Highlight the specific model
                    if model_name.lower() == mentioned_model.lower():
                        html_parts.append(f"<li><strong>{_html.escape(model_name)} → {rate:.1f}%</strong> ← Your query</li>")
                    else:
                        html_parts.append(f"<li>{_html.escape(model_name)} → {rate:.1f}%</li>")
                
                html_parts.append("</ul>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Model-specific rate calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't compute {label} for {mentioned_model.title()} (error: {_html.escape(str(e))}).</p>"


class ModelSpecificCountHandler(QueryHandler):
    """
    Handle count queries for specific models (e.g., "how many failures for Sentra").
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        # Check if query mentions a specific model and asks for counts
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        query_lower = context.query_lower
        
        has_model = any(model in query_lower for model in models)
        has_count = any(word in query_lower for word in ["how many", "count", "total", "number of"])
        
        return has_model and has_count
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ModelSpecificCountHandler processing")
        
        # Extract the specific model from the query
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        mentioned_model = None
        for model in models:
            if model in context.query_lower:
                mentioned_model = model
                break
        
        if not mentioned_model:
            return "<p>Could not identify the specific model in your query.</p>"
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            return "<p>Requested metric not available to compute counts.</p>"
        
        # Find the model column
        model_col = _safe_column(df, ["model"])
        if not model_col:
            return "<p>Model information not available in the dataset.</p>"
        
        try:
            # Filter data for the specific model
            model_data = df[df[model_col].str.lower() == mentioned_model.lower()]
            
            if model_data.empty:
                return f"<p>No data found for {mentioned_model.title()} in the dataset.</p>"
            
            # Calculate counts for the specific model
            model_total = int(pd.to_numeric(model_data[metric_col], errors="coerce").fillna(0).sum())
            model_rows = len(model_data)
            
            # Calculate overall counts for comparison
            overall_total = int(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            overall_rows = len(df)
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build the response
            html_parts = []
            
            # Main answer
            html_parts.append(f"<p><strong>{mentioned_model.title()}: {model_total} {label}</strong></p>")
            
            # Context
            html_parts.append(f"<p>This represents {model_total} {label} across {model_rows} {mentioned_model.title()} records.</p>")
            
            # Comparison with overall
            if overall_total > 0:
                percentage = (model_total / overall_total * 100.0) if overall_total > 0 else 0.0
                html_parts.append(f"<p><em>{mentioned_model.title()} accounts for {percentage:.1f}% of all {label} in the dataset.</em></p>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Model-specific count calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't compute {label} count for {mentioned_model.title()} (error: {_html.escape(str(e))}).</p>"
