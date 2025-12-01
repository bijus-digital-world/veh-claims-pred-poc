"""
Helper functions for generating user-friendly error messages with contextual suggestions.
"""

import re
import html as _html
import pandas as pd
from typing import Optional, Dict, List

from chat.handlers import QueryContext


def generate_user_friendly_error(
    error: Exception,
    query: str,
    context: Optional[QueryContext] = None,
    error_type: Optional[str] = None,
    handler_name: Optional[str] = None
) -> str:
    """
    Generate a user-friendly error message with contextual suggestions.
    
    Args:
        error: The exception that occurred
        query: The user's query
        context: Query context (optional, for data-aware suggestions)
        error_type: Type of error (optional, for specific error handling)
        handler_name: Name of handler that failed (optional)
    
    Returns:
        HTML-formatted error message with helpful suggestions
    """
    error_msg = str(error).lower()
    query_lower = query.lower()
    
    # Initialize error parts
    error_parts = []
    
    # Main error message - user-friendly based on error type
    if error_type == "sql_generation_failed":
        error_parts.append(
            "<p>I had trouble understanding your question in a way that I can query the data.</p>"
        )
    elif error_type == "sql_execution_failed":
        error_parts.append(
            "<p>I couldn't execute the query based on your question. This might be because the data doesn't match what you're looking for.</p>"
        )
    elif error_type == "no_results":
        error_parts.append(
            "<p>Your query didn't return any results. This might mean there's no data matching your criteria.</p>"
        )
    elif error_type == "prescriptive_failed":
        error_parts.append(
            "<p>I couldn't generate prescriptive recommendations for your request.</p>"
        )
    else:
        # Generic error
        error_parts.append(
            "<p>I encountered an issue processing your question.</p>"
        )
    
    # Add contextual suggestions if context is available
    if context and context.df_history is not None:
        suggestions = _generate_contextual_suggestions(query, context.df_history, error_msg)
        if suggestions:
            error_parts.append(suggestions)
    
    # Add general helpful tips
    error_parts.append(_generate_general_tips(query))
    
    # Add error details for debugging (if technical error detected)
    if "column" in error_msg or "sql" in error_msg:
        error_parts.append(
            "<p style='font-size: 0.85em; color: #64748b; margin-top: 10px;'>"
            "<em>Tip: Try rephrasing your question or asking about specific models, parts, or metrics.</em></p>"
        )
    
    return "".join(error_parts)


def _generate_contextual_suggestions(query: str, df: pd.DataFrame, error_msg: str) -> str:
    """Generate contextual suggestions based on the query and available data."""
    query_lower = query.lower()
    suggestions_parts = []
    
    # Check for model-related issues
    model_match = re.search(r'\b(leaf|sentra|ariya|altima|rogue|pathfinder|frontier|titan)\b', query_lower)
    if model_match:
        model_name = model_match.group(1).title()
        if 'model' in df.columns:
            model_data = df[df.get('model', pd.Series(dtype=str)).str.lower() == model_name.lower()]
            if model_data.empty:
                suggestions_parts.append(
                    f"<p style='color: #fca5a5; margin-top: 10px;'>"
                    f"<strong>Note:</strong> No data found for model '{model_name}'. "
                    f"Available models in your dataset include: {_get_available_models(df)}</p>"
                )
    
    # Check for part-related issues
    part_keywords = ['battery', 'engine', 'brake', 'cooling', 'transmission', 'clutch']
    mentioned_part = [kw for kw in part_keywords if kw in query_lower]
    if mentioned_part and 'primary_failed_part' in df.columns:
        available_parts = df['primary_failed_part'].dropna().unique()[:5]
        suggestions_parts.append(
            f"<p style='color: #cfe9ff; margin-top: 10px;'>"
            f"<strong>Available parts:</strong> {', '.join([str(p) for p in available_parts])}</p>"
        )
    
    # Check for column-related issues
    if "column" in error_msg or "no such column" in error_msg:
        available_cols = list(df.columns)[:10]
        suggestions_parts.append(
            f"<p style='color: #cfe9ff; margin-top: 10px;'>"
            f"<strong>Available columns in your dataset:</strong> {', '.join(available_cols)}</p>"
        )
    
    return "".join(suggestions_parts)


def _generate_general_tips(query: str) -> str:
    """Generate general helpful tips based on query type."""
    query_lower = query.lower()
    
    tips = []
    
    if any(kw in query_lower for kw in ['prescribe', 'recommend', 'advice']):
        tips.append(
            "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 10px;'>"
            "<strong>Tip:</strong> For prescriptive queries, try: 'Prescribe for model [ModelName] part [PartName]'</p>"
        )
    elif any(kw in query_lower for kw in ['rate', 'percentage', 'percent']):
        tips.append(
            "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 10px;'>"
            "<strong>Tip:</strong> Try asking: 'What's the failure rate for [Model]?'</p>"
        )
    elif any(kw in query_lower for kw in ['compare', 'vs', 'versus']):
        tips.append(
            "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 10px;'>"
            "<strong>Tip:</strong> Try asking: 'Compare [Model1] vs [Model2] failure rates'</p>"
        )
    else:
        tips.append(
            "<p style='font-size: 0.9em; color: #94a3b8; margin-top: 10px;'>"
            "<strong>Tip:</strong> Try rephrasing your question or ask about specific models, parts, or metrics. "
            "You can also ask 'What columns are available?' to see what data is in the dataset.</p>"
        )
    
    return "".join(tips)


def _get_available_models(df: pd.DataFrame) -> str:
    """Get list of available models as a formatted string."""
    if 'model' not in df.columns:
        return "N/A"
    models = df['model'].dropna().unique()[:5]
    return ", ".join([str(m) for m in models])

