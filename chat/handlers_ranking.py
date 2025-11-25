"""
Handlers for ranking and top/bottom queries.
"""

import re
import html as _html
import pandas as pd
from typing import Optional

from chat.handlers import QueryHandler, QueryContext
from chat_helper import _detect_metric_from_text, _metric_or_fallback_column, _safe_column
from utils.logger import chat_logger as logger


def _extract_number_from_query(query: str) -> Optional[int]:
    """Extract number from query (e.g., 'top 3' -> 3)."""
    patterns = [
        r"top\s+(\d+)",
        r"first\s+(\d+)",
        r"worst\s+(\d+)",
        r"best\s+(\d+)",
        r"bottom\s+(\d+)",
        r"last\s+(\d+)"
    ]

    lowered_query = query.lower()
    for pattern in patterns:
        match = re.search(pattern, lowered_query)
        if match:
            return int(match.group(1))

    return None


class ModelRankingHandler(QueryHandler):
    """
    Handle model ranking queries (e.g., "Which model has the most failures?", "Top 3 models").
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        
        # Skip if asking for individual records/vehicles (e.g., "top 5 vehicles", "top 10 Sentra vehicles")
        # This should be handled by Text-to-SQL, not ranking handler
        # Allow words between number and "vehicles" (e.g., "top 10 Sentra vehicles")
        if re.search(r'\btop\s+\d+.*?\b(vehicles?|records?|rows?|entries?|items?)\b', query_lower):
            return False
        
        # Check for ranking/which/top/bottom keywords
        ranking_keywords = [
            "which model", "which models", "top", "bottom", "worst", "best",
            "most", "least", "highest", "lowest", "rank", "ranking",
            "number one", "number 1", "first", "last"
        ]
        
        has_ranking = any(keyword in query_lower for keyword in ranking_keywords)
        
        # Check for model-related terms
        model_terms = ["model", "vehicle", "car", "sentra", "leaf", "ariya"]
        has_model_context = any(term in query_lower for term in model_terms)
        
        return has_ranking and has_model_context
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ModelRankingHandler processing")
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            metric_col = "failures_count"
        
        # Find the model column
        model_col = _safe_column(df, ["model"])
        if not model_col:
            return "<p>Model information not available in the dataset.</p>"
        
        try:
            # Calculate model rankings
            model_stats = df.groupby(model_col).agg({
                metric_col: ["sum", "count"]
            }).reset_index()
            
            # Flatten column names
            model_stats.columns = ["model", "total_metric", "record_count"]
            model_stats["rate"] = (model_stats["total_metric"] / model_stats["record_count"] * 100.0)
            
            # Sort by total metric (descending for "most")
            model_stats = model_stats.sort_values("total_metric", ascending=False)
            
            if model_stats.empty:
                return "<p>No model data available for ranking.</p>"
            
            # Determine if asking for top or bottom
            query_lower = context.query_lower
            is_bottom_query = any(word in query_lower for word in ["worst", "least", "lowest", "bottom"])
            is_top_query = any(word in query_lower for word in ["best", "most", "highest", "top"])
            
            # Determine number of results
            n_results = _extract_number_from_query(context.query)
            if n_results is None:
                n_results = 1 if "which" in query_lower else 3
            
            # Get the requested results
            # For "most failures" queries, we want the models with most failures (worst reliability)
            # For "least failures" queries, we want the models with least failures (best reliability)
            if "most" in query_lower or "highest" in query_lower:
                # Most failures = worst reliability = head of sorted list (descending)
                results = model_stats.head(n_results)
                position = "worst"
                is_bottom_query = True  # Override for "most failures" queries
            elif "least" in query_lower or "lowest" in query_lower or "best" in query_lower:
                # Least failures = best reliability = tail of sorted list (ascending)
                results = model_stats.tail(n_results).iloc[::-1]  # Reverse to show best first
                position = "best"
                is_bottom_query = False  # Override for "least failures" queries
            elif "top" in query_lower and ("failures" in query_lower or "fail" in query_lower):
                # "Top X by failures" = most failures = worst reliability = head of sorted list
                results = model_stats.head(n_results)
                position = "worst"
                is_bottom_query = True  # Override for "top by failures" queries
            elif is_bottom_query:
                results = model_stats.tail(n_results).iloc[::-1]  # Reverse to show worst first
                position = "worst"
            else:
                results = model_stats.head(n_results)
                position = "best"
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build response
            html_parts = []
            
            if n_results == 1:
                top_model = results.iloc[0]
                if is_bottom_query:
                    # For "worst" queries, we want the model with most failures (worst reliability)
                    html_parts.extend([
                        f"<p><strong>{top_model['model']} has the {position} reliability (most {label}).</strong></p>",
                        f"<p><strong>Details:</strong></p>",
                        "<ul style='margin-top:6px;'>",
                        f"<li><strong>Total {label}:</strong> {int(top_model['total_metric']):,}</li>",
                        f"<li><strong>Records:</strong> {int(top_model['record_count']):,}</li>",
                        f"<li><strong>Failure rate:</strong> {top_model['rate']:.1f}%</li>",
                        "</ul>"
                    ])
                else:
                    # For "best" queries, we want the model with least failures (best reliability)
                    html_parts.extend([
                        f"<p><strong>{top_model['model']} has the {position} reliability (least {label}).</strong></p>",
                        f"<p><strong>Details:</strong></p>",
                        "<ul style='margin-top:6px;'>",
                        f"<li><strong>Total {label}:</strong> {int(top_model['total_metric']):,}</li>",
                        f"<li><strong>Records:</strong> {int(top_model['record_count']):,}</li>",
                        f"<li><strong>Failure rate:</strong> {top_model['rate']:.1f}%</li>",
                        "</ul>"
                    ])
            else:
                if is_bottom_query:
                    html_parts.extend([
                        f"<p><strong>Top {n_results} Models with Most {label.title()} (Worst Reliability):</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                else:
                    html_parts.extend([
                        f"<p><strong>Top {n_results} Models with Least {label.title()} (Best Reliability):</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    reliability_desc = "worst" if is_bottom_query else "best"
                    html_parts.append(
                        f"<li><strong>#{i} {row['model']}:</strong> {int(row['total_metric']):,} {label} "
                        f"({row['rate']:.1f}% failure rate across {int(row['record_count']):,} records) - {reliability_desc} reliability</li>"
                    )
                
                html_parts.append("</ul>")
            
            # Add context about all models
            total_models = len(model_stats)
            total_metric = int(model_stats['total_metric'].sum())
            avg_rate = model_stats['rate'].mean()
            
            html_parts.extend([
                f"<p><strong>Context:</strong></p>",
                "<ul style='margin-top:6px;'>",
                f"<li><strong>Total models analyzed:</strong> {total_models}</li>",
                f"<li><strong>Total {label} across all models:</strong> {total_metric:,}</li>",
                f"<li><strong>Average failure rate:</strong> {avg_rate:.1f}%</li>",
                "</ul>"
            ])
            
            # Add performance comparison
            if n_results == 1:
                top_model = results.iloc[0]
                # model_stats is sorted by total_metric DESC (most failures first)
                # So iloc[0] = most failures (worst), iloc[-1] = least failures (best)
                model_with_most_failures = model_stats.iloc[0]  # Worst reliability
                model_with_least_failures = model_stats.iloc[-1]  # Best reliability
                
                if is_bottom_query:
                    # For "worst" queries, we're showing the model with most failures
                    if top_model['model'] == model_with_most_failures['model']:
                        html_parts.append(f"<p><em>{top_model['model']} has the worst reliability overall (most failures).</em></p>")
                    elif top_model['model'] == model_with_least_failures['model']:
                        html_parts.append(f"<p><em>{top_model['model']} actually has the best reliability overall (least failures).</em></p>")
                    else:
                        # Find position in overall ranking (1 = most failures/worst, last = least failures/best)
                        position = model_stats[model_stats['model'] == top_model['model']].index[0] + 1
                        html_parts.append(f"<p><em>{top_model['model']} ranks #{position} out of {total_models} models for reliability (lower is better).</em></p>")
                else:
                    # For "best" queries, we're showing the model with least failures
                    if top_model['model'] == model_with_least_failures['model']:
                        html_parts.append(f"<p><em>{top_model['model']} has the best reliability overall (least failures).</em></p>")
                    elif top_model['model'] == model_with_most_failures['model']:
                        html_parts.append(f"<p><em>{top_model['model']} actually has the worst reliability overall (most failures).</em></p>")
                    else:
                        # Find position in overall ranking (1 = most failures/worst, last = least failures/best)
                        position = model_stats[model_stats['model'] == top_model['model']].index[0] + 1
                        html_parts.append(f"<p><em>{top_model['model']} ranks #{position} out of {total_models} models for reliability (lower is better).</em></p>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Model ranking calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't determine model rankings: {_html.escape(str(e))}</p>"
    
class PartRankingHandler(QueryHandler):
    """
    Handle part ranking queries (e.g., "Which part fails most?", "Top failed parts").
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        
        # Skip if asking for individual records/vehicles (e.g., "top 5 vehicles", "top 10 Sentra vehicles")
        # This should be handled by Text-to-SQL, not ranking handler
        # Allow words between number and "vehicles" (e.g., "top 10 Sentra vehicles")
        if re.search(r'\btop\s+\d+.*?\b(vehicles?|records?|rows?|entries?|items?)\b', query_lower):
            return False
        
        # Check for ranking keywords
        ranking_keywords = [
            "which part", "which parts", "top", "bottom", "worst", "best",
            "most", "least", "highest", "lowest", "rank", "ranking",
            "failed part", "failing part", "problematic part"
        ]
        
        has_ranking = any(keyword in query_lower for keyword in ranking_keywords)
        
        # Check for part-related terms
        part_terms = ["part", "component", "battery", "engine", "transmission", "brake"]
        has_part_context = any(term in query_lower for term in part_terms)
        
        return has_ranking and has_part_context
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("PartRankingHandler processing")
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            metric_col = "failures_count"
        
        # Find the part column
        part_col = _safe_column(df, ["primary_failed_part", "failed_part", "part"])
        if not part_col:
            return "<p>Part information not available in the dataset.</p>"
        
        try:
            # Calculate part rankings
            part_stats = df.groupby(part_col).agg({
                metric_col: ["sum", "count"]
            }).reset_index()
            
            # Flatten column names
            part_stats.columns = ["part", "total_metric", "record_count"]
            part_stats["rate"] = (part_stats["total_metric"] / part_stats["record_count"] * 100.0)
            
            # Sort by total metric (descending for "most")
            part_stats = part_stats.sort_values("total_metric", ascending=False)
            
            if part_stats.empty:
                return "<p>No part data available for ranking.</p>"
            
            # Determine if asking for top or bottom
            query_lower = context.query_lower
            is_bottom_query = any(word in query_lower for word in ["worst", "least", "lowest", "bottom"])
            is_top_query = any(word in query_lower for word in ["best", "most", "highest", "top"])
            
            # Determine number of results
            n_results = _extract_number_from_query(context.query)
            if n_results is None:
                n_results = 1 if "which" in query_lower else 3
            
            # Get the requested results
            # For "most failures" queries, we want the parts with most failures (most problematic)
            # For "least failures" queries, we want the parts with least failures (most reliable)
            if "most" in query_lower or "highest" in query_lower:
                # Most failures = most problematic = head of sorted list (descending)
                results = part_stats.head(n_results)
                position = "most problematic"
                is_bottom_query = True  # Override for "most failures" queries
            elif "least" in query_lower or "lowest" in query_lower or "best" in query_lower:
                # Least failures = most reliable = tail of sorted list (ascending)
                results = part_stats.tail(n_results).iloc[::-1]  # Reverse to show best first
                position = "most reliable"
                is_bottom_query = False  # Override for "least failures" queries
            elif "top" in query_lower and ("failures" in query_lower or "fail" in query_lower):
                # "Top X by failures" = most failures = most problematic = head of sorted list
                results = part_stats.head(n_results)
                position = "most problematic"
                is_bottom_query = True  # Override for "top by failures" queries
            elif is_bottom_query:
                results = part_stats.tail(n_results).iloc[::-1]  # Reverse to show worst first
                position = "most problematic"
            else:
                results = part_stats.head(n_results)
                position = "most reliable"
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build response
            html_parts = []
            
            if n_results == 1:
                top_part = results.iloc[0]
                if is_bottom_query:
                    # For "worst" queries, we want the part with most failures (most problematic)
                    html_parts.extend([
                        f"<p><strong>{top_part['part']} is the most problematic part (most {label}).</strong></p>",
                        f"<p><strong>Details:</strong></p>",
                        "<ul style='margin-top:6px;'>",
                        f"<li><strong>Total {label}:</strong> {int(top_part['total_metric']):,}</li>",
                        f"<li><strong>Records:</strong> {int(top_part['record_count']):,}</li>",
                        f"<li><strong>Failure rate:</strong> {top_part['rate']:.1f}%</li>",
                        "</ul>"
                    ])
                else:
                    # For "best" queries, we want the part with least failures (most reliable)
                    html_parts.extend([
                        f"<p><strong>{top_part['part']} is the most reliable part (least {label}).</strong></p>",
                        f"<p><strong>Details:</strong></p>",
                        "<ul style='margin-top:6px;'>",
                        f"<li><strong>Total {label}:</strong> {int(top_part['total_metric']):,}</li>",
                        f"<li><strong>Records:</strong> {int(top_part['record_count']):,}</li>",
                        f"<li><strong>Failure rate:</strong> {top_part['rate']:.1f}%</li>",
                        "</ul>"
                    ])
            else:
                if is_bottom_query:
                    html_parts.extend([
                        f"<p><strong>Top {n_results} Most Problematic Parts (Most {label.title()}):</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                else:
                    html_parts.extend([
                        f"<p><strong>Top {n_results} Most Reliable Parts (Least {label.title()}):</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                
                for i, (_, row) in enumerate(results.iterrows(), 1):
                    reliability_desc = "most problematic" if is_bottom_query else "most reliable"
                    html_parts.append(
                        f"<li><strong>#{i} {row['part']}:</strong> {int(row['total_metric']):,} {label} "
                        f"({row['rate']:.1f}% failure rate across {int(row['record_count']):,} records) - {reliability_desc}</li>"
                    )
                
                html_parts.append("</ul>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Part ranking calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't determine part rankings: {_html.escape(str(e))}</p>"
    
