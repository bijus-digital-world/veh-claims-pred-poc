"""
Handlers for model comparison queries.
"""

import re
import html as _html
import pandas as pd
from typing import List, Tuple

from chat.handlers import QueryHandler, QueryContext
from chat_helper import _detect_metric_from_text, _metric_or_fallback_column, _safe_column
from utils.logger import chat_logger as logger


class ModelComparisonHandler(QueryHandler):
    """
    Handle direct model comparison queries (e.g., "Compare Leaf vs Ariya failure rate").
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        
        # Check for comparison keywords
        comparison_keywords = ["compare", "versus", "vs", "vs.", "against", "difference between"]
        has_comparison = any(keyword in query_lower for keyword in comparison_keywords)
        
        # Check for model mentions (at least 2 models)
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        mentioned_models = [model for model in models if model in query_lower]
        
        return has_comparison and len(mentioned_models) >= 2
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("ModelComparisonHandler processing")
        
        # Extract models from query
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        mentioned_models = [model for model in models if model in context.query_lower]
        
        if len(mentioned_models) < 2:
            return "<p>Please specify at least two models to compare (e.g., 'Compare Leaf vs Ariya failure rate').</p>"
        
        # Take first two mentioned models
        model1, model2 = mentioned_models[:2]
        
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
            # Get data for both models
            model1_data = df[df[model_col].str.lower() == model1.lower()]
            model2_data = df[df[model_col].str.lower() == model2.lower()]
            
            if model1_data.empty:
                return f"<p>No data found for {model1.title()} in the dataset.</p>"
            if model2_data.empty:
                return f"<p>No data found for {model2.title()} in the dataset.</p>"
            
            # Calculate metrics for both models
            model1_metrics = self._calculate_model_metrics(model1_data, metric_col)
            model2_metrics = self._calculate_model_metrics(model2_data, metric_col)
            
            # Calculate overall metrics for context
            overall_total = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            overall_rows = len(df)
            overall_rate = (overall_total / overall_rows * 100.0) if overall_rows > 0 else 0.0
            
            label = "failure rate" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build comparison response
            html_parts = [
                f"<p><strong>{model1.title()} vs {model2.title()} {label.title()} Comparison</strong></p>",
                "<table style='border-collapse: collapse; width: 100%; margin: 10px 0;'>",
                "<tr style='background-color: #f8f9fa;'>",
                f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{model1.title()}</th>",
                f"<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{model2.title()}</th>",
                "<th style='border: 1px solid #ddd; padding: 8px; text-align: left;'>Difference</th>",
                "</tr>",
                "<tr>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'><strong>{model1_metrics['rate']:.1f}%</strong></td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'><strong>{model2_metrics['rate']:.1f}%</strong></td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'><strong>{model1_metrics['rate'] - model2_metrics['rate']:+.1f}%</strong></td>",
                "</tr>",
                "<tr>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model1_metrics['total']:,} {label.replace(' rate', 's')}</td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model2_metrics['total']:,} {label.replace(' rate', 's')}</td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model1_metrics['total'] - model2_metrics['total']:+,}</td>",
                "</tr>",
                "<tr>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model1_metrics['records']:,} records</td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model2_metrics['records']:,} records</td>",
                f"<td style='border: 1px solid #ddd; padding: 8px;'>{model1_metrics['records'] - model2_metrics['records']:+,}</td>",
                "</tr>",
                "</table>"
            ]
            
            # Add detailed analysis
            better_model = model1 if model1_metrics['rate'] < model2_metrics['rate'] else model2
            worse_model = model2 if model1_metrics['rate'] < model2_metrics['rate'] else model1
            better_rate = min(model1_metrics['rate'], model2_metrics['rate'])
            worse_rate = max(model1_metrics['rate'], model2_metrics['rate'])
            difference = worse_rate - better_rate
            
            html_parts.extend([
                f"<p><strong>Analysis:</strong></p>",
                f"<ul style='margin-top:6px;'>",
                f"<li><strong>{better_model.title()}</strong> has the better {label} at <strong>{better_rate:.1f}%</strong></li>",
                f"<li><strong>{worse_model.title()}</strong> has a {label} of <strong>{worse_rate:.1f}%</strong></li>",
                f"<li><strong>Performance gap:</strong> {difference:.1f} percentage points</li>",
                f"<li><strong>Relative performance:</strong> {better_model.title()} is {((worse_rate - better_rate) / worse_rate * 100):.1f}% better than {worse_model.title()}</li>",
                "</ul>"
            ])
            
            # Add context with overall rate
            html_parts.extend([
                f"<p><strong>Context:</strong></p>",
                f"<ul style='margin-top:6px;'>",
                f"<li>Overall {label} across all models: <strong>{overall_rate:.1f}%</strong></li>",
                f"<li>{model1.title()}: {self._get_performance_vs_overall(model1_metrics['rate'], overall_rate)}</li>",
                f"<li>{model2.title()}: {self._get_performance_vs_overall(model2_metrics['rate'], overall_rate)}</li>",
                "</ul>"
            ])
            
            # Add ranking information if we have more models
            all_models = df[model_col].str.lower().unique()
            if len(all_models) > 2:
                ranking = self._get_model_ranking(df, model_col, metric_col)
                if ranking:
                    html_parts.extend([
                        f"<p><strong>Overall Ranking:</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                    for i, (model_name, rate) in enumerate(ranking[:5], 1):
                        is_model1 = model_name.lower() == model1.lower()
                        is_model2 = model_name.lower() == model2.lower()
                        if is_model1 or is_model2:
                            html_parts.append(f"<li><strong>#{i} {model_name.title()}: {rate:.1f}%</strong> {'← Your comparison' if is_model1 or is_model2 else ''}</li>")
                        else:
                            html_parts.append(f"<li>#{i} {model_name.title()}: {rate:.1f}%</li>")
                    html_parts.append("</ul>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}", exc_info=True)
            return f"<p>Couldn't compare {model1.title()} vs {model2.title()}: {_html.escape(str(e))}</p>"
    
    def _calculate_model_metrics(self, model_data: pd.DataFrame, metric_col: str) -> dict:
        """Calculate key metrics for a model."""
        total_metric = int(pd.to_numeric(model_data[metric_col], errors="coerce").fillna(0).sum())
        total_records = len(model_data)
        rate = (total_metric / total_records * 100.0) if total_records > 0 else 0.0
        
        return {
            'total': total_metric,
            'records': total_records,
            'rate': rate
        }
    
    def _get_performance_vs_overall(self, model_rate: float, overall_rate: float) -> str:
        """Get performance description vs overall rate."""
        if model_rate < overall_rate:
            diff = overall_rate - model_rate
            return f"{diff:.1f} percentage points below average"
        elif model_rate > overall_rate:
            diff = model_rate - overall_rate
            return f"{diff:.1f} percentage points above average"
        else:
            return "exactly at average"
    
    def _get_model_ranking(self, df: pd.DataFrame, model_col: str, metric_col: str) -> List[Tuple[str, float]]:
        """Get model ranking by rate."""
        try:
            grp = df.groupby(model_col).agg(
                total_metric=(metric_col, "sum"),
                rows=(metric_col, "count")
            )
            grp = grp[grp["rows"] > 0]
            grp["rate"] = (grp["total_metric"] / grp["rows"]) * 100.0
            ranking = grp.sort_values("rate", ascending=True)[["rate"]].reset_index()
            return [(row[model_col], row["rate"]) for _, row in ranking.iterrows()]
        except Exception:
            return []
