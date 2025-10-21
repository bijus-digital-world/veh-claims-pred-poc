"""
chat/handlers_analysis.py

Handlers for analysis queries (monthly, rates, trends, top parts).
"""

import re
import html as _html
import pandas as pd
import numpy as np

from chat.handlers import QueryHandler, QueryContext
from chat_helper import (
    _detect_metric_from_text,
    _metric_or_fallback_column,
    ensure_failures_column,
    _safe_column,
    _has_word,
    summarize_top_failed_parts,
    summarize_incident_details,
    _compute_model_trends
)
from utils.logger import chat_logger as logger


class MonthlyAggregateHandler(QueryHandler):
    """Handle monthly aggregate queries (per month, monthly)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        return bool(re.search(r"\b(per month|monthly|claims per month|repairs per month|recalls per month)\b", 
                             context.query_lower))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("MonthlyAggregateHandler processing")
        
        # Choose metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        # Determine metric column
        if metric_col is None:
            if "claim" in context.query_lower:
                metric_col = "claims_count"
            elif "repair" in context.query_lower:
                metric_col = "repairs_count"
            elif "recall" in context.query_lower:
                metric_col = "recalls_count"
            else:
                metric_col = "failures_count"
        
        # Ensure synthetic column if needed
        if metric_col == "failures_count" and "failures_count" not in df.columns:
            df = ensure_failures_column(df, out_col="failures_count")
        
        if metric_col not in df.columns:
            return "<p>Requested metric not available to compute monthly aggregates. Try 'claims', 'repairs', or 'failures'.</p>"
        
        try:
            df = df.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            
            if df_month.empty:
                return "<p>Not enough date information to compute monthly aggregates.</p>"
            
            monthly_vals = df_month.astype(int)
            first_month = monthly_vals.index[0].strftime("%b %Y")
            last_month = monthly_vals.index[-1].strftime("%b %Y")
            total_over_period = int(monthly_vals.sum())
            avg_per_month = int(round(monthly_vals.mean()))
            min_val = int(monthly_vals.min())
            max_val = int(monthly_vals.max())
            min_month = monthly_vals.idxmin().strftime("%b %Y")
            max_month = monthly_vals.idxmax().strftime("%b %Y")
            
            # Trend calculation
            y = monthly_vals.values.astype(float)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # By-model breakdown if requested
            if "by model" in context.query_lower or "per model" in context.query_lower:
                grp = df.dropna(subset=["date_parsed"]).set_index("date_parsed").groupby(
                    [pd.Grouper(freq="M"), "model"]
                )[metric_col].sum()
                
                if not grp.empty:
                    pivot = grp.unstack(fill_value=0)
                    avg_by_model = pivot.mean(axis=0).sort_values(ascending=False).head(6)
                    lines = [f"{_html.escape(str(idx))} → {val:.1f} /month" for idx, val in avg_by_model.items()]
                    
                    return (f"<p>Overall {label} between {first_month} and {last_month}: "
                           f"total = {total_over_period}, average ≈ {avg_per_month} per month. "
                           f"Monthly {label} ranged between {min_val} – {max_val}. "
                           f"The overall trend is {trend}.</p>"
                           f"<p><strong>Average per month by model (top):</strong><br>{'<br>'.join(lines)}</p>")
            
            return (f"<p>Between {first_month} and {last_month}, monthly {label} totals ranged "
                   f"between {min_val} – {max_val} per month. "
                   f"{max_month} showed the highest activity (≈{max_val}) while {min_month} was the lowest (≈{min_val}). "
                   f"Total over the period = {total_over_period}, average ≈ {avg_per_month} per month. "
                   f"The overall trend is {trend} (slope={slope:.2f} {label}/month).</p>")
        
        except Exception as e:
            logger.error(f"Monthly aggregate calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't compute monthly aggregates: {_html.escape(str(e))}</p>"


class RateHandler(QueryHandler):
    """Handle rate queries (per 100, rate calculations)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        return "rate" in context.query_lower or "per 100" in context.query_lower
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("RateHandler processing")
        
        # Determine grouping column
        if "by model" in context.query_lower or (_has_word(context.query_lower, "model") and 
                                                  not (_has_word(context.query_lower, "age") or 
                                                      _has_word(context.query_lower, "mileage"))):
            group_col = _safe_column(context.df_history, ["model"])
        elif "age" in context.query_lower or "age bucket" in context.query_lower:
            group_col = _safe_column(context.df_history, ["age_bucket"])
        elif "mileage" in context.query_lower or "mileage bucket" in context.query_lower:
            group_col = _safe_column(context.df_history, ["mileage_bucket"])
        elif "dealer" in context.query_lower or "service center" in context.query_lower:
            group_col = _safe_column(context.df_history, ["dealer", "service_center"])
        else:
            group_col = _safe_column(context.df_history, ["model"])
        
        if not group_col:
            return "<p>Could not determine grouping column for rate. Try: 'failure rate by model' or 'failure rate by age bucket'.</p>"
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            return "<p>Requested metric not available to compute rates.</p>"
        
        try:
            overall_total = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            overall_rows = len(df)
            overall_per100 = (overall_total / overall_rows) * 100.0 if overall_rows > 0 else 0.0
            
            label = "failure rate" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            grp = df.groupby(group_col).agg(total_metric=(metric_col, "sum"), rows=(metric_col, "count"))
            grp = grp[grp["rows"] > 0]
            grp["rate_per_100"] = (grp["total_metric"] / grp["rows"]) * 100.0
            top_n = grp.sort_values("rate_per_100", ascending=False).head(6)
            
            lines = [f"{_html.escape(str(idx))} → {row['rate_per_100']:.1f}%" for idx, row in top_n.iterrows()]
            
            return (f"<p>Overall {label}: total = {int(overall_total)} across {overall_rows} records "
                   f"(~{overall_per100:.1f} per 100 records)."
                   f"<br><strong>Breakdown by {group_col}:</strong><br>" + "<br>".join(lines) + "</p>")
        
        except Exception as e:
            logger.error(f"Rate calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't compute rates for the requested category (error: {_html.escape(str(e))}).</p>"


class TrendHandler(QueryHandler):
    """Handle trend queries (increasing, decreasing, rising trends)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        return any(tok in context.query_lower for tok in 
                   ["trend", "increasing", "rising", "declining", "decreasing"])
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("TrendHandler processing")
        
        # Detect metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            df = df_with_metric
        else:
            df = context.df_history
        
        if metric_col is None:
            metric_col = "failures_count"
        
        # Model-specific trend?
        mentioned_model = None
        if "model" in df.columns:
            for m in df["model"].dropna().unique():
                if str(m).lower() in context.query_lower:
                    mentioned_model = str(m)
                    break
        
        if mentioned_model:
            return self._handle_model_trend(df, mentioned_model, metric_col)
        
        # Which models rising?
        if ("which" in context.query_lower and "model" in context.query_lower) or ("which models" in context.query_lower):
            return self._handle_which_models_rising(df, metric_col)
        
        # Overall trend
        return self._handle_overall_trend(df, metric_col)
    
    def _handle_model_trend(self, df, mentioned_model, metric_col):
        """Handle trend for specific model"""
        try:
            df = df.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_model = df[df["model"].str.lower() == mentioned_model.lower()]
            
            if df_model.empty:
                return f"<p>No data found for model {mentioned_model}.</p>"
            
            df_month = df_model.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            
            if len(df_month) < 3:
                return f"<p>Not enough months of data for {mentioned_model} to determine a reliable trend.</p>"
            
            y = df_month.values.astype(float)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            
            first_month = df_month.index[0].strftime("%b %Y")
            last_month = df_month.index[-1].strftime("%b %Y")
            max_month = df_month.idxmax().strftime("%b %Y")
            min_month = df_month.idxmin().strftime("%b %Y")
            max_val = int(df_month.max())
            min_val = int(df_month.min())
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")
            
            return (f"<p><strong>{_html.escape(mentioned_model)}</strong> shows a {trend} trend in {label} "
                   f"from {first_month} to {last_month} (slope={slope:.2f} per month).</p>"
                   f"<p>Monthly {label} ranged from {min_val} in {min_month} to {max_val} in {max_month}.</p>"
                   f"<p>This {direction} trend may indicate changes in failure incidence or detection "
                   f"for {_html.escape(mentioned_model)}.</p>")
        
        except Exception as e:
            logger.error(f"Model trend calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't compute trend for {mentioned_model}: {_html.escape(str(e))}</p>"
    
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
                   + "<br><em>Note:</em> slopes are fitted over each model's monthly series; "
                   "increase measured in metric units/month.</p>")
        
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


class TopFailedPartsHandler(QueryHandler):
    """Handle top failed parts queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        pattern = r"top failed parts|top parts|top failure|top failures|most frequent parts"
        return bool(re.search(pattern, context.query_lower))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("TopFailedPartsHandler processing")
        requested_metric = _detect_metric_from_text(context.query)
        return summarize_top_failed_parts(context.df_history, metric=requested_metric, top_n=6)


class IncidentDetailsHandler(QueryHandler):
    """Handle incident details / time to resolution queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        pattern = r"incident details|failure details|time to resolution|resolution time|avg resolution"
        return bool(re.search(pattern, context.query_lower))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("IncidentDetailsHandler processing")
        requested_metric = _detect_metric_from_text(context.query)
        return summarize_incident_details(context.df_history, metric=requested_metric)

