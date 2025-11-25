"""
chat/handlers_metrics.py

Handlers for metric-related queries (totals, counts, averages, time to resolution).
"""

import re
import html as _html
import pandas as pd

from chat.handlers import QueryHandler, QueryContext
from chat_helper import (
    _detect_metric_from_text,
    _metric_or_fallback_column,
    _detect_count_or_average_intent,
    _detect_incident_or_failure_request,
    ensure_failures_column,
    summarize_overall_metric,
    compute_count_and_average_html,
    format_total_incidents,
    format_total_failures,
    _safe_column,
    _has_word
)
from utils.logger import chat_logger as logger


class TotalMetricHandler(QueryHandler):
    """Handle total/overall metric requests"""
    
    def can_handle(self, context: QueryContext) -> bool:
        # Skip if monthly query
        if re.search(r"\b(per month|monthly)\b", context.query_lower):
            return False
        
        total_trigger = re.search(
            r"\b(total(?: number)?(?: of)?|total count|how many(?: total)?|give me the total|what is the total|total number of)\b",
            context.query_lower
        )
        wants_failures_word = any(k in context.query_lower for k in 
                                  ["failur", "claim", "repair", "recall", "total"])
        
        return bool(total_trigger and wants_failures_word)
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("TotalMetricHandler processing")
        
        # Detect and resolve metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            context.df_history = df_with_metric
        
        if metric_col is None:
            available = ", ".join(sorted(context.df_history.columns))
            return (f"<p>You asked for totals but the requested metric is not available. "
                    f"Available columns: <strong>{available}</strong>.</p>")
        
        if metric_col == "failures_count" and "failures_count" not in context.df_history.columns:
            context.df_history = ensure_failures_column(context.df_history, out_col="failures_count")
        
        # Check if user explicitly mentions incidents
        if (re.search(r"\bincident(s)?\b", context.query_lower) or 
            re.search(r"\bincident count\b", context.query_lower) or 
            re.search(r"\brows with\b", context.query_lower)):
            return compute_count_and_average_html(context.df_history, requested_metric, context.query, sample_df=None)
        
        # Default: long-format total (sum) with top models
        return summarize_overall_metric(context.df_history, metric_col, top_n=6)


class CountAndAverageHandler(QueryHandler):
    """Handle count and average queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        wants_count, wants_avg = _detect_count_or_average_intent(context.query)
        return wants_count or wants_avg
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("CountAndAverageHandler processing")
        
        # Detect and resolve metric
        requested_metric = _detect_metric_from_text(context.query)
        metric_col, df_with_metric = _metric_or_fallback_column(context.df_history, requested_metric)
        
        if df_with_metric is not None:
            context.df_history = df_with_metric
        
        if metric_col == "failures_count" and "failures_count" not in context.df_history.columns:
            context.df_history = ensure_failures_column(context.df_history, out_col="failures_count")
        
        # Strong detection: incidents vs failures
        explicit_intent = _detect_incident_or_failure_request(context.query)
        
        if explicit_intent == "incidents":
            return format_total_incidents(context.df_history, metric_col, top_n=6)
        
        # Default: failures summary
        return format_total_failures(context.df_history, metric_col, top_n=6)


class TimeToResolutionHandler(QueryHandler):
    """Handle time to resolution / average resolution time queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        pattern1 = r"\b(avg|average|mean|median|typical|what is the)\b.*\b(time to (claim )?resolution|resolution time|time to (?:close|resolve)|days to resolve|time to claim)\b"
        pattern2 = r"\b(time to (claim )?resolution|resolution time|time to resolve|days to resolve|time to claim)\b"
        return bool(re.search(pattern1, context.query_lower) or re.search(pattern2, context.query_lower))
    
    def handle(self, context: QueryContext) -> str:
        logger.debug("TimeToResolutionHandler processing")
        
        time_col = _safe_column(context.df_history, [
            "time_to_resolution", "resolution_days", "days_to_resolve", "time_to_resolve",
            "resolution_time", "time_to_close", "time_to_claim_resolution", "time_to_claim"
        ])
        
        if not time_col:
            return ("<p>I can't find a time-to-resolution column in your dataset. "
                    "Look for columns named like <em>time_to_resolution</em>, "
                    "<em>resolution_days</em>, or <em>days_to_resolve</em>.</p>")
        
        df_tmp = context.df_history.copy()
        df_tmp[time_col] = pd.to_numeric(df_tmp.get(time_col), errors="coerce")
        df_valid = df_tmp.dropna(subset=[time_col])
        
        if df_valid.empty:
            return f"<p>I found column <strong>{_html.escape(time_col)}</strong> but it contains no numeric values I can use.</p>"
        
        cnt = int(len(df_valid))
        mean_v = float(df_valid[time_col].mean())
        median_v = float(df_valid[time_col].median())
        std_v = float(df_valid[time_col].std(ddof=0)) if cnt > 1 else 0.0
        q25 = float(df_valid[time_col].quantile(0.25))
        q75 = float(df_valid[time_col].quantile(0.75))
        min_v = float(df_valid[time_col].min())
        max_v = float(df_valid[time_col].max())
        
        # Determine unit
        tcol_lower = time_col.lower()
        if "day" in tcol_lower:
            unit = "days"
        elif "hour" in tcol_lower:
            unit = "hours"
        else:
            unit = "units (as recorded)"
        
        reply_lines = [
            f"<p>Average time to resolution (based on <strong>{_html.escape(time_col)}</strong>, {cnt} records):</p>",
            f"<ul style='margin-top:6px;'>",
            f"<li><strong>Mean:</strong> {mean_v:.1f} {unit}</li>",
            f"<li><strong>Median:</strong> {median_v:.1f} {unit} (IQR: {q25:.1f}–{q75:.1f})</li>",
            f"<li><strong>Range:</strong> {min_v:.1f} – {max_v:.1f} {unit}</li>",
            f"<li><strong>Std. dev:</strong> {std_v:.1f} {unit}</li>",
            f"</ul>"
        ]
        
        # Optional group breakdowns
        if "by model" in context.query_lower or "per model" in context.query_lower or _has_word(context.query_lower, "model"):
            if "model" in df_tmp.columns:
                grp = df_valid.groupby("model")[time_col].agg(["count", "mean"]).sort_values("mean", ascending=False).head(6)
                if not grp.empty:
                    grp_lines = [f"{_html.escape(str(idx))} → mean {row['mean']:.1f} {unit} (n={int(row['count'])})" 
                                for idx, row in grp.iterrows()]
                    reply_lines.append("<p><strong>Average by model (top):</strong><br>" + "<br>".join(grp_lines) + "</p>")
        
        if "by part" in context.query_lower or _has_word(context.query_lower, "part"):
            part_col = _safe_column(df_tmp, ["primary_failed_part", "failed_part", "part"])
            if part_col:
                grp = df_valid.groupby(part_col)[time_col].agg(["count", "mean"]).sort_values("mean", ascending=False).head(6)
                if not grp.empty:
                    grp_lines = [f"{_html.escape(str(idx))} → mean {row['mean']:.1f} {unit} (n={int(row['count'])})" 
                                for idx, row in grp.iterrows()]
                    reply_lines.append("<p><strong>Average by part (top):</strong><br>" + "<br>".join(grp_lines) + "</p>")
        
        reply_lines.append("<p style='color:#94a3b8; margin-top:6px;'>If you want a different grouping "
                          "(e.g. 'by model and part') ask: 'average time to resolution by model and part'.</p>")
        
        return "".join(reply_lines)

