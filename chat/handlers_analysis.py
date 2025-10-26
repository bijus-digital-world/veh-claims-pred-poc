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
        # Only handle specific monthly aggregate queries, not trend queries
        monthly_keywords = r"\b(per month|monthly|claims per month|repairs per month|recalls per month)\b"
        trend_keywords = r"\b(trend|trends|increasing|decreasing|rising|declining)\b"
        
        has_monthly = bool(re.search(monthly_keywords, context.query_lower))
        has_trend = bool(re.search(trend_keywords, context.query_lower))
        
        # Only handle if it has monthly keywords but NOT trend keywords
        return has_monthly and not has_trend
    
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
                   ["trend", "trends", "increasing", "rising", "declining", "decreasing"])
    
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
        
        # Which models rising? (detect various phrasings)
        rising_phrases = [
            "which models", "any model", "any vehicle model", "growing trend", 
            "increasing trend", "rising trend", "models with", "vehicle model"
        ]
        if any(phrase in context.query_lower for phrase in rising_phrases):
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
            # Get all model trends (both rising and falling)
            model_trends = _compute_model_trends(df, metric_col, min_months=6, slope_threshold=-999.0, top_n=20)
            
            if not model_trends:
                return ("<p>No models have sufficient monthly data to determine trends "
                       "(need at least 6 months of data per model).</p>")
            
            # Separate rising vs falling trends
            rising_models = [(m, s, mon, last) for (m, s, mon, last) in model_trends if s > 0]
            falling_models = [(m, s, mon, last) for (m, s, mon, last) in model_trends if s <= 0]
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            response_parts = []
            
            if rising_models:
                # Plain English summary
                model_names = [m for (m, s, mon, last) in rising_models[:3]]
                if len(rising_models) == 1:
                    summary = f"Yes, the {model_names[0]} model is showing concerning growth in {label}."
                elif len(rising_models) == 2:
                    summary = f"Yes, both the {model_names[0]} and {model_names[1]} models are experiencing growing {label} trends."
                else:
                    summary = f"Yes, {len(rising_models)} vehicle models including {', '.join(model_names)} are showing increasing {label} patterns."
                
                response_parts.append(f"<p><strong>{summary}</strong></p>")
                response_parts.append(f"<p><strong>Detailed Analysis:</strong></p>")
                lines = [f"<li><strong>{_html.escape(str(m))}</strong> → increasing by {s:.2f} {label}/month "
                        f"(based on {mon} months of data, latest: {int(last)})</li>" 
                        for (m, s, mon, last) in rising_models[:6]]
                response_parts.append("<ul>" + "".join(lines) + "</ul>")
            else:
                response_parts.append(f"<p><strong>No, I do not observe any growing trends in {label} for vehicle models.</strong></p>")
            
            if falling_models:
                response_parts.append(f"<p><strong>Models showing decreasing trends:</strong></p>")
                lines = [f"<li><strong>{_html.escape(str(m))}</strong> → decreasing by {abs(s):.2f} {label}/month "
                        f"(based on {mon} months of data)</li>" 
                        for (m, s, mon, last) in falling_models[:3]]
                response_parts.append("<ul>" + "".join(lines) + "</ul>")
            
            response_parts.append("<p><em>Note:</em> Trends are calculated using linear regression on monthly data. "
                                 "A positive slope indicates increasing failures over time.</p>")
            
            return "".join(response_parts)
        
        except Exception as e:
            logger.error(f"Model trends calculation failed: {e}", exc_info=True)
            return f"<p>Couldn't determine per-model trends: {_html.escape(str(e))}</p>"
    
    def _handle_overall_trend(self, df, metric_col):
        """Handle overall trend calculation with detailed analysis"""
        try:
            df = df.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            
            if len(df_month) < 3:
                return "<p>Not enough months to determine a reliable overall trend (need at least 3 months of data).</p>"
            
            # Calculate trend statistics
            y = df_month.values
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            
            # Calculate additional statistics
            total_failures = int(y.sum())
            avg_monthly = float(y.mean())
            min_month = int(y.min())
            max_month = int(y.max())
            recent_3_months = int(y[-3:].sum()) if len(y) >= 3 else int(y.sum())
            previous_3_months = int(y[-6:-3].sum()) if len(y) >= 6 else int(y[-3:].sum()) if len(y) >= 3 else 0
            
            # Calculate trend strength
            if abs(slope) < 0.5:
                trend_strength = "weak"
            elif abs(slope) < 2.0:
                trend_strength = "moderate"
            else:
                trend_strength = "strong"
            
            # Calculate percentage change
            if previous_3_months > 0:
                pct_change = ((recent_3_months - previous_3_months) / previous_3_months) * 100
            else:
                pct_change = 0
            
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            
            # Build comprehensive response
            html_parts = [
                f"<p><strong>Overall {label.title()} Trend Analysis</strong></p>",
                f"<p><strong>Trend Direction:</strong> {trend.title()} ({trend_strength} trend, slope={slope:.1f} per month)</p>",
                f"<p><strong>Summary Statistics:</strong></p>",
                "<ul style='margin-top:6px;'>",
                f"<li><strong>Total {label}:</strong> {total_failures:,}</li>",
                f"<li><strong>Average per month:</strong> {avg_monthly:.1f}</li>",
                f"<li><strong>Peak month:</strong> {max_month:,} {label}</li>",
                f"<li><strong>Lowest month:</strong> {min_month:,} {label}</li>",
                f"<li><strong>Data period:</strong> {len(y)} months</li>",
                "</ul>"
            ]
            
            # Add recent vs previous comparison if we have enough data
            if len(y) >= 6:
                html_parts.extend([
                    f"<p><strong>Recent Performance:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Last 3 months:</strong> {recent_3_months:,} {label}</li>",
                    f"<li><strong>Previous 3 months:</strong> {previous_3_months:,} {label}</li>",
                    f"<li><strong>Change:</strong> {pct_change:+.1f}%</li>",
                    "</ul>"
                ])
            
            # Add trend interpretation
            if trend == "increasing":
                if trend_strength == "strong":
                    html_parts.append("<p style='color:#fca5a5;'><strong>Concern:</strong> Strong increasing trend indicates potential quality issues that need immediate attention.</p>")
                elif trend_strength == "moderate":
                    html_parts.append("<p style='color:#fbbf24;'><strong>Watch:</strong> Moderate increasing trend suggests monitoring is needed.</p>")
                else:
                    html_parts.append("<p style='color:#94a3b8;'><strong>Note:</strong> Slight increasing trend, continue monitoring.</p>")
            elif trend == "decreasing":
                if trend_strength == "strong":
                    html_parts.append("<p style='color:#10b981;'><strong>Good:</strong> Strong decreasing trend indicates quality improvements are working.</p>")
                elif trend_strength == "moderate":
                    html_parts.append("<p style='color:#10b981;'><strong>Positive:</strong> Moderate decreasing trend shows progress.</p>")
                else:
                    html_parts.append("<p style='color:#94a3b8;'><strong>Note:</strong> Slight decreasing trend, continue current practices.</p>")
            else:
                html_parts.append("<p style='color:#94a3b8;'><strong>Stable:</strong> Consistent performance with no significant trend.</p>")
            
            # Add monthly breakdown if not too many months
            if len(y) <= 12:
                html_parts.extend([
                    f"<p><strong>Monthly Breakdown:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ])
                for i, (date, value) in enumerate(df_month.items()):
                    month_name = date.strftime("%B %Y")
                    html_parts.append(f"<li><strong>{month_name}:</strong> {int(value):,} {label}</li>")
                html_parts.append("</ul>")
            
            return "".join(html_parts)
        
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

