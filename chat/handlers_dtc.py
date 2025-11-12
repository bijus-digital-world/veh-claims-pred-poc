"""
chat/handlers_dtc.py

DTC (Diagnostic Trouble Code) specific query handlers.
Handles queries related to DTC codes, error codes, and diagnostic data.
"""

import re
import html as _html
import pandas as pd
import numpy as np
from typing import Optional

from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger


class DTCCommonCodesHandler(QueryHandler):
    """Handle queries about most common DTC codes"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        
        # Don't handle if a specific model is mentioned (let ModelSpecificDTCHandler handle it)
        models = ["sentra", "leaf", "ariya", "altima", "rogue", "pathfinder", "frontier", "titan"]
        has_model = any(model in query_lower for model in models)
        if has_model:
            return False
        
        has_dtc = any(kw in query_lower for kw in ["dtc", "diagnostic trouble", "trouble code", "error code", "fault code", "obd code"])
        has_common = any(kw in query_lower for kw in ["most", "common", "frequent", "top"])
        
        return has_dtc and has_common and "dtc_code" in context.df_history.columns
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        
        if "dtc_code" not in df.columns:
            return "<p>DTC code data is not available in the dataset.</p>"
        
        dtc_data = df[df["dtc_code"].notna()].copy()
        
        if dtc_data.empty:
            return "<p>No DTC codes found in the dataset.</p>"
        
        dtc_counts = dtc_data["dtc_code"].value_counts().head(10)
        total_dtc = len(dtc_data)
        
        html = f"<p><strong>Most Common DTC Codes:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for idx, (dtc_code, count) in enumerate(dtc_counts.items(), 1):
            pct = (count / total_dtc * 100) if total_dtc > 0 else 0
            html += f"<li><strong>{_html.escape(str(dtc_code))}</strong>: {count:,} occurrences ({pct:.1f}%)</li>"
        
        html += "</ul>"
        html += f"<p style='margin-top:8px;color:#94a3b8;'>Total DTC records: {total_dtc:,}</p>"
        
        return html


class DTCByModelHandler(QueryHandler):
    """Handle queries about DTC codes by model"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        dtc_keywords = ["dtc by model", "dtc codes by model", "error codes by model",
                       "trouble codes by model", "dtc per model", "which model has dtc"]
        
        has_dtc = any(kw in query_lower for kw in ["dtc", "diagnostic trouble", "trouble code", "error code", "fault code"])
        has_model = "model" in query_lower and ("by" in query_lower or "per" in query_lower)
        
        return has_dtc and has_model and "dtc_code" in context.df_history.columns and "model" in context.df_history.columns
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        
        if "dtc_code" not in df.columns or "model" not in df.columns:
            return "<p>DTC code or model data is not available in the dataset.</p>"
        
        dtc_data = df[df["dtc_code"].notna()].copy()
        
        if dtc_data.empty:
            return "<p>No DTC codes found in the dataset.</p>"
        
        model_dtc = dtc_data.groupby("model").agg(
            dtc_count=("dtc_code", "count"),
            unique_dtc=("dtc_code", "nunique")
        ).reset_index()
        model_dtc = model_dtc.sort_values("dtc_count", ascending=False)
        
        total_dtc = len(dtc_data)
        
        html = f"<p><strong>DTC Codes by Model:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for _, row in model_dtc.iterrows():
            model_name = row["model"]
            count = int(row["dtc_count"])
            unique = int(row["unique_dtc"])
            pct = (count / total_dtc * 100) if total_dtc > 0 else 0
            html += f"<li><strong>{_html.escape(str(model_name))}</strong>: {count:,} DTC occurrences ({pct:.1f}%), {unique} unique DTC codes</li>"
        
        html += "</ul>"
        html += f"<p style='margin-top:8px;color:#94a3b8;'>Total DTC records: {total_dtc:,}</p>"
        
        return html


class DTCByManufacturingYearHandler(QueryHandler):
    """Handle queries about DTC codes by manufacturing year"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        has_dtc = any(kw in query_lower for kw in ["dtc", "diagnostic trouble", "trouble code", "error code", "fault code"])
        has_year = any(kw in query_lower for kw in ["manufacturing year", "manufacture year", "production year", "by year"])
        
        return has_dtc and has_year and "dtc_code" in context.df_history.columns and "manufacturing_date" in context.df_history.columns
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        
        if "dtc_code" not in df.columns or "manufacturing_date" not in df.columns:
            return "<p>DTC code or manufacturing date data is not available in the dataset.</p>"
        
        dtc_data = df[df["dtc_code"].notna()].copy()
        
        if dtc_data.empty:
            return "<p>No DTC codes found in the dataset.</p>"
        
        dtc_data["manufacturing_date"] = pd.to_datetime(dtc_data["manufacturing_date"], errors="coerce")
        dtc_data = dtc_data[dtc_data["manufacturing_date"].notna()]
        
        if dtc_data.empty:
            return "<p>No valid manufacturing date data available for DTC analysis.</p>"
        
        dtc_data["manufacturing_year"] = dtc_data["manufacturing_date"].dt.year
        year_dtc = dtc_data.groupby("manufacturing_year").agg(
            dtc_count=("dtc_code", "count"),
            unique_dtc=("dtc_code", "nunique")
        ).reset_index()
        year_dtc = year_dtc.sort_values("manufacturing_year")
        
        total_dtc = len(dtc_data)
        
        html = f"<p><strong>DTC Codes by Manufacturing Year:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for _, row in year_dtc.iterrows():
            year = int(row["manufacturing_year"])
            count = int(row["dtc_count"])
            unique = int(row["unique_dtc"])
            pct = (count / total_dtc * 100) if total_dtc > 0 else 0
            html += f"<li><strong>{year}</strong>: {count:,} DTC occurrences ({pct:.1f}%), {unique} unique DTC codes</li>"
        
        html += "</ul>"
        html += f"<p style='margin-top:8px;color:#94a3b8;'>Total DTC records: {total_dtc:,}</p>"
        
        return html


class ModelSpecificDTCHandler(QueryHandler):
    """Handle queries about DTC codes for a specific model"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        models = ["sentra", "leaf", "ariya"]
        mentioned_model = any(model in query_lower for model in models)
        has_dtc = any(kw in query_lower for kw in ["dtc", "diagnostic trouble", "trouble code", "error code", "fault code"])
        
        return mentioned_model and has_dtc and "dtc_code" in context.df_history.columns and "model" in context.df_history.columns
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query_lower
        
        if "dtc_code" not in df.columns or "model" not in df.columns:
            return "<p>DTC code or model data is not available in the dataset.</p>"
        
        models = ["sentra", "leaf", "ariya"]
        mentioned_model = None
        for model in models:
            if model in query_lower:
                mentioned_model = model.title()
                break
        
        if not mentioned_model:
            return "<p>Could not identify the model in your query.</p>"
        
        model_data = df[(df["model"].str.lower() == mentioned_model.lower()) & (df["dtc_code"].notna())].copy()
        
        if model_data.empty:
            return f"<p>No DTC codes found for {mentioned_model}.</p>"
        
        dtc_counts = model_data["dtc_code"].value_counts().head(10)
        total_dtc = len(model_data)
        
        html = f"<p><strong>DTC Codes for {mentioned_model}:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for dtc_code, count in dtc_counts.items():
            pct = (count / total_dtc * 100) if total_dtc > 0 else 0
            html += f"<li><strong>{_html.escape(str(dtc_code))}</strong>: {count:,} occurrences ({pct:.1f}%)</li>"
        
        html += "</ul>"
        html += f"<p style='margin-top:8px;color:#94a3b8;'>Total DTC records for {mentioned_model}: {total_dtc:,}</p>"
        
        return html


class DTCTrendHandler(QueryHandler):
    """Handle queries about DTC trends over time"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query_lower
        has_dtc = any(kw in query_lower for kw in ["dtc", "diagnostic trouble", "trouble code", "error code", "fault code"])
        has_trend = any(kw in query_lower for kw in ["trend", "over time", "over the", "increasing", "decreasing", "change"])
        
        return has_dtc and has_trend and "dtc_code" in context.df_history.columns and "date" in context.df_history.columns
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        
        if "dtc_code" not in df.columns or "date" not in df.columns:
            return "<p>DTC code or date data is not available in the dataset.</p>"
        
        dtc_data = df[df["dtc_code"].notna()].copy()
        
        if dtc_data.empty:
            return "<p>No DTC codes found in the dataset.</p>"
        
        dtc_data["date"] = pd.to_datetime(dtc_data["date"], errors="coerce")
        dtc_data = dtc_data[dtc_data["date"].notna()]
        
        if dtc_data.empty:
            return "<p>No valid date data available for DTC trend analysis.</p>"
        
        daily_dtc = dtc_data.groupby(dtc_data["date"].dt.date).agg(
            dtc_count=("dtc_code", "count")
        ).reset_index()
        daily_dtc = daily_dtc.sort_values("date")
        
        if len(daily_dtc) < 2:
            return "<p>Insufficient data points for trend analysis.</p>"
        
        dates = daily_dtc["date"].values
        counts = daily_dtc["dtc_count"].values
        
        slope = np.polyfit(range(len(counts)), counts, 1)[0]
        total_dtc = counts.sum()
        avg_daily = counts.mean()
        peak_day = daily_dtc.loc[daily_dtc["dtc_count"].idxmax(), "date"]
        peak_count = int(daily_dtc["dtc_count"].max())
        
        trend_direction = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
        trend_strength = "strong" if abs(slope) > 1.0 else "moderate" if abs(slope) > 0.5 else "weak"
        
        html = f"<p><strong>DTC Trend Analysis:</strong></p>"
        html += f"<p><strong>Trend Direction:</strong> {trend_direction.title()} ({trend_strength} trend, slope={slope:.2f} per day)</p>"
        html += f"<p><strong>Summary Statistics:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        html += f"<li>Total DTC occurrences: {total_dtc:,}</li>"
        html += f"<li>Average per day: {avg_daily:.1f}</li>"
        html += f"<li>Peak day: {peak_day} ({peak_count:,} DTCs)</li>"
        html += f"<li>Data period: {dates[0]} to {dates[-1]}</li>"
        html += "</ul>"
        
        return html

