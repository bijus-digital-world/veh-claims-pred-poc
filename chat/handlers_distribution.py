"""
Distribution analysis handlers for mileage and age bucket queries.

Handles queries like:
- "At what mileage do X parts fail most?"
- "At what age do X parts fail most frequently?"
- "Show me failure distribution by mileage"
"""

import html as _html
import pandas as pd
from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger


class MileageDistributionHandler(QueryHandler):
    """Handle 'at what mileage do X fail' queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return any(phrase in query_lower for phrase in [
            "what mileage", "which mileage", "mileage range",
            "mileage do", "mileage does", "at what mileage",
            "mileage bucket", "by mileage", "per mileage"
        ]) and any(word in query_lower for word in [
            "fail", "failure", "failures", "problem", "issue"
        ])
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"MileageDistributionHandler processing: '{context.query[:50]}...'")
        
        if "mileage_bucket" not in df.columns:
            return "<p>Mileage bucket information is not available in the dataset.</p>"
        
        # Detect part family filter
        part_families = ["Battery", "Brakes", "Transmission", "Engine", "Electrical", 
                        "Lighting", "HVAC", "Safety", "Steering", "Tires"]
        part_family = None
        for family in part_families:
            if family.lower() in query_lower:
                part_family = family
                break
        
        # Also check for specific part names
        specific_part = None
        if not part_family and "part_family" in df.columns:
            for part in df["primary_failed_part"].unique():
                if str(part).lower() in query_lower:
                    specific_part = str(part)
                    break
        
        # Filter dataset
        if part_family and "part_family" in df.columns:
            filtered = df[df["part_family"] == part_family]
            context_msg = f"for {part_family} parts"
        elif specific_part:
            filtered = df[df["primary_failed_part"] == specific_part]
            context_msg = f"for {specific_part}"
        else:
            filtered = df
            context_msg = "overall"
        
        if filtered.empty:
            return f"<p>No data found {context_msg}.</p>"
        
        # Calculate failures by mileage bucket
        filtered["failures"] = (
            filtered["claims_count"].fillna(0) +
            filtered["repairs_count"].fillna(0) +
            filtered["recalls_count"].fillna(0)
        )
        
        mileage_stats = filtered.groupby("mileage_bucket").agg({
            "failures": "sum",
            "vin": "count"
        }).reset_index()
        mileage_stats.columns = ["Mileage Bucket", "Total Failures", "Total Incidents"]
        mileage_stats["Failure Rate %"] = (
            (mileage_stats["Total Failures"] / mileage_stats["Total Incidents"] * 100)
            .round(1)
        )
        
        # Sort by standard mileage bucket order
        bucket_order = {"0-10k": 1, "10-30k": 2, "30-60k": 3, "60k+": 4}
        mileage_stats["sort_order"] = mileage_stats["Mileage Bucket"].map(bucket_order).fillna(99)
        mileage_stats = mileage_stats.sort_values("sort_order")
        
        # Find bucket with most failures
        max_failures = mileage_stats["Total Failures"].max()
        max_bucket = mileage_stats[mileage_stats["Total Failures"] == max_failures]["Mileage Bucket"].iloc[0]
        
        # Find bucket with highest rate
        max_rate = mileage_stats["Failure Rate %"].max()
        max_rate_bucket = mileage_stats[mileage_stats["Failure Rate %"] == max_rate]["Mileage Bucket"].iloc[0]
        
        html = f"<p><strong>Failure distribution by mileage {context_msg}:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for _, row in mileage_stats.iterrows():
            is_max = row["Mileage Bucket"] == max_bucket
            style = "font-weight:700; color:#ef4444;" if is_max else ""
            html += (f"<li style='{style}'><strong>{row['Mileage Bucket']}</strong> — "
                    f"{int(row['Total Failures'])} failures "
                    f"({row['Failure Rate %']:.1f}% rate, "
                    f"{int(row['Total Incidents'])} incidents)</li>")
        html += "</ul>"
        
        html += (f"<p style='margin-top:8px;'><strong>{context_msg.capitalize()}, failures occur most frequently in the "
                f"<span style='color:#ef4444; font-weight:700;'>{max_bucket}</span> mileage range "
                f"with <strong>{int(max_failures)} total failures</strong>.</strong>")
        
        if max_rate_bucket != max_bucket:
            html += (f" However, the <strong>highest failure rate</strong> is in "
                    f"<span style='color:#f59e0b'>{max_rate_bucket}</span> "
                    f"({max_rate:.1f}%).")
        html += "</p>"
        
        logger.info(f"Mileage analysis {context_msg}: Most failures in {max_bucket} ({int(max_failures)} failures)")
        return html


class AgeDistributionHandler(QueryHandler):
    """Handle 'at what age do X fail' queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return any(phrase in query_lower for phrase in [
            "what age", "which age", "age range",
            "age do", "age does", "at what age",
            "age bucket", "by age", "per age", "vehicle age"
        ]) and any(word in query_lower for word in [
            "fail", "failure", "failures", "problem", "issue", "occur"
        ])
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"AgeDistributionHandler processing: '{context.query[:50]}...'")
        
        if "age_bucket" not in df.columns:
            return "<p>Age bucket information is not available in the dataset.</p>"
        
        # Detect part family filter
        part_families = ["Battery", "Brakes", "Transmission", "Engine", "Electrical", 
                        "Lighting", "HVAC", "Safety", "Steering", "Tires"]
        part_family = None
        for family in part_families:
            if family.lower() in query_lower:
                part_family = family
                break
        
        # Also check for specific part names
        specific_part = None
        if not part_family and "part_family" in df.columns:
            for part in df["primary_failed_part"].unique():
                if str(part).lower() in query_lower:
                    specific_part = str(part)
                    break
        
        # Filter dataset
        if part_family and "part_family" in df.columns:
            filtered = df[df["part_family"] == part_family]
            context_msg = f"for {part_family} parts"
        elif specific_part:
            filtered = df[df["primary_failed_part"] == specific_part]
            context_msg = f"for {specific_part}"
        else:
            filtered = df
            context_msg = "overall"
        
        if filtered.empty:
            return f"<p>No data found {context_msg}.</p>"
        
        # Calculate failures by age bucket
        filtered["failures"] = (
            filtered["claims_count"].fillna(0) +
            filtered["repairs_count"].fillna(0) +
            filtered["recalls_count"].fillna(0)
        )
        
        age_stats = filtered.groupby("age_bucket").agg({
            "failures": "sum",
            "vin": "count"
        }).reset_index()
        age_stats.columns = ["Age Bucket", "Total Failures", "Total Incidents"]
        age_stats["Failure Rate %"] = (
            (age_stats["Total Failures"] / age_stats["Total Incidents"] * 100)
            .round(1)
        )
        
        # Sort by standard age bucket order
        bucket_order = {"<1yr": 1, "1-3yr": 2, "3-5yr": 3, "5+yr": 4}
        age_stats["sort_order"] = age_stats["Age Bucket"].map(bucket_order).fillna(99)
        age_stats = age_stats.sort_values("sort_order")
        
        # Find bucket with most failures
        max_failures = age_stats["Total Failures"].max()
        max_bucket = age_stats[age_stats["Total Failures"] == max_failures]["Age Bucket"].iloc[0]
        
        # Find bucket with highest rate
        max_rate = age_stats["Failure Rate %"].max()
        max_rate_bucket = age_stats[age_stats["Failure Rate %"] == max_rate]["Age Bucket"].iloc[0]
        
        html = f"<p><strong>Failure distribution by vehicle age {context_msg}:</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for _, row in age_stats.iterrows():
            is_max = row["Age Bucket"] == max_bucket
            style = "font-weight:700; color:#ef4444;" if is_max else ""
            html += (f"<li style='{style}'><strong>{row['Age Bucket']}</strong> — "
                    f"{int(row['Total Failures'])} failures "
                    f"({row['Failure Rate %']:.1f}% rate, "
                    f"{int(row['Total Incidents'])} incidents)</li>")
        html += "</ul>"
        
        html += (f"<p style='margin-top:8px;'><strong>{context_msg.capitalize()}, failures occur most frequently in the "
                f"<span style='color:#ef4444; font-weight:700;'>{max_bucket}</span> age range "
                f"with <strong>{int(max_failures)} total failures</strong>.</strong>")
        
        if max_rate_bucket != max_bucket:
            html += (f" However, the <strong>highest failure rate</strong> is in "
                    f"<span style='color:#f59e0b'>{max_rate_bucket}</span> "
                    f"({max_rate:.1f}%).")
        html += "</p>"
        
        logger.info(f"Age analysis {context_msg}: Most failures in {max_bucket} ({int(max_failures)} failures)")
        return html

