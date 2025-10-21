"""
Supplier, VIN, and Location-related query handlers for enhanced dataset.

Handles queries about:
- Supplier lists and quality analysis
- VIN tracking and lookup
- Vehicle location queries
- Service center proximity
"""

import re
import html as _html
import pandas as pd
from typing import Optional
from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger


class SupplierListHandler(QueryHandler):
    """Handle 'list suppliers for X part' queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return any(phrase in query_lower for phrase in [
            "list suppliers", "which suppliers", "suppliers for",
            "who supplies", "supplier list", "show suppliers",
            "suppliers provide", "suppliers supply", "suppliers who",
            "list of suppliers", "supplier who supply"
        ]) or ("supplier" in query_lower and "supply" in query_lower)
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"SupplierListHandler processing: '{context.query[:50]}...'")
        
        # Check if supplier columns exist
        if "supplier_name" not in df.columns:
            return "<p>Supplier information is not available in the current dataset. Please regenerate with the enhanced dataset.</p>"
        
        # Detect requested part family
        part_families = ["Battery", "Brakes", "Transmission", "Engine", "Electrical", 
                        "Lighting", "HVAC", "Safety", "Steering", "Tires"]
        part_family = None
        for family in part_families:
            if family.lower() in query_lower:
                part_family = family
                break
        
        # Filter by part family if specified
        if part_family and "part_family" in df.columns:
            filtered = df[df["part_family"] == part_family]
            context_msg = f"for {part_family} parts"
        else:
            filtered = df
            context_msg = "across all parts"
        
        if filtered.empty:
            return f"<p>No suppliers found {context_msg}.</p>"
        
        # Group by supplier
        supplier_summary = filtered.groupby("supplier_name").agg({
            "supplier_id": "first",
            "supplier_quality_score": "first",
            "defect_rate": "first",
            "claims_count": "sum",
            "primary_failed_part": lambda x: x.nunique()
        }).reset_index()
        
        supplier_summary.columns = ["Supplier", "ID", "Quality", "Defect Rate", "Total Claims", "Part Types"]
        supplier_summary = supplier_summary.sort_values("Quality", ascending=False)
        
        html = f"<p><strong>Suppliers {context_msg}:</strong></p><ul style='margin-top:6px;'>"
        for _, row in supplier_summary.head(10).iterrows():
            defect_pct = row['Defect Rate'] * 100 if pd.notna(row['Defect Rate']) else 0
            quality_color = "#16a34a" if row['Quality'] >= 90 else ("#f59e0b" if row['Quality'] >= 80 else "#ef4444")
            html += (f"<li><strong>{_html.escape(str(row['Supplier']))}</strong> "
                    f"<span style='color:#94a3b8;'>(ID: {row['ID']})</span> — "
                    f"Quality: <span style='color:{quality_color}'>{int(row['Quality'])}/100</span>, "
                    f"Defect Rate: {defect_pct:.1f}%, "
                    f"Claims: {int(row['Total Claims'])}, "
                    f"Part Types: {int(row['Part Types'])}</li>")
        html += "</ul>"
        
        logger.info(f"Found {len(supplier_summary)} suppliers {context_msg}")
        return html


class DefectiveSupplierHandler(QueryHandler):
    """Handle 'which supplier is defective' or 'worst supplier' queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return any(phrase in query_lower for phrase in [
            "defective supplier", "worst supplier", "highest defect",
            "supplier causing", "supplier quality", "supplier with most",
            "poor quality supplier", "bad supplier", "problematic supplier",
            "supplier performance", "supplier issues", "defective part",
            "supplier suppling", "supplier is suppling", "which supplier"
        ]) or ("supplier" in query_lower and any(w in query_lower for w in ["defective", "defect", "worst", "bad", "problem"]))
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        
        logger.info(f"DefectiveSupplierHandler processing: '{context.query[:50]}...'")
        
        # Check if supplier columns exist
        if "supplier_name" not in df.columns:
            return "<p>Supplier information is not available in the current dataset.</p>"
        
        # Calculate supplier performance
        supplier_stats = df.groupby(["supplier_name", "part_family"]).agg({
            "supplier_quality_score": "first",
            "defect_rate": "first",
            "claims_count": "sum",
            "repairs_count": "sum",
            "recalls_count": "sum"
        }).reset_index()
        
        supplier_stats["total_failures"] = (
            supplier_stats["claims_count"] + 
            supplier_stats["repairs_count"] + 
            supplier_stats["recalls_count"]
        )
        
        # Sort by worst performers (highest defect rate and most failures)
        worst = supplier_stats.sort_values(["defect_rate", "total_failures"], ascending=[False, False]).head(8)
        
        html = "<p><strong>Suppliers with highest defect rates:</strong></p><ul style='margin-top:6px;'>"
        for _, row in worst.iterrows():
            defect_pct = row['defect_rate'] * 100 if pd.notna(row['defect_rate']) else 0
            html += (f"<li><span style='color:#ef4444; font-weight:600'>{_html.escape(str(row['supplier_name']))}</span> "
                    f"<span style='color:#94a3b8;'>({row['part_family']})</span> — "
                    f"Defect Rate: <strong>{defect_pct:.1f}%</strong>, "
                    f"Quality Score: {int(row['supplier_quality_score'])}/100, "
                    f"Total Failures: {int(row['total_failures'])}</li>")
        html += "</ul>"
        
        # Add recommendation
        if not worst.empty:
            worst_supplier = worst.iloc[0]
            html += (f"<p style='color:#fca5a5; margin-top:8px;'>"
                    f"<strong>⚠️ Recommendation:</strong> Consider reviewing contract with "
                    f"{_html.escape(str(worst_supplier['supplier_name']))} for {worst_supplier['part_family']} parts "
                    f"(Defect Rate: {worst_supplier['defect_rate']*100:.1f}%).</p>")
        
        logger.info(f"Identified {len(worst)} problematic suppliers")
        return html


class VINQueryHandler(QueryHandler):
    """Handle VIN-related queries (location, affected vehicles, etc.)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return "vin" in query_lower or re.search(r'1N4[A-Z0-9]{8,}', context.query)
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"VINQueryHandler processing: '{context.query[:50]}...'")
        
        # Check if VIN column exists
        if "vin" not in df.columns:
            return "<p>VIN tracking is not available in the current dataset. Please regenerate with the enhanced dataset.</p>"
        
        # Extract specific VIN if mentioned (17-character format)
        vin_match = re.search(r'1N4[A-Z0-9]{8,17}', context.query)
        
        if vin_match:
            # Specific VIN query
            vin = vin_match.group(0)
            vin_data = df[df["vin"].str.contains(vin, case=False, na=False)]
            
            if vin_data.empty:
                return f"<p>VIN <span style='font-family:monospace'>{vin}</span> not found in the dataset.</p>"
            
            row = vin_data.iloc[0]
            
            # Determine what type of VIN query
            if any(word in query_lower for word in ["location", "where", "coordinates", "city"]):
                # Location query
                html = f"<p><strong>Location for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>City:</strong> {row.get('city', 'N/A')}</li>"
                if pd.notna(row.get('current_lat')) and pd.notna(row.get('current_lon')):
                    html += f"<li><strong>Coordinates:</strong> {row['current_lat']:.6f}, {row['current_lon']:.6f}</li>"
                if pd.notna(row.get('dealer_name')):
                    html += f"<li><strong>Nearest Dealer:</strong> {row['dealer_name']}</li>"
                if pd.notna(row.get('dealer_distance_km')):
                    html += f"<li><strong>Distance to Dealer:</strong> {row['dealer_distance_km']:.2f} km</li>"
                html += "</ul>"
                
            elif any(word in query_lower for word in ["service center", "dealer", "nearest"]):
                # Service center query
                html = f"<p><strong>Nearest service center for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>Dealer:</strong> {row.get('dealer_name', 'N/A')}</li>"
                html += f"<li><strong>Distance:</strong> {row.get('dealer_distance_km', 0):.2f} km</li>"
                if pd.notna(row.get('dealer_lat')) and pd.notna(row.get('dealer_lon')):
                    html += f"<li><strong>Dealer Location:</strong> {row['dealer_lat']:.6f}, {row['dealer_lon']:.6f}</li>"
                html += f"<li><strong>Vehicle City:</strong> {row.get('city', 'N/A')}</li>"
                html += "</ul>"
                
            else:
                # General VIN info
                failures = int(row.get('claims_count', 0) + row.get('repairs_count', 0) + row.get('recalls_count', 0))
                html = f"<p><strong>Information for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>Model:</strong> {row.get('model', 'N/A')}</li>"
                html += f"<li><strong>Location:</strong> {row.get('city', 'N/A')}</li>"
                html += f"<li><strong>Total Failures:</strong> {failures}</li>"
                html += f"<li><strong>Failed Part:</strong> {row.get('primary_failed_part', 'N/A')}</li>"
                if pd.notna(row.get('supplier_name')):
                    html += f"<li><strong>Supplier:</strong> {row['supplier_name']}</li>"
                if pd.notna(row.get('failure_description')) and row.get('failure_description') != "No failure detected":
                    html += f"<li><strong>Failure Reason:</strong> {row['failure_description']}</li>"
                html += "</ul>"
            
            logger.info(f"Retrieved info for VIN {vin}")
            return html
            
        else:
            # List VINs matching criteria
            if any(word in query_lower for word in ["affected", "failure", "failures", "problem", "issue"]):
                # VINs with failures
                affected = df[(df["claims_count"] > 0) | (df["repairs_count"] > 0) | (df["recalls_count"] > 0)]
                
                # Filter by part family if mentioned
                for family in ["Battery", "Brakes", "Transmission", "Engine"]:
                    if family.lower() in query_lower and "part_family" in df.columns:
                        affected = affected[affected["part_family"] == family]
                        break
                
                if affected.empty:
                    return "<p>No VINs found matching the failure criteria.</p>"
                
                vin_list = affected["vin"].head(15).tolist()
                total_affected = len(affected)
                
                html = f"<p><strong>VINs affected by failures ({total_affected} total):</strong></p>"
                html += "<p style='font-family:monospace; font-size:13px; color:#cfe9ff;'>"
                for i, vin in enumerate(vin_list[:10]):
                    html += f"{vin}<br>"
                if len(vin_list) > 10:
                    html += f"<span style='color:#94a3b8;'>... and {total_affected - 10} more</span>"
                html += "</p>"
                
                logger.info(f"Found {total_affected} affected VINs")
                return html
            
            return "<p>Please specify a VIN (e.g., '1N4AZMA5100456') or search criteria (e.g., 'VINs with Battery failures').</p>"


class FailureReasonHandler(QueryHandler):
    """Handle 'why did X fail' or 'failure reasons' queries"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        return any(phrase in query_lower for phrase in [
            "why", "reason", "reasons behind", "root cause", "cause of failure",
            "failure description", "what caused", "why did", "failure reason"
        ])
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"FailureReasonHandler processing: '{context.query[:50]}...'")
        
        # Check if failure_description column exists
        if "failure_description" not in df.columns:
            return "<p>Failure descriptions are not available in the current dataset.</p>"
        
        # Filter to actual failures only
        failures_df = df[(df["claims_count"] > 0) | (df["repairs_count"] > 0) | (df["recalls_count"] > 0)]
        failures_df = failures_df[failures_df["failure_description"] != "No failure detected"]
        
        if failures_df.empty:
            return "<p>No failure descriptions found in the dataset.</p>"
        
        # Detect part family filter
        part_families = ["Battery", "Brakes", "Transmission", "Engine", "Electrical"]
        part_family = None
        for family in part_families:
            if family.lower() in query_lower:
                part_family = family
                break
        
        if part_family and "part_family" in failures_df.columns:
            failures_df = failures_df[failures_df["part_family"] == part_family]
            context_msg = f"for {part_family} failures"
        else:
            context_msg = "across all failures"
        
        if failures_df.empty:
            return f"<p>No failure descriptions found {context_msg}.</p>"
        
        # Group by failure description
        reason_counts = failures_df.groupby("failure_description").agg({
            "vin": "count",
            "supplier_name": lambda x: x.mode()[0] if len(x.mode()) > 0 else "Various"
        }).reset_index()
        reason_counts.columns = ["Failure Reason", "Count", "Primary Supplier"]
        reason_counts = reason_counts.sort_values("Count", ascending=False)
        
        total_failures = len(failures_df)
        html = f"<p><strong>Failure reasons {context_msg} ({total_failures} total failures):</strong></p>"
        html += "<ul style='margin-top:6px;'>"
        
        for _, row in reason_counts.head(8).iterrows():
            pct = (row['Count'] / total_failures * 100) if total_failures > 0 else 0
            html += (f"<li><strong>{_html.escape(str(row['Failure Reason']))}</strong> "
                    f"— {int(row['Count'])} incidents ({pct:.1f}%) "
                    f"<span style='color:#94a3b8;'>| Supplier: {_html.escape(str(row['Primary Supplier']))}</span></li>")
        html += "</ul>"
        
        logger.info(f"Found {len(reason_counts)} unique failure reasons {context_msg}")
        return html


class LocationQueryHandler(QueryHandler):
    """Handle location-based queries (vehicles in city, nearby failures, etc.)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        cities = ["california", "texas", "new york", "chicago", "miami", "seattle", 
                 "boston", "atlanta", "denver", "toronto", "vancouver", "montreal"]
        
        return (
            any(city in query_lower for city in cities) or
            any(phrase in query_lower for phrase in [
                "in city", "location", "where are", "vehicles in",
                "failures in", "near", "nearby"
            ])
        )
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"LocationQueryHandler processing: '{context.query[:50]}...'")
        
        # Check if city column exists
        if "city" not in df.columns:
            return "<p>Location information is not available in the current dataset.</p>"
        
        # Detect city mentioned in query
        mentioned_city = None
        for city_name in df["city"].dropna().unique():
            if str(city_name).lower() in query_lower:
                mentioned_city = str(city_name)
                break
        
        if mentioned_city:
            # City-specific query
            city_data = df[df["city"] == mentioned_city]
            
            if city_data.empty:
                return f"<p>No data found for {mentioned_city}.</p>"
            
            total_vehicles = len(city_data)
            total_failures = int((city_data["claims_count"] + city_data["repairs_count"] + city_data["recalls_count"]).sum())
            failure_rate = (total_failures / total_vehicles * 100) if total_vehicles > 0 else 0
            
            # Top parts failing in this city
            part_stats = city_data[city_data["claims_count"] > 0].groupby("primary_failed_part").size().sort_values(ascending=False).head(5)
            
            html = f"<p><strong>Analysis for {mentioned_city}:</strong></p>"
            html += f"<ul style='margin-top:6px;'>"
            html += f"<li><strong>Total Vehicles:</strong> {total_vehicles}</li>"
            html += f"<li><strong>Total Failures:</strong> {total_failures} ({failure_rate:.1f}%)</li>"
            html += "</ul>"
            
            if not part_stats.empty:
                html += "<p><strong>Top failed parts in this city:</strong></p><ul>"
                for part, count in part_stats.items():
                    html += f"<li>{_html.escape(str(part))} — {int(count)} failures</li>"
                html += "</ul>"
            
            logger.info(f"Retrieved data for city: {mentioned_city}")
            return html
        else:
            # General location summary
            city_summary = df.groupby("city").agg({
                "vin": "count",
                "claims_count": "sum"
            }).reset_index()
            city_summary.columns = ["City", "Vehicles", "Claims"]
            city_summary = city_summary.sort_values("Claims", ascending=False)
            
            html = "<p><strong>Vehicles by city:</strong></p><ul style='margin-top:6px;'>"
            for _, row in city_summary.head(10).iterrows():
                html += f"<li>{_html.escape(str(row['City']))} — {int(row['Vehicles'])} vehicles, {int(row['Claims'])} claims</li>"
            html += "</ul>"
            
            return html

