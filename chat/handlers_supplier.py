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
from chat.handlers import QueryHandler, QueryContext
from utils.logger import chat_logger as logger
from helper import km_to_miles


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
        
        # Plain English summary
        best_supplier = supplier_summary.iloc[0] if not supplier_summary.empty else None
        if best_supplier is not None:
            summary = f"We work with {len(supplier_summary)} suppliers {context_msg}, with {best_supplier['Supplier']} being the highest quality (score: {int(best_supplier['Quality'])}/100)."
        else:
            summary = f"Supplier information {context_msg} is available."
        
        html = f"<p><strong>{summary}</strong></p>"
        html += f"<p><strong>Complete supplier list {context_msg}:</strong></p><ul style='margin-top:6px;'>"
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
        
        # Plain English summary
        if not worst.empty:
            worst_supplier = worst.iloc[0]
            defect_pct = worst_supplier['defect_rate'] * 100 if pd.notna(worst_supplier['defect_rate']) else 0
            summary = f"The most problematic supplier is {worst_supplier['supplier_name']} with a {defect_pct:.1f}% defect rate in {worst_supplier['part_family']} parts."
        else:
            summary = "Supplier performance analysis shows varying quality levels across different part families."
        
        html = f"<p><strong>{summary}</strong></p>"
        html += "<p><strong>Detailed supplier performance analysis:</strong></p><ul style='margin-top:6px;'>"
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
                    f"<strong>Recommendation:</strong> Consider reviewing contract with "
                    f"{_html.escape(str(worst_supplier['supplier_name']))} for {worst_supplier['part_family']} parts "
                    f"(Defect Rate: {worst_supplier['defect_rate']*100:.1f}%).</p>")
        
        logger.info(f"Identified {len(worst)} problematic suppliers")
        return html


class VINQueryHandler(QueryHandler):
    """Handle VIN-related queries (location, affected vehicles, etc.)"""
    
    def can_handle(self, context: QueryContext) -> bool:
        query_lower = context.query.lower()
        # Check for "vin" as a whole word or VIN pattern
        return re.search(r'\bvin\b', query_lower) or re.search(r'1N4[A-Z0-9]{8,}', context.query)
    
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
                    miles_val = km_to_miles(row['dealer_distance_km'])
                    if miles_val is not None:
                        html += f"<li><strong>Distance to Dealer:</strong> {miles_val:.2f} mi</li>"
                html += "</ul>"
                
            elif any(word in query_lower for word in ["service center", "dealer", "nearest"]):
                # Service center query
                html = f"<p><strong>Nearest service center for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>Dealer:</strong> {row.get('dealer_name', 'N/A')}</li>"
                miles_val = km_to_miles(row.get('dealer_distance_km'))
                distance_text = f"{miles_val:.2f} mi" if miles_val is not None else "N/A"
                html += f"<li><strong>Distance:</strong> {distance_text}</li>"
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
        
        # Plain English summary
        top_reason = reason_counts.iloc[0] if not reason_counts.empty else None
        if top_reason is not None:
            top_pct = (top_reason['Count'] / total_failures * 100) if total_failures > 0 else 0
            summary = f"The most common issue is {top_reason['Failure Reason'].lower()}, affecting {top_pct:.1f}% of all failures."
        else:
            summary = f"Analysis of {total_failures} total failures shows various root causes."
        
        html = f"<p><strong>{summary}</strong></p>"
        html += f"<p><strong>Detailed breakdown {context_msg} ({total_failures} total failures):</strong></p>"
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
                "failures in", "near", "nearby", "dealers", "dealer",
                "service center", "which dealers", "dealer issues",
                "dealer problems", "dealer failures", "major issues",
                "issues from", "problems from", "failures from",
                "failures by", "by region", "by city", "by location",
                "by area", "regional", "regional analysis", "city analysis",
                "location analysis", "area analysis", "geographic",
                "which region", "which city", "which area", "which location"
            ]) or
            # Check if query mentions a specific dealer name
            self._mentions_dealer_name(context.df_history, query_lower)
        )
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"LocationQueryHandler processing: '{context.query[:50]}...'")
        
        # Handle region-based queries
        if any(phrase in query_lower for phrase in ["failures by", "by region", "by city", "by location", "by area", "regional", "city analysis", "location analysis", "area analysis", "geographic", "which region", "which city", "which area", "which location"]):
            return self._handle_region_analysis(df, query_lower)
        
        # Handle dealer-related queries
        if (any(phrase in query_lower for phrase in ["dealers", "dealer", "which dealers", "dealer issues", "dealer problems", "dealer failures"]) or
            self._mentions_dealer_name(df, query_lower)):
            return self._handle_dealer_analysis(df, query_lower)
        
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
    
    def _mentions_dealer_name(self, df: pd.DataFrame, query_lower: str) -> bool:
        """Check if the query mentions a specific dealer name."""
        if "dealer_name" not in df.columns:
            return False
        
        # Get unique dealer names
        dealer_names = df["dealer_name"].dropna().unique()
        
        # Check if any dealer name is mentioned in the query
        for dealer_name in dealer_names:
            if str(dealer_name).lower() in query_lower:
                return True
        
        return False
    
    def _handle_region_analysis(self, df: pd.DataFrame, query_lower: str) -> str:
        """Handle region-based failure analysis queries."""
        try:
            # Check if city column exists
            if "city" not in df.columns:
                return "<p>Location information is not available in the current dataset.</p>"
            
            # Calculate regional failure statistics
            region_stats = df.groupby("city").agg({
                "claims_count": "sum",
                "repairs_count": "sum",
                "recalls_count": "sum",
                "city": "count"
            }).rename(columns={"city": "total_vehicles"})
            
            # Calculate total failures per region
            region_stats["total_failures"] = (
                region_stats["claims_count"] + 
                region_stats["repairs_count"] + 
                region_stats["recalls_count"]
            )
            
            # Calculate failure rate
            region_stats["failure_rate"] = (
                region_stats["total_failures"] / region_stats["total_vehicles"] * 100
            )
            
            # Sort by total failures (most problematic regions first)
            region_stats = region_stats.sort_values("total_failures", ascending=False)
            
            if region_stats.empty:
                return "<p>No regional data available for analysis.</p>"
            
            # Build response based on query type
            if "most failures" in query_lower or "which region" in query_lower or "which city" in query_lower:
                # Show regions with most failures
                top_regions = region_stats.head(10)
                
                html_parts = [
                    "<p><strong>Regions with Most Failures:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ]
                
                for i, (region_name, stats) in enumerate(top_regions.iterrows(), 1):
                    html_parts.append(
                        f"<li><strong>#{i} {region_name}:</strong> "
                        f"{int(stats['total_failures'])} total failures "
                        f"({stats['failure_rate']:.1f}% rate across {int(stats['total_vehicles'])} vehicles)</li>"
                    )
                
                html_parts.append("</ul>")
                
                # Add summary statistics
                total_failures = int(region_stats['total_failures'].sum())
                total_vehicles = int(region_stats['total_vehicles'].sum())
                avg_failure_rate = region_stats['failure_rate'].mean()
                
                html_parts.extend([
                    f"<p><strong>Summary:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Total regions analyzed:</strong> {len(region_stats)}</li>",
                    f"<li><strong>Total failures across all regions:</strong> {total_failures:,}</li>",
                    f"<li><strong>Average failure rate:</strong> {avg_failure_rate:.1f}%</li>",
                    "</ul>"
                ])
                
                return "".join(html_parts)
            
            else:
                # General regional overview
                total_regions = len(region_stats)
                total_failures = int(region_stats['total_failures'].sum())
                total_vehicles = int(region_stats['total_vehicles'].sum())
                avg_failure_rate = region_stats['failure_rate'].mean()
                
                # Top 5 most problematic regions
                top_5 = region_stats.head(5)
                
                html_parts = [
                    "<p><strong>Regional Failure Analysis:</strong></p>",
                    f"<p>Analyzed <strong>{total_regions} regions</strong> with <strong>{total_vehicles:,} total vehicles</strong> and <strong>{total_failures:,} total failures</strong>.</p>",
                    f"<p>Average failure rate across all regions: <strong>{avg_failure_rate:.1f}%</strong></p>",
                    "<p><strong>Top 5 Most Problematic Regions:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ]
                
                for i, (region_name, stats) in enumerate(top_5.iterrows(), 1):
                    html_parts.append(
                        f"<li><strong>#{i} {region_name}:</strong> "
                        f"{int(stats['total_failures'])} failures "
                        f"({stats['failure_rate']:.1f}% rate across {int(stats['total_vehicles'])} vehicles)</li>"
                    )
                
                html_parts.append("</ul>")
                
                # Add breakdown by failure type
                total_claims = int(region_stats['claims_count'].sum())
                total_repairs = int(region_stats['repairs_count'].sum())
                total_recalls = int(region_stats['recalls_count'].sum())
                
                html_parts.extend([
                    "<p><strong>Failure Breakdown:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Claims:</strong> {total_claims:,} ({total_claims/total_failures*100:.1f}%)</li>",
                    f"<li><strong>Repairs:</strong> {total_repairs:,} ({total_repairs/total_failures*100:.1f}%)</li>",
                    f"<li><strong>Recalls:</strong> {total_recalls:,} ({total_recalls/total_failures*100:.1f}%)</li>",
                    "</ul>"
                ])
                
                return "".join(html_parts)
                
        except Exception as e:
            logger.error(f"Region analysis failed: {e}", exc_info=True)
            return f"<p>Could not analyze regional data: {_html.escape(str(e))}</p>"
    
    def _handle_dealer_analysis(self, df: pd.DataFrame, query_lower: str) -> str:
        """Handle dealer-related queries and analysis."""
        try:
            # Check if dealer columns exist
            if "dealer_name" not in df.columns:
                return "<p>Dealer information is not available in the current dataset.</p>"
            
            # Calculate dealer statistics
            dealer_stats = df.groupby("dealer_name").agg({
                "claims_count": "sum",
                "repairs_count": "sum", 
                "recalls_count": "sum",
                "dealer_name": "count"
            }).rename(columns={"dealer_name": "total_vehicles"})
            
            # Calculate total issues per dealer
            dealer_stats["total_issues"] = (
                dealer_stats["claims_count"] + 
                dealer_stats["repairs_count"] + 
                dealer_stats["recalls_count"]
            )
            
            # Calculate issue rate
            dealer_stats["issue_rate"] = (
                dealer_stats["total_issues"] / dealer_stats["total_vehicles"] * 100
            )
            
            # Sort by total issues (most problematic dealers first)
            dealer_stats = dealer_stats.sort_values("total_issues", ascending=False)
            
            if dealer_stats.empty:
                return "<p>No dealer data available for analysis.</p>"
            
            # Check if a specific dealer is mentioned
            mentioned_dealer = None
            if "dealer_name" in df.columns:
                dealer_names = df["dealer_name"].dropna().unique()
                for dealer_name in dealer_names:
                    if str(dealer_name).lower() in query_lower:
                        mentioned_dealer = str(dealer_name)
                        break
            
            # Handle specific dealer queries
            if mentioned_dealer:
                return self._handle_specific_dealer_analysis(df, mentioned_dealer, query_lower)
            
            # Build response based on query type
            if "most issues" in query_lower or "most problems" in query_lower:
                # Show dealers with most issues
                top_dealers = dealer_stats.head(5)
                
                html_parts = [
                    "<p><strong>Dealers with Most Issues:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ]
                
                for i, (dealer_name, stats) in enumerate(top_dealers.iterrows(), 1):
                    html_parts.append(
                        f"<li><strong>#{i} {dealer_name}:</strong> "
                        f"{int(stats['total_issues'])} total issues "
                        f"({stats['issue_rate']:.1f}% rate across {int(stats['total_vehicles'])} vehicles)</li>"
                    )
                
                html_parts.append("</ul>")
                
                # Add summary statistics
                total_issues = int(dealer_stats['total_issues'].sum())
                total_vehicles = int(dealer_stats['total_vehicles'].sum())
                avg_issue_rate = dealer_stats['issue_rate'].mean()
                
                html_parts.extend([
                    f"<p><strong>Summary:</strong></p>",
                    "<ul style='margin-top:6px;'>",
                    f"<li><strong>Total dealers analyzed:</strong> {len(dealer_stats)}</li>",
                    f"<li><strong>Total issues across all dealers:</strong> {total_issues:,}</li>",
                    f"<li><strong>Average issue rate:</strong> {avg_issue_rate:.1f}%</li>",
                    "</ul>"
                ])
                
                return "".join(html_parts)
            
            else:
                # General dealer overview
                total_dealers = len(dealer_stats)
                total_issues = int(dealer_stats['total_issues'].sum())
                total_vehicles = int(dealer_stats['total_vehicles'].sum())
                avg_issue_rate = dealer_stats['issue_rate'].mean()
                
                # Top 3 most problematic dealers
                top_3 = dealer_stats.head(3)
                
                html_parts = [
                    "<p><strong>Dealer Analysis Overview:</strong></p>",
                    f"<p>Analyzed <strong>{total_dealers} dealers</strong> with <strong>{total_vehicles:,} total vehicles</strong> and <strong>{total_issues:,} total issues</strong>.</p>",
                    f"<p>Average issue rate across all dealers: <strong>{avg_issue_rate:.1f}%</strong></p>",
                    "<p><strong>Top 3 Most Problematic Dealers:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ]
                
                for i, (dealer_name, stats) in enumerate(top_3.iterrows(), 1):
                    html_parts.append(
                        f"<li><strong>#{i} {dealer_name}:</strong> "
                        f"{int(stats['total_issues'])} issues "
                        f"({stats['issue_rate']:.1f}% rate)</li>"
                    )
                
                html_parts.append("</ul>")
                
                return "".join(html_parts)
                
        except Exception as e:
            logger.error(f"Dealer analysis failed: {e}", exc_info=True)
            return f"<p>Could not analyze dealer data: {_html.escape(str(e))}</p>"
    
    def _handle_specific_dealer_analysis(self, df: pd.DataFrame, dealer_name: str, query_lower: str) -> str:
        """Handle analysis for a specific dealer."""
        try:
            # Filter data for the specific dealer
            dealer_data = df[df["dealer_name"] == dealer_name]
            
            if dealer_data.empty:
                return f"<p>No data found for dealer '{dealer_name}'.</p>"
            
            # Calculate dealer statistics
            total_vehicles = len(dealer_data)
            total_claims = int(dealer_data["claims_count"].sum())
            total_repairs = int(dealer_data["repairs_count"].sum())
            total_recalls = int(dealer_data["recalls_count"].sum())
            total_issues = total_claims + total_repairs + total_recalls
            issue_rate = (total_issues / total_vehicles * 100) if total_vehicles > 0 else 0
            
            # Get top failing parts for this dealer
            part_stats = dealer_data.groupby("primary_failed_part").agg({
                "claims_count": "sum",
                "repairs_count": "sum",
                "recalls_count": "sum"
            })
            part_stats["total_issues"] = part_stats["claims_count"] + part_stats["repairs_count"] + part_stats["recalls_count"]
            part_stats = part_stats.sort_values("total_issues", ascending=False)
            
            # Get model breakdown for this dealer
            model_stats = dealer_data.groupby("model").agg({
                "claims_count": "sum",
                "repairs_count": "sum", 
                "recalls_count": "sum",
                "model": "count"
            }).rename(columns={"model": "vehicle_count"})
            model_stats["total_issues"] = model_stats["claims_count"] + model_stats["repairs_count"] + model_stats["recalls_count"]
            model_stats["issue_rate"] = (model_stats["total_issues"] / model_stats["vehicle_count"] * 100)
            model_stats = model_stats.sort_values("total_issues", ascending=False)
            
            # Build response
            html_parts = [
                f"<p><strong>{dealer_name} - Detailed Analysis</strong></p>",
                f"<p><strong>Overall Performance:</strong></p>",
                "<ul style='margin-top:6px;'>",
                f"<li><strong>Total Vehicles:</strong> {total_vehicles:,}</li>",
                f"<li><strong>Total Issues:</strong> {total_issues:,} (Issue Rate: {issue_rate:.1f}%)</li>",
                f"<li><strong>Claims:</strong> {total_claims:,}</li>",
                f"<li><strong>Repairs:</strong> {total_repairs:,}</li>",
                f"<li><strong>Recalls:</strong> {total_recalls:,}</li>",
                "</ul>"
            ]
            
            # Add top failing parts
            if not part_stats.empty:
                html_parts.extend([
                    "<p><strong>Top Failing Parts:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ])
                
                for i, (part_name, stats) in enumerate(part_stats.head(5).iterrows(), 1):
                    html_parts.append(
                        f"<li><strong>#{i} {part_name}:</strong> {int(stats['total_issues'])} total issues "
                        f"({int(stats['claims_count'])} claims, {int(stats['repairs_count'])} repairs, {int(stats['recalls_count'])} recalls)</li>"
                    )
                
                html_parts.append("</ul>")
            
            # Add model breakdown
            if not model_stats.empty:
                html_parts.extend([
                    "<p><strong>Issues by Model:</strong></p>",
                    "<ul style='margin-top:6px;'>"
                ])
                
                for model, stats in model_stats.iterrows():
                    html_parts.append(
                        f"<li><strong>{model}:</strong> {int(stats['total_issues'])} issues "
                        f"({stats['issue_rate']:.1f}% rate across {int(stats['vehicle_count'])} vehicles)</li>"
                    )
                
                html_parts.append("</ul>")
            
            # Add location info if available
            if "city" in dealer_data.columns:
                cities = dealer_data["city"].dropna().unique()
                if len(cities) > 0:
                    html_parts.extend([
                        "<p><strong>Service Areas:</strong></p>",
                        "<ul style='margin-top:6px;'>"
                    ])
                    for city in cities[:5]:  # Show top 5 cities
                        city_count = len(dealer_data[dealer_data["city"] == city])
                        html_parts.append(f"<li><strong>{city}:</strong> {city_count} vehicles</li>")
                    html_parts.append("</ul>")
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Specific dealer analysis failed for {dealer_name}: {e}", exc_info=True)
            return f"<p>Could not analyze {dealer_name} data: {_html.escape(str(e))}</p>"

