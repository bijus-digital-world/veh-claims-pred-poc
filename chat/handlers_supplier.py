"""
Vehicle Location Tracking and Location Analysis query handlers.

Handles queries about:
- Vehicle location tracking (WHERE IS a specific vehicle) - VehicleLocationHandler
- Location-based data analysis (WHERE DO failures occur) - LocationAnalysisHandler
"""

import re
import html as _html
import pandas as pd
from typing import Optional
from chat.handlers import QueryHandler, QueryContext
from chat.context_utils import (
    extract_context_filters_from_memory,
    extract_referenced_entity_from_memory
)
from utils.logger import chat_logger as logger
from helper import km_to_miles, fetch_nearest_dealers
from config import config


class VehicleLocationHandler(QueryHandler):
    """
    Handle vehicle location tracking and service center queries for specific VINs.
    
    This handler tracks WHERE A SPECIFIC VEHICLE IS located:
    - Current location queries (integrates with inference log for most recent location data)
    - Service center queries (dynamically fetches nearest dealers via AWS Location Service)
    - Context-aware queries (extracts VIN from conversation memory for follow-up questions)
    
    This is VEHICLE-CENTRIC (asks about a specific vehicle's location).
    
    General VIN data queries (show VINs, count VINs, VINs by model, etc.) 
    are handled by TextToSQLHandler.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        """
        Only handle VIN queries that require special logic:
        - Specific VIN location queries (needs inference log integration)
        - Specific VIN service center queries (needs dynamic dealer fetching)
        - "VINs affected by failures" (needs context extraction from conversation)
        
        General VIN data queries (show VINs, count VINs, VINs by model, etc.) 
        should be handled by TextToSQLHandler.
        """
        query_lower = context.query.lower()
        
        has_vin = (re.search(r'\bvin\b', query_lower) or 
                   re.search(r'\bvins\b', query_lower) or 
                   re.search(r'1N4[A-Z0-9]{8,}', context.query))
        
        if not has_vin:
            return False
        
        # Only handle if it's a specific VIN query with special requirements:
        # 1. Specific VIN location query (with actual VIN pattern or "this VIN")
        has_specific_vin = (
            re.search(r'1N4[A-Z0-9]{8,}', context.query) or  # Actual VIN in query
            re.search(r'\b(this|that|the)\s+vin\b', query_lower) or  # "this VIN" reference
            re.search(r'\bvin\s+(mentioned|above|before|earlier)\b', query_lower)  # VIN reference
        )
        
        # 2. Location-related query for specific VIN
        is_location_query = any(keyword in query_lower for keyword in [
            "location", "where", "coordinates", "current location", 
            "current position", "where is", "service center", "dealer", "nearest"
        ])
        
        # 3. "VINs affected by failures" - needs context extraction
        is_affected_query = any(phrase in query_lower for phrase in [
            "affected", "vins affected", "vehicles affected", 
            "affected by", "affected by these", "affected by failures"
        ])
        
        # Only handle if it matches one of these specific patterns
        return (has_specific_vin and is_location_query) or is_affected_query
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"VehicleLocationHandler processing: '{context.query[:50]}...'")
        
        # Check if VIN column exists
        if "vin" not in df.columns:
            return "<p>VIN tracking is not available in the current dataset. Please regenerate with the enhanced dataset.</p>"
        
        # Extract specific VIN if mentioned (17-character format)
        vin_match = re.search(r'1N4[A-Z0-9]{8,17}', context.query)
        vin = None
        
        # If no VIN in current query, check conversation context for "this VIN" references
        if not vin_match:
            # Check if query references "this VIN", "that VIN", "the VIN", etc.
            vin_reference_patterns = [r'\b(this|that|the)\s+vin\b', r'\bvin\s+(mentioned|above|before|earlier)\b']
            is_vin_reference = any(re.search(pattern, query_lower) for pattern in vin_reference_patterns)
            
            if is_vin_reference and context.conversation_context:
                logger.info("Detected VIN reference in query, extracting from conversation memory...")
                recent_exchanges = context.conversation_context.get_recent_context(window_size=5)
                vin = extract_referenced_entity_from_memory(context, recent_exchanges, entity_type="VIN")
                if vin:
                    logger.info(f"Found VIN {vin} from conversation memory")
                else:
                    logger.warning("VIN reference detected but no VIN found in conversation context")
        
        # If we found a VIN (either in query or context), use it
        if vin_match:
            vin = vin_match.group(0)
        
        if vin:
            # Specific VIN query (from current query or conversation context)
            # Try exact match first, then fall back to contains
            vin_data = df[df["vin"].astype(str).str.strip().str.upper() == vin.upper()]
            if vin_data.empty:
                # Fall back to contains for partial matches
                vin_data = df[df["vin"].astype(str).str.contains(vin, case=False, na=False)]
            
            if vin_data.empty:
                return f"<p>VIN <span style='font-family:monospace'>{vin}</span> not found in the dataset.</p>"
            
            row = vin_data.iloc[0]
            
            # Determine what type of VIN query
            location_keywords = ["location", "where", "coordinates", "city", "current location", "current position", "where is"]
            is_current_location = any(keyword in query_lower for keyword in ["current location", "current position"])
            
            if any(keyword in query_lower for keyword in location_keywords):
                # Location query - prioritize inference log for "current location"
                current_lat = None
                current_lon = None
                location_source = "historical"
                
                # For "current location" queries, check inference log first (most recent data)
                if is_current_location and context.df_inference is not None and not context.df_inference.empty:
                    if "vin" in context.df_inference.columns and "lat" in context.df_inference.columns and "lon" in context.df_inference.columns:
                        # Find most recent inference log entry for this VIN
                        vin_inference = context.df_inference[
                            context.df_inference["vin"].astype(str).str.strip().str.upper() == vin.upper()
                        ]
                        if not vin_inference.empty:
                            # Get most recent entry (sorted by timestamp)
                            if "timestamp" in vin_inference.columns:
                                latest_inference = vin_inference.sort_values("timestamp", ascending=False).iloc[0]
                            else:
                                latest_inference = vin_inference.iloc[-1]  # Last row if no timestamp
                            
                            if pd.notna(latest_inference.get('lat')) and pd.notna(latest_inference.get('lon')):
                                current_lat = latest_inference['lat']
                                current_lon = latest_inference['lon']
                                location_source = "inference_log"
                                logger.info(f"Using current location from inference log for VIN {vin}")
                
                # Fall back to historical data if inference log doesn't have location
                if current_lat is None or current_lon is None:
                    if pd.notna(row.get('current_lat')) and pd.notna(row.get('current_lon')):
                        current_lat = row['current_lat']
                        current_lon = row['current_lon']
                        location_source = "historical"
                    elif pd.notna(row.get('vehicle_lat')) and pd.notna(row.get('vehicle_lon')):
                        current_lat = row['vehicle_lat']
                        current_lon = row['vehicle_lon']
                        location_source = "historical"
                    elif pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                        current_lat = row['lat']
                        current_lon = row['lon']
                        location_source = "historical"
                
                # Build location response
                html = f"<p><strong>Location for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>City:</strong> {row.get('city', 'N/A')}</li>"
                if current_lat is not None and current_lon is not None:
                    html += f"<li><strong>Coordinates:</strong> {current_lat:.6f}, {current_lon:.6f}"
                    if is_current_location and location_source == "inference_log":
                        html += " <span style='color:#94a3b8; font-size:0.85em;'>(most recent)</span>"
                    html += "</li>"
                else:
                    html += f"<li><strong>Coordinates:</strong> Not available</li>"
                
                # For current location from inference log, fetch nearest dealers dynamically
                dealer_name = None
                dealer_distance_miles = None
                if location_source == "inference_log" and current_lat is not None and current_lon is not None:
                    try:
                        # Fetch nearest dealers dynamically using AWS Location Service
                        nearest_dealers, from_aws = fetch_nearest_dealers(
                            current_lat=current_lat,
                            current_lon=current_lon,
                            place_index_name=config.aws.place_index_name,
                            aws_region=config.aws.region,
                            text_query="Nissan Service Center",
                            top_n=1,  # Get only the nearest one
                        )
                        if nearest_dealers and len(nearest_dealers) > 0:
                            nearest_dealer = nearest_dealers[0]
                            dealer_name = nearest_dealer.get('name', 'N/A')
                            dealer_distance_miles = nearest_dealer.get('distance_miles')
                            logger.info(f"Fetched nearest dealer dynamically: {dealer_name} ({dealer_distance_miles} mi)")
                    except Exception as e:
                        logger.warning(f"Failed to fetch nearest dealers dynamically: {e}")
                        # Fall through to use historical data
                
                # Use historical dealer data if we didn't fetch dynamically
                if dealer_name is None:
                    if pd.notna(row.get('dealer_name')):
                        dealer_name = row['dealer_name']
                    if pd.notna(row.get('dealer_distance_km')):
                        miles_val = km_to_miles(row['dealer_distance_km'])
                        if miles_val is not None:
                            dealer_distance_miles = miles_val
                
                # Display dealer information
                if dealer_name:
                    html += f"<li><strong>Nearest Dealer:</strong> {dealer_name}</li>"
                if dealer_distance_miles is not None:
                    html += f"<li><strong>Distance to Dealer:</strong> {dealer_distance_miles:.2f} mi</li>"
                html += "</ul>"
                
            elif any(word in query_lower for word in ["service center", "dealer", "nearest"]):
                # Service center query - get current location first, then fetch nearest dealer
                current_lat = None
                current_lon = None
                location_source = "historical"
                
                # Check inference log first for most recent location
                if context.df_inference is not None and not context.df_inference.empty:
                    if "vin" in context.df_inference.columns and "lat" in context.df_inference.columns and "lon" in context.df_inference.columns:
                        vin_inference = context.df_inference[
                            context.df_inference["vin"].astype(str).str.strip().str.upper() == vin.upper()
                        ]
                        if not vin_inference.empty:
                            if "timestamp" in vin_inference.columns:
                                latest_inference = vin_inference.sort_values("timestamp", ascending=False).iloc[0]
                            else:
                                latest_inference = vin_inference.iloc[-1]
                            
                            if pd.notna(latest_inference.get('lat')) and pd.notna(latest_inference.get('lon')):
                                current_lat = latest_inference['lat']
                                current_lon = latest_inference['lon']
                                location_source = "inference_log"
                                logger.info(f"Using current location from inference log for service center query, VIN {vin}")
                
                # Fall back to historical data if inference log doesn't have location
                if current_lat is None or current_lon is None:
                    if pd.notna(row.get('current_lat')) and pd.notna(row.get('current_lon')):
                        current_lat = row['current_lat']
                        current_lon = row['current_lon']
                        location_source = "historical"
                    elif pd.notna(row.get('vehicle_lat')) and pd.notna(row.get('vehicle_lon')):
                        current_lat = row['vehicle_lat']
                        current_lon = row['vehicle_lon']
                        location_source = "historical"
                    elif pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                        current_lat = row['lat']
                        current_lon = row['lon']
                        location_source = "historical"
                
                # Fetch nearest dealer dynamically based on current location
                dealer_name = None
                dealer_distance_miles = None
                dealer_lat = None
                dealer_lon = None
                
                if current_lat is not None and current_lon is not None:
                    try:
                        nearest_dealers, from_aws = fetch_nearest_dealers(
                            current_lat=current_lat,
                            current_lon=current_lon,
                            place_index_name=config.aws.place_index_name,
                            aws_region=config.aws.region,
                            text_query="Nissan Service Center",
                            top_n=1,  # Get only the nearest one
                        )
                        if nearest_dealers and len(nearest_dealers) > 0:
                            nearest_dealer = nearest_dealers[0]
                            dealer_name = nearest_dealer.get('name', 'N/A')
                            dealer_distance_miles = nearest_dealer.get('distance_miles')
                            dealer_lat = nearest_dealer.get('lat')
                            dealer_lon = nearest_dealer.get('lon')
                            logger.info(f"Fetched nearest dealer dynamically: {dealer_name} ({dealer_distance_miles} mi)")
                    except Exception as e:
                        logger.warning(f"Failed to fetch nearest dealers dynamically: {e}")
                        # Fall through to use historical data
                
                # Use historical dealer data if we didn't fetch dynamically
                if dealer_name is None:
                    if pd.notna(row.get('dealer_name')):
                        dealer_name = row['dealer_name']
                    if pd.notna(row.get('dealer_distance_km')):
                        miles_val = km_to_miles(row['dealer_distance_km'])
                        if miles_val is not None:
                            dealer_distance_miles = miles_val
                    if pd.notna(row.get('dealer_lat')) and pd.notna(row.get('dealer_lon')):
                        dealer_lat = row['dealer_lat']
                        dealer_lon = row['dealer_lon']
                
                # Build service center response
                html = f"<p><strong>Nearest service center for VIN {vin}:</strong></p><ul style='margin-top:6px;'>"
                html += f"<li><strong>Dealer:</strong> {dealer_name if dealer_name else 'N/A'}</li>"
                if dealer_distance_miles is not None:
                    html += f"<li><strong>Distance:</strong> {dealer_distance_miles:.2f} mi</li>"
                else:
                    html += f"<li><strong>Distance:</strong> N/A</li>"
                if dealer_lat is not None and dealer_lon is not None:
                    html += f"<li><strong>Dealer Location:</strong> {dealer_lat:.6f}, {dealer_lon:.6f}</li>"
                html += f"<li><strong>Vehicle City:</strong> {row.get('city', 'N/A')}</li>"
                if location_source == "inference_log":
                    html += f"<li style='color:#94a3b8; font-size:0.85em;'>Based on most recent vehicle location</li>"
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
            # No VIN found in query or context
            # Check if this is a reference query that failed to find VIN
            reference_patterns = [r'\b(this|that|the)\s+vin\b', r'\bvin\s+(mentioned|above|before|earlier)\b']
            is_reference = any(re.search(pattern, query_lower) for pattern in reference_patterns)
            
            if is_reference:
                if not context.conversation_context:
                    return "<p>I don't have access to conversation history. Please provide a specific VIN number (e.g., '1N4AZMA1800004').</p>"
                elif len(context.conversation_context.memory) == 0:
                    return "<p>I couldn't find a VIN in our conversation. Please ask about a specific VIN first, then I can help with follow-up questions.</p>"
                else:
                    return "<p>I couldn't find a VIN in our recent conversation. Please provide a specific VIN number (e.g., '1N4AZMA1800004') or ask about a VIN first.</p>"
            
            # Check if this is a service center query without VIN
            if any(word in query_lower for word in ["service center", "dealer", "nearest"]):
                return "<p>Please specify a VIN to find the nearest service center. For example: 'nearest service center for VIN 1N4AZMA1800004'</p>"
            
            # General case - no VIN specified
            # List VINs matching criteria
            if any(word in query_lower for word in ["affected", "failure", "failures", "problem", "issue"]):
                # VINs with failures
                affected = df[(df["claims_count"] > 0) | (df["repairs_count"] > 0) | (df["recalls_count"] > 0)]
                
                # Check if query references "these failures" - use LLM to extract context from memory
                failure_reference_patterns = [
                    r'\b(these|those|the)\s+failures?\b',
                    r'\bfailures?\s+(mentioned|above|before|earlier)\b'
                ]
                is_failure_reference = any(re.search(pattern, query_lower) for pattern in failure_reference_patterns)
                
                context_filters = {}
                if is_failure_reference and context.conversation_context:
                    logger.info("Detected failure reference, using LLM to extract context from conversation memory...")
                    recent_exchanges = context.conversation_context.get_recent_context(window_size=3)
                    context_filters = extract_context_filters_from_memory(context, recent_exchanges)
                
                # Apply filters from current query first
                for family in ["Battery", "Brakes", "Transmission", "Engine", "Electrical", "Lighting", "HVAC", "Safety", "Steering", "Tires"]:
                    if family.lower() in query_lower and "part_family" in df.columns:
                        affected = affected[affected["part_family"] == family]
                        break
                
                # Apply context filters from previous exchange
                if context_filters:
                    if "part_family" in context_filters and "part_family" in affected.columns:
                        affected = affected[affected["part_family"] == context_filters["part_family"]]
                        logger.info(f"Applied part_family filter: {context_filters['part_family']}")
                    if "age_bucket" in context_filters and "age_bucket" in affected.columns:
                        affected = affected[affected["age_bucket"] == context_filters["age_bucket"]]
                        logger.info(f"Applied age_bucket filter: {context_filters['age_bucket']}")
                    if "mileage_bucket" in context_filters and "mileage_bucket" in affected.columns:
                        affected = affected[affected["mileage_bucket"] == context_filters["mileage_bucket"]]
                        logger.info(f"Applied mileage_bucket filter: {context_filters['mileage_bucket']}")
                
                if affected.empty:
                    return "<p>No VINs found matching the failure criteria.</p>"
                
                vin_list = affected["vin"].head(15).tolist()
                total_affected = len(affected)
                
                # Build description with context info
                description = "failures"
                if context_filters:
                    parts = []
                    if "part_family" in context_filters:
                        parts.append(context_filters["part_family"].lower())
                    if "age_bucket" in context_filters:
                        parts.append(f"aged {context_filters['age_bucket']}")
                    if parts:
                        description = f"{' '.join(parts)} {description}"
                
                html = f"<p><strong>VINs affected by {description} ({total_affected} total):</strong></p>"
                html += "<p style='font-family:monospace; font-size:13px; color:#cfe9ff;'>"
                for i, vin in enumerate(vin_list[:10]):
                    html += f"{vin}<br>"
                if len(vin_list) > 10:
                    html += f"<span style='color:#94a3b8;'>... and {total_affected - 10} more</span>"
                html += "</p>"
                
                logger.info(f"Found {total_affected} affected VINs with context filters: {context_filters}")
                return html
            
            return "<p>Please specify a VIN (e.g., '1N4AZMA5100456') or search criteria (e.g., 'VINs with Battery failures').</p>"
class LocationAnalysisHandler(QueryHandler):
    """
    Handle location-based data analysis queries (geographic/dealer analysis).
    
    This handler analyzes data BY LOCATION:
    - Specific city analysis with detailed stats
    - Dealer-specific analysis
    - Region analysis with special formatting
    
    This is GEOGRAPHIC-CENTRIC (asks about WHERE failures/issues occur).
    
    General location breakdown queries ("failures by city", "vehicles by location", etc.)
    should be handled by TextToSQLHandler.
    """
    
    def can_handle(self, context: QueryContext) -> bool:
        """
        Only handle location queries that require special formatting or analysis:
        - Specific city analysis with detailed stats
        - Dealer-specific analysis
        - Region analysis with special formatting
        
        General location breakdown queries ("failures by city", "vehicles by location", etc.)
        should be handled by TextToSQLHandler.
        """
        query_lower = context.query.lower()
        df = context.df_history
        
        # Check for location keywords
        has_location_keyword = any(phrase in query_lower for phrase in [
            "city", "cities", "location", "locations", "where",
            "coordinates", "latitude", "longitude", "lat", "lon",
            "dealers", "dealer", "service center", "region", "regional"
        ])
        
        if not has_location_keyword:
            return False
        
        # Only handle if it's a specific analysis query (not a simple breakdown):
        # 1. Specific city mentioned (not just "by city")
        if "city" in df.columns:
            mentioned_city = None
            for city_name in df["city"].dropna().unique():
                if str(city_name).lower() in query_lower:
                    mentioned_city = str(city_name)
                    break
            if mentioned_city:
                # Specific city query - handle with special formatting
                return True
        
        # 2. Specific dealer mentioned
        if self._mentions_dealer_name(df, query_lower):
            return True
        
        # 3. Region/dealer analysis queries (not simple "by X" breakdowns)
        is_analysis_query = any(phrase in query_lower for phrase in [
            "dealer issues", "dealer problems", "dealer failures", 
            "major issues", "issues from", "problems from", "failures from",
            "regional analysis", "city analysis", "location analysis", 
            "area analysis", "geographic", "which dealers", "which region"
        ])
        
        # Don't handle simple breakdown queries - let TextToSQL handle them
        is_simple_breakdown = bool(re.search(r'\b(by|per)\s+(city|location|region|area)\b', query_lower))
        
        # Don't handle time-based breakdowns - let TextToSQL handle them
        is_time_breakdown = bool(re.search(r'\b(by|per)\s+(month|quarter|year|week|day|time)\b', query_lower))
        
        return is_analysis_query and not is_simple_breakdown and not is_time_breakdown
    
    def handle(self, context: QueryContext) -> str:
        df = context.df_history
        query_lower = context.query.lower()
        
        logger.info(f"LocationAnalysisHandler processing: '{context.query[:50]}...'")
        
        # Don't handle time-based breakdowns - these should go to TextToSQL
        is_time_breakdown = bool(re.search(r'\b(by|per)\s+(month|quarter|year|week|day|time)\b', query_lower))
        if is_time_breakdown:
            # This shouldn't happen if can_handle is working correctly, but add defensive check
            logger.warning(f"LocationAnalysisHandler received time-based query, should have been handled by TextToSQL: {context.query}")
            return None  # Return None to let it fall through to TextToSQL
        
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

