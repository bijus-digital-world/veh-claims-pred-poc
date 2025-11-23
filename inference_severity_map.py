"""
Severity map visualization for inference log data.
Shows North America map with colored bubbles based on prediction severity.
"""

import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Tuple
import math

from config import config
try:
    from utils.logger import app_logger as logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


def calculate_risk_level(claim_pct: float) -> str:
    """Calculate risk level from claim percentage."""
    if claim_pct > config.risk.high_threshold:
        return "High"
    elif claim_pct >= config.risk.medium_threshold:
        return "Medium"
    return "Low"


def get_severity_color(severity: str, alpha: int = 200) -> List[int]:
    """Get RGB color for severity level."""
    colors = {
        "High": [239, 68, 68, alpha],      # Red
        "Medium": [251, 191, 36, alpha],   # Yellow
        "Low": [34, 197, 94, alpha]        # Green
    }
    return colors.get(severity, [156, 163, 175, alpha])  # Gray fallback


def group_by_location(df: pd.DataFrame, grid_size: float = 0.5) -> pd.DataFrame:
    """
    Group inference log data by location using a grid.
    
    Args:
        df: DataFrame with lat, lon, and pred_prob_pct columns
        grid_size: Grid size in degrees (0.5 = ~55km at equator)
    
    Returns:
        DataFrame with aggregated data per location
    """
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return pd.DataFrame()
    
    # Filter out rows without valid coordinates
    df_clean = df.dropna(subset=["lat", "lon"]).copy()
    if df_clean.empty:
        return pd.DataFrame()
    
    # Check for pred_prob_pct column (may be named differently)
    pred_col = None
    for col in ["pred_prob_pct", "pred_prob", "predictive_pct", "pred_pct"]:
        if col in df_clean.columns:
            pred_col = col
            break
    
    if pred_col is None:
        # Try to calculate from pred_prob if available
        if "pred_prob" in df_clean.columns:
            df_clean["pred_prob_pct"] = df_clean["pred_prob"] * 100.0
            pred_col = "pred_prob_pct"
        else:
            # No prediction data available
            return pd.DataFrame()
    
    # Ensure pred_prob_pct is numeric (but don't drop rows yet - we want to preserve model/mileage data)
    df_clean[pred_col] = pd.to_numeric(df_clean[pred_col], errors='coerce')
    # Only drop rows where pred_prob_pct is NaN AFTER we've preserved other column data
    df_clean = df_clean.dropna(subset=[pred_col])
    
    if df_clean.empty:
        return pd.DataFrame()
    
    # Round coordinates to grid
    df_clean["lat_grid"] = (df_clean["lat"] / grid_size).round() * grid_size
    df_clean["lon_grid"] = (df_clean["lon"] / grid_size).round() * grid_size
    
    # Aggregate by location grid
    agg_dict = {
        pred_col: ["mean", "max", "count"],
    }
    
    # Add optional columns if they exist
    # Helper functions for aggregation
    def get_mode_or_first(series):
        """Get mode (most common value) or first value if mode is empty."""
        # Drop NaN values before computing mode
        series_clean = series.dropna()
        if len(series_clean) > 0:
            # Convert to string to handle any type
            series_clean_str = series_clean.astype(str)
            # Remove empty strings
            series_clean_str = series_clean_str[series_clean_str.str.strip() != ""]
            if len(series_clean_str) > 0:
                mode_values = series_clean_str.mode()
                if not mode_values.empty:
                    result = str(mode_values.iloc[0]).strip()
                    return result if result else "N/A"
                else:
                    result = str(series_clean_str.iloc[0]).strip()
                    return result if result else "N/A"
        return "N/A"
    
    def get_mean_or_zero(series):
        """Get mean value or 0 if all NaN."""
        # Convert to numeric, coercing errors to NaN
        series_clean = pd.to_numeric(series, errors='coerce').dropna()
        if len(series_clean) > 0:
            mean_val = float(series_clean.mean())
            # Return 0.0 if mean is NaN or invalid
            return mean_val if pd.notna(mean_val) and mean_val >= 0 else 0.0
        return 0.0
    
    # Always try to aggregate these columns if they exist, even if some values are NaN
    # The aggregation functions will handle NaN values appropriately
    if "model" in df_clean.columns:
        agg_dict["model"] = get_mode_or_first
    if "primary_failed_part" in df_clean.columns:
        agg_dict["primary_failed_part"] = get_mode_or_first
    if "timestamp" in df_clean.columns:
        agg_dict["timestamp"] = "max"  # Most recent event
    if "mileage" in df_clean.columns:
        # Use custom function to handle NaN values properly
        agg_dict["mileage"] = get_mean_or_zero
    if "age" in df_clean.columns:
        # Use custom function to handle NaN values properly
        agg_dict["age"] = get_mean_or_zero
    
    agg_data = df_clean.groupby(["lat_grid", "lon_grid"]).agg(agg_dict).reset_index()
    
    # Debug: Print column names before flattening
    # logger.debug(f"Columns before flattening: {list(agg_data.columns)}")
    
    # Flatten multi-level column names from aggregation
    # pandas agg with multiple functions creates MultiIndex columns like (pred_col, 'mean'), (pred_col, 'max'), etc.
    # Custom aggregation functions create MultiIndex columns like ('model', 'get_mode_or_first')
    # For custom functions, we want to keep just the column name (first part), not the function name
    new_columns = []
    column_mapping = {}  # Track original column names
    
    for col in agg_data.columns:
        if isinstance(col, tuple):
            if len(col) == 2:
                col_name, func_name = col
                # For prediction columns with built-in functions (mean, max, count), join with underscore
                if func_name in ['mean', 'max', 'count']:
                    new_col = '_'.join(str(c) for c in col if c)
                    new_columns.append(new_col)
                # For custom aggregation functions, use just the column name
                elif func_name in ['get_mode_or_first', 'get_mean_or_zero']:
                    new_col = str(col_name)
                    new_columns.append(new_col)
                # For built-in functions like 'max' on timestamp, join with underscore
                else:
                    new_col = '_'.join(str(c) for c in col if c)
                    new_columns.append(new_col)
            else:
                # Fallback: join all parts
                new_col = '_'.join(str(c) for c in col if c)
                new_columns.append(new_col)
            column_mapping[new_col] = col  # Store mapping for reference
        else:
            # Regular column name (grouping columns like lat_grid, lon_grid)
            new_col = str(col)
            new_columns.append(new_col)
            column_mapping[new_col] = col  # Store mapping for reference
    
    agg_data.columns = new_columns
    
    # Debug: Print column names after flattening
    # logger.debug(f"Columns after flattening: {list(agg_data.columns)}")
    
    # Rename columns to standard names
    rename_dict = {}
    if "lat_grid" in agg_data.columns:
        rename_dict["lat_grid"] = "lat"
    if "lon_grid" in agg_data.columns:
        rename_dict["lon_grid"] = "lon"
    
    # Handle prediction columns (mean, max, count)
    for col in agg_data.columns:
        if pred_col in col:
            if "mean" in col.lower():
                rename_dict[col] = "avg_pred_pct"
            elif "max" in col.lower():
                rename_dict[col] = "max_pred_pct"
            elif "count" in col.lower():
                rename_dict[col] = "count"
    
    # Handle optional columns - check what we actually got after aggregation
    # Note: Columns aggregated with custom functions appear as-is (no suffix)
    # Columns aggregated with built-in functions may have suffixes
    
    # Model column (aggregated with get_mode_or_first) - should appear as "model"
    if "model" not in agg_data.columns:
        # Check if it's in the rename_dict (shouldn't be, but check anyway)
        agg_data["model"] = "N/A"
    else:
        # Clean up model column - handle NaN and convert to string
        agg_data["model"] = agg_data["model"].fillna("N/A")
        agg_data["model"] = agg_data["model"].astype(str)
        agg_data["model"] = agg_data["model"].replace("nan", "N/A").replace("None", "N/A").replace("", "N/A")
    
    # Handle primary_failed_part (aggregated with get_mode_or_first) - should appear as "primary_failed_part"
    pfp_col = None
    for col in agg_data.columns:
        if col.lower() == "primary_failed_part" or "primary_failed_part" in col.lower():
            pfp_col = col
            rename_dict[col] = "primary_part"
            break
    if pfp_col is None:
        agg_data["primary_part"] = "N/A"
    
    # Handle timestamp (aggregated with "max") - will appear as "timestamp_max" after flattening
    timestamp_col = None
    for col in agg_data.columns:
        if "timestamp" in col.lower() and "lat" not in col.lower() and "lon" not in col.lower():
            timestamp_col = col
            rename_dict[col] = "event_timestamp"
            break
    if timestamp_col is None:
        # Check if it's just "timestamp" (shouldn't happen with "max" aggregation, but check anyway)
        if "timestamp" in agg_data.columns:
            rename_dict["timestamp"] = "event_timestamp"
        else:
            agg_data["event_timestamp"] = None
    
    # Handle mileage (aggregated with get_mean_or_zero) - should appear as "mileage"
    mileage_col = None
    for col in agg_data.columns:
        if col.lower() == "mileage":
            mileage_col = col
            rename_dict[col] = "avg_mileage"
            break
    if mileage_col is None:
        agg_data["avg_mileage"] = 0.0
    
    # Handle age (aggregated with get_mean_or_zero) - should appear as "age"
    age_col = None
    for col in agg_data.columns:
        if col.lower() == "age":
            age_col = col
            rename_dict[col] = "avg_age"
            break
    if age_col is None:
        agg_data["avg_age"] = 0.0
    
    # Apply renames
    agg_data = agg_data.rename(columns=rename_dict)
    
    # Final check: ensure all required columns exist after renaming
    if "model" not in agg_data.columns:
        agg_data["model"] = "N/A"
    else:
        # Final cleanup of model column
        agg_data["model"] = agg_data["model"].fillna("N/A").astype(str).replace("nan", "N/A").replace("None", "N/A").replace("", "N/A")
    
    if "primary_part" not in agg_data.columns:
        agg_data["primary_part"] = "N/A"
    else:
        agg_data["primary_part"] = agg_data["primary_part"].fillna("N/A").astype(str).replace("nan", "N/A").replace("None", "N/A").replace("", "N/A")
    
    if "event_timestamp" not in agg_data.columns:
        agg_data["event_timestamp"] = None
    
    if "avg_mileage" not in agg_data.columns:
        agg_data["avg_mileage"] = 0.0
    else:
        # Ensure avg_mileage is numeric
        agg_data["avg_mileage"] = pd.to_numeric(agg_data["avg_mileage"], errors='coerce').fillna(0.0)
    
    if "avg_age" not in agg_data.columns:
        agg_data["avg_age"] = 0.0
    else:
        # Ensure avg_age is numeric
        agg_data["avg_age"] = pd.to_numeric(agg_data["avg_age"], errors='coerce').fillna(0.0)
    
    # Calculate severity based on average prediction percentage
    agg_data["severity"] = agg_data["avg_pred_pct"].apply(calculate_risk_level)
    
    # Calculate weighted severity (considering both average and count)
    agg_data["severity_score"] = agg_data.apply(
        lambda row: row["avg_pred_pct"] * (1 + math.log10(max(row["count"], 1))),
        axis=1
    )
    
    return agg_data


def _apply_filters(df_log: pd.DataFrame, date_range=None, text_filter: str = "") -> pd.DataFrame:
    """
    Apply date range and text filters to the inference log DataFrame.
    
    Args:
        df_log: Full inference log DataFrame
        date_range: Tuple of (start_date, end_date), single date, list, or None
        text_filter: Optional text filter for model/part
    
    Returns:
        Filtered DataFrame
    """
    if df_log.empty:
        return df_log.copy()
    
    # Ensure timestamp column is datetime
    if "timestamp" not in df_log.columns:
        return pd.DataFrame()
    
    if not pd.api.types.is_datetime64_any_dtype(df_log["timestamp"]):
        try:
            df_log = df_log.copy()
            df_log["timestamp"] = pd.to_datetime(df_log["timestamp"])
        except Exception:
            return pd.DataFrame()
    
    # Apply date range filter
    mask = pd.Series([True] * len(df_log), index=df_log.index)
    
    if date_range is not None:
        try:
            # Handle different date_range formats from Streamlit
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                # Date range: (start_date, end_date)
                dr_start, dr_end = date_range
                if dr_start is not None and dr_end is not None:
                    mask = (df_log["timestamp"].dt.date >= dr_start) & (df_log["timestamp"].dt.date <= dr_end)
            elif isinstance(date_range, (tuple, list)) and len(date_range) == 1:
                # Single date in a list/tuple
                mask = df_log["timestamp"].dt.date == date_range[0]
            else:
                # Single date object
                mask = df_log["timestamp"].dt.date == date_range
        except Exception as e:
            logger.warning(f"Error applying date range filter: {e}")
            # Continue without date filter if there's an error
    
    # Apply text filter
    if text_filter and text_filter.strip():
        try:
            t = text_filter.strip().lower()
            text_mask = pd.Series([False] * len(df_log), index=df_log.index)
            
            # Check model column
            if "model" in df_log.columns:
                model_mask = df_log["model"].astype(str).str.lower().str.contains(t, na=False, regex=False)
                text_mask = text_mask | model_mask
            
            # Check primary_failed_part column
            if "primary_failed_part" in df_log.columns:
                part_mask = df_log["primary_failed_part"].astype(str).str.lower().str.contains(t, na=False, regex=False)
                text_mask = text_mask | part_mask
            
            mask = mask & text_mask
        except Exception as e:
            logger.warning(f"Error applying text filter: {e}")
            # Continue without text filter if there's an error
    
    return df_log[mask].copy()


def build_severity_map_data(df_log: pd.DataFrame, date_range=None, text_filter: str = "") -> List[Dict]:
    """
    Build map data points from inference log.
    
    Args:
        df_log: Full inference log DataFrame
        date_range: Tuple of (start_date, end_date), single date, list, or None
        text_filter: Optional text filter for model/part (searches in model and primary_failed_part columns)
    
    Returns:
        List of map point dictionaries with filtered data
    """
    if df_log.empty:
        return []
    
    # Check if lat/lon columns exist
    lat_cols = [col for col in df_log.columns if col.lower() in ['lat', 'latitude']]
    lon_cols = [col for col in df_log.columns if col.lower() in ['lon', 'longitude']]
    
    if not lat_cols or not lon_cols:
        return []
    
    lat_col = lat_cols[0]
    lon_col = lon_cols[0]
    
    # Apply filters using shared function
    df_filtered = _apply_filters(df_log, date_range, text_filter)
    
    if df_filtered.empty:
        return []
    
    # Rename columns for consistency
    df_filtered = df_filtered.rename(columns={lat_col: "lat", lon_col: "lon"})
    df_filtered = df_filtered.dropna(subset=["lat", "lon"])
    if df_filtered.empty:
        return []

    # Ensure prediction percentage column
    pred_col = None
    for col in ["pred_prob_pct", "pred_prob", "predictive_pct", "pred_pct"]:
        if col in df_filtered.columns:
            pred_col = col
            break
    if pred_col is None:
        return []

    if pred_col == "pred_prob":
        df_filtered["pred_prob_pct"] = df_filtered[pred_col] * 100.0
    elif pred_col != "pred_prob_pct":
        df_filtered["pred_prob_pct"] = df_filtered[pred_col]
    df_filtered["pred_prob_pct"] = pd.to_numeric(df_filtered["pred_prob_pct"], errors="coerce")
    df_filtered = df_filtered.dropna(subset=["pred_prob_pct"])

    if df_filtered.empty:
        return []

    # Filter to North America bounds (roughly)
    na_mask = (
        (df_filtered["lat"] >= 25) & (df_filtered["lat"] <= 70) &
        (df_filtered["lon"] >= -130) & (df_filtered["lon"] <= -50)
    )
    df_filtered = df_filtered[na_mask]

    if df_filtered.empty:
        return []

    radius_lookup = {"High": 2500, "Medium": 2000, "Low": 1500}

    # Build map points (one per VIN/record)
    map_points = []
    for _, row in df_filtered.iterrows():
        pred_pct = float(row.get("pred_prob_pct", 0))
        severity = calculate_risk_level(pred_pct)
        fill_color = get_severity_color(severity, alpha=180)
        line_color = get_severity_color(severity, alpha=255)
        radius_m = radius_lookup.get(severity, 1500)

        event_timestamp = row.get("timestamp", None)
        if pd.notna(event_timestamp) and event_timestamp:
            if isinstance(event_timestamp, pd.Timestamp):
                event_str = event_timestamp.strftime("%Y-%m-%d %H:%M")
            elif isinstance(event_timestamp, str):
                # Try to parse and format the timestamp string
                try:
                    if len(event_timestamp) >= 16:
                        event_str = event_timestamp[:16].replace("T", " ")
                    else:
                        event_str = event_timestamp
                except:
                    event_str = str(event_timestamp)[:16]
            else:
                event_str = str(event_timestamp)[:16]
        else:
            event_str = "N/A"
        
        model = row.get("model", "N/A")
        if pd.isna(model) or model == "" or model is None or str(model).lower() in ["nan", "none", "n/a"]:
            model = "N/A"
        else:
            model = str(model).strip()
            if model == "":
                model = "N/A"
        
        pfp = row.get("primary_failed_part", row.get("primary_part", "N/A"))
        if pd.isna(pfp) or pfp == "" or pfp is None or str(pfp).lower() in ["nan", "none", "n/a"]:
            pfp = "N/A"
        else:
            pfp = str(pfp).strip()
            if pfp == "":
                pfp = "N/A"
        
        mileage = row.get("mileage", row.get("avg_mileage", None))
        if mileage is None or pd.isna(mileage):
            mileage_str = "N/A"
        else:
            try:
                mileage_val = float(mileage)
                # Show 0 or negative as "N/A" since they indicate missing/invalid data
                # Also check for unreasonable values (e.g., > 1 million miles)
                if mileage_val <= 0 or mileage_val > 1000000:
                    mileage_str = "N/A"
                else:
                    mileage_str = f"{mileage_val:,.0f}"
            except (ValueError, TypeError):
                mileage_str = "N/A"
        
        age = row.get("age", row.get("avg_age", 0))
        if pd.isna(age) or age is None or age == 0:
            age_str = "N/A"
        else:
            try:
                age_val = float(age)
                age_str = f"{age_val:.1f}"
            except:
                age_str = "N/A"
        
        pred_pct_str = f"{pred_pct:.1f}%"
        vin = row.get("vin", "N/A")
        
        # Create tooltip with all requested fields
        tooltip_html = (
            f"<div style='padding:12px; max-width:320px; background:rgba(15,15,15,0.95); border-radius:6px;'>"
            f"<div style='font-weight:700; font-size:15px; margin-bottom:10px; color:{'#ef4444' if severity=='High' else '#fbbf24' if severity=='Medium' else '#22c55e'};'>"
            f"{severity} Risk</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>VIN:</strong> {vin}</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>Location:</strong> {row['lat']:.3f}°N, {row['lon']:.3f}°W</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>Event:</strong> {event_str}</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>Model:</strong> {model}</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>PFP:</strong> {pfp}</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>Mileage:</strong> {mileage_str}</div>"
            f"<div style='font-size:12px; color:#d1d5db; margin-bottom:5px; line-height:1.6;'>"
            f"<strong>Age:</strong> {age_str}</div>"
            f"<div style='font-size:12px; color:#d1d5db; line-height:1.6;'>"
            f"<strong>Pred %:</strong> {pred_pct_str}</div>"
            f"</div>"
        )
        
        map_points.append({
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "position": [float(row["lon"]), float(row["lat"])],
            "severity": severity,
            "avg_pred_pct": float(pred_pct),
            "radius_m": radius_m,
            "fill_color": fill_color,
            "line_color": line_color,
            "tooltip": tooltip_html
        })
    
    return map_points


def render_severity_map(df_log: pd.DataFrame, date_range=None, text_filter: str = "", skip_header: bool = False):
    """
    Render severity map showing North America with colored bubbles.
    
    Args:
        df_log: Full inference log DataFrame
        date_range: Tuple of (start_date, end_date) or single date, or None
        text_filter: Optional text filter for model/part
        skip_header: If True, skip rendering the header (header rendered externally)
    """
    # Card header with professional styling (only if not skipped)
    if not skip_header:
        st.markdown("""
        <style>
        .severity-map-container {
            margin-top: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(
            '<div class="card severity-map-container" style="margin-top:0;"><div class="card-header" style="margin-bottom:0; padding:8px 12px;">Regional Risk Analysis</div>',
            unsafe_allow_html=True
        )
    else:
        # When header is skipped, the card container was already started in app.py with the header
        # We just continue with the content - no need to start a new container
        pass
    
    # Check if df_log has required columns
    if df_log.empty:
        st.markdown(
            "<div style='padding:20px; color:#94a3b8; text-align:center;'>"
            "No inference log data available.</div>",
            unsafe_allow_html=True
        )
        if not skip_header:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Check for location columns
    lat_cols = [col for col in df_log.columns if col.lower() in ['lat', 'latitude']]
    lon_cols = [col for col in df_log.columns if col.lower() in ['lon', 'longitude']]
    
    if not lat_cols or not lon_cols:
        st.markdown(
            "<div style='padding:20px; color:#94a3b8; text-align:center;'>"
            "Location data (latitude/longitude) not found in inference log. "
            "Map visualization requires coordinates to display regional risk analysis.</div>",
            unsafe_allow_html=True
        )
        if not skip_header:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Build map data
    try:
        map_points = build_severity_map_data(df_log, date_range, text_filter)
    except Exception as e:
        logger.error(f"Error building severity map data: {e}", exc_info=True)
        st.markdown(
            f"<div style='padding:20px; color:#ef4444; text-align:center;'>"
            f"Error building map data: {str(e)}</div>",
            unsafe_allow_html=True
        )
        if not skip_header:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    if not map_points:
        st.markdown(
            "<div style='padding:20px; color:#94a3b8; text-align:center;'>"
            "No location data available for the selected filters. "
            "Ensure inference log contains latitude and longitude coordinates.</div>",
            unsafe_allow_html=True
        )
        if not skip_header:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Summary statistics
    total_locations = len(map_points)
    high_count = sum(1 for p in map_points if p["severity"] == "High")
    medium_count = sum(1 for p in map_points if p["severity"] == "Medium")
    low_count = sum(1 for p in map_points if p["severity"] == "Low")
    
    # Summary KPI
    st.markdown("""
    <div style="display:flex; gap:12px; margin-bottom:12px; flex-wrap:wrap;">
        <div style="flex:1; min-width:120px; padding:8px; background:rgba(239,68,68,0.15); border-radius:6px; border:1px solid rgba(239,68,68,0.3);">
            <div style="font-size:11px; color:#94a3b8; margin-bottom:2px;">High Risk</div>
            <div style="font-size:18px; font-weight:700; color:#ef4444;">{high_count}</div>
        </div>
        <div style="flex:1; min-width:120px; padding:8px; background:rgba(251,191,36,0.15); border-radius:6px; border:1px solid rgba(251,191,36,0.3);">
            <div style="font-size:11px; color:#94a3b8; margin-bottom:2px;">Medium Risk</div>
            <div style="font-size:18px; font-weight:700; color:#fbbf24;">{medium_count}</div>
        </div>
        <div style="flex:1; min-width:120px; padding:8px; background:rgba(34,197,94,0.15); border-radius:6px; border:1px solid rgba(34,197,94,0.3);">
            <div style="font-size:11px; color:#94a3b8; margin-bottom:2px;">Low Risk</div>
            <div style="font-size:18px; font-weight:700; color:#22c55e;">{low_count}</div>
        </div>
    </div>
    """.format(
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count
    ), unsafe_allow_html=True)
    
    # Legend - balloon sizes match map bubble sizes (radius_min_pixels to radius_max_pixels)
    # Map bubbles: Low (3-35px radius), Medium (4-40px radius), High (5-45px radius)
    # Use smaller representative sizes that match typical bubble appearance on map
    legend_html = """
    <div style="display:flex; gap:16px; align-items:center; margin-bottom:10px; flex-wrap:wrap;">
      <div style="display:flex; gap:6px; align-items:center;">
        <div style="width:10px; height:10px; border-radius:50%; background:#ef4444; border:1px solid rgba(255,255,255,0.15); box-shadow:0 1px 2px rgba(0,0,0,0.2);"></div>
        <div style="color:#cbd5e1; font-size:12px; font-weight:500;">High Risk</div>
      </div>
      <div style="display:flex; gap:6px; align-items:center;">
        <div style="width:8px; height:8px; border-radius:50%; background:#fbbf24; border:1px solid rgba(255,255,255,0.15); box-shadow:0 1px 2px rgba(0,0,0,0.2);"></div>
        <div style="color:#cbd5e1; font-size:12px; font-weight:500;">Medium Risk</div>
      </div>
      <div style="display:flex; gap:6px; align-items:center;">
        <div style="width:6px; height:6px; border-radius:50%; background:#22c55e; border:1px solid rgba(255,255,255,0.15); box-shadow:0 1px 2px rgba(0,0,0,0.2);"></div>
        <div style="color:#cbd5e1; font-size:12px; font-weight:500;">Low Risk</div>
      </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Calculate map center (center of USA for full view)
    if map_points:
        lats = [p["lat"] for p in map_points]
        lons = [p["lon"] for p in map_points]
        if len(lats) > 1:
            # Center on data but zoom out to show full USA
            center_lat = (min(lats) + max(lats)) / 2.0
            center_lon = (min(lons) + max(lons)) / 2.0
            # Zoom out to show full USA area (lower zoom = more zoomed out)
            zoom_level = 2.0
        else:
            # Single point - center on USA with zoomed out view
            center_lat = 39.8283  # Center of USA
            center_lon = -98.5795
            zoom_level = 2.0
    else:
        # Default to center of USA, zoomed out
        center_lat = 39.8283
        center_lon = -98.5795
        zoom_level = 2.0
    
    # Create layers by severity for better rendering
    high_points = [p for p in map_points if p["severity"] == "High"]
    medium_points = [p for p in map_points if p["severity"] == "Medium"]
    low_points = [p for p in map_points if p["severity"] == "Low"]
    
    layers = []
    
    # Low risk layer (bottom)
    if low_points:
        low_layer = pdk.Layer(
            "ScatterplotLayer",
            data=low_points,
            get_position="position",
            get_fill_color="fill_color",
            get_line_color="line_color",
            get_radius="radius_m",
            radius_min_pixels=3,
            radius_max_pixels=35,
            radius_scale=1,
            pickable=True,
            stroked=True,
            line_width_min_pixels=1,
            opacity=0.7,
        )
        layers.append(low_layer)
    
    # Medium risk layer (middle)
    if medium_points:
        medium_layer = pdk.Layer(
            "ScatterplotLayer",
            data=medium_points,
            get_position="position",
            get_fill_color="fill_color",
            get_line_color="line_color",
            get_radius="radius_m",
            radius_min_pixels=4,
            radius_max_pixels=40,
            radius_scale=1,
            pickable=True,
            stroked=True,
            line_width_min_pixels=1,
            opacity=0.75,
        )
        layers.append(medium_layer)
    
    # High risk layer (top - most visible)
    if high_points:
        high_layer = pdk.Layer(
            "ScatterplotLayer",
            data=high_points,
            get_position="position",
            get_fill_color="fill_color",
            get_line_color="line_color",
            get_radius="radius_m",
            radius_min_pixels=5,
            radius_max_pixels=45,
            radius_scale=1,
            pickable=True,
            stroked=True,
            line_width_min_pixels=1,
            opacity=0.8,
        )
        layers.append(high_layer)
    
    if not layers:
        st.markdown("<div style='padding:20px; color:#94a3b8;'>No map data to display.</div>", unsafe_allow_html=True)
        if not skip_header:
            st.markdown("</div>", unsafe_allow_html=True)
        return
    
    # Tooltip configuration
    tooltip = {
        "html": "{tooltip}",
        "style": {
            "backgroundColor": "rgba(15,15,15,0.95)",
            "color": "white",
            "font-family": "Inter, Arial, sans-serif",
            "font-size": "13px",
            "padding": "8px",
            "borderRadius": "6px",
            "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"
        }
    }
    
    # Create deck
    try:
        deck = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_level,
                pitch=0,
                bearing=0
            ),
            tooltip=tooltip,
            map_style='dark'
        )
        
        # Render map with reduced height for 40% column
        st.pydeck_chart(deck, use_container_width=True, height=200)
        
    except Exception as e:
        logger.error(f"Severity map rendering failed: {e}", exc_info=True)
        st.error(f"Map rendering failed: {e}")
        # Fallback to simple map
        try:
            map_df = pd.DataFrame([{"lat": p["lat"], "lon": p["lon"]} for p in map_points])
            st.map(map_df)
        except Exception as e2:
            st.error(f"Fallback map also failed: {e2}")
    
    # Add horizontal gradient divider between map and donut charts
    st.markdown("""
    <div style="
        height: 3px;
        background: linear-gradient(to right, #c3002f 0%, #7a0000 50%, #0b0f13 100%);
        margin: 20px 0 16px 0;
        border-radius: 2px;
        box-shadow: 0 1px 3px rgba(195, 0, 47, 0.3);
    "></div>
    """, unsafe_allow_html=True)
    
    # Add CSS to reduce spacing between labels and charts and hide any title elements
    st.markdown("""
    <style>
    /* Ensure both chart columns (Model and PFP) have IDENTICAL top alignment */
    div[data-testid="column"]:has([data-testid="stPlotlyChart"]) {
        padding-top: 0px !important;
        margin-top: 0px !important;
        vertical-align: top !important;
        align-items: flex-start !important;
    }
    /* Ensure all element containers in chart columns start at the same top position */
    div[data-testid="column"]:has([data-testid="stPlotlyChart"]) > div {
        padding-top: 0px !important;
        margin-top: 0px !important;
    }
    /* Ensure label containers in both columns have IDENTICAL top alignment - CRITICAL */
    div[data-testid="column"]:has([data-testid="stPlotlyChart"]) div[data-testid="stMarkdownContainer"]:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        vertical-align: top !important;
    }
    /* Ensure the actual label divs have identical positioning */
    div[data-testid="column"]:has([data-testid="stPlotlyChart"]) div[data-testid="stMarkdownContainer"]:first-child > div {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    /* Target specific label text containers to ensure exact alignment */
    div[data-testid="column"]:has([data-testid="stPlotlyChart"]) div[data-testid="stMarkdownContainer"]:first-child > div > div {
        margin-top: 0px !important;
        padding-top: 0px !important;
        line-height: 1.2 !important;
    }
    /* Ensure donut chart labels have identical positioning */
    .donut-chart-label {
        margin-top: 0px !important;
        margin-bottom: -6px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        vertical-align: top !important;
        line-height: 1.2 !important;
        font-size: 12px !important;
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    /* Ensure charts in both columns have IDENTICAL top margin */
    div[data-testid="stPlotlyChart"] {
        margin-top: -15px !important;
        padding-top: 0px !important;
    }
    /* Position PFP legend to align with Model legend */
    /* Model legend is at y=-0.18, chart height is 280px, so legend starts at ~230px from chart top */
    /* Chart has 15px top margin, plot area starts at ~15px, plot area is ~195px tall */
    /* Legend at y=-0.18 means it's 50.4px below plot bottom, so ~245px from chart container top */
    div[data-testid="column"]:has-text("By PFP") .pfp-legend {
        margin-top: -50px !important;
        position: relative;
    }
    /* Ensure PFP chart and legend are in same container for positioning */
    div[data-testid="column"]:has-text("By PFP") [data-testid="stPlotlyChart"] {
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    /* Hide any Plotly title elements that might show "undefined" */
    .js-plotly-plot .plotly .gtitle,
    .js-plotly-plot .plotly .gtitletext,
    .plotly .gtitle,
    .plotly .gtitletext {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    </style>
    <script>
    // Aggressively remove any text containing "undefined" from the page, especially in chart titles
    (function() {
        function removeUndefined() {
            try {
                // First, remove from all SVG text elements in Plotly charts (most common location)
                document.querySelectorAll('svg text, svg tspan, .js-plotly-plot text, .plotly text').forEach(el => {
                    if (el && el.textContent && typeof el.textContent === 'string') {
                        if (el.textContent.includes('undefined')) {
                            // Replace "undefined" with empty string, or hide if result is empty
                            const newText = el.textContent.replace(/\s*undefined\s*/g, ' ').trim();
                            if (newText) {
                                el.textContent = newText;
                            } else {
                                el.style.display = 'none';
                                el.style.visibility = 'hidden';
                                el.setAttribute('display', 'none');
                            }
                        }
                    }
                });
                
                // Remove from all title elements
                document.querySelectorAll('.gtitle, .gtitletext, [class*="title"]').forEach(el => {
                    if (el && el.textContent && typeof el.textContent === 'string') {
                        if (el.textContent.includes('undefined')) {
                            el.style.display = 'none';
                            el.style.visibility = 'hidden';
                            el.style.opacity = '0';
                            el.style.height = '0';
                            el.style.margin = '0';
                            el.style.padding = '0';
                        }
                    }
                });
                
                // Remove from all text nodes (but be careful with script/style tags)
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;
                            if (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            if (node.textContent && node.textContent.includes('undefined')) {
                                return NodeFilter.FILTER_ACCEPT;
                            }
                            return NodeFilter.FILTER_REJECT;
                        }
                    },
                    false
                );
                
                let node;
                const nodesToFix = [];
                while (node = walker.nextNode()) {
                    nodesToFix.push(node);
                }
                
                nodesToFix.forEach(textNode => {
                    if (textNode.textContent && textNode.textContent.includes('undefined')) {
                        const newText = textNode.textContent.replace(/\s*undefined\s*/g, ' ').trim();
                        textNode.textContent = newText;
                    }
                });
                
            } catch (e) {
                console.error('Error removing undefined:', e);
            }
        }
        
        // Run immediately when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', removeUndefined);
        } else {
            removeUndefined();
        }
        
        // Run repeatedly to catch dynamically rendered content
        setTimeout(removeUndefined, 10);
        setTimeout(removeUndefined, 50);
        setTimeout(removeUndefined, 100);
        setTimeout(removeUndefined, 200);
        setTimeout(removeUndefined, 500);
        setTimeout(removeUndefined, 1000);
        setTimeout(removeUndefined, 2000);
        
        // Run every 300ms to catch any dynamically rendered content
        const intervalId = setInterval(removeUndefined, 300);
        
        // Watch for DOM changes with MutationObserver
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver(function(mutations) {
                let shouldRun = false;
                mutations.forEach(function(mutation) {
                    if (mutation.addedNodes.length > 0 || mutation.type === 'characterData') {
                        shouldRun = true;
                    }
                });
                if (shouldRun) {
                    removeUndefined();
                }
            });
            
            observer.observe(document.body, { 
                childList: true, 
                subtree: true, 
                characterData: true,
                attributes: false
            });
        }
        
        // Also run when Plotly charts are rendered
        if (window.Plotly) {
            const originalNewPlot = window.Plotly.newPlot;
            window.Plotly.newPlot = function() {
                const result = originalNewPlot.apply(this, arguments);
                setTimeout(removeUndefined, 50);
                return result;
            };
        }
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Render donut charts for model and PFP in the same row with separator
    chart_col1, separator_col, chart_col2 = st.columns([1, 0.15, 1], gap="small")
    
    with chart_col1:
        render_model_distribution_chart(df_log, date_range, text_filter)
    
    with separator_col:
        # Professional vertical separator line - visible and centered
        st.markdown("""
        <div style="
            display: flex;
            justify-content: center;
            align-items: flex-start;
            width: 100%;
            padding-top: 30px;
        ">
            <div style="
                width: 4px;
                height: 280px;
                background: linear-gradient(to bottom, 
                    rgba(195, 0, 47, 0.2) 0%, 
                    rgba(195, 0, 47, 0.5) 15%, 
                    rgba(195, 0, 47, 0.8) 30%,
                    #c3002f 50%,
                    rgba(195, 0, 47, 0.8) 70%,
                    rgba(195, 0, 47, 0.5) 85%, 
                    rgba(195, 0, 47, 0.2) 100%);
                border-radius: 2px;
                box-shadow: 0 0 8px rgba(195, 0, 47, 0.6), 
                            inset 0 0 3px rgba(0, 0, 0, 0.4);
            "></div>
        </div>
        """, unsafe_allow_html=True)
    
    with chart_col2:
        # Render PFP chart directly (wrapper removed to ensure identical structure to Model chart)
        render_pfp_distribution_chart(df_log, date_range, text_filter)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_model_distribution_chart(df_log: pd.DataFrame, date_range=None, text_filter: str = ""):
    """
    Render a professional donut chart showing inference count by model.
    
    Args:
        df_log: Full inference log DataFrame
        date_range: Tuple of (start_date, end_date) or single date, or None
        text_filter: Optional text filter for model/part
    """
    # Add label matching Streamlit input label style with minimal spacing
    # Use identical styling for both labels to ensure perfect top alignment
    st.markdown(
        '<div class="donut-chart-label" style="font-size: 12px; color: #94a3b8; margin-top: 0px; margin-bottom: -6px; padding-top: 0px; padding-bottom: 0px; font-weight: 500; line-height: 1.2; vertical-align: top;">By Model</div>',
        unsafe_allow_html=True
    )
    
    if df_log.empty:
        return
    
    # Apply filters using shared function (same as map)
    df_filtered = _apply_filters(df_log, date_range, text_filter)
    
    if df_filtered.empty or "model" not in df_filtered.columns:
        return
    
    # Group by model and count
    model_counts = df_filtered["model"].value_counts().reset_index()
    model_counts.columns = ["model", "count"]
    # Filter out NaN/None values and fill any remaining NaN with "Unknown"
    model_counts = model_counts.dropna(subset=["model"])
    model_counts["model"] = model_counts["model"].fillna("Unknown").astype(str)
    model_counts = model_counts.sort_values("count", ascending=False)
    
    if model_counts.empty:
        return
    
    # Calculate percentages
    total_count = int(model_counts["count"].sum() or 0)
    if total_count == 0:
        return
    model_counts["percentage"] = (model_counts["count"] / total_count * 100).round(1)
    
    # Professional, modern color palette with excellent contrast and visual appeal
    # Using sophisticated color scheme that works well in dark theme
    professional_colors = [
        "#c3002f",  # Nissan Red (primary brand color)
        "#3b82f6",  # Vibrant Blue
        "#10b981",  # Emerald Green
        "#f59e0b",  # Amber Gold
        "#8b5cf6",  # Purple
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#f97316",  # Orange
        "#14b8a6",  # Teal
        "#6366f1",  # Indigo
        "#ef4444",  # Red
        "#84cc16",  # Lime
    ]
    
    # Cycle through colors if we have more models than colors
    colors = professional_colors * ((len(model_counts) // len(professional_colors)) + 1)
    colors = colors[:len(model_counts)]
    
    # Create donut chart using plotly with enhanced styling
    fig = go.Figure(data=[go.Pie(
        labels=model_counts["model"],
        values=model_counts["count"],
        hole=0.65,  # Slightly larger hole for more modern, elegant look
        marker=dict(
            colors=colors,
            line=dict(
                color="#0f172a",  # Deep dark border for contrast
                width=2.5  # Slightly thicker for better definition
            ),
            pattern=dict(
                shape="",  # Solid fill for cleaner look
            )
        ),
        textinfo="percent",  # Show percentage on slices
        textposition="outside",  # Position text outside for better visibility
        texttemplate="%{percent:.0%}",  # Format as percentage (multiplies by 100 and adds %)
        textfont=dict(
            size=11,
            color="#e2e8f0",  # Light gray text for good contrast on dark background
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
        ),
        insidetextorientation="auto",  # Auto-orientation for optimal text placement
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Count: %{value:,.0f}<br>" +
            "Percentage: %{percent:.1%}<br>" +
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.98)",  # Deep dark background
            bordercolor="rgba(195, 0, 47, 0.5)",  # Subtle red border
            font=dict(
                size=13,
                family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                color="#ffffff"
            ),
            align="left"
        ),
        rotation=0,  # Start from top
        sort=False  # Maintain data order
    )])
    
    # Update layout for professional dark theme with enhanced styling
    layout_dict = {
        "title": {
            "text": "",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 1, "color": "rgba(0,0,0,0)"}
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.18,
            "xanchor": "center",
            "x": 0.5,
            "font": {
                "size": 10,
                "color": "#94a3b8",
                "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
            },
            "itemclick": "toggleothers",
            "itemdoubleclick": "toggle",
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.05)",
            "borderwidth": 1,
            "itemwidth": 30,
            "tracegroupgap": 8
        },
        "margin": {"l": 10, "r": 10, "t": 15, "b": 70},
        "height": 280,
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "color": "#e2e8f0"
        },
        "hovermode": "closest",
        "uniformtext": {
            "mode": "hide",
            "minsize": 10  # Hide text on slices smaller than this size for cleaner look
        }
    }
    fig.update_layout(**layout_dict)
    
    # Double-check: ensure title text is absolutely empty
    if hasattr(fig.layout, 'title') and fig.layout.title:
        if hasattr(fig.layout.title, 'text'):
            fig.layout.title.text = ""
        try:
            fig.layout.title.font.size = 1
            fig.layout.title.font.color = "rgba(0,0,0,0)"
        except:
            pass
    
    # Add professional center annotation with enhanced styling
    total_display = int(total_count) if pd.notna(total_count) and total_count > 0 else 0
    fig.add_annotation(
        text=(
            f"<span style='font-size: 11px; color: #94a3b8; font-weight: 500; letter-spacing: 0.5px;'>TOTAL</span><br>" +
            f"<span style='font-size: 28px; color: #ffffff; font-weight: 700; line-height: 1.2;'>{total_display:,}</span><br>" +
            f"<span style='font-size: 10px; color: #64748b; font-weight: 400;'>Records</span>"
        ),
        x=0.5,
        y=0.5,
        font_size=14,
        font_color="#ffffff",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="center",
        bgcolor="rgba(15, 23, 42, 0.3)",
        bordercolor="rgba(195, 0, 47, 0.2)",
        borderwidth=1,
        borderpad=8
    )
    
    # Render the chart with config to prevent any title rendering
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={
            "displayModeBar": False,
            "staticPlot": False,
            "responsive": True
        }
    )


def render_pfp_distribution_chart(df_log: pd.DataFrame, date_range=None, text_filter: str = ""):
    """
    Render a professional donut chart showing inference count by Primary Failed Part (PFP).
    
    Args:
        df_log: Full inference log DataFrame
        date_range: Tuple of (start_date, end_date) or single date, or None
        text_filter: Optional text filter for model/part
    """
    # Add label matching Streamlit input label style with minimal spacing
    # Use IDENTICAL styling as "By Model" label for perfect top alignment
    st.markdown(
        '<div class="donut-chart-label" style="font-size: 12px; color: #94a3b8; margin-top: 0px; margin-bottom: -6px; padding-top: 0px; padding-bottom: 0px; font-weight: 500; line-height: 1.2; vertical-align: top;">By PFP</div>',
        unsafe_allow_html=True
    )
    
    if df_log.empty:
        return
    
    # Apply filters using shared function (same as map and model chart)
    df_filtered = _apply_filters(df_log, date_range, text_filter)
    
    if df_filtered.empty or "primary_failed_part" not in df_filtered.columns:
        return
    
    # Group by primary_failed_part and count
    pfp_counts = df_filtered["primary_failed_part"].value_counts().reset_index()
    pfp_counts.columns = ["part", "count"]
    # Filter out NaN/None values and fill any remaining NaN with "Unknown"
    pfp_counts = pfp_counts.dropna(subset=["part"])
    pfp_counts["part"] = pfp_counts["part"].fillna("Unknown").astype(str)
    pfp_counts = pfp_counts.sort_values("count", ascending=False)
    
    # Calculate total count from ALL parts BEFORE limiting to top 8
    # This ensures the total matches the model chart (all records with PFP value)
    total_count = int(pfp_counts["count"].sum() or 0)
    if total_count == 0:
        return
    
    # Limit to top 8 parts for better visualization (too many slices make chart cluttered)
    pfp_counts = pfp_counts.head(8)
    
    if pfp_counts.empty:
        return
    
    # Calculate percentages based on the full total (not just top 8)
    pfp_counts["percentage"] = (pfp_counts["count"] / total_count * 100).round(1)
    
    # Professional, modern color palette for parts - complementary to model colors
    # Using a slightly different but harmonious color scheme for visual distinction
    part_colors = [
        "#f59e0b",  # Amber Gold (warmer tone)
        "#3b82f6",  # Vibrant Blue
        "#10b981",  # Emerald Green
        "#c3002f",  # Nissan Red
        "#8b5cf6",  # Purple
        "#06b6d4",  # Cyan
        "#ec4899",  # Pink
        "#f97316",  # Orange
        "#14b8a6",  # Teal
        "#6366f1",  # Indigo
        "#ef4444",  # Red
        "#84cc16",  # Lime
    ]
    
    # Cycle through colors if we have more parts than colors
    colors = part_colors * ((len(pfp_counts) // len(part_colors)) + 1)
    colors = colors[:len(pfp_counts)]
    
    # Create donut chart using plotly with enhanced styling
    fig = go.Figure(data=[go.Pie(
        labels=pfp_counts["part"],
        values=pfp_counts["count"],
        hole=0.65,  # Slightly larger hole for more modern, elegant look
        marker=dict(
            colors=colors,
            line=dict(
                color="#0f172a",  # Deep dark border for contrast
                width=2.5  # Slightly thicker for better definition
            ),
            pattern=dict(
                shape="",  # Solid fill for cleaner look
            )
        ),
        textinfo="percent",  # Show percentage on slices
        textposition="outside",  # Position text outside for better visibility
        texttemplate="%{percent:.0%}",  # Format as percentage (multiplies by 100 and adds %)
        textfont=dict(
            size=10,  # Slightly smaller for part names (often longer)
            color="#e2e8f0",  # Light gray text for good contrast on dark background
            family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"
        ),
        insidetextorientation="auto",  # Auto-orientation for optimal text placement
        hovertemplate=(
            "<b>%{label}</b><br>" +
            "Count: %{value:,.0f}<br>" +
            "Percentage: %{percent:.1%}<br>" +
            "<extra></extra>"
        ),
        hoverlabel=dict(
            bgcolor="rgba(15, 23, 42, 0.98)",  # Deep dark background
            bordercolor="rgba(195, 0, 47, 0.5)",  # Subtle red border
            font=dict(
                size=13,
                family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                color="#ffffff"
            ),
            align="left"
        ),
        rotation=0,  # Start from top
        sort=False  # Maintain data order
    )])
    
    # Update layout for professional dark theme with enhanced styling
    layout_dict = {
        "title": {
            "text": "",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 1, "color": "rgba(0,0,0,0)"}
        },
        "showlegend": False,  # Disable default legend - using custom multi-column legend instead
        "margin": {"l": 10, "r": 10, "t": 15, "b": 70},  # Match Model chart - 70px bottom margin for legend space
        "height": 280,
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {
            "family": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "color": "#e2e8f0"
        },
        "hovermode": "closest",
        "uniformtext": {
            "mode": "hide",
            "minsize": 10  # Hide text on slices smaller than this size for cleaner look
        }
    }
    fig.update_layout(**layout_dict)
    
    # Double-check: ensure title text is absolutely empty
    if hasattr(fig.layout, 'title') and fig.layout.title:
        if hasattr(fig.layout.title, 'text'):
            fig.layout.title.text = ""
        try:
            fig.layout.title.font.size = 1
            fig.layout.title.font.color = "rgba(0,0,0,0)"
        except:
            pass
    
    # Add professional center annotation with enhanced styling
    total_display = int(total_count) if pd.notna(total_count) and total_count > 0 else 0
    fig.add_annotation(
        text=(
            f"<span style='font-size: 11px; color: #94a3b8; font-weight: 500; letter-spacing: 0.5px;'>TOTAL</span><br>" +
            f"<span style='font-size: 28px; color: #ffffff; font-weight: 700; line-height: 1.2;'>{total_display:,}</span><br>" +
            f"<span style='font-size: 10px; color: #64748b; font-weight: 400;'>Records</span>"
        ),
        x=0.5,
        y=0.5,
        font_size=14,
        font_color="#ffffff",
        showarrow=False,
        xref="paper",
        yref="paper",
        align="center",
        bgcolor="rgba(15, 23, 42, 0.3)",
        bordercolor="rgba(195, 0, 47, 0.2)",
        borderwidth=1,
        borderpad=8
    )
    
    # Render the chart with config to prevent any title rendering
    st.plotly_chart(
        fig, 
        use_container_width=True, 
        config={
            "displayModeBar": False,
            "staticPlot": False,
            "responsive": True
        }
    )
    
    # Add JavaScript to align labels and remove spacing after render
    st.markdown("""
    <script>
    (function() {
        function alignLabelsAndReduceSpacing() {
            try {
                // Find all columns in the donut chart row
                const columns = Array.from(document.querySelectorAll('[data-testid="column"]'));
                
                // Find the columns containing the charts by looking for Plotly charts
                let modelColumn = null;
                let pfpColumn = null;
                let modelChart = null;
                let pfpChart = null;
                
                columns.forEach(col => {
                    const chart = col.querySelector('[data-testid="stPlotlyChart"]');
                    if (chart) {
                        // Check if this column has "By Model" or "By PFP" label
                        const labelText = col.textContent || '';
                        if (labelText.includes('By Model')) {
                            modelColumn = col;
                            modelChart = chart;
                        } else if (labelText.includes('By PFP')) {
                            pfpColumn = col;
                            pfpChart = chart;
                        }
                    }
                });
                
                // If we found both columns, align the labels to be top-aligned
                if (modelColumn && pfpColumn) {
                    // Get ALL markdown containers and find the ones containing "By Model" and "By PFP"
                    const modelLabelContainers = Array.from(modelColumn.querySelectorAll('div[data-testid="stMarkdownContainer"]'));
                    const pfpLabelContainers = Array.from(pfpColumn.querySelectorAll('div[data-testid="stMarkdownContainer"]'));
                    
                    // Find the container with "By Model" text
                    let modelLabelContainer = null;
                    for (const container of modelLabelContainers) {
                        if (container.textContent && container.textContent.includes('By Model')) {
                            modelLabelContainer = container;
                            break;
                        }
                    }
                    
                    // Find the container with "By PFP" text
                    let pfpLabelContainer = null;
                    for (const container of pfpLabelContainers) {
                        if (container.textContent && container.textContent.includes('By PFP')) {
                            pfpLabelContainer = container;
                            break;
                        }
                    }
                    
                    // If we found both labels, align them to EXACT same top position
                    if (modelLabelContainer && pfpLabelContainer) {
                        // Get column top positions
                        const modelColRect = modelColumn.getBoundingClientRect();
                        const pfpColRect = pfpColumn.getBoundingClientRect();
                        
                        // Get label top positions (top edge of the label element)
                        const modelLabelRect = modelLabelContainer.getBoundingClientRect();
                        const pfpLabelRect = pfpLabelContainer.getBoundingClientRect();
                        
                        // Calculate top positions relative to column tops
                        const modelLabelTop = modelLabelRect.top - modelColRect.top;
                        const pfpLabelTop = pfpLabelRect.top - pfpColRect.top;
                        
                        // Calculate adjustment needed to align tops exactly
                        const adjustment = modelLabelTop - pfpLabelTop;
                        
                        // Always apply alignment (even if difference is small) to ensure perfect alignment
                        if (Math.abs(adjustment) > 0.1) {
                            // Get computed styles from Model label
                            const modelStyles = window.getComputedStyle(modelLabelContainer);
                            
                            // Reset PFP label transform
                            pfpLabelContainer.style.transform = '';
                            pfpLabelContainer.style.position = 'relative';
                            
                            // Apply exact same styles from Model label to PFP label
                            pfpLabelContainer.style.marginTop = modelStyles.marginTop;
                            pfpLabelContainer.style.paddingTop = modelStyles.paddingTop;
                            pfpLabelContainer.style.paddingBottom = modelStyles.paddingBottom;
                            pfpLabelContainer.style.marginBottom = modelStyles.marginBottom;
                            pfpLabelContainer.style.verticalAlign = modelStyles.verticalAlign || 'top';
                            
                            // Also copy line-height and other text properties
                            pfpLabelContainer.style.lineHeight = modelStyles.lineHeight || '1.2';
                            pfpLabelContainer.style.fontSize = modelStyles.fontSize || '12px';
                            
                            // Apply fine-tuned adjustment if still needed
                            if (Math.abs(adjustment) > 0.5) {
                                const currentMarginTop = parseFloat(window.getComputedStyle(pfpLabelContainer).marginTop) || 0;
                                pfpLabelContainer.style.marginTop = `${currentMarginTop + adjustment}px`;
                            }
                        }
                        
                        // Also ensure parent containers are aligned
                        const modelParent = modelLabelContainer.parentElement;
                        const pfpParent = pfpLabelContainer.parentElement;
                        if (modelParent && pfpParent) {
                            const modelParentStyles = window.getComputedStyle(modelParent);
                            pfpParent.style.paddingTop = modelParentStyles.paddingTop;
                            pfpParent.style.marginTop = modelParentStyles.marginTop;
                        }
                    }
                    
                    // Align chart positions - use EXACT same spacing as Model chart
                    if (modelChart && pfpChart) {
                        const modelChartStyles = window.getComputedStyle(modelChart);
                        // Apply same top margin as model chart
                        pfpChart.style.marginTop = modelChartStyles.marginTop;
                        pfpChart.style.paddingTop = modelChartStyles.paddingTop;
                    }
                }
                
                // Align PFP legend to be TOP-ALIGNED with Model legend
                const pfpLegend = document.querySelector('.pfp-legend');
                if (pfpLegend && pfpChart && modelChart) {
                    // Find Model legend - it may take time to render
                    const findAndAlign = () => {
                        const modelLegend = modelChart.querySelector('.legend, .legendgroup, .js-plotly-plot .legend');
                        if (!modelLegend) {
                            setTimeout(findAndAlign, 50);
                            return;
                        }
                        
                        // Get column containers
                        const modelColumn = modelChart.closest('[data-testid="column"]');
                        const pfpColumn = pfpChart.closest('[data-testid="column"]');
                        if (!modelColumn || !pfpColumn) return;
                        
                        // Get bounding rectangles
                        const modelColRect = modelColumn.getBoundingClientRect();
                        const pfpColRect = pfpColumn.getBoundingClientRect();
                        const modelLegendRect = modelLegend.getBoundingClientRect();
                        const pfpLegendRect = pfpLegend.getBoundingClientRect();
                        
                        // Calculate top positions relative to their columns
                        const modelTop = modelLegendRect.top - modelColRect.top;
                        const pfpTop = pfpLegendRect.top - pfpColRect.top;
                        
                        // Calculate adjustment needed
                        const adjustment = modelTop - pfpTop;
                        
                        // Apply adjustment if significant
                        if (Math.abs(adjustment) > 1) {
                            const baseMargin = -50;
                            pfpLegend.style.marginTop = `${baseMargin + adjustment}px`;
                            pfpLegend.style.transform = '';
                        }
                    };
                    
                    // Start finding and aligning
                    findAndAlign();
                }
            } catch(e) {
                console.error('Error aligning labels:', e);
            }
        }
        
        // Dedicated function to align PFP legend with Model legend - robust version
        function alignPfpLegendRobust() {
            try {
                // Find all columns and charts
                const columns = Array.from(document.querySelectorAll('[data-testid="column"]'));
                let modelColumn = null;
                let pfpColumn = null;
                let modelChart = null;
                let pfpChart = null;
                let pfpLegend = null;
                
                columns.forEach(col => {
                    const text = col.textContent || '';
                    const chart = col.querySelector('[data-testid="stPlotlyChart"]');
                    
                    if (text.includes('By Model') && chart) {
                        modelColumn = col;
                        modelChart = chart;
                    } else if (text.includes('By PFP') && chart) {
                        pfpColumn = col;
                        pfpChart = chart;
                        pfpLegend = col.querySelector('.pfp-legend');
                    }
                });
                
                if (!modelColumn || !pfpColumn || !modelChart || !pfpChart || !pfpLegend) {
                    return false;
                }
                
                // Find Model legend with retry logic
                const modelLegend = modelChart.querySelector('.legend, .legendgroup, .js-plotly-plot .legend, .plotly .legend');
                if (!modelLegend) {
                    return false; // Legend not rendered yet
                }
                
                // Get all bounding rectangles
                const modelColRect = modelColumn.getBoundingClientRect();
                const pfpColRect = pfpColumn.getBoundingClientRect();
                const modelLegendRect = modelLegend.getBoundingClientRect();
                const pfpLegendRect = pfpLegend.getBoundingClientRect();
                
                // Calculate top positions relative to column tops
                const modelLegendTop = modelLegendRect.top - modelColRect.top;
                const pfpLegendTop = pfpLegendRect.top - pfpColRect.top;
                
                // Calculate adjustment needed
                const adjustment = modelLegendTop - pfpLegendTop;
                
                // Apply adjustment if significant (more than 2px difference)
                if (Math.abs(adjustment) > 2) {
                    // Reset transform
                    pfpLegend.style.transform = '';
                    // Apply new margin (base -50px + adjustment)
                    const newMargin = -50 + adjustment;
                    pfpLegend.style.marginTop = `${newMargin}px`;
                    return true; // Alignment applied
                }
                
                return false; // Already aligned
            } catch(e) {
                console.error('Error in alignPfpLegendRobust:', e);
                return false;
            }
        }
        
        // Run alignment with retries - enhanced version
        function runAlignmentWithRetries() {
            // First align labels
            alignLabelsAndReduceSpacing();
            
            // Then align legend with retries
            let retries = 0;
            const maxRetries = 30;
            const tryAlign = () => {
                const legendAligned = alignPfpLegendRobust();
                if (legendAligned || retries >= maxRetries) {
                    return; // Success or max retries reached
                }
                retries++;
                setTimeout(tryAlign, 50); // Faster retry for labels
            };
            tryAlign();
        }
        
        // Initial run immediately
        runAlignmentWithRetries();
        
        // Run at various intervals to catch all render states
        [50, 100, 200, 400, 600, 1000, 1500, 2000, 3000, 5000].forEach(delay => {
            setTimeout(runAlignmentWithRetries, delay);
        });
        
        // Watch for DOM changes - more aggressive
        if (typeof MutationObserver !== 'undefined') {
            const observer = new MutationObserver(() => {
                // Debounce to avoid too many calls
                clearTimeout(window.alignmentTimeout);
                window.alignmentTimeout = setTimeout(runAlignmentWithRetries, 50);
            });
            observer.observe(document.body, { 
                childList: true, 
                subtree: true,
                attributes: true,
                attributeFilter: ['style', 'class']
            });
        }
        
        // Listen for Plotly render events
        if (window.Plotly) {
            const originalNewPlot = window.Plotly.newPlot;
            window.Plotly.newPlot = function() {
                const result = originalNewPlot.apply(this, arguments);
                setTimeout(runAlignmentWithRetries, 100);
                setTimeout(runAlignmentWithRetries, 300);
                return result;
            };
        }
        
        // Also listen for window load and resize
        window.addEventListener('load', runAlignmentWithRetries);
        window.addEventListener('resize', () => {
            setTimeout(runAlignmentWithRetries, 100);
        });
    })();
    </script>
    """, unsafe_allow_html=True)
    
    # Create custom multi-column legend below the chart using CSS Grid
    st.markdown("""
    <style>
    .pfp-legend {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 6px 12px;
        margin-top: -50px !important;
        margin-bottom: 10px;
        padding: 0px 0;
        font-size: 10px;
        color: #94a3b8;
        font-family: Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif;
        position: relative;
    }
    .pfp-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .pfp-legend-color {
        width: 12px;
        height: 12px;
        border-radius: 2px;
        flex-shrink: 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Build legend HTML with colors matching the chart
    legend_items = []
    for i, (idx, row) in enumerate(pfp_counts.iterrows()):
        color = colors[i % len(colors)]  # Use enumerate index to match chart colors
        part_name = row['part']
        legend_items.append(
            f'<div class="pfp-legend-item">'
            f'<div class="pfp-legend-color" style="background-color: {color};"></div>'
            f'<span>{part_name}</span>'
            f'</div>'
        )
    
    # Render custom multi-column legend
    if legend_items:
        st.markdown(
            f'<div class="pfp-legend">{"".join(legend_items)}</div>',
            unsafe_allow_html=True
        )

