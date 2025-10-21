"""
components_col2.py

Modular components for Column 2 (Predictive & Prescriptive Analysis)
Extracted from the monolithic render_col2() function for better maintainability.
"""

import streamlit as st
import pandas as pd
import pydeck as pdk
import html as _html
import math
import os
from typing import Dict, List, Tuple, Optional

from config import config
from utils.logger import components_logger as logger

from helper import (
    load_model,
    random_inference_row_from_df,
    append_inference_log,
    append_inference_log_s3,
    fetch_nearest_dealers,
    reverse_geocode
)


# Module-level constants (imported from config in parent)
def get_constants():
    """Get constants from config"""
    return {
        'NISSAN_RED': config.colors.nissan_red,
        'NISSAN_GOLD': config.colors.nissan_gold,
        'MODEL_PATH': config.paths.model_path,
        'IS_POC': config.ui.is_poc,
        'USE_S3': config.aws.use_s3,
        'S3_BUCKET': config.aws.s3_bucket,
        'LOG_FILE_S3_KEY': config.paths.inference_log_s3_key,
        'LOG_FILE_LOCAL': config.paths.inference_log_local,
        'PLACE_INDEX_NAME': config.aws.place_index_name,
        'AWS_REGION': config.aws.region,
    }


def render_predictive_controls():
    """
    Render predictive analysis header with controls (refresh interval and threshold).
    Returns: refresh_interval (seconds)
    """
    constants = get_constants()
    
    header_col, slider_col, dropdown_col = st.columns([2.75, 1.25, 1], gap="medium")
    
    with header_col:
        st.markdown(
            '<div class="card-header" style="height:34px; display:flex; align-items:center;">Predictive Analysis</div>',
            unsafe_allow_html=True
        )
    
    # Refresh interval slider
    interval_map = config.ui.refresh_intervals
    if "predictive_interval_label" not in st.session_state:
        st.session_state.predictive_interval_label = config.ui.default_refresh_interval
    
    with slider_col:
        st.select_slider(
            "⏱️",
            options=list(interval_map.keys()),
            key="predictive_interval_label",
            format_func=lambda x: f"⏱ {x}",
            label_visibility="collapsed",
            help="Inference data generation interval",
        )
    
    # Threshold dropdown
    threshold_options = config.model.threshold_options
    if "predictive_threshold_pct" not in st.session_state:
        st.session_state.predictive_threshold_pct = config.model.default_threshold_pct
    
    with dropdown_col:
        st.selectbox(
            "⚠️",
            options=threshold_options,
            key="predictive_threshold_pct",
            label_visibility="collapsed",
            help="Show actionable summary when predicted claim probability ≥ this value",
        )
    
    refresh_interval = interval_map.get(st.session_state.predictive_interval_label, 900)
    return refresh_interval


def generate_prediction(df_history: pd.DataFrame, model_pipe) -> Tuple[Dict, float]:
    """
    Generate a prediction for a random inference row.
    
    Args:
        df_history: Historical data DataFrame
        model_pipe: Loaded ML model pipeline (or None)
    
    Returns:
        Tuple of (inference_row dict, predicted_probability float)
    """
    inf_row = random_inference_row_from_df(df_history)
    logger.debug(f"Generated random inference row: {inf_row.get('model')}/{inf_row.get('primary_failed_part')}")
    
    pred_prob = None
    if model_pipe is not None:
        try:
            test_df = pd.DataFrame([{
                "model": inf_row["model"],
                "primary_failed_part": inf_row["primary_failed_part"],
                "mileage": float(inf_row["mileage"]),
                "age": float(inf_row["age"]),
            }])
            pred_val = model_pipe.predict(test_df)[0]
            pred_prob = float(max(0.0, min(1.0, pred_val)))
            logger.info(f"[OK] ML prediction: {pred_prob*100:.1f}% (mileage={inf_row['mileage']:.1f}, age={inf_row['age']:.2f})")
        except Exception as e:
            logger.warning(f"ML prediction failed, using fallback: {e}")
            pred_prob = None
    
    # Fallback prediction logic if model unavailable (using continuous values)
    if pred_prob is None:
        def safe_sorted_unique(series):
            vals = [v for v in pd.unique(series) if pd.notna(v)]
            vals = [str(v) for v in vals]
            return sorted(vals, key=lambda s: s.lower())
        
        # Base risk by model
        model_risk = {"Leaf": 0.03, "Ariya": 0.04, "Sentra": 0.06}
        
        # Part severity mapping
        unique_parts = safe_sorted_unique(df_history["primary_failed_part"])
        part_risk = {p: 0.03 + (idx % 5) * 0.01 for idx, p in enumerate(unique_parts)}
        
        # Normalize continuous values to 0-1 scale
        mileage_normalized = inf_row["mileage"] / 150000.0  # Max 150k miles
        age_normalized = inf_row["age"] / 15.0  # Max 15 years
        
        # Calculate risk with continuous effects
        base = (
            0.5 * model_risk.get(inf_row["model"], 0.04)
            + 0.8 * (0.02 + 0.08 * mileage_normalized)  # Linear mileage effect
            + 0.6 * part_risk.get(inf_row["primary_failed_part"], 0.03)
            + 0.3 * (0.02 + 0.06 * age_normalized)  # Linear age effect
        )
        pred_prob = float(max(0.0, min(0.99, base)))
        logger.info(f"Using fallback prediction: {pred_prob*100:.1f}% (mileage={inf_row['mileage']:.1f}, age={inf_row['age']:.2f})")
    
    return inf_row, pred_prob


def log_inference(inf_row: Dict, pred_prob: float) -> bool:
    """
    Log inference to S3 or local file.
    
    Args:
        inf_row: Inference row dictionary
        pred_prob: Predicted probability
    
    Returns:
        bool indicating if logging was successful
    """
    constants = get_constants()
    
    try:
        if constants['USE_S3']:
            try:
                appended = append_inference_log_s3(
                    inf_row, pred_prob,
                    s3_bucket=constants['S3_BUCKET'],
                    s3_key=constants['LOG_FILE_S3_KEY'],
                    local_fallback=constants['LOG_FILE_LOCAL']
                )
                if appended:
                    logger.debug(f"Inference logged to S3: {inf_row.get('model')}/{inf_row.get('primary_failed_part')}")
            except Exception as e:
                logger.warning(f"S3 logging failed, using local: {e}")
                appended = append_inference_log(inf_row, pred_prob, filepath=constants['LOG_FILE_LOCAL'])
        else:
            appended = append_inference_log(inf_row, pred_prob, filepath=constants['LOG_FILE_LOCAL'])
            if appended:
                logger.debug("Inference logged locally")
        return appended
    except Exception as e:
        logger.error(f"Inference logging failed: {e}")
        return False


def render_vehicle_info_left(inf_row: Dict):
    """
    Render left side of vehicle information (model and part).
    This function is called inside a column context from app.py.
    
    Args:
        inf_row: Inference row dictionary
    """
    constants = get_constants()
    
    if constants['IS_POC']:
        st.markdown(
            """<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Model</div>
               <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">Sentra</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            """<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Part</div>
               <div style="font-weight:700; font-size:14px; color:#e6eef8;">Engine Coolant System</div>""",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Model</div>
                <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">{inf_row['model']}</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Part</div>
                <div style="font-weight:700; font-size:14px; color:#e6eef8;">{inf_row['primary_failed_part']}</div>""",
            unsafe_allow_html=True
        )


def render_vehicle_info_right(inf_row: Dict):
    """
    Render right side of vehicle information (mileage and age).
    This function is called inside a column context from app.py.
    
    Args:
        inf_row: Inference row dictionary
    """
    constants = get_constants()
    
    if constants['IS_POC']:
        st.markdown(
            """<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Mileage</div>
               <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">10,200 miles</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            """<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Age</div>
               <div style="font-weight:700; font-size:14px; color:#e6eef8;">6 months</div>""",
            unsafe_allow_html=True
        )
    else:
        # Format continuous values for display
        mileage_display = f"{inf_row['mileage']:,.1f} mi"
        age_display = f"{inf_row['age']:.1f} yrs"
        
        st.markdown(
            f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Mileage</div>
                <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">{mileage_display}</div>""",
            unsafe_allow_html=True
        )
        st.markdown(
            f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Age</div>
                <div style="font-weight:700; font-size:14px; color:#e6eef8;">{age_display}</div>""",
            unsafe_allow_html=True
        )


def render_divider():
    """
    Render vertical divider between vehicle info and prediction KPI.
    This function is called inside a column context from app.py.
    """
    st.markdown(
        """
        <div style="display:flex; align-items:center; height:100%; justify-content:center;">
            <div style="
                width:3px; 
                height:86px;
                border-radius:3px;
                background:linear-gradient(to bottom, rgba(255,255,255,0.35), rgba(255,255,255,0.08));
                box-shadow:0 0 8px rgba(255,255,255,0.15);
            ">
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_prediction_kpi(pred_prob: float):
    """
    Render the predicted claim probability KPI.
    This renders directly in the KPI column (must be called inside the right column context).
    
    Args:
        pred_prob: Predicted probability (0.0 to 1.0)
    """
    constants = get_constants()
    pct_text = f"{pred_prob*100:.1f}%"
    
    if constants['IS_POC']:
        pct_text = f"{80.8:.1f}%"
    
    st.markdown(
        f"""
        <div style="padding:10px; border-radius:8px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
        <div style="text-align:center;">
            <div class="kpi-label">Predicted Claim Probability</div>
            <div class="kpi-wrap" style="margin-top:6px;">
            <div class="kpi-num" style="color:{constants['NISSAN_RED']}; font-size:34px; line-height:1;">{pct_text}</div>
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-top:6px;">
                Alert if ≥ {int(st.session_state.get('predictive_threshold_pct', 80))}%
            </div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prescriptive_section(inf_row: Dict, pred_prob: float, render_summary_ui_func):
    """
    Render prescriptive summary section with dealer recommendations.
    
    Args:
        inf_row: Inference row dictionary
        pred_prob: Predicted probability (0.0 to 1.0)
        render_summary_ui_func: Function to render prescriptive summary UI
    """
    constants = get_constants()
    
    st.markdown(
        '<div class="card-header" style="height:34px; display:flex; align-items:center;">Prescriptive Summary</div>',
        unsafe_allow_html=True
    )
    
    # Fetch nearest dealers
    current_lat = inf_row.get("lat", 38.4405)
    current_lon = inf_row.get("lon", -122.7144)
    
    logger.debug(f"Fetching nearest dealers for location: lat={current_lat}, lon={current_lon}")
    
    try:
        nearest, from_aws = fetch_nearest_dealers(
            current_lat=current_lat,
            current_lon=current_lon,
            place_index_name=constants['PLACE_INDEX_NAME'],
            aws_region=constants['AWS_REGION'],
            fallback_dealers=None,
            text_query="Nissan Service Center",
            top_n=3,
        )
        logger.info(f"Found {len(nearest)} dealers (from_aws={from_aws})")
    except Exception as e:
        logger.error(f"fetch_nearest_dealers failed: {e}", exc_info=True)
        if config.debug:
            st.write("Debug: fetch_nearest_dealers failed:", e)
        nearest = []
    
    # Render summary
    if constants['IS_POC']:
        render_summary_ui_func(
            'Sentra',
            'Engine Cooling System',
            '10200 miles',
            '6 months',
            80.8
        )
    else:
        dealer_name = nearest[0]['name'] if nearest else "N/A"
        # Format continuous values for Bedrock summary
        mileage_display = f"{inf_row['mileage']:,.0f} miles"
        age_display = f"{inf_row['age']:.1f} years"
        
        render_summary_ui_func(
            inf_row['model'],
            inf_row['primary_failed_part'],
            mileage_display,
            age_display,
            pred_prob * 100,
            dealer_name
        )
    
    return nearest


def build_map_points(inf_row: Dict, nearest_dealers: List[Dict], pred_prob: float) -> Tuple[List[Dict], List[Dict]]:
    """
    Build map points for dealers and current vehicle location.
    
    Args:
        inf_row: Inference row dictionary
        nearest_dealers: List of nearest dealer dictionaries
        pred_prob: Predicted probability
    
    Returns:
        Tuple of (all_map_points, dealers_to_plot)
    """
    constants = get_constants()
    MAX_KM_20_MILES = config.data.max_dealer_distance_km
    
    # Filter dealers within configured distance
    nearby_dealers = [d for d in nearest_dealers if d.get("distance_km", 99999) <= MAX_KM_20_MILES]
    dealers_to_plot = nearby_dealers if nearby_dealers else nearest_dealers[:3]
    
    map_points = []
    
    # Add dealer points (red)
    if constants['IS_POC']:
        tooltip_html = (
            f"<div style='padding:8px; max-width:260px;'>"
            f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>United Nissan</div>"
            f"<div style='font-size:13px; color:#d1d5db; margin-bottom:6px;'>"
            f"United Nissan, 3025 E Sahara Ave, Las Vegas, NV 89104, United States</div>"
            f"<div style='font-size:12px; color:#94a3b8;'>Distance: 19 mi — ETA: 5 min</div>"
            f"</div>"
        )
        map_points.append({
            "name": "United Nissan",
            "short_name": "United Nissan",
            "lat": 36.1434,
            "lon": -115.108,
            "distance_km": 19,
            "eta_min": 5,
            "type": "Dealer",
            "tooltip": tooltip_html
        })
    else:
        for d in dealers_to_plot:
            try:
                lat = float(d.get("lat"))
                lon = float(d.get("lon"))
            except Exception:
                continue
            
            name_full = d.get("name", "Nissan Dealer")
            short_name = name_full.split(",")[0]
            dist_km = d.get("distance_km", "N/A")
            eta = d.get("eta_min", "N/A")
            
            tooltip_html = (
                f"<div style='padding:8px; max-width:260px;'>"
                f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>{_html.escape(short_name)}</div>"
                f"<div style='font-size:13px; color:#d1d5db; margin-bottom:6px;'>"
                f"{_html.escape(name_full)}</div>"
                f"<div style='font-size:12px; color:#94a3b8;'>Distance: {dist_km} km — ETA: {eta} min</div>"
                f"</div>"
            )
            
            map_points.append({
                "name": name_full,
                "short_name": short_name,
                "lat": lat,
                "lon": lon,
                "distance_km": dist_km,
                "eta_min": eta,
                "type": "Dealer",
                "tooltip": tooltip_html
            })
    
    # Add current vehicle point (green)
    current_lat = inf_row.get("lat", 38.4405)
    current_lon = inf_row.get("lon", -122.7144)
    
    # Resolve place name
    place_name = reverse_geocode(current_lat, current_lon)
    inf_row["place_name"] = place_name or "Current Location"
    
    if constants['IS_POC']:
        vehicle_tooltip = (
            f"<div style='padding:8px; max-width:260px;'>"
            f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>Current Location</div>"
            f"<div style='font-size:13px; color:#d1d5db; margin-bottom:6px;'>73WJ+PP2 North Las Vegas, Nevada, USA</div>"
            f"<div style='font-size:12px; color:#94a3b8; line-height:1.45;'>"
            f"<b>Vehicle:</b> Sentra<br/>"
            f"<b>Part:</b> Engine Coolant System<br/>"
            f"<b>Mileage:</b> 10,200 miles<br/>"
            f"<b>Age:</b> 6 months<br/>"
            f"<b>Claim Prob:</b> 80.8%"
        )
        current_lat = 36.20
        current_lon = -115.05
    else:
        # Format continuous values for tooltip
        mileage_display = f"{inf_row.get('mileage', 0):,.1f} miles"
        age_display = f"{inf_row.get('age', 0):.1f} years"
        
        vehicle_tooltip = (
            f"<div style='padding:8px; max-width:260px;'>"
            f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>Current Location</div>"
            f"<div style='font-size:13px; color:#d1d5db; margin-bottom:6px;'>{_html.escape(inf_row.get('place_name','Current Location'))}</div>"
            f"<div style='font-size:12px; color:#94a3b8; line-height:1.45;'>"
            f"<b>Vehicle:</b> {_html.escape(str(inf_row.get('model','N/A')))}<br/>"
            f"<b>Part:</b> {_html.escape(str(inf_row.get('primary_failed_part','N/A')))}<br/>"
            f"<b>Mileage:</b> {_html.escape(mileage_display)}<br/>"
            f"<b>Age:</b> {_html.escape(age_display)}<br/>"
            f"<b>Claim Prob:</b> {pred_prob*100:.1f}%"
            f"</div></div>"
        )
    
    display_label = inf_row.get("place_name", "Current Location").split(",")[0]
    
    map_points.append({
        "name": "Current Location",
        "short_name": display_label,
        "lat": float(current_lat),
        "lon": float(current_lon),
        "type": "Current",
        "tooltip": vehicle_tooltip
    })
    
    # Add helper fields for pydeck
    for p in map_points:
        p["position"] = [p["lon"], p["lat"]]
        p["_radius_m"] = 900 if p.get("type") == "Current" else 520
        if p.get("type") == "Current":
            p["_fill_color"] = [29, 158, 106, 230]  # green
            p["_line_color"] = [8, 62, 40, 200]
        else:
            p["_fill_color"] = [211, 47, 47, 220]   # red
            p["_line_color"] = [90, 20, 20, 200]
    
    return map_points, dealers_to_plot


def compute_map_center_and_zoom(map_points: List[Dict], viewport_width_px: int = 520) -> Tuple[float, float, int]:
    """
    Compute optimal map center and zoom level for given points.
    
    Args:
        map_points: List of map point dictionaries with lat/lon
        viewport_width_px: Viewport width in pixels
    
    Returns:
        Tuple of (center_lat, center_lon, zoom_level)
    """
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.asin(math.sqrt(a))
    
    lats = [p["lat"] for p in map_points]
    lons = [p["lon"] for p in map_points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    
    d1 = _haversine_km(min_lat, min_lon, max_lat, max_lon)
    max_dist_km = max(d1, 0.0)
    
    if max_dist_km < 0.02:  # Lower threshold for wider view
        return center_lat, center_lon, 9
    
    max_dist_m = max_dist_km * 1000.0
    meters_per_pixel_required = max_dist_m / (viewport_width_px * 0.85)
    lat_rad = math.radians(center_lat)
    meters_per_pixel_at_zoom0 = 156543.03392 * math.cos(lat_rad)
    
    try:
        zoom = math.log2(meters_per_pixel_at_zoom0 / meters_per_pixel_required)
        zoom = max(6, min(10, int(round(zoom - 2))))  # Zoom OUT by 2 levels for wider view
    except Exception:
        zoom = 8  # Lower default zoom for wider view
    
    return center_lat, center_lon, zoom


def render_map_visualization(map_points: List[Dict], dealers_to_plot: List[Dict]):
    """
    Render the interactive map with dealers and current vehicle location.
    
    Args:
        map_points: List of all map points (dealers + current)
        dealers_to_plot: List of dealers to show in legend/expander
    """
    constants = get_constants()
    
    st.markdown(
        '<div class="card" style="margin-top:-5px;"><div class="card-header">Vehicle Location & Nearest Dealers</div>',
        unsafe_allow_html=True
    )
    
    # Show warning if no nearby dealers
    MAX_KM_20_MILES = config.data.max_dealer_distance_km
    has_nearby = any(d.get("type") == "Dealer" for d in map_points if d.get("distance_km", 99999) <= MAX_KM_20_MILES)
    
    if not has_nearby and dealers_to_plot:
        st.markdown(
            f"<div style='font-size:14px; color:{constants['NISSAN_GOLD']};'>No dealers found within 20 miles. Showing nearest results.</div>",
            unsafe_allow_html=True
        )
        st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    
    # Legend
    legend_html = """
    <div style="display:flex; gap:12px; align-items:center; margin-bottom:8px;">
      <div style="display:flex; gap:8px; align-items:center;">
        <div style="width:14px; height:14px; border-radius:50%; background:#1da05b; border:2px solid rgba(255,255,255,0.08);"></div>
        <div style="color:#94a3b8; font-size:13px;">Current vehicle</div>
      </div>
      <div style="display:flex; gap:8px; align-items:center;">
        <div style="width:14px; height:14px; border-radius:50%; background:#d32f2f; border:2px solid rgba(255,255,255,0.08);"></div>
        <div style="color:#94a3b8; font-size:13px;">Dealer (service center)</div>
      </div>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)
    
    # Compute map center and zoom
    center_lat, center_lon, zoom_level = compute_map_center_and_zoom(map_points, viewport_width_px=400)
    
    # Render map with pydeck
    try:
        dealers_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[p for p in map_points if p["type"] == "Dealer"],
            get_position="position",
            get_fill_color="_fill_color",
            get_radius="_radius_m",
            get_line_color="_line_color",
            pickable=True,
            stroked=True,
            radius_min_pixels=8,
            radius_max_pixels=60,
        )
        
        current_layer = pdk.Layer(
            "ScatterplotLayer",
            data=[p for p in map_points if p["type"] == "Current"],
            get_position="position",
            get_fill_color="_fill_color",
            get_radius="_radius_m",
            get_line_color="_line_color",
            pickable=True,
            stroked=True,
            radius_min_pixels=10,
            radius_max_pixels=80,
        )
        
        text_layer = pdk.Layer(
            "TextLayer",
            data=map_points,
            get_position="position",
            get_text="short_name",
            get_color=[230, 230, 230],
            get_size=16,
            get_angle=0,
            get_text_anchor="start",
            get_alignment_baseline="center",
            billboard=True,
            sizeUnits="pixels",
        )
        
        tooltip = {
            "html": "{tooltip}",
            "style": {
                "backgroundColor": "rgba(15,15,15,0.95)",
                "color": "white",
                "font-family": "Inter, Arial, sans-serif",
                "font-size": "13px",
                "padding": "6px"
            }
        }
        
        deck = pdk.Deck(
            layers=[dealers_layer, current_layer, text_layer],
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom_level,
                pitch=0
            ),
            tooltip=tooltip,
            map_style='dark'
        )
        st.pydeck_chart(deck, use_container_width=True, height=250)
    
    except Exception as e:
        st.write("Debug: pydeck failed, falling back to st.map — error:", e)
        map_df = pd.DataFrame([{"lat": p["lat"], "lon": p["lon"]} for p in map_points])
        st.map(map_df)
    
    st.markdown('<div style="height:-5px;"></div>', unsafe_allow_html=True)
    
    # Dealer list expander
    with st.expander("Nearest dealers", expanded=False):
        if not dealers_to_plot:
            st.markdown("<div style='color:#94a3b8;'>No dealers within 20 miles.</div>", unsafe_allow_html=True)
        for d in dealers_to_plot:
            st.markdown(
                f"- **{d.get('name','Nissan Dealer')}** — {d.get('distance_km','N/A')} km ({d.get('eta_min','N/A')} min). Phone: {d.get('phone','N/A')}"
            )
    
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_download_button():
    """Render download button for inference log if available."""
    constants = get_constants()
    
    if os.path.isfile(constants['LOG_FILE_LOCAL']):
        with open(constants['LOG_FILE_LOCAL'], "rb") as f:
            st.download_button(
                label="⬇️ Download Real-Time Vehicle Feed",
                data=f,
                file_name="inference_log.csv",
                mime="text/csv"
            )

