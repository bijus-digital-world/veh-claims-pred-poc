"""
Digital Twin Component
Renders a comprehensive vehicle Digital Twin dashboard showing:
- Vehicle Overview
- Health Score Gauge
- Key Sensor Snapshot
- DTC Fault Panel
- Recent Anomalies Timeline
- Component Health Cards (Battery, Engine, Brakes)
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import base64
import os
import textwrap

try:
	import config
except ImportError:
	config = None

logger = logging.getLogger(__name__)

# Color scheme matching the design
COLORS = {
	"healthy": "#10b981",  # Green
	"warning": "#f59e0b",  # Amber/Yellow
	"critical": "#ef4444",  # Red
	"normal": "#10b981",
	"bg_dark": "#0f172a",
	"text_light": "#e2e8f0",
	"text_muted": "#94a3b8",
	"card_bg": "#1e293b",
	"border": "#334155"
}

GLASS_CARD_STYLE = (
	"background: linear-gradient(165deg, rgba(17,24,39,0.95), rgba(10,15,25,0.82)); "
	"border: 1px solid rgba(148,163,184,0.18); "
	"box-shadow: 0 18px 35px rgba(2,6,23,0.55), inset 0 1px 0 rgba(255,255,255,0.05); "
	"backdrop-filter: blur(22px); "
	"-webkit-backdrop-filter: blur(22px); "
)


def _build_status_chip(label: str) -> str:
	"""Return a styled status chip HTML snippet."""
	if not label:
		label = "Status"
	label_lower = str(label).strip().lower()
	if any(token in label_lower for token in ["healthy", "normal", "good", "low"]):
		variant = "status-chip--healthy"
	elif any(token in label_lower for token in ["warn", "medium"]):
		variant = "status-chip--warning"
	else:
		variant = "status-chip--critical"
	return f'<div class="status-chip {variant}"><span class="status-dot"></span>{str(label).upper()}</div>'

def calculate_health_score(pred_prob_pct: float, battery_soc: Optional[float] = None, 
						  dtc_severity: Optional[str] = None) -> Tuple[float, str]:
	"""
	Calculate overall health score (0-100) and status.
	
	Args:
		pred_prob_pct: Prediction probability percentage (0-100)
		battery_soc: Battery state of charge (0-100)
		dtc_severity: DTC severity (LOW, MEDIUM, HIGH, CRITICAL)
	
	Returns:
		Tuple of (health_score, status_text)
	"""
	# Base score from prediction (inverse: high pred = low health)
	base_score = 100 - pred_prob_pct
	
	# Adjustments
	if battery_soc is not None:
		if battery_soc < 20:
			base_score -= 20
		elif battery_soc < 50:
			base_score -= 10
	
	if dtc_severity:
		severity_penalty = {"CRITICAL": -30, "HIGH": -20, "MEDIUM": -10, "LOW": -5}
		base_score += severity_penalty.get(dtc_severity, 0)
	
	# Clamp to 0-100
	health_score = max(0, min(100, base_score))
	
	# Determine status
	if health_score >= 70:
		status = "Healthy"
	elif health_score >= 40:
		status = "Warning"
	else:
		status = "Critical"
	
	return round(health_score, 1), status

def format_time_ago(timestamp: pd.Timestamp) -> str:
	"""Format timestamp as 'X days ago' or 'X hours ago'"""
	if pd.isna(timestamp):
		return "N/A"
	
	now = pd.Timestamp.now()
	delta = now - timestamp
	
	if delta.days > 0:
		return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
	elif delta.seconds >= 3600:
		hours = delta.seconds // 3600
		return f"{hours} hour{'s' if hours > 1 else ''} ago"
	elif delta.seconds >= 60:
		minutes = delta.seconds // 60
		return f"{minutes} min(s) ago"
	else:
		return "Just now"

def get_row_value(row: Dict, *keys, default=None):
	"""Return the first non-null value found in the provided keys."""
	for key in keys:
		if not key:
			continue
		val = row.get(key)
		if val is None:
			continue
		try:
			if pd.isna(val):
				continue
		except Exception:
			pass
		return val
	return default

def format_mileage_display(row: Dict) -> str:
	"""Format mileage by preferring inference mileage and falling back to historical bucket."""
	mileage_val = get_row_value(row, "mileage")
	if mileage_val is not None:
		try:
			mileage_int = int(float(mileage_val))
			return f"{mileage_int:,} miles"
		except Exception:
			pass
	
	mileage_bucket = get_row_value(row, "mileage_bucket")
	if mileage_bucket:
		mileage_bucket = str(mileage_bucket).strip()
		if mileage_bucket:
			suffix = "" if "mile" in mileage_bucket.lower() else " miles"
			return f"{mileage_bucket}{suffix}"
	return "N/A"

def join_telematics_with_inference(df_history: pd.DataFrame, df_inference: pd.DataFrame, 
								   vin: Optional[str] = None) -> pd.DataFrame:
	"""
	Join historical telematics data with inference log data using LEFT JOIN approach.
	
	LEFT JOIN: Start with telematics data (primary), optionally join inference data if available.
	This ensures all VINs with telematics data are shown, even if they don't have inference records.
	
	Args:
		df_history: Historical CSV with telematics data (LEFT side)
		df_inference: Inference log CSV with predictions (RIGHT side, optional)
		vin: Optional VIN to filter
	
	Returns:
		Merged DataFrame with combined data (telematics always present, inference if available)
	"""
	if df_history.empty:
		return pd.DataFrame()
	
	# Ensure timestamp columns are datetime
	if "telematics_timestamp" in df_history.columns:
		df_history["telematics_timestamp"] = pd.to_datetime(df_history["telematics_timestamp"], errors="coerce")
	elif "date" in df_history.columns:
		df_history["telematics_timestamp"] = pd.to_datetime(df_history["date"], errors="coerce")
	
	if not df_inference.empty and "timestamp" in df_inference.columns:
		df_inference["timestamp"] = pd.to_datetime(df_inference["timestamp"], errors="coerce")
	
	# LEFT JOIN: Start with telematics data (primary source)
	if vin and pd.notna(vin):
		# Use string comparison to handle type mismatches
		df_history_vin = df_history[
			df_history["vin"].notna() & 
			(df_history["vin"].astype(str).str.strip() == str(vin).strip())
		].copy()
		if df_history_vin.empty:
			return pd.DataFrame()
		# Get most recent telematics record for this VIN
		if "telematics_timestamp" in df_history_vin.columns:
			closest_telematics = df_history_vin.sort_values("telematics_timestamp", ascending=False).iloc[0]
		elif "date" in df_history_vin.columns:
			closest_telematics = df_history_vin.sort_values("date", ascending=False).iloc[0]
		else:
			closest_telematics = df_history_vin.iloc[0]
	else:
		# Use most recent from history
		if "telematics_timestamp" in df_history.columns:
			closest_telematics = df_history.sort_values("telematics_timestamp", ascending=False).iloc[0]
		elif "date" in df_history.columns:
			closest_telematics = df_history.sort_values("date", ascending=False).iloc[0]
		else:
			closest_telematics = df_history.iloc[0]
		vin = closest_telematics.get("vin")
	
	# Try to find matching inference data (optional - LEFT JOIN)
	inference_data = {}
	if not df_inference.empty and vin and pd.notna(vin):
		# Filter inference by VIN (handle NaN VINs and string comparison)
		df_inference_vin = df_inference[
			df_inference["vin"].notna() & 
			(df_inference["vin"].astype(str).str.strip() == str(vin).strip())
		].copy()
		if not df_inference_vin.empty:
			# Found inference data - get closest timestamp match
			if "telematics_timestamp" in df_history.columns and pd.notna(closest_telematics.get("telematics_timestamp")):
				df_inference_vin = df_inference_vin.copy()
				df_inference_vin["time_diff"] = abs(
					df_inference_vin["timestamp"] - closest_telematics["telematics_timestamp"]
				)
				df_inference_filtered = df_inference_vin[df_inference_vin["time_diff"] <= pd.Timedelta(days=7)]
				
				if not df_inference_filtered.empty:
					latest_inference = df_inference_filtered.nsmallest(1, "time_diff").iloc[0]
				else:
					latest_inference = df_inference_vin.sort_values("timestamp", ascending=False).iloc[0]
			else:
				latest_inference = df_inference_vin.sort_values("timestamp", ascending=False).iloc[0]
			inference_data = latest_inference.to_dict()
		else:
			# No inference match by VIN - try model + timestamp fallback
			inference_model = closest_telematics.get("model")
			if inference_model and pd.notna(inference_model):
				df_inference_model = df_inference[
					df_inference["model"].notna() & 
					(df_inference["model"].astype(str).str.strip() == str(inference_model).strip())
				].copy()
				if not df_inference_model.empty:
					# Match by timestamp (within 7 days) if telematics_timestamp is available
					if "telematics_timestamp" in df_history.columns and pd.notna(closest_telematics.get("telematics_timestamp")):
						df_inference_model = df_inference_model.copy()
						df_inference_model["time_diff"] = abs(
							df_inference_model["timestamp"] - closest_telematics["telematics_timestamp"]
						)
						df_inference_filtered = df_inference_model[df_inference_model["time_diff"] <= pd.Timedelta(days=7)]
						
						if not df_inference_filtered.empty:
							latest_inference = df_inference_filtered.nsmallest(1, "time_diff").iloc[0]
						else:
							latest_inference = df_inference_model.sort_values("timestamp", ascending=False).iloc[0]
					else:
						latest_inference = df_inference_model.sort_values("timestamp", ascending=False).iloc[0]
					inference_data = latest_inference.to_dict()
	
	# Merge telematics data (always present) with inference data (if available)
	merged = closest_telematics.to_dict()
	# Always preserve the historical model as model_history (source of truth)
	history_model = merged.get("model")
	if inference_data:
		merged.update(inference_data)
	# Always set model_history from historical data (even if None, to ensure consistency)
	merged["model_history"] = history_model
	# Also ensure the merged model field uses history model if inference doesn't have it
	if not merged.get("model") and history_model:
		merged["model"] = history_model
	
	return pd.DataFrame([merged])

def load_vehicle_image(model: str) -> str:
	"""Load vehicle image from images folder as base64"""
	model_lower = model.lower()
	
	# Map models to image files (JPG/PNG)
	image_map = {
		"ariya": "nissan_ariya.png",
		"leaf": "nissan_leaf.png",
		"sentra": "nissan_sentra.png"
	}
	
	image_file = image_map.get(model_lower, "nissan_ariya.png")
	image_path = os.path.join("images", image_file)
	
	if os.path.exists(image_path):
		with open(image_path, "rb") as f:
			image_data = f.read()
			base64_data = base64.b64encode(image_data).decode("utf-8")
			
			# Determine MIME type based on file extension
			if image_file.endswith('.png'):
				mime_type = "image/png"
			else:
				mime_type = "image/jpeg"
			
			return f"data:{mime_type};base64,{base64_data}"
	
	return ""

def render_vehicle_overview_card(row: Dict) -> None:
	"""Render Vehicle Overview card (top-left) - matching the exact design from image"""
	model = get_row_value(row, "model", "model_history", "model_inference", default="N/A")
	model_year = get_row_value(row, "model_year")
	if model_year and pd.notna(model_year):
		model_year_str = str(int(model_year))
	else:
		# Try to extract from manufacturing_date
		mfg_date = row.get("manufacturing_date")
		if mfg_date and pd.notna(mfg_date):
			try:
				model_year_str = str(pd.Timestamp(mfg_date).year)
			except:
				model_year_str = ""
		else:
			model_year_str = ""
	
	vin = get_row_value(row, "vin", default="")
	
	# Health status
	pred_pct = get_row_value(row, "pred_prob_pct", default=0) or 0
	health_score, status = calculate_health_score(
		float(pred_pct),
		get_row_value(row, "battery_soc"),
		get_row_value(row, "dtc_severity")
	)
	
	status_chip = _build_status_chip(status)
	
	# Last seen
	timestamp = get_row_value(row, "timestamp", "telematics_timestamp", "date")
	if timestamp and pd.notna(timestamp):
		try:
			last_seen = format_time_ago(pd.Timestamp(timestamp))
		except:
			last_seen = "N/A"
	else:
		last_seen = "N/A"
	
	# Location
	city = get_row_value(row, "city", default="N/A")
	state = get_row_value(row, "state", default="")
	if state and state != "N/A":
		location_str = f"{city}, {state}"
	else:
		location_str = city
	
	# Mileage (from inference log or historical)
	mileage_str = format_mileage_display(row)
	
	# Battery SOC (used as energy/fuel indicator)
	battery_soc = get_row_value(row, "battery_soc")
	if battery_soc is not None and pd.notna(battery_soc):
		try:
			battery_val = float(battery_soc)
			battery_str = f"{battery_val:.0f}%"
		except:
			battery_str = "N/A"
	else:
		battery_str = "N/A"
	
	model_normalized = str(model).strip().lower()
	ev_models = {"leaf", "ariya"}
	is_ev_model = model_normalized in ev_models
	energy_value_str = battery_str
	
	# Load vehicle image from images folder
	vehicle_image_data = load_vehicle_image(model)
	if vehicle_image_data:
		vehicle_html = f'<img src="{vehicle_image_data}" style="width: 100%; max-width: 200px; height: 105px; object-fit: contain; margin: 14px auto 10px auto; display: block;" />'
	else:
		vehicle_html = '<div style="width: 200px; height: 105px; background: rgba(255,255,255,0.1); border-radius: 8px; margin: 14px auto 10px auto;"></div>'
	
	battery_icon_svg = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
		<rect x="2" y="7" width="16" height="10" rx="2" stroke="{color}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
		<rect x="4" y="9" width="12" height="6" rx="1" fill="{color}" opacity="0.3"/>
		<path d="M18 11V13" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
		<path d="M20 5V7" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
		<path d="M20 17V19" stroke="{color}" stroke-width="2" stroke-linecap="round"/>
	</svg>'''.replace("{color}", COLORS['text_light'])
	
	fuel_icon_svg = '''<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
		<path d="M6 3H13C13.5523 3 14 3.44772 14 4V20H6V3Z" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
		<path d="M14 8H17L20 11V17C20 18.1046 19.1046 19 18 19H14" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
		<path d="M18 11V8H17" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
		<rect x="8" y="5" width="3" height="6" rx="1.5" fill="{color}" opacity="0.3"/>
	</svg>'''.replace("{color}", COLORS['text_light'])
	
	energy_icon_svg = battery_icon_svg if is_ev_model else fuel_icon_svg
	
	st.markdown(f"""
	<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 0; min-height: 280px; height: 280px; display: flex; flex-direction: column; overflow: hidden;">
		<div style="width: 100%; height: 3px; background: linear-gradient(90deg, #c3002f 0%, #000000 100%); flex-shrink: 0;"></div>
		<div style="padding: 20px 24px; flex: 1; display: flex; flex-direction: column;">
			<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
				<div style="font-size: 20px; font-weight: 600; color: {COLORS['text_light']};">
					{model_year_str + " " if model_year_str else ""}Nissan {model}
				</div>
				{status_chip}
			</div>
			<div style="display: flex; justify-content: center; align-items: center; margin: 0 0 16px 0;">
				{vehicle_html}
			</div>
			<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: auto;">
			<div style="display: flex; flex-direction: column; gap: 12px;">
				<div style="font-size: 13px; color: {COLORS['text_light']};">
					Last seen {last_seen}
				</div>
				<div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: {COLORS['text_light']};">
					<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
						<path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" stroke="{COLORS['text_light']}" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
						<circle cx="12" cy="9" r="2.5" stroke="{COLORS['text_light']}" stroke-width="2" fill="none"/>
					</svg>
					{mileage_str}
				</div>
			</div>
			<div style="display: flex; flex-direction: column; gap: 12px; align-items: flex-end;">
				<div style="font-size: 13px; color: {COLORS['text_light']}; text-align: right;">
					{location_str}
				</div>
				<div style="display: flex; align-items: center; gap: 8px; font-size: 13px; color: {COLORS['text_light']};">
					{energy_icon_svg}
					{energy_value_str}
				</div>
			</div>
		</div>
		</div>
	</div>
	""", unsafe_allow_html=True)

def render_health_score_gauge(health_score: float, status: str) -> None:
	"""Render Overall Health Score gauge (top-right) with animated dashboard feel"""
	status_color = COLORS["healthy"] if status == "Healthy" else (COLORS["warning"] if status == "Warning" else COLORS["critical"])
	prev_score = st.session_state.get("last_health_score", health_score)
	
	# Maintain history for sparkline (last 7 values)
	history_key = "health_score_history"
	if history_key not in st.session_state:
		st.session_state[history_key] = []
	history = st.session_state[history_key]
	if len(history) >= 7:
		history.pop(0)
	history.append(health_score)
	st.session_state[history_key] = history
	st.session_state["last_health_score"] = health_score
	
	if not st.session_state.get("health_gauge_css_injected"):
		st.session_state["health_gauge_css_injected"] = True
		st.markdown(
			f"""
			<style>
				.health-gauge-card {{
					{GLASS_CARD_STYLE}
					border-radius: 14px;
					padding: 22px 26px 18px 26px;
					min-height: 280px;
					height: 280px;
					display: flex;
					flex-direction: column;
					overflow: hidden;
					position: relative;
				}}
				.health-gauge-card > * {{
					position: relative;
					z-index: 1;
				}}
				.health-gauge-wrapper {{
					position: relative;
					display: flex;
					align-items: center;
					justify-content: center;
					margin: 8px 0;
				}}
				.health-gauge-frame {{
					position: absolute;
					width: 180px;
					height: 180px;
					border-radius: 50%;
					background: radial-gradient(circle at center, rgba(195,0,47,0.15) 0%, rgba(0,0,0,0.3) 70%, transparent 100%);
					box-shadow: 0 0 30px rgba(195,0,47,0.2), inset 0 0 20px rgba(0,0,0,0.3);
					pointer-events: none;
					z-index: 0;
				}}
				.health-gauge-needle-shadow {{
					position: absolute;
					width: 4px;
					height: 80px;
					background: linear-gradient(to bottom, rgba(0,0,0,0.6) 0%, rgba(0,0,0,0.2) 50%, transparent 100%);
					border-radius: 2px;
					transform-origin: bottom center;
					transform: rotate(-90deg) translateY(-40px);
					pointer-events: none;
					z-index: 2;
					filter: blur(2px);
				}}
				.health-gauge-chart-wrapper {{
					position: relative;
					z-index: 1;
				}}
				.health-panel-header {{
					display: flex;
					justify-content: space-between;
					align-items: center;
					margin-bottom: 8px;
					padding: 12px 4px 0 4px;
					gap: 12px;
					flex-wrap: nowrap;
					width: 100%;
				}}
				.health-gauge-title {{
					font-size: 14px;
					font-weight: 600;
					color: #e2e8f0;
					letter-spacing: 0.4px;
					margin-bottom: 0;
					margin-right: auto;
					text-transform: uppercase;
					display: inline-flex;
					align-items: center;
					white-space: nowrap;
				}}
				.health-status-chip {{
					text-align: center;
					font-size: 14px;
					font-weight: 600;
					margin-top: 0;
					margin-left: auto;
					padding: 6px 10px;
					border-radius: 999px;
					display: inline-flex;
					align-items: center;
					justify-content: center;
					gap: 6px;
					min-width: 120px;
					flex-shrink: 0;
				}}
			</style>
			""",
			unsafe_allow_html=True,
		)

	fig = go.Figure(
		go.Indicator(
			mode="gauge+number",
			value=health_score,
			number={'font': {'size': 44, 'color': '#ff8c00', 'family': 'Arial Black'}, 'suffix': " pts"},
			gauge={
				'shape': 'angular',
				'axis': {
					'range': [0, 100],
					'tickmode': 'array',
					'tickvals': [20, 40, 60, 80],
					'ticktext': ['20', '40', '60', '80'],
					'tickfont': {'color': 'white', 'size': 12},
					'tickwidth': 2
				},
				'bar': {'color': 'white', 'thickness': 0.15},
				'bgcolor': 'rgba(255,255,255,0.0)',
				'borderwidth': 0,
				'steps': [
					{'range': [0, 40], 'color': 'rgba(139, 0, 0, 0.8)'},  # Dark red
					{'range': [40, 60], 'color': 'rgba(204, 85, 0, 0.8)'},  # Dark orange
					{'range': [60, 100], 'color': 'rgba(0, 128, 128, 0.8)'}  # Dark teal
				],
				'threshold': {
					'line': {'color': 'white', 'width': 6},
					'thickness': 0.95,
					'value': health_score
				}
			}
		)
	)
	
	fig.update_layout(
		height=200,
		paper_bgcolor='rgba(0,0,0,0)',
		plot_bgcolor='rgba(0,0,0,0)',
		margin=dict(l=10, r=10, t=5, b=5),
		transition={'duration': 800, 'easing': 'cubic-in-out'}
	)
	
	# Render gauge inside card with enhancements
	status_chip = _build_status_chip(status)
	
	st.markdown(
		f"""
		<div class="health-panel-header" style="display: flex; justify-content: space-between; align-items: center; width: 100%; flex-wrap: nowrap;">
			<span class='health-gauge-title' style="margin-right: auto;">Health Score</span>
			<div style="margin-left: auto; flex-shrink: 0;">{status_chip}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	
	st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
	
	# Sparkline removed per latest design

def render_sensor_snapshot(row: Dict) -> None:
	"""Render Key Sensor Snapshot card (bottom-left)"""
	def safe_get(key, default=None):
		val = row.get(key, default)
		if val is None or (isinstance(val, float) and pd.isna(val)):
			return None
		return val
	
	battery_soc = get_row_value(row, "battery_soc")
	battery_voltage = get_row_value(row, "battery_voltage")
	ambient_temp = get_row_value(row, "ambient_temperature")
	engine_rpm = get_row_value(row, "engine_rpm")
	coolant_temp = get_row_value(row, "coolant_temperature")
	vehicle_speed = get_row_value(row, "vehicle_speed")
	oil_pressure = get_row_value(row, "oil_pressure")
	engine_load = get_row_value(row, "engine_load")
	battery_temp_c = get_row_value(row, "battery_temperature")
	battery_temp_f = None
	if battery_temp_c is not None:
		try:
			battery_temp_f = float(battery_temp_c) * 9 / 5 + 32
		except Exception:
			battery_temp_f = None
	
	def format_sensor_value(value, unit, is_good=True):
		if value is None:
			return ""
		try:
			if isinstance(value, (int, float)):
				val_str = f"{float(value):.0f}"
			else:
				val_str = str(value)
		except:
			val_str = "N/A"
		
		color = COLORS["healthy"] if is_good else COLORS["text_muted"]
		unit_spacer = " " if unit.strip() else ""
		return f'<span class="sensor-value-number" style="color: {color}; font-size: 16px; font-weight: 600;">{val_str}{unit_spacer}{unit.strip()}</span>'
	
	# Always use model_history for EV/non-EV determination since it's the source of truth
	model_name = str(get_row_value(row, "model_history", "model") or "").strip().lower()
	ev_models = {"leaf", "ariya"}
	is_ev_model = model_name in ev_models

	if is_ev_model:
		metrics = [
			("Battery SOC", battery_soc, "%", battery_soc and battery_soc > 50),
			("Battery Voltage", battery_voltage, " v", battery_voltage and (300 <= battery_voltage <= 400 or 11 <= battery_voltage <= 14)),
			("Ambient Temp", ambient_temp, " °F", True),
			("Battery Temp", battery_temp_f, " °F", battery_temp_f and 68 <= battery_temp_f <= 113),
			("Motor RPM", engine_rpm, " rpm", False),
			("Coolant Temp", coolant_temp, " °F", coolant_temp and 180 <= coolant_temp <= 210),
			("Speed", vehicle_speed, " mph", False),
		]
	else:
		metrics = [
			("Engine Coolant Temp", coolant_temp, " °F", coolant_temp and 180 <= coolant_temp <= 210),
			("Oil Pressure", oil_pressure, " psi", oil_pressure and 25 <= oil_pressure <= 80),
			("Engine RPM", engine_rpm, " rpm", False),
			("Fuel Level", battery_soc, " %", battery_soc and battery_soc > 30),
			("Engine Load", engine_load, " %", engine_load and engine_load < 90),
			("Speed", vehicle_speed, " mph", False),
		]

	# Icon mapping for sensors
	icon_map = {
		"Battery SOC": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="4" width="12" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><line x1="18" y1="8" x2="20" y2="8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="18" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="18" y1="16" x2="20" y2="16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Battery Voltage": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L13.09 8.26L18 9.14L14 13.78L14.36 20.86L12 18.14L9.64 20.86L10 13.78L6 9.14L10.91 8.26L12 2Z" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
		"Ambient Temp": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2V6M12 18V22M4 12H2M6.31412 6.31412L4.8999 4.8999M17.6859 6.31412L19.1001 4.8999M6.31412 17.69L4.8999 19.1042M17.6859 17.69L19.1001 19.1042M22 12H20M17 12C17 14.7614 14.7614 17 12 17C9.23858 17 7 14.7614 7 12C7 9.23858 9.23858 7 12 7C14.7614 7 17 9.23858 17 12Z" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
		"Battery Temp": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="3" width="14" height="18" rx="3" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M9 7H15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M9 11H15" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M9 15H13" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Motor RPM": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="6" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M12 4V7.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M12 16.5V20" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M4 12H7.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M16.5 12H20" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M17.3 6.7L15.2 8.8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M8.8 15.2L6.7 17.3" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Coolant Temp": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
		"Engine Coolant Temp": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
		"Speed": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>',
		"Oil Pressure": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M8 12L10.5 14.5L16 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
		"Fuel Level": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M6 3H13C13.5523 3 14 3.44772 14 4V20H6V3Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M14 8H17L20 11V17C20 18.1046 19.1046 19 18 19H14" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><rect x="8" y="5" width="3" height="6" rx="1.5" fill="currentColor" opacity="0.3"/></svg>',
		"Engine Load": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M3 12H21M12 3V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Engine RPM": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.5" fill="none"/><line x1="12" y1="4" x2="12" y2="8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="12" y1="16" x2="12" y2="20" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="4" y1="12" x2="8" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="16" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
	}
	
	rows_html = []
	for idx, (label, val, unit, is_good) in enumerate(metrics):
		value_html = format_sensor_value(val, unit, bool(is_good))
		if not value_html:
			value_html = f"<span style='color:{COLORS['text_muted']}; font-size:14px;'>N/A</span>"
		
		icon = icon_map.get(label, '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>')
		
		row_html = (
			f"<div class='sensor-grid-item' data-index='{idx}'>"
			f"<div class='sensor-item-content'>"
			f"<div class='sensor-icon'>{icon}</div>"
			f"<div class='sensor-item-text'>"
			f"<div class='sensor-row-label'>{label}</div>"
			f"<div class='sensor-row-value'>{value_html}</div>"
			f"</div>"
			f"</div>"
			f"</div>"
		)
		rows_html.append(row_html)

	card_html = textwrap.dedent(f"""
		<style>
			@keyframes fadeInUp {{
				from {{
					opacity: 0;
					transform: translateY(8px);
				}}
				to {{
					opacity: 1;
					transform: translateY(0);
				}}
			}}
			.sensor-grid {{
				display: grid;
				grid-template-columns: 1fr 1fr;
				gap: 12px;
			}}
			.sensor-grid-item {{
				position: relative;
				padding: 10px 0;
			}}
			.sensor-grid-item:not(:last-child):not(:nth-last-child(2))::after {{
				content: "";
				position: absolute;
				bottom: 0;
				left: 0;
				right: 0;
				height: 1px;
				background: rgba(255,255,255,0.06);
			}}
			.sensor-item-content {{
				display: flex;
				align-items: center;
				gap: 10px;
			}}
			.sensor-icon {{
				display: flex;
				align-items: center;
				justify-content: center;
				width: 28px;
				height: 28px;
				flex-shrink: 0;
				color: {COLORS['text_muted']};
				opacity: 0.7;
			}}
			.sensor-item-text {{
				flex: 1;
				display: flex;
				flex-direction: column;
				gap: 4px;
			}}
			.sensor-row-label {{
				color: {COLORS['text_muted']};
				font-size: 11px;
				letter-spacing: 0.3px;
				text-transform: uppercase;
			}}
			.sensor-row-value {{
				font-size: 14px;
			}}
			.sensor-value-number {{
				animation: fadeInUp 0.5s ease-out;
				animation-fill-mode: both;
			}}
			.sensor-grid-item[data-index="0"] .sensor-value-number {{
				animation-delay: 0.1s;
			}}
			.sensor-grid-item[data-index="1"] .sensor-value-number {{
				animation-delay: 0.15s;
			}}
			.sensor-grid-item[data-index="2"] .sensor-value-number {{
				animation-delay: 0.2s;
			}}
			.sensor-grid-item[data-index="3"] .sensor-value-number {{
				animation-delay: 0.25s;
			}}
			.sensor-grid-item[data-index="4"] .sensor-value-number {{
				animation-delay: 0.3s;
			}}
			.sensor-grid-item[data-index="5"] .sensor-value-number {{
				animation-delay: 0.35s;
			}}
		</style>
		<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; min-height: 280px; height: 280px; display: flex; flex-direction: column;">
			<div style="font-size: 14px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 16px;">
				Key Sensor Snapshot
			</div>
			<div class="sensor-grid">
				{''.join(rows_html)}
			</div>
		</div>
	""")

	st.markdown(card_html, unsafe_allow_html=True)

def render_dtc_panel(row: Dict) -> None:
	"""Render DTC Fault Panel (bottom-right) with enhanced styling"""
	dtc_code = get_row_value(row, "dtc_code")
	dtc_subsystem = get_row_value(row, "dtc_subsystem")
	dtc_recommendation = get_row_value(row, "dtc_recommendation")
	dtc_explanation = get_row_value(row, "dtc_explanation")
	dtc_severity = get_row_value(row, "dtc_severity")
	
	# DTC icon SVG - Wrench/tool icon representing diagnostics and troubleshooting
	dtc_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14.7 6.3C15.1 6.7 15.1 7.3 14.7 7.7L9.7 12.7C9.3 13.1 8.7 13.1 8.3 12.7L5.3 9.7C4.9 9.3 4.9 8.7 5.3 8.3L10.3 3.3C10.7 2.9 11.3 2.9 11.7 3.3L14.7 6.3Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M19.5 19.5L16.5 16.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
	
	# Subsystem icon SVG
	subsystem_icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M3 9H21M9 3V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
	
	# Recommendation icon SVG
	recommendation_icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" stroke="currentColor" stroke-width="1.5" fill="none"/></svg>'
	
	if not dtc_code or pd.isna(dtc_code):
		# DTC icon for empty state - Wrench/tool icon representing diagnostics
		dtc_icon_empty = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14.7 6.3C15.1 6.7 15.1 7.3 14.7 7.7L9.7 12.7C9.3 13.1 8.7 13.1 8.3 12.7L5.3 9.7C4.9 9.3 4.9 8.7 5.3 8.3L10.3 3.3C10.7 2.9 11.3 2.9 11.7 3.3L14.7 6.3Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M19.5 19.5L16.5 16.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
		
		st.markdown(f"""
		<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px;">
			<div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 12px; color: {COLORS['text_muted']};">
				{dtc_icon_empty}
			</div>
			<div style="font-size: 14px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 8px;">
				DTC Fault Panel
			</div>
			<div style="color: {COLORS['text_muted']}; font-size: 13px; text-align: center;">
				No active DTC codes
			</div>
		</div>
		""", unsafe_allow_html=True)
		return
	
	severity_color = COLORS["critical"] if dtc_severity == "CRITICAL" else (COLORS["warning"] if dtc_severity in ["HIGH", "MEDIUM"] else COLORS["text_muted"])
	severity_chip = _build_status_chip(dtc_severity or "Info")
	
	# Details icon SVG
	details_icon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
	
	# DTC code icon SVG (inline)
	dtc_code_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
	
	# Inject CSS only once
	if not st.session_state.get("dtc_panel_css_injected"):
		st.session_state["dtc_panel_css_injected"] = True
		st.markdown(f"""
		<style>
			.dtc-panel-code {{
				display: inline-flex;
				align-items: center;
				gap: 10px;
				padding: 12px 16px;
				background: rgba(195,0,47,0.1);
				border: 1px solid rgba(195,0,47,0.2);
				border-radius: 8px;
				margin-bottom: 16px;
			}}
			.dtc-panel-section {{
				margin-bottom: 16px;
				padding-bottom: 16px;
				border-bottom: 1px solid rgba(255,255,255,0.06);
			}}
			.dtc-panel-section:last-child {{
				border-bottom: none;
				margin-bottom: 0;
				padding-bottom: 0;
			}}
			.dtc-panel-label {{
				display: flex;
				align-items: center;
				gap: 6px;
				color: {COLORS['text_muted']};
				font-size: 11px;
				text-transform: uppercase;
				letter-spacing: 0.5px;
				margin-bottom: 6px;
			}}
			.dtc-panel-value {{
				color: {COLORS['text_light']};
				font-size: 14px;
				line-height: 1.5;
			}}
			.dtc-panel-recommendation {{
				display: flex;
				align-items: flex-start;
				gap: 10px;
				padding: 12px;
				background: rgba(195,0,47,0.08);
				border-radius: 6px;
				margin-top: 8px;
			}}
		</style>
		""", unsafe_allow_html=True)
	
	# Enhanced DTC Panel with better visual hierarchy - using triple quotes to avoid quote conflicts
	text_light = COLORS['text_light']
	text_muted = COLORS['text_muted']
	card_html = f'''<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; border-left: 4px solid {severity_color}; position: relative; overflow: hidden;">
		<div style="position: absolute; top: 0; right: 0; width: 100px; height: 100px; background: radial-gradient(circle, {severity_color}15 0%, transparent 70%); pointer-events: none;"></div>
		<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;position:relative;z-index:1;">
			<div style="display: flex; align-items: center; gap: 8px;">
				<div style="width: 32px; height: 32px; border-radius: 8px; background: rgba(195,0,47,0.15); border: 1px solid rgba(195,0,47,0.3); display: flex; align-items: center; justify-content: center; color: {severity_color};">
					{dtc_icon}
				</div>
				<div style="font-size: 14px; font-weight: 600; color: {text_light};">
					DTC Fault Panel
				</div>
			</div>
			{severity_chip}
		</div>
		<div class="dtc-panel-section" style="position:relative;z-index:1;">
			<div class="dtc-panel-code" style="color: {severity_color};">
				{dtc_code_icon}
				<span style="font-size: 28px; font-weight: 700; letter-spacing: 2px;">{dtc_code}</span>
			</div>
		</div>
		<div class="dtc-panel-section" style="position:relative;z-index:1;">
			<div class="dtc-panel-label">
				{subsystem_icon}
				Subsystem
			</div>
			<div class="dtc-panel-value" style="font-weight: 600;">
				{dtc_subsystem or 'N/A'}
			</div>
		</div>
		<div class="dtc-panel-section" style="position:relative;z-index:1;">
			<div class="dtc-panel-label">
				{recommendation_icon}
				Recommended Action
			</div>
			<div class="dtc-panel-recommendation" style="border-left: 3px solid {severity_color};">
				<div style="flex: 1;">
					<div class="dtc-panel-value" style="font-weight: 600; color: {severity_color};">
						{dtc_recommendation or 'Inspection recommended'}
					</div>
				</div>
			</div>
		</div>
		<div style="position:relative;z-index:1;">
			<div class="dtc-panel-label">
				{details_icon}
				Details
			</div>
			<div class="dtc-panel-value" style="color: {text_muted}; font-size: 12px; line-height: 1.6;">
				{dtc_explanation or 'Diagnostic trouble code detected. Please refer to service manual for detailed diagnostics.'}
			</div>
		</div>
	</div>'''
	
	st.markdown(card_html, unsafe_allow_html=True)

def render_anomalies_timeline(df_history: pd.DataFrame, vin: str) -> None:
	"""Render Recent Anomalies Timeline"""
	# Filter anomalies for this VIN
	if "vin" not in df_history.columns:
		# Anomaly icon SVG - Timeline/activity graph representing anomaly detection over time
		anomaly_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 3V21H21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 16L10 13L14 17L21 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="7" cy="16" r="1.5" fill="currentColor"/><circle cx="10" cy="13" r="1.5" fill="currentColor"/><circle cx="14" cy="17" r="1.5" fill="currentColor"/><circle cx="21" cy="10" r="1.5" fill="currentColor"/></svg>'
		
		st.markdown(f"""
		<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px;">
			<div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 12px; color: {COLORS['text_muted']};">
				{anomaly_icon}
			</div>
			<div style="font-size: 14px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 8px;">
				Recent Anomalies Timeline
			</div>
			<div style="color: {COLORS['text_muted']}; font-size: 13px; text-align: center;">
				No anomaly data available
			</div>
		</div>
		""", unsafe_allow_html=True)
		return
	
	vin_data = df_history[df_history["vin"] == vin].copy()
	
	# Get anomalies (non-null anomaly_type)
	if "anomaly_type" not in vin_data.columns:
		anomalies = pd.DataFrame()
	else:
		anomalies = vin_data[vin_data["anomaly_type"].notna()].copy()
	
	if anomalies.empty:
		# Anomaly icon SVG - Timeline/activity graph representing anomaly detection over time
		anomaly_icon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 3V21H21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 16L10 13L14 17L21 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><circle cx="7" cy="16" r="1.5" fill="currentColor"/><circle cx="10" cy="13" r="1.5" fill="currentColor"/><circle cx="14" cy="17" r="1.5" fill="currentColor"/><circle cx="21" cy="10" r="1.5" fill="currentColor"/></svg>'
		
		st.markdown(f"""
		<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 200px;">
			<div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; margin-bottom: 12px; color: {COLORS['text_muted']};">
				{anomaly_icon}
			</div>
			<div style="font-size: 14px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 8px;">
				Recent Anomalies Timeline
			</div>
			<div style="color: {COLORS['text_muted']}; font-size: 13px; text-align: center;">
				No anomalies detected
			</div>
		</div>
		""", unsafe_allow_html=True)
		return
	
	# Sort by timestamp (most recent first) and take top 4
	anomalies = anomalies.sort_values("anomaly_timestamp", ascending=False).head(4)
	
	# SVG Icon mapping with color coding
	icon_map = {
		"SOC drop": {
			"svg": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M13 2L3 14H12L11 22L21 10H12L13 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
			"color": "#fbbf24"
		},
		"High temperature": {
			"svg": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C8.13 2 5 5.13 5 9C5 14.25 12 22 12 22C12 22 19 14.25 19 9C19 5.13 15.87 2 12 2Z" stroke="currentColor" stroke-width="2" fill="none"/><circle cx="12" cy="9" r="2.5" stroke="currentColor" stroke-width="2" fill="none"/></svg>',
			"color": "#ef4444"
		},
		"DTC logged": {
			"svg": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="12" cy="12" r="2" fill="currentColor"/></svg>',
			"color": "#f59e0b"
		},
		"Service event": {
			"svg": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M14.7 6.3C15.1 6.7 15.1 7.3 14.7 7.7L9.7 12.7C9.3 13.1 8.7 13.1 8.3 12.7L5.3 9.7C4.9 9.3 4.9 8.7 5.3 8.3C5.7 7.9 6.3 7.9 6.7 8.3L9 10.6L13.3 6.3C13.7 5.9 14.3 5.9 14.7 6.3Z" fill="currentColor"/><path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="currentColor"/></svg>',
			"color": "#10b981"
		}
	}
	
	# Get severity colors
	severity_map = {
		"CRITICAL": "#ef4444",
		"HIGH": "#f59e0b",
		"MEDIUM": "#fbbf24",
		"LOW": "#10b981",
		"INFO": COLORS['text_muted']
	}
	
	# Inject CSS only once
	if not st.session_state.get("anomalies_timeline_css_injected"):
		st.session_state["anomalies_timeline_css_injected"] = True
		st.markdown(f"""
		<style>
			.anomaly-timeline-item {{
				position: relative;
				z-index: 1;
				flex: 1;
				text-align: center;
				transition: transform 0.2s ease;
			}}
			.anomaly-timeline-item:hover {{
				transform: translateY(-4px);
			}}
			.anomaly-icon-container {{
				width: 48px;
				height: 48px;
				border-radius: 50%;
				display: flex;
				align-items: center;
				justify-content: center;
				margin: 0 auto 10px;
				position: relative;
				transition: all 0.3s ease;
			}}
			.anomaly-icon-container::before {{
				content: "";
				position: absolute;
				inset: -2px;
				border-radius: 50%;
				background: linear-gradient(135deg, currentColor, transparent);
				opacity: 0.3;
				z-index: -1;
			}}
			.anomaly-icon-container:hover {{
				transform: scale(1.1);
			}}
			.anomaly-time {{
				color: {COLORS['text_muted']};
				font-size: 10px;
				text-transform: uppercase;
				letter-spacing: 0.5px;
				margin-bottom: 6px;
				font-weight: 600;
			}}
			.anomaly-type {{
				color: {COLORS['text_light']};
				font-size: 12px;
				font-weight: 500;
			}}
		</style>
		""", unsafe_allow_html=True)
	
	timeline_html = f'''<div class="dt-card" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px;">
		<div style="font-size: 14px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 20px; display: flex; align-items: center; gap: 8px;">
			<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
				<path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="currentColor"/>
			</svg>
			Recent Anomalies Timeline
		</div>
		<div style="display: flex; align-items: center; gap: 16px; position: relative; padding: 24px 0;">
			<div style="position: absolute; top: 50%; left: 0; right: 0; height: 2px; background: linear-gradient(to right, transparent, {COLORS['border']}, transparent); z-index: 0; transform: translateY(-50%);"></div>'''
	
	for idx, (_, anomaly) in enumerate(anomalies.iterrows()):
		anomaly_type = anomaly.get("anomaly_type", "Unknown")
		anomaly_time = anomaly.get("anomaly_timestamp")
		anomaly_severity = anomaly.get("anomaly_severity", "INFO")
		anomaly_description = anomaly.get("anomaly_description", "")
		
		icon_data = icon_map.get(anomaly_type, {
			"svg": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2" fill="none"/><path d="M12 8V12M12 16H12.01" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
			"color": COLORS['text_muted']
		})
		
		icon_svg = icon_data["svg"]
		icon_color = severity_map.get(anomaly_severity, icon_data["color"])
		
		if pd.notna(anomaly_time):
			time_str = format_time_ago(pd.Timestamp(anomaly_time))
		else:
			time_str = "Unknown"
		
		timeline_html += f'''
			<div class="anomaly-timeline-item">
				<div class="anomaly-icon-container" style="background: {icon_color}15; border: 2px solid {icon_color}40; color: {icon_color};">
					{icon_svg}
				</div>
				<div class="anomaly-time">{time_str}</div>
				<div class="anomaly-type">{anomaly_type}</div>
				{f'<div style="color: {COLORS["text_muted"]}; font-size: 10px; margin-top: 4px; line-height: 1.3;">{anomaly_description[:30]}{"..." if len(anomaly_description) > 30 else ""}</div>' if anomaly_description else ''}
			</div>'''
	
	timeline_html += '</div></div>'
	
	st.markdown(timeline_html, unsafe_allow_html=True)

def render_component_health_card(component: str, row: Dict) -> None:
	"""Render component health card (Battery, Engine, or Brakes)"""
	def safe_get(key, default=None):
		return get_row_value(row, key, default=default)
	
	# Helper function to calculate metric health percentage (defined early for use in status calculation)
	def get_metric_health(metric_name, value_str, component):
		"""Calculate health percentage for a metric based on component and value"""
		try:
			# Extract numeric value
			value = float(''.join(filter(str.isdigit, value_str.replace(',', ''))))
			
			if component == "Battery":
				if "Temperature" in metric_name:
					# Battery temp: 20-40°C is optimal (100%), 0-20 or 40-60 is 50%, >60 is 0%
					if 20 <= value <= 40:
						return 100
					elif (0 <= value < 20) or (40 < value <= 60):
						return 50
					else:
						return 0
				elif "Voltage" in metric_name:
					# Battery voltage: 300-400V is optimal
					if 300 <= value <= 400:
						return 100
					elif 250 <= value < 300 or 400 < value <= 450:
						return 60
					else:
						return 30
			elif component == "Fuel System":
				if "Fuel Level" in metric_name:
					# Fuel level: >50% is 100%, 20-50% is 60%, <20% is 20%
					if value > 50:
						return 100
					elif 20 <= value <= 50:
						return 60
					else:
						return 20
				elif "Oil Pressure" in metric_name:
					# Oil pressure: 25-80 psi is optimal
					if 25 <= value <= 80:
						return 100
					elif 15 <= value < 25 or 80 < value <= 100:
						return 60
					else:
						return 30
				elif "Engine Load" in metric_name:
					# Engine load: <70% is optimal, 70-90% is moderate, >90% is high
					if value < 70:
						return 100
					elif 70 <= value <= 90:
						return 60
					else:
						return 30
			elif component == "Engine":
				if "Water pump speed" in metric_name:
					# Water pump: >1500 RPM is optimal
					if value > 1500:
						return 100
					elif 1000 <= value <= 1500:
						return 70
					else:
						return 40
				elif "Coolant temp" in metric_name:
					# Coolant: 180-210°C is optimal
					if 180 <= value <= 210:
						return 100
					elif 160 <= value < 180 or 210 < value <= 230:
						return 60
					else:
						return 30
			elif component == "Brakes":
				if "Pressure" in metric_name:
					# Brake pressure: 25-35 bar is optimal
					if 25 <= value <= 35:
						return 100
					elif 20 <= value < 25 or 35 < value <= 40:
						return 70
					else:
						return 40
				elif "Pad wear" in metric_name:
					# Pad wear: <50% is good, 50-70% is moderate, >70% is critical
					if value < 50:
						return 100
					elif 50 <= value <= 70:
						return 60
					else:
						return 30
		except:
			pass
		return 75  # Default moderate health
	
	# Component-specific data extraction
	if component == "Battery":
		# Get raw health status from data (fallback if we can't calculate)
		raw_health_status = safe_get("battery_health_status", "Normal")
		battery_temp = safe_get("battery_temperature")
		# Try multiple possible column names for battery voltage
		battery_voltage = get_row_value(row, "battery_voltage", "voltage", "battery_voltage_v", default=None)
		metrics = {}
		# Add Temperature metric
		if battery_temp is not None:
			try:
				temp_val = float(battery_temp)
				if not (pd.isna(temp_val) or temp_val == 0):
					metrics["Temperature"] = f"{temp_val:.0f}°C"
			except (ValueError, TypeError):
				pass
		# Add Voltage metric - CRITICAL: Always add if value exists and is valid
		if battery_voltage is not None:
			try:
				voltage_val = float(battery_voltage)
				# Only skip if it's NaN, but allow 0 as valid (though unlikely for battery)
				if not pd.isna(voltage_val):
					# Format voltage appropriately based on value range
					# EV batteries are typically 300-400V, 12V batteries are 11-14V
					if voltage_val >= 100:
						metrics["Voltage"] = f"{voltage_val:.1f}V"
					else:
						metrics["Voltage"] = f"{voltage_val:.2f}V"
			except (ValueError, TypeError) as e:
				# Log error for debugging
				if config and hasattr(config, 'debug') and config.debug:
					logger.error(f"Error adding Voltage metric: {e}, value={battery_voltage}, type={type(battery_voltage)}")
				pass
		
		# Calculate overall health from metrics to derive status
		health_scores = [get_metric_health(k, v, component) for k, v in metrics.items()]
		overall_health = int(sum(health_scores) / len(health_scores)) if health_scores else 75
		
		# Derive health_status from calculated overall_health
		if overall_health >= 80:
			health_status = "Normal"
		elif overall_health >= 50:
			health_status = "Warning"
		else:
			health_status = "Critical"
		
		status_text = "No major issues detected" if health_status == "Normal" else (
			"Voltage below optimal range" if battery_voltage and battery_voltage < 300 else "Capacity degradation detected"
		)
	elif component == "Fuel System":
		# For non-EV models: Fuel System component
		raw_health_status = safe_get("engine_health_status", "Normal")  # Use engine health as proxy
		fuel_level = safe_get("battery_soc", 0)  # battery_soc is used as fuel level for non-EV
		oil_pressure = safe_get("oil_pressure", 0)
		engine_load = safe_get("engine_load", 0)
		metrics = {}
		if fuel_level is not None:
			metrics["Fuel Level"] = f"{float(fuel_level):.0f}%"
		if oil_pressure is not None:
			metrics["Oil Pressure"] = f"{float(oil_pressure):.0f} psi"
		if engine_load is not None:
			metrics["Engine Load"] = f"{float(engine_load):.0f}%"
		
		# Calculate overall health from metrics to derive status
		health_scores = [get_metric_health(k, v, component) for k, v in metrics.items()]
		overall_health = int(sum(health_scores) / len(health_scores)) if health_scores else 75
		
		# Derive health_status from calculated overall_health
		if overall_health >= 80:
			health_status = "Normal"
		elif overall_health >= 50:
			health_status = "Warning"
		else:
			health_status = "Critical"
		
		# Determine status text based on actual conditions
		if fuel_level and fuel_level < 20:
			status_text = "Low fuel level detected"
		elif oil_pressure and oil_pressure < 25:
			status_text = "Low oil pressure detected"
		elif engine_load and engine_load > 90:
			status_text = "High engine load detected"
		elif health_status == "Normal":
			status_text = "Fuel system operating within normal parameters"
		else:
			status_text = "Fuel system requires attention"
	elif component == "Engine":
		raw_health_status = safe_get("engine_health_status", "Normal")
		water_pump = safe_get("water_pump_speed", 0)
		coolant_temp = safe_get("coolant_temperature", 0)
		metrics = {}
		if water_pump is not None:
			metrics["Water pump speed"] = f"{int(water_pump):,} RPM"
		if coolant_temp is not None:
			metrics["Coolant temp"] = f"{float(coolant_temp):.0f}°C"
		
		# Calculate overall health from metrics to derive status
		health_scores = [get_metric_health(k, v, component) for k, v in metrics.items()]
		overall_health = int(sum(health_scores) / len(health_scores)) if health_scores else 75
		
		# Derive health_status from calculated overall_health
		if overall_health >= 80:
			health_status = "Normal"
		elif overall_health >= 50:
			health_status = "Warning"
		else:
			health_status = "Critical"
		
		# Determine status text based on actual conditions
		if health_status == "Normal":
			status_text = "Engine operating within normal parameters"
		elif water_pump and water_pump < 1000:
			status_text = "Water pump operating at reduced speed"
		elif coolant_temp and coolant_temp > 220:
			status_text = "Coolant temperature exceeds safe limit"
		else:
			status_text = "Engine requires attention"
	elif component == "Brakes":
		raw_health_status = safe_get("brake_health_status", "Normal")
		brake_pressure = safe_get("brake_pressure", 0)
		pad_wear = safe_get("brake_pad_wear_pct", 0)
		metrics = {}
		if brake_pressure is not None:
			metrics["Pressure"] = f"{float(brake_pressure):.0f} bar"
		if pad_wear is not None:
			metrics["Pad wear"] = f"{float(pad_wear):.0f}%"
		
		# Calculate overall health from metrics to derive status
		health_scores = [get_metric_health(k, v, component) for k, v in metrics.items()]
		overall_health = int(sum(health_scores) / len(health_scores)) if health_scores else 75
		
		# Derive health_status from calculated overall_health
		if overall_health >= 80:
			health_status = "Normal"
		elif overall_health >= 50:
			health_status = "Warning"
		else:
			health_status = "Critical"
		
		status_text = "Brakes functioning within normal parameters" if health_status == "Normal" else "Brake performance degradation detected"
	else:
		return
	
	# Status color and styling
	status_color = COLORS["healthy"] if health_status == "Normal" else (COLORS["warning"] if health_status == "Warning" else COLORS["critical"])
	
	# SVG Icons for components
	icon_svgs = {
		"Battery": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="6" y="4" width="12" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><line x1="18" y1="8" x2="20" y2="8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="18" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="18" y1="16" x2="20" y2="16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Fuel System": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 12C3 7.02944 7.02944 3 12 3C16.9706 3 21 7.02944 21 12C21 16.9706 16.9706 21 12 21C7.02944 21 3 16.9706 3 12Z" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M12 8V16M8 12H16" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><path d="M12 6L14 10L10 10L12 6Z" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
		"Engine": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.5" fill="none"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.5" fill="none"/><line x1="12" y1="4" x2="12" y2="8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="12" y1="16" x2="12" y2="20" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="4" y1="12" x2="8" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><line x1="16" y1="12" x2="20" y2="12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>',
		"Brakes": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="8" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M8 12L10.5 14.5L16 9" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'
	}
	icon_svg = icon_svgs.get(component, "")
	
	# Status badge HTML
	status_badge = _build_status_chip(health_status or "Status")
	
	# Inject CSS only once
	if not st.session_state.get("component_health_css_injected"):
		st.session_state["component_health_css_injected"] = True
		st.markdown(f"""
		<style>
			.component-metric-row {{
				display: flex;
				flex-direction: column;
				gap: 6px;
				padding: 10px 0;
				border-bottom: 1px solid rgba(255,255,255,0.05);
			}}
			.component-metric-row:last-child {{
				border-bottom: none;
			}}
			.component-metric-header {{
				display: flex;
				justify-content: space-between;
				align-items: center;
				gap: 12px;
				width: 100%;
				box-sizing: border-box;
			}}
			.component-metric-label {{
				color: {COLORS['text_muted']};
				font-size: 12px;
				font-weight: 600;
				text-transform: uppercase;
				letter-spacing: 0.5px;
				display: flex;
				align-items: center;
				gap: 8px;
				flex: 1;
				min-width: 0;
			}}
			.component-metric-label > span:last-child::after {{
				content: ": ";
				margin-left: 4px;
				color: {COLORS['text_muted']};
			}}
			.component-metric-value {{
				color: {COLORS['text_light']};
				font-size: 15px;
				font-weight: 600;
				white-space: nowrap;
				font-family: inherit;
				flex-shrink: 0;
				margin-left: auto;
				text-align: right;
				min-width: fit-content;
			}}
			.component-metric-bar-container {{
				width: 100%;
				height: 4px;
				background: rgba(255,255,255,0.05);
				border-radius: 2px;
				overflow: hidden;
				margin-top: 4px;
			}}
			.component-metric-bar {{
				height: 100%;
				border-radius: 2px;
				transition: width 0.5s ease;
			}}
			.component-health-indicator {{
				width: 8px;
				height: 8px;
				border-radius: 50%;
				display: inline-block;
				margin-right: 6px;
			}}
		</style>
		""", unsafe_allow_html=True)
	
	# Metrics HTML with progress bars
	metrics_html = ""
	metrics_list = list(metrics.items())
	for idx, (k, v) in enumerate(metrics_list):
		health_pct = get_metric_health(k, v, component)
		bar_color = COLORS["healthy"] if health_pct >= 80 else (COLORS["warning"] if health_pct >= 50 else COLORS["critical"])
		indicator_color = bar_color
		# Remove border for last item
		border_style = "" if idx == len(metrics_list) - 1 else "border-bottom: 1px solid rgba(255,255,255,0.05);"
		
		metrics_html += f'<div class="component-metric-row" style="display: flex; flex-direction: column; gap: 6px; padding: 10px 0; {border_style}"><div class="component-metric-header" style="display: flex; justify-content: space-between; align-items: center; gap: 12px; width: 100%; box-sizing: border-box;"><span class="component-metric-label" style="color: {COLORS["text_muted"]}; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 8px; flex: 1; min-width: 0;"><span class="component-health-indicator" style="background: {indicator_color}; box-shadow: 0 0 6px {indicator_color}60; width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 6px;"></span><span style="margin-right: 4px;">{k}:</span></span><span class="component-metric-value" style="color: {COLORS["text_light"]}; font-size: 15px; font-weight: 600; white-space: nowrap; font-family: inherit; flex-shrink: 0; margin-left: auto; text-align: right; min-width: fit-content;">{v}</span></div><div class="component-metric-bar-container" style="width: 100%; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; margin-top: 4px;"><div class="component-metric-bar" style="width: {health_pct}%; height: 100%; border-radius: 2px; transition: width 0.5s ease; background: linear-gradient(90deg, {bar_color}, {bar_color}80);"></div></div></div>'
	
	# overall_health is already calculated earlier for each component
	
	# Extract color values to avoid quote conflicts
	text_light = COLORS['text_light']
	text_muted = COLORS['text_muted']
	
	# Clock icon SVG
	clock_icon = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2Z" stroke="{status_color}" stroke-width="1.5" fill="none"/><path d="M12 6V12L16 14" stroke="{status_color}" stroke-width="1.5" stroke-linecap="round"/></svg>'
	
	# Grid icon SVG
	grid_icon = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M3 9H21M9 3V21" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>'
	
	# Enhanced card with gradient accent - using single line to avoid quote conflicts
	# Increase height for all component cards to accommodate multiple metrics
	card_height = "300px"
	# Get VIN to include in HTML for uniqueness (helps Streamlit detect changes)
	vin = get_row_value(row, "vin", default="")
	# Include VIN as data attribute to help with re-rendering
	card_html = f'<div class="dt-card" data-vin="{vin}" data-component="{component}" style="{GLASS_CARD_STYLE} border-radius: 14px; padding: 20px; margin-bottom: 12px; min-height: {card_height}; height: {card_height}; display: flex; flex-direction: column; position: relative; overflow: hidden; border-left: 3px solid {status_color};"><div style="position: absolute; top: 0; right: 0; width: 80px; height: 80px; background: radial-gradient(circle, {status_color}10 0%, transparent 70%); pointer-events: none;"></div><div style="display: flex; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.08); position: relative; z-index: 1;"><div style="display: flex; align-items: center; gap: 10px;"><div style="color: {status_color}; display: flex; align-items: center; justify-content: center; width: 36px; height: 36px; background: {status_color}15; border: 1px solid {status_color}30; border-radius: 10px; flex-shrink: 0; box-shadow: 0 0 12px {status_color}20;">{icon_svg}</div><span style="font-size: 15px; font-weight: 600; color: {text_light}; letter-spacing: 0.2px;">{component}</span></div>{status_badge}</div><div style="margin-bottom: 16px; flex-grow: 1; position: relative; z-index: 1;"><div style="color: {text_light}; font-size: 12px; line-height: 1.6; margin-bottom: 12px;">{status_text}</div><div style="display: flex; align-items: center; gap: 8px; padding: 8px 12px; background: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);">{clock_icon}<span style="color: {text_muted}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;">Health Score:</span><span style="color: {status_color}; font-size: 14px; font-weight: 700;">{overall_health}%</span></div></div><div style="margin-top: auto; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.08); position: relative; z-index: 1;"><div style="font-size: 11px; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 12px; font-weight: 600; display: flex; align-items: center; gap: 6px;">{grid_icon}Key Metrics</div><div style="display: flex; flex-direction: column;">{metrics_html if metrics_html else "<div style=\'color: rgba(148,163,184,0.6); font-size: 12px; padding: 8px 0;\'>No metrics available</div>"}</div></div></div>'
	
	st.markdown(card_html, unsafe_allow_html=True)

def render_digital_twin_dashboard(df_history: pd.DataFrame, df_inference: pd.DataFrame, 
								  selected_vin: Optional[str] = None) -> None:
	"""
	Main function to render the Digital Twin dashboard.
	
	Args:
		df_history: Historical CSV with telematics data
		df_inference: Inference log CSV with predictions
		selected_vin: Optional VIN to display (if None, uses most recent)
	"""
	try:
		if df_history.empty or df_inference.empty:
			st.warning("No data available for Digital Twin view. Please ensure both historical and inference data are loaded.")
			return
	except Exception as e:
		logger.error(f"Error checking data availability: {e}", exc_info=True)
		st.error(f"Error loading Digital Twin data: {e}")
		return
	
	# Add container styling for professional centered layout with narrower width and more margins
	st.markdown("""
	<style>
	@keyframes fadeInUp {
		from {
			opacity: 0;
			transform: translateY(20px);
		}
		to {
			opacity: 1;
			transform: translateY(0);
		}
	}
	@keyframes fadeIn {
		from {
			opacity: 0;
		}
		to {
			opacity: 1;
		}
	}
	@keyframes slideInRight {
		from {
			opacity: 0;
			transform: translateX(20px);
		}
		to {
			opacity: 1;
			transform: translateX(0);
		}
	}
	@keyframes pulse {
		0%, 100% {
			opacity: 1;
		}
		50% {
			opacity: 0.7;
		}
	}
	.digital-twin-container {
		max-width: 1100px;
		margin: 0 auto;
		padding: 0 80px;
		animation: fadeIn 0.6s ease-out;
	}
		@media (max-width: 1600px) {
			.digital-twin-container {
				max-width: 1000px;
				padding: 0 60px;
			}
		}
		@media (max-width: 1400px) {
			.digital-twin-container {
				max-width: 900px;
				padding: 0 50px;
			}
		}
		@media (max-width: 1200px) {
			.digital-twin-container {
				max-width: 100%;
				padding: 0 40px;
			}
		}
		@media (max-width: 768px) {
			.digital-twin-container {
				padding: 0 20px;
			}
		}
	.status-chip {
		display: inline-flex;
		align-items: center;
		gap: 6px;
		padding: 6px 14px;
		border-radius: 999px;
		font-size: 12px;
		font-weight: 600;
		letter-spacing: 0.2px;
		text-transform: uppercase;
		position: relative;
		overflow: hidden;
		transition: transform 0.2s ease, box-shadow 0.2s ease;
	}
	.status-chip:hover {
		transform: scale(1.05);
		box-shadow: 0 0 12px currentColor;
	}
	.status-chip .status-dot {
		width: 9px;
		height: 9px;
		border-radius: 50%;
		display: inline-block;
		box-shadow: 0 0 8px currentColor;
	}
	.status-chip--healthy {
		color: #10b981;
		border: 1px solid rgba(16,185,129,0.4);
		background: rgba(16,185,129,0.12);
	}
	.status-chip--warning {
		color: #fbbf24;
		border: 1px solid rgba(251,191,36,0.4);
		background: rgba(251,191,36,0.12);
	}
	.status-chip--critical {
		color: #ef4444;
		border: 1px solid rgba(239,68,68,0.4);
		background: rgba(239,68,68,0.12);
	}
	.dt-card {
		position: relative;
		border-radius: 14px;
		padding: 22px;
		overflow: hidden;
		animation: fadeInUp 0.5s ease-out;
		animation-fill-mode: both;
		transition: transform 0.3s ease, box-shadow 0.3s ease;
	}
	.dt-card:hover {
		transform: translateY(-2px);
		box-shadow: 0 8px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.1);
	}
	.dt-card::before {
		content: "";
		position: absolute;
		inset: 0;
		border-radius: inherit;
		border: 1px solid rgba(255,255,255,0.08);
		pointer-events: none;
		transition: border-color 0.3s ease;
	}
	.dt-card:hover::before {
		border-color: rgba(255,255,255,0.15);
	}
	.dt-card::after {
		content: "";
		position: absolute;
		width: 160%;
		height: 140%;
		top: -60%;
		left: -20%;
		background: radial-gradient(circle at top, rgba(255,255,255,0.12), transparent 60%);
		opacity: 0.35;
		pointer-events: none;
		transition: opacity 0.3s ease;
	}
	.dt-card:hover::after {
		opacity: 0.5;
	}
	.dt-card-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 12px;
	}
	.dt-card-divider {
		width: 100%;
		height: 1px;
		background: rgba(255,255,255,0.06);
		margin: 12px 0;
	}
	.dt-section {
		animation: fadeInUp 0.5s ease-out;
		animation-fill-mode: both;
	}
	.dt-section:nth-of-type(1) {
		animation-delay: 0.1s;
	}
	.dt-section:nth-of-type(2) {
		animation-delay: 0.2s;
	}
	.dt-section:nth-of-type(3) {
		animation-delay: 0.3s;
	}
	.dt-section:nth-of-type(4) {
		animation-delay: 0.4s;
	}
	.vin-selector-container {
		transition: all 0.3s ease;
	}
	.vin-selector-container:hover {
		transform: translateY(-1px);
	}
	.component-health-section {
		animation: fadeInUp 0.6s ease-out;
		animation-delay: 0.5s;
		animation-fill-mode: both;
	}
	.component-health-section [data-testid="column"],
	.component-health-section [data-testid="column"] > div,
	.component-health-section [data-testid="column"] > div > div,
	.component-health-section .stColumn,
	.component-health-section .stColumn > div,
	.component-health-section .stColumn > div > div,
	.component-health-section ~ div [data-testid="column"],
	.component-health-section ~ div [data-testid="column"] > div {
		align-items: flex-start !important;
		vertical-align: top !important;
		display: flex !important;
		flex-direction: column !important;
	}
	.component-health-section [data-testid="column"] > div > div,
	.component-health-section [data-testid="column"] > div > div > div {
		align-self: flex-start !important;
		margin-top: 0 !important;
	}
	.stSelectbox > div > div {
		transition: all 0.2s ease;
	}
	.stSelectbox > div > div:hover {
		box-shadow: 0 0 0 2px rgba(195,0,47,0.3);
	}
	</style>
	<div class="digital-twin-container">
	""", unsafe_allow_html=True)
	
	# Header - left-aligned with first metric card
	header_col_left, header_col, header_col2, header_col3, header_col_right = st.columns([.25, 1, 1, 1, .25], gap="medium")
	
	with header_col:
		st.markdown('<div class="card"><div class="card-header">Vehicle Digital Twin</div>', unsafe_allow_html=True)
	
	# Get VIN selection - label and dropdown on same row
	# Use LEFT JOIN approach: show all VINs with telematics data (from historical CSV)
	# Inference data will be joined if available, but not required
	# CRITICAL: Always show dropdowns and use session state - never use a default VIN
	# This ensures consistent behavior between initial load and explicit selection
	if selected_vin is None or "digital_twin_vin_select" not in st.session_state:
		if "vin" not in df_history.columns:
			st.warning("No VIN column found in historical data. Telematics data cannot be matched.")
			return

		model_options = []
		selected_model = None
		if "model" in df_history.columns:
			model_values = (
				df_history["model"]
				.astype(str)
				.str.strip()
			)
			valid_models = sorted(
				[v for v in model_values.unique().tolist() if v],
				key=lambda x: x.lower()
			)
			model_options = valid_models

		st.markdown('<div class="vin-selector-container dt-section">', unsafe_allow_html=True)
		selector_cols = st.columns([.15, .07, .25, .05, .07, .35, 1], gap="small")

		with selector_cols[1]:
			st.markdown(
				'<div style="font-size: 12px; color: #94a3b8; font-weight: 500; line-height: 1.4;">Model:</div>',
				unsafe_allow_html=True
			)

		with selector_cols[2]:
			selected_model = st.selectbox(
				"",
				options=model_options,
				index=0 if model_options else None,
				key="digital_twin_model_select",
				label_visibility="collapsed"
			)

		vin_df = df_history[df_history["vin"].notna()].copy()
		if selected_model and "model" in df_history.columns:
			selected_model_lower = selected_model.strip().lower()
			vin_df = vin_df[
				vin_df["model"]
				.astype(str)
				.str.strip()
				.str.lower()
				.eq(selected_model_lower)
			]

		all_vins = vin_df["vin"].astype(str).str.strip().unique().tolist()
		all_vins.sort()

		if not all_vins:
			st.warning("No VINs found in historical data for the selected model.")
			return

		# Check if the current VIN in session state is valid for the selected model
		# If not, reset it to the first VIN in the list
		current_vin = None
		if "digital_twin_vin_select" in st.session_state:
			current_vin = st.session_state.digital_twin_vin_select
			if current_vin and current_vin not in all_vins:
				# VIN from previous model selection is not valid, reset it
				current_vin = None
				if "digital_twin_vin_select" in st.session_state:
					del st.session_state.digital_twin_vin_select

		# Determine the index for the selectbox
		vin_index = 0
		if current_vin and current_vin in all_vins:
			vin_index = all_vins.index(current_vin)

		with selector_cols[4]:
			st.markdown(
				'<div style="font-size: 12px; color: #94a3b8; font-weight: 500; line-height: 1.4;">VIN:</div>',
				unsafe_allow_html=True
			)

		with selector_cols[5]:
			selected_vin = st.selectbox(
				"",
				options=all_vins,
				index=vin_index,
				key="digital_twin_vin_select",
				label_visibility="collapsed",
				on_change=None
			)

		st.markdown('</div>', unsafe_allow_html=True)
	
	# CRITICAL: Always use the VIN from the selectbox (via session state)
	# Never use a default or fallback - wait for user selection
	if "digital_twin_vin_select" not in st.session_state:
		# Selectbox hasn't been set yet - don't render anything
		return
	
	selected_vin = st.session_state.digital_twin_vin_select
	
	# Validate the VIN is valid
	if not selected_vin or (isinstance(selected_vin, str) and not selected_vin.strip()) or pd.isna(selected_vin):
		return
	
	# CRITICAL: Ensure the selected VIN is in the current model's VIN list
	# This prevents using a stale VIN from a previous model selection
	if "digital_twin_model_select" not in st.session_state:
		# Model selectbox hasn't been set yet - don't proceed
		return
		
	selected_model = st.session_state.digital_twin_model_select
	if not selected_model:
		return
		
	# Filter VINs for the selected model
	vin_df = df_history[df_history["vin"].notna()].copy()
	selected_model_lower = str(selected_model).strip().lower()
	vin_df = vin_df[
		vin_df["model"].astype(str).str.strip().str.lower() == selected_model_lower
	]
	valid_vins = vin_df["vin"].astype(str).str.strip().unique().tolist()
	
	# If the selected VIN is not in the valid list, don't proceed
	# This ensures we never use a VIN from a different model
	if selected_vin not in valid_vins:
		# VIN doesn't match selected model - clear it and return
		# This will force the user to select a valid VIN
		if "digital_twin_vin_select" in st.session_state:
			del st.session_state.digital_twin_vin_select
		return
	
	# Join data for selected VIN - ensure we're using the selected VIN
	try:
		merged_data = join_telematics_with_inference(df_history, df_inference, selected_vin)
	except Exception as e:
		logger.error(f"Error joining data for VIN {selected_vin}: {e}", exc_info=True)
		st.error(f"Error loading data for VIN {selected_vin}: {e}")
		return
	
	# Debug: Log the selected VIN and merged data
	if config and hasattr(config, 'debug') and config.debug:
		logger.debug(f"Selected VIN: {selected_vin}, Merged data rows: {len(merged_data)}")
	
	if merged_data.empty:
		# Provide more helpful error message
		try:
			has_vin_in_history = "vin" in df_history.columns and (
				df_history["vin"].notna() & 
				(df_history["vin"].astype(str).str.strip() == str(selected_vin).strip())
			).any()
			has_vin_in_inference = False
			if "vin" in df_inference.columns:
				has_vin_in_inference = (
					df_inference["vin"].notna() & 
					(df_inference["vin"].astype(str).str.strip() == str(selected_vin).strip())
				).any()
		except Exception as e:
			logger.error(f"Error checking VIN existence: {e}", exc_info=True)
			has_vin_in_history = False
			has_vin_in_inference = False
		
		if not has_vin_in_history and not has_vin_in_inference:
			st.warning(f"VIN '{selected_vin}' not found in historical or inference data. Please select a VIN that exists in the data.")
		elif not has_vin_in_history:
			st.warning(f"VIN '{selected_vin}' found in inference log but no matching telematics data in historical CSV. The historical CSV may need to be regenerated with telematics data.")
		else:
			st.warning(f"Unable to join telematics data for VIN: {selected_vin}. Please try another VIN.")
		return
	
	row = merged_data.iloc[0].to_dict()
	
	# CRITICAL: Ensure model_history is always set from the actual historical record for this VIN
	# This prevents inconsistencies when inference data has a different model
	vin_history = df_history[
		df_history["vin"].notna() & 
		(df_history["vin"].astype(str).str.strip() == str(selected_vin).strip())
	]
	if not vin_history.empty:
		# Get the most recent historical record for this VIN
		if "telematics_timestamp" in vin_history.columns:
			vin_history_record = vin_history.sort_values("telematics_timestamp", ascending=False).iloc[0]
		elif "date" in vin_history.columns:
			vin_history_record = vin_history.sort_values("date", ascending=False).iloc[0]
		else:
			vin_history_record = vin_history.iloc[0]
		
		# Force model_history to be the model from the actual historical record
		historical_model = vin_history_record.get("model")
		if historical_model:
			row["model_history"] = historical_model
			# Also ensure the model field uses history if inference doesn't have it or has wrong one
			if not row.get("model") or str(row.get("model")).strip().lower() != str(historical_model).strip().lower():
				row["model"] = historical_model
	
	# Top row: Vehicle Overview, Key Sensor Snapshot, and Health Score
	st.markdown('<div class="dt-section">', unsafe_allow_html=True)
	tcol_left, col1, col2, col3, col_right = st.columns([.25, 1, 1, 1, .25], gap="medium")
	
	with col1:
		render_vehicle_overview_card(row)
	
	with col2:
		render_sensor_snapshot(row)
	
	with col3:
		health_score, status = calculate_health_score(
			row.get("pred_prob_pct", 0),
			row.get("battery_soc"),
			row.get("dtc_severity")
		)
		render_health_score_gauge(health_score, status)
	
	st.markdown('</div>', unsafe_allow_html=True)
	
	# Second row: DTC Panel (left-aligned)
	st.markdown('<div class="dt-section" style="margin-top: 24px;">', unsafe_allow_html=True)
	tcol_left, col_dtc, col_test, col_right = st.columns([.17, 1, 1, .17], gap="medium")
	
	with col_dtc:
		render_dtc_panel(row)
	
	with col_test:
		render_anomalies_timeline(df_history, selected_vin)
	
	st.markdown('</div>', unsafe_allow_html=True)

	# Anomalies Timeline
	# st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
	# render_anomalies_timeline(df_history, selected_vin)
	
	# Component Health Cards
	# Inject CSS first to ensure it's applied before columns are rendered
	st.markdown("""
	<style>
	.component-health-section [data-testid="column"],
	.component-health-section [data-testid="column"] > div,
	.component-health-section [data-testid="column"] > div > div,
	.component-health-section [data-testid="column"] > div > div > div {
		align-items: flex-start !important;
		display: flex !important;
		flex-direction: column !important;
		vertical-align: top !important;
	}
	.component-health-section [data-testid="column"] > div > div {
		align-self: flex-start !important;
		margin-top: 0 !important;
	}
	</style>
	<script>
	// Force alignment on first load
	setTimeout(function() {
		var section = document.querySelector('.component-health-section');
		if (section) {
			var columns = section.querySelectorAll('[data-testid="column"]');
			columns.forEach(function(col) {
				col.style.alignItems = 'flex-start';
				col.style.display = 'flex';
				col.style.flexDirection = 'column';
				if (col.children.length > 0) {
					col.children[0].style.alignItems = 'flex-start';
					col.children[0].style.display = 'flex';
					col.children[0].style.flexDirection = 'column';
				}
			});
		}
	}, 100);
	</script>
	""", unsafe_allow_html=True)
	st.markdown('<div class="component-health-section" style="margin-top: 24px;">', unsafe_allow_html=True)
	col_left_header, col_header, col_empty1, col_empty2, col_right_header = st.columns([.25, 1, 1, 1, .25], gap="medium")
	
	with col_header:
		st.markdown(f"""
		<div style="font-size: 16px; font-weight: 600; color: {COLORS['text_light']}; margin-bottom: 16px;">
			Component Health
		</div>
		""", unsafe_allow_html=True)
	
	col_left, col5, col6, col7, col_right = st.columns([.25, 1, 1, 1, .25], gap="medium")
	
	# Determine if EV or non-EV model - always use model_history as source of truth
	model_name = str(get_row_value(row, "model_history", "model") or "").strip().lower()
	ev_models = {"leaf", "ariya"}
	is_ev_model = model_name in ev_models
	
	# Get VIN for unique container keys to force re-render on VIN change
	vin = get_row_value(row, "vin", default="")
	
	with col5:
		# Use container with unique key based on VIN and component type
		component_type = "Battery" if is_ev_model else "Fuel System"
		with st.container(key=f"component_card_1_{vin}_{component_type}"):
			if is_ev_model:
				render_component_health_card("Battery", row)
			else:
				render_component_health_card("Fuel System", row)
	
	with col6:
		# Use container with unique key based on VIN
		with st.container(key=f"component_card_2_{vin}_Engine"):
			render_component_health_card("Engine", row)
	
	with col7:
		# Use container with unique key based on VIN
		with st.container(key=f"component_card_3_{vin}_Brakes"):
			render_component_health_card("Brakes", row)
	
	st.markdown('</div>', unsafe_allow_html=True)
	
	# Close container div
	st.markdown("</div>", unsafe_allow_html=True)


