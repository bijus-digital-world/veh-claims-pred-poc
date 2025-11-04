import os
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from streamlit_autorefresh import st_autorefresh
import re
import html as _html
from datetime import datetime, timezone
import numpy as np  
import math
import streamlit.components.v1 as components
from pathlib import Path
import time

# Configuration
from config import config

# Logging
from utils.logger import app_logger as logger, log_dataframe_info, log_performance

# Voice services
try:
    from voice_service import create_voice_service
    voice_service = create_voice_service()
    VOICE_AVAILABLE = voice_service is not None and config.model.voice_enabled
    if not VOICE_AVAILABLE:
        if voice_service is None:
            logger.warning("Voice service creation returned None - check Streamlit logs for initialization errors")
        if not config.model.voice_enabled:
            logger.warning(f"Voice is disabled in config (voice_enabled={config.model.voice_enabled})")
    else:
        logger.info(f"Voice service initialized successfully (region: {config.aws.region}, bucket: {config.aws.s3_bucket})")
except Exception as e:
    logger.error(f"Voice services initialization failed: {e}", exc_info=True)
    voice_service = None
    VOICE_AVAILABLE = False

# Configure pydub to use FFmpeg if available (for audio processing)
# This helps streamlit-audiorecorder work properly
try:
    from pydub import AudioSegment
    import os
    # Try to find ffmpeg in common locations
    ffmpeg_paths = [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
    ]
    for path in ffmpeg_paths:
        if os.path.exists(path):
            AudioSegment.converter = path
            AudioSegment.ffprobe = path.replace("ffmpeg.exe", "ffprobe.exe")
            logger.info(f"Configured pydub to use FFmpeg at {path}")
            break
except Exception as e:
    logger.debug(f"Could not configure pydub FFmpeg paths: {e}")

if config.debug:
    logger.info(f"Voice available: {VOICE_AVAILABLE}, Voice enabled in config: {config.model.voice_enabled}")

# Local helper imports
from helper import (
    load_history_data,
    load_model,
    append_inference_log,
    append_inference_log_s3,
    random_inference_row_from_df,
    get_bedrock_summary,
    fetch_nearest_dealers,
    reverse_geocode,
    generate_enhanced_prescriptive_summary
)

from chat_helper import (
    build_tf_idf_index,
    generate_reply as chat_generate_reply,
    ensure_failures_column,
    retrieve_with_faiss_or_tfidf
)

try:
    import faiss
    HAS_FAISS = True
except Exception as e:
    HAS_FAISS = False
    st.warning("FAISS not installed — falling back to TF-IDF. To enable vector store, pip install faiss-cpu")


# Load configuration paths and constants
VECTOR_DIR = config.paths.vector_dir
IDX_PATH = config.paths.faiss_index_path
EMB_PATH = config.paths.embeddings_path
META_PATH = config.paths.metadata_path
JSON_META = config.paths.metadata_json_path

# Embedding model name
EMBED_MODEL_NAME = config.model.embedding_model_name

# AWS and S3 configuration
USE_S3 = config.aws.use_s3
S3_BUCKET = config.aws.s3_bucket
S3_KEY = config.paths.s3_data_key
AWS_REGION = config.aws.region
PLACE_INDEX_NAME = config.aws.place_index_name

# Model and log paths
MODEL_PATH = config.paths.model_path
LOG_FILE_LOCAL = config.paths.inference_log_local
LOG_FILE_S3_KEY = config.paths.inference_log_s3_key

# Chat log path
LOG_DIR = config.paths.logs_dir
CHAT_LOG_PATH = Path(config.paths.chat_log_file)

# Location and UI settings
LOCATION_PROB_THRESHOLD = config.data.location_prob_threshold
SHOW_REPAIR_COST = config.ui.show_repair_cost
IS_POC = config.ui.is_poc

# Data constants
MODELS = config.data.models
MILEAGE_BUCKETS = config.data.mileage_buckets
AGE_BUCKETS = config.data.age_buckets

# Color constants
NISSAN_RED = config.colors.nissan_red
NISSAN_GOLD = config.colors.nissan_gold
NISSAN_HEATMAP_SCALE = config.colors.heatmap_scale

# ------------------------
# Page config and styles
# ------------------------
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout,
    initial_sidebar_state=config.ui.initial_sidebar_state
)
# apply_style is in styles.py; we assume it's already imported/used as before
from styles import apply_style
apply_style()

# ------------------------
# Helper UI functions (added)
# ------------------------
def safe_sorted_unique(series):
    """Return sorted unique string values from a pandas Series."""
    vals = [v for v in pd.unique(series) if pd.notna(v)]
    # cast to str for consistent comparisons and sort case-insensitively
    vals = [str(v) for v in vals]
    return sorted(vals, key=lambda s: s.lower())


def calculate_risk_level(claim_pct: float) -> str:
    """
    Calculate risk level from claim percentage using configured thresholds.
    HIGH % = High likelihood of claim = More urgent action needed
    
    Args:
        claim_pct: Claim percentage (0-100)
    
    Returns:
        Risk level: "High", "Medium", or "Low" (based on claim likelihood)
    """
    if claim_pct >= config.risk.high_threshold:
        return "High"
    elif claim_pct >= config.risk.medium_threshold:
        return "Medium"
    return "Low"

@st.cache_resource(show_spinner=True)
def load_persisted_faiss():
    """Load FAISS index, embeddings, and metadata if available on disk."""
    # Attempting to load persisted FAISS index
    
    # Check if FAISS is available before attempting to load
    if not HAS_FAISS:
        logger.warning("FAISS library not installed - vector search will use TF-IDF fallback")
        return {
            "available": False,
            "index": None,
            "embs": None,
            "meta": None,
            "d": None,
            "message": "FAISS library not installed. Install with: pip install faiss-cpu"
        }
    
    if not IDX_PATH.exists() or not EMB_PATH.exists() or not META_PATH.exists():
        logger.warning(f"FAISS index files not found at {VECTOR_DIR}")
        # Run build_faiss_index.py to create FAISS index for faster retrieval
        return {
            "available": False,
            "index": None,
            "embs": None,
            "meta": None,
            "d": None,
            "message": "Persisted FAISS files missing. Run build_faiss_index.py first."
        }

    try:
        start_time = time.time()
        index = faiss.read_index(str(IDX_PATH))
        embs = np.load(EMB_PATH)
        meta = list(np.load(META_PATH, allow_pickle=True))
        d = embs.shape[1]
        duration_ms = (time.time() - start_time) * 1000
        
        # FAISS index loaded successfully
        
        return {
            "available": True,
            "index": index,
            "embs": embs,
            "meta": meta,
            "d": d,
            "message": f"Loaded persisted FAISS index: {len(meta)} vectors · dim={d}"
        }
    except Exception as e:
        logger.error(f"Failed to load persisted FAISS index: {e}", exc_info=True)
        return {
            "available": False,
            "index": None,
            "embs": None,
            "meta": None,
            "d": None,
            "message": f"Failed to load persisted FAISS: {e}"
        }

# call this once at startup
faiss_res = load_persisted_faiss()


def _extract_plain_and_bullets(summary_html):
    """
    From the HTML-safe summary produced by get_bedrock_summary, return:
      - plain_paragraph: the first narrative line with HTML tags stripped
      - bullets_html: the remaining text (bullets) kept as HTML (so bold/span remain)
      - risk_label: 'Low'|'Medium'|'High'
      - pct: percentage string if found (e.g., '65%') or empty
    """
    plain = re.sub(r'<[^>]+>', '', summary_html).strip()
    parts = re.split(r'\n\s*\n', plain, maxsplit=1)
    plain_paragraph = parts[0].strip() if parts else plain.strip()
    bullets_html = summary_html
    if len(parts) > 1:
        html_parts = re.split(r'\n\s*\n', summary_html, maxsplit=1)
        bullets_html = html_parts[1].strip() if len(html_parts) > 1 else summary_html

    m = re.search(r'\b(Low|Medium|High)\b', plain_paragraph, flags=re.IGNORECASE)
    risk_token = (m.group(1).capitalize() if m else "Medium")
    pct_match = re.search(r'\((\d{1,3})%\)', plain_paragraph)
    pct = f"{pct_match.group(1)}%" if pct_match else ""
    return plain_paragraph, bullets_html, risk_token, pct


def _badge_html(risk_token, pct):
    colors = {
        "Low": "#16a34a",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }
    color = colors.get(risk_token, "#f59e0b")
    pct_text = f" {pct}" if pct else ""
    return (
        f'<div style="flex: none; display:inline-flex; align-items:center; justify-content:center; '
        f'white-space:nowrap; padding:6px 12px; border-radius:10px; background:{color}; color:#fff; '
        f'font-weight:700; font-size:0.95rem; min-width:88px; text-align:center;">'
        f'{risk_token} risk{pct_text}</div>'
    )


def render_summary_ui(model_name, part_name, mileage_bucket, age_bucket, claim_pct, nearest_dealer=None, llm_model_id=None, region="us-east-1"):
    try:
        if claim_pct >= int(st.session_state.get('predictive_threshold_pct', 50)):
            # Use enhanced prescriptive summary for high-risk predictions (above threshold)
            summary_html = generate_enhanced_prescriptive_summary(
                model_name, part_name, mileage_bucket, age_bucket, claim_pct, 
                df_history, nearest_dealer
            )
        else:
            # Show hardcoded value (for predictions below threshold)
            risk_token = calculate_risk_level(claim_pct)
            pct = f"{round(claim_pct)}%"
            fallback = (
                f"The predicted claim probability is {round(claim_pct,1)}% for {part_name} in {model_name}. "
                "Continue routine monitoring and standard maintenance protocols."
            )
            summary_html = f"<strong>{risk_token} risk ({pct})</strong>: {_html.escape(fallback)}"

        if not summary_html:
            raise ValueError("Empty summary returned from generator")
    except Exception as e:
        logger.error(f"Enhanced prescriptive summary generation failed for {model_name}/{part_name}: {e}", exc_info=config.debug)
        if config.debug:
            st.warning(f"Enhanced prescriptive summary generation failed: {e}")
        
        # Fallback to Bedrock if enhanced summary fails
        try:
            if claim_pct >= int(st.session_state.get('predictive_threshold_pct', 50)):
                summary_html = get_bedrock_summary(model_name, part_name, mileage_bucket, age_bucket, claim_pct, 
                                                llm_model_id=llm_model_id, region=region)
            else:
                fallback = (
                    f"The predicted claim probability is {round(claim_pct,1)}% for {part_name} in {model_name}. "
                    "No immediate action recommended; monitor for trend changes."
                )
                risk_token = calculate_risk_level(claim_pct)
                pct = f"{round(claim_pct)}%"
                summary_html = f"<strong>{risk_token} risk ({pct})</strong>: {_html.escape(fallback)}"
        except Exception as e2:
            logger.error(f"Bedrock fallback also failed: {e2}")
            fallback = (
                f"The predicted claim probability is {round(claim_pct,1)}% for {part_name} in {model_name}. "
                "Continue routine monitoring and standard maintenance protocols."
            )
            risk_token = calculate_risk_level(claim_pct)
            pct = f"{round(claim_pct)}%"
            summary_html = f"<strong>{risk_token} risk ({pct})</strong>: {_html.escape(fallback)}"

    split_html = re.split(r'\n\s*\n', summary_html, maxsplit=1)
    first_para_html = split_html[0].strip()
    bullets_html = split_html[1].strip() if len(split_html) > 1 else ""

    m = re.search(r'\b(Low|Medium|High)\b', re.sub(r'<[^>]+>', '', first_para_html), flags=re.IGNORECASE)
    risk_token = (m.group(1).capitalize() if m else calculate_risk_level(claim_pct))
    pct_match = re.search(r'\((\d{1,3})%\)', re.sub(r'<[^>]+>', '', first_para_html))
    pct = f"{pct_match.group(1)}%" if pct_match else f"{round(claim_pct)}%"

    # Use enhanced summary for all modes (it handles threshold logic internally)
    combined_html = (
        '<div style="display:flex; align-items:center; gap:12px;">'
        f'{_badge_html(risk_token, pct)}'
        f'<div style="flex:1; min-width:0; font-size:1.02rem; line-height:1.35; color:#e6eef8; '
        f'overflow-wrap:break-word; word-break:break-word;">'
        f'{summary_html}'
        f'</div>'
        '</div>'
    )

    st.markdown(combined_html, unsafe_allow_html=True)
    if not IS_POC:
        st.write("")
        if bullets_html:
            # with st.expander("Details / Analyst Evidence", expanded=False):
            st.markdown(bullets_html, unsafe_allow_html=True)
        # else:
        #     # with st.expander("Details / Analyst Evidence", expanded=False):
        #     st.markdown("<div style='color:#94a3b8;'>No additional details.</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_history_cached(use_s3=None, s3_bucket=None, s3_key=None, local_path=None):
    """Load historical data with config defaults"""
    use_s3 = use_s3 if use_s3 is not None else config.aws.use_s3
    s3_bucket = s3_bucket or config.aws.s3_bucket
    s3_key = s3_key or config.paths.s3_data_key
    local_path = local_path or config.paths.local_data_file
    
    df_history = load_history_data(use_s3=use_s3, s3_bucket=s3_bucket, s3_key=s3_key, local_path=local_path)
    df_history = ensure_failures_column(df_history)
    return df_history


def persist_chat_to_disk(history):
    """
    Persist chat history to disk as CSV.
    
    Args:
        history: List of chat message dictionaries
    """
    df = pd.DataFrame(history)
    try:
        df.to_csv(CHAT_LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Failed to persist chat log: {e}")

# ------------------------
# Load data (S3 fallback to local)
# ------------------------
# Starting application - loading historical data
start_time = time.time()

try:
    df_history = load_history_cached()
    duration_ms = (time.time() - start_time) * 1000
    # Historical records loaded successfully
    log_dataframe_info(logger, "df_history", df_history)
except FileNotFoundError as e:
    logger.critical(f"Data load failed - file not found: {e}", exc_info=True)
    st.error(f"❌ Data load failed: {e}")
    st.stop()
except Exception as e:
    logger.critical(f"Unexpected error loading data: {e}", exc_info=True)
    st.error(f"❌ Unexpected error loading data: {e}")
    st.stop()

REQUIRED_COLS = config.data.required_columns
if not REQUIRED_COLS.issubset(df_history.columns):
    missing_cols = REQUIRED_COLS - set(df_history.columns)
    logger.error(f"Data validation failed - missing required columns: {missing_cols}")
    logger.error(f"Available columns: {list(df_history.columns)}")
    st.error(f"CSV data missing required columns: {missing_cols}. Please check your data file.")
    st.stop()
else:
    # Data validation passed - all required columns present
    pass

# Note: FAISS index is already loaded via load_persisted_faiss() at line ~152
# and stored in faiss_res global variable. No need to rebuild here.
# For TF-IDF fallback, chat helper will build it on-demand when needed.


# ------------------------
# Navbar (lightweight)
# ------------------------
icon_path = os.path.join("images", "maintenance_icon.svg")
logo_path = os.path.join("images", "nissan_logo.svg")
icon_b64 = ""
logo_b64 = ""
if os.path.exists(icon_path):
    with open(icon_path, "rb") as f:
        import base64
        icon_b64 = base64.b64encode(f.read()).decode("utf-8")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        import base64
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")

st.markdown(
    f"""
    <div class="navbar">
      <div style="display:flex;align-items:center;gap:10px;">
        <img src="data:image/svg+xml;base64,{icon_b64}" style="height:22px"/>
        <div class="title-column" style="display:flex;flex-direction:column;line-height:1;">
          <div class="title">Vehicle Predictive Insights</div>
          <div class="subtitle">Turning diagnostics into foresight</div>
        </div>
      </div>

      <div style="display:flex;align-items:center;gap:18px; font-size:12px; color:#374151;">
        <a href="?page=dashboard" style="text-decoration:none; color:inherit;">Dashboard</a>
        <a href="?page=inference" style="text-decoration:none; color:inherit;">Real-Time Vehicle Feed</a>
        <img src="data:image/svg+xml;base64,{logo_b64}" style="height:26px"/>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Routing
# ------------------------
query_params = st.query_params
page = query_params.get("page", "dashboard")

# ------------------------
# Email Confirmation Modal (render at top level)
# ------------------------
# Import email modal component
from email_modal_component import render_email_confirmation_modal

# Check if email confirmation should be shown and render dialog
modal_keys = [key for key in st.session_state.keys() if key.startswith("show_email_confirm_")]
if modal_keys:
    render_email_confirmation_modal()


# ------------------------
# Layout columns
# ------------------------
# Using 2 columns: Col1 (historical) and Col2 (predictive + map)
# Col2 is wider to accommodate both predictive analysis and map side by side
col1, col2 = st.columns([2, 3], gap="medium")
# col1_container = col1.empty()

# ------------------------
# Session state defaults
# ------------------------
available_models = ["All"] + safe_sorted_unique(df_history["model"])
available_parts = ["All"] + safe_sorted_unique(df_history["primary_failed_part"])

if "model_sel" not in st.session_state:
    st.session_state.model_sel = "All"
if "part_sel" not in st.session_state:
    st.session_state.part_sel = "All"
if "sev_abs" not in st.session_state:
    st.session_state.sev_abs = False

# ------------------------
# Rendering: Column 1 (Historical)
# ------------------------
@st.fragment
def render_col1():
    st.markdown('<div class="card"><div class="card-header">Historical Data Analysis</div>', unsafe_allow_html=True)
    dd1, dd2, dd3 = st.columns([1, 1, 0.6], gap="small")
    with dd1:
        st.selectbox("Model", options=available_models, key="model_sel")
    with dd2:
        st.selectbox("Primary Failed Part", options=available_parts, key="part_sel")
    with dd3:
        st.markdown("<div style='font-size:12px; color:#94a3b8; margin-bottom:6px;'>Show absolute claims</div>", unsafe_allow_html=True)
        st.checkbox("", value=st.session_state.sev_abs, key="sev_abs")

    model = st.session_state.model_sel
    part = st.session_state.part_sel
    sev_abs = st.session_state.sev_abs

    mask = pd.Series(True, index=df_history.index)
    if model != "All":
        mask &= df_history["model"] == model
    if part != "All":
        mask &= df_history["primary_failed_part"] == part
    filtered = df_history.loc[mask].copy()
    filtered["failures_count"] = (
                                    filtered[["claims_count", "repairs_count", "recalls_count"]]
                                    .fillna(0)
                                    .sum(axis=1)
                                    .astype(int)   
                                )

    if filtered.empty:
        st.markdown("<div style='padding:8px; color:#94a3b8;'>No historical records for this selection.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Number of incidents (rows)
    total_incidents = int(filtered.shape[0])

    # Total failure events (sum of claims + repairs + recalls)
    total_failures = int(filtered["failures_count"].sum())

    #  claims
    total_claims = int(filtered["claims_count"].sum())
    claim_rate = (total_claims / total_incidents * 100.0) if total_incidents > 0 else 0.0
    claim_tooltip_text = f"{total_claims} claims out of {total_incidents} incidents"

    # repairs
    total_repairs = int(filtered["repairs_count"].sum())
    repair_rate = (total_repairs / total_incidents * 100.0) if total_incidents > 0 else 0.0
    repair_tooltip_text = f"{total_repairs} repairs out of {total_incidents} incidents"

    # recalls
    total_recalls = int(filtered["recalls_count"].sum())
    recall_rate = (total_recalls / total_incidents * 100.0) if total_incidents > 0 else 0.0
    recall_tooltip_text = f"{total_recalls} recalls out of {total_incidents} incidents"

    top_repairs, top_claims, top_recalls = st.columns([1, 1, 1], gap="small")
    with top_repairs:
        label_text = "Total Repairs" if sev_abs else "Repair rate"
        main_value = f"{total_repairs}" if sev_abs else f"{repair_rate:.1f}%"
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div class="kpi-label">{label_text}</div>
              <div class="kpi-wrap" style="margin-top:2px;">
                <div class="kpi-num" style="color:{NISSAN_RED};;">{main_value}</div>
                <div class="kpi-tooltip">{repair_tooltip_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_claims:
        label_text = "Total Claims" if sev_abs else "Claim rate"
        main_value = f"{total_claims}" if sev_abs else f"{claim_rate:.1f}%"
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div class="kpi-label">{label_text}</div>
              <div class="kpi-wrap" style="margin-top:2px;">
                <div class="kpi-num" style="color:{NISSAN_RED};;">{main_value}</div>
                <div class="kpi-tooltip">{claim_tooltip_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_recalls:
        label_text = "Total Recalls" if sev_abs else "Recall rate"
        main_value = f"{total_recalls}" if sev_abs else f"{recall_rate:.1f}%"
        st.markdown(
            f"""
            <div style="text-align:center;">
              <div class="kpi-label">{label_text}</div>
              <div class="kpi-wrap" style="margin-top:2px;">
                <div class="kpi-num" style="color:{NISSAN_RED};;">{main_value}</div>
                <div class="kpi-tooltip">{recall_tooltip_text}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    bot_failures, bot_repair, bot_claim, bot_recall = st.columns([1, 1, 1, 1], gap="small")
    with bot_failures:
        st.markdown(
            f"""
            <div class="stat-centered">
              <div class="stat-label">Total Failures</div>
              <div class="stat-value" style="color:{NISSAN_RED};">{total_failures}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bot_repair:
        st.markdown(
            f"""
            <div class="stat-centered">
              <div class="stat-label">Repairs</div>
              <div class="stat-value">{total_repairs}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bot_claim:
        st.markdown(
            f"""
            <div class="stat-centered">
              <div class="stat-label">Claims</div>
              <div class="stat-value">{total_claims}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bot_recall:
        st.markdown(
            f"""
            <div class="stat-centered">
              <div class="stat-label">Recalls</div>
              <div class="stat-value">{total_recalls}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('<div style="height:3px; margin:12px 0; background: linear-gradient(90deg, #c3002f, #000000); border-radius:2px;"></div>', unsafe_allow_html=True)

    mileage_grp = filtered.groupby("mileage_bucket").agg(incidents=("failures_count", "size"), failures=("failures_count", "sum")).reindex(MILEAGE_BUCKETS, fill_value=0).reset_index()
    mileage_grp["rate_per_100"] = mileage_grp.apply(lambda r: (r["failures"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0, axis=1)

    age_grp = filtered.groupby("age_bucket").agg(incidents=("failures_count", "size"), failures=("failures_count", "sum")).reindex(AGE_BUCKETS, fill_value=0).reset_index()
    age_grp["rate_per_100"] = age_grp.apply(lambda r: (r["failures"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0, axis=1)

    r1c1, r1c2 = st.columns(2, gap="small")
    common_plot_kwargs = dict(template="plotly_dark")

    with r1c1:
        st.markdown('<div class="chart-header">Mileage:</div>', unsafe_allow_html=True)
        if sev_abs:
            fig_mileage = px.bar(mileage_grp, x="mileage_bucket", y="failures", labels={"mileage_bucket": "Mileage Bucket", "failures": "Failures"}, **common_plot_kwargs)
            fig_mileage.update_traces(width=0.33, marker_color="#9aa3ad")
        else:
            fig_mileage = px.bar(mileage_grp, x="mileage_bucket", y="rate_per_100", labels={"mileage_bucket": "Mileage Bucket", "rate_per_100": "Failure rate"}, **common_plot_kwargs)
            fig_mileage.update_traces(width=0.33, marker_color="#9aa3ad")
            fig_mileage.update_yaxes(tickformat=".1f")
        fig_mileage.update_layout(bargap=0.42, margin=dict(l=4, r=4, t=6, b=4), height=200, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_mileage.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=11, color="#94a3b8"))
        fig_mileage.update_xaxes(tickfont=dict(size=11, color="#94a3b8"))
        st.plotly_chart(fig_mileage, use_container_width=True)

    with r1c2:
        st.markdown('<div class="chart-header">Age:</div>', unsafe_allow_html=True)
        if sev_abs:
            fig_age = px.bar(age_grp, x="age_bucket", y="failures", labels={"age_bucket": "Age Bucket", "failures": "Failures"}, **common_plot_kwargs)
            fig_age.update_traces(width=0.33, marker_color="#93a0ad")
        else:
            fig_age = px.bar(age_grp, x="age_bucket", y="rate_per_100", labels={"age_bucket": "Age Bucket", "rate_per_100": "Failure rate"}, **common_plot_kwargs)
            fig_age.update_traces(width=0.33, marker_color="#93a0ad")
            fig_age.update_yaxes(tickformat=".1f")
        fig_age.update_layout(bargap=0.42, margin=dict(l=4, r=4, t=6, b=4), height=200, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_age.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.04)", zeroline=False, tickfont=dict(size=11, color="#94a3b8"))
        fig_age.update_xaxes(tickfont=dict(size=11, color="#94a3b8"))
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown('<div class="chart-header">Age vs. Mileage:</div>', unsafe_allow_html=True)
    pivot = filtered.groupby(["age_bucket", "mileage_bucket"]).agg(incidents=("failures_count", "size"), failures=("failures_count", "sum")).reset_index()
    all_cells = [{"age_bucket": a, "mileage_bucket": m} for a in AGE_BUCKETS for m in MILEAGE_BUCKETS]
    pivot = pd.DataFrame(all_cells).merge(pivot, on=["age_bucket", "mileage_bucket"], how="left").fillna(0)
    pivot["rate_per_100"] = pivot.apply(lambda r: (r["failures"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0, axis=1)

    heatmap_counts = pivot.pivot(index="age_bucket", columns="mileage_bucket", values="failures").reindex(index=AGE_BUCKETS, columns=MILEAGE_BUCKETS).fillna(0)
    heatmap_rate = pivot.pivot(index="age_bucket", columns="mileage_bucket", values="rate_per_100").reindex(index=AGE_BUCKETS, columns=MILEAGE_BUCKETS).fillna(0)

    z = heatmap_counts.values.astype(float) if sev_abs else heatmap_rate.values.astype(float)
    color_label = "Failures" if sev_abs else "Failure rate"

    fig_heat = px.imshow(z, labels=dict(x="Mileage Bucket", y="Age Bucket", color=color_label), x=heatmap_counts.columns.tolist(), y=heatmap_counts.index.tolist(), aspect="auto", color_continuous_scale=NISSAN_HEATMAP_SCALE, origin="lower", template="plotly_dark")
    fig_heat.update_layout(margin=dict(l=6, r=6, t=8, b=6), height=260, coloraxis_colorbar=dict(title=color_label), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

    max_val = float(z.max()) if z.size > 0 else 0.0
    annotations = []
    for i, age_label in enumerate(heatmap_counts.index.tolist()):
        for j, mileage_label in enumerate(heatmap_counts.columns.tolist()):
            val = float(z[i, j])
            ann_color = "#ffffff" if (max_val > 0 and val >= (0.45 * max_val)) else "#000000"
            txt = f"{int(val)}" if sev_abs else f"{val:.1f}%"
            annotations.append(dict(x=j, y=i, text=txt, showarrow=False, font=dict(size=10, color=ann_color)))
    fig_heat.update_layout(annotations=annotations)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="chart-header">Failure Trend:</div>', unsafe_allow_html=True)
    daily = filtered.groupby("date").agg(incidents=("failures_count", "size"), failures=("failures_count", "sum")).reset_index().sort_values("date")
    if daily.empty:
        st.markdown("<div style='padding:6px;color:#94a3b8;'>No trend data available for this selection.</div>", unsafe_allow_html=True)
    else:
        if sev_abs:
            fig_trend = px.line(daily, x="date", y="failures", labels={"date": "Date", "failures": "Failures"}, template="plotly_dark")
            fig_trend.update_traces(mode="lines+markers", line_color=NISSAN_RED, marker=dict(size=6))
            fig_trend.update_yaxes(title_text="Failures")
        else:
            daily["failure_pct"] = daily.apply(lambda r: (r["failures"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0, axis=1)
            fig_trend = px.line(daily, x="date", y="failure_pct", labels={"date": "Date", "failure_pct": "Failure rate"}, template="plotly_dark")
            fig_trend.update_traces(mode="lines+markers", line_color=NISSAN_RED, marker=dict(size=6))
            fig_trend.update_yaxes(title_text="Failure rate", tickformat=".1f")

        fig_trend.update_layout(margin=dict(l=4, r=4, t=6, b=4), height=220, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        fig_trend.update_xaxes(tickfont=dict(size=11, color="#94a3b8"))
        fig_trend.update_yaxes(tickfont=dict(size=11, color="#94a3b8"))
        st.plotly_chart(fig_trend, use_container_width=True)


    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# Column 2: Predictive + Prescriptive
# ------------------------
# Import modular components
from components_col2 import (
    render_predictive_controls,
    generate_prediction,
    log_inference,
    render_prescriptive_section,
    build_map_points,
    render_map_visualization
)

@st.fragment
def render_chat_interface():
    """
    Render chat interface as a separate fragment to minimize reruns.
    This will only refresh the chat section when chat interactions occur.
    """
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    
    # Render chat interface
    st.markdown('<div class="card"><div class="card-header">Vehicle Insights Companion</div>', unsafe_allow_html=True)

    # show faiss status 
    try:
        if faiss_res.get("available"):
            pass
        else:
            st.markdown(
                f"<div style='color:#fca5a5; font-size:12px'>FAISS unavailable: {faiss_res.get('message','missing')}</div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        logger.error(f"FAISS status check failed: {e}", exc_info=True)
        if config.debug:
            st.warning(f"FAISS status check failed: {e}")
        st.markdown("<div style='color:#fca5a5; font-size:12px'>FAISS status unknown</div>", unsafe_allow_html=True)
    
    # Show conversation context info
    try:
        if (hasattr(st.session_state, 'conversation_memory') and 
            st.session_state.conversation_memory and 
            hasattr(st.session_state.conversation_memory, 'memory') and 
            st.session_state.conversation_memory.memory):
            conv_summary = st.session_state.conversation_memory.get_conversation_summary()
            if conv_summary['total_exchanges'] > 0:
                st.markdown(
                    f"<div style='color:#94a3b8; font-size:11px; margin-bottom:8px;'>"
                    f"Conversation: {conv_summary['total_exchanges']} exchanges, "
                    f"{conv_summary['session_duration']:.0f}s duration</div>",
                    unsafe_allow_html=True,
                )
    except Exception as e:
        logger.debug(f"Failed to display conversation context: {e}")
        # Silently fail - this is just a display feature

    # ensure session-state chat history exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize conversation memory
    try:
        if "conversation_memory" not in st.session_state:
            from chat.conversation_memory import ConversationContext
            st.session_state.conversation_memory = ConversationContext(context_window=10)
        else:
            from chat.conversation_memory import ConversationContext
            # Check if it's already a ConversationContext object
            if not isinstance(st.session_state.conversation_memory, ConversationContext):
                st.session_state.conversation_memory = ConversationContext.load_from_session_state(st.session_state)
    except Exception as e:
        logger.error(f"Failed to initialize conversation memory: {e}", exc_info=True)
        from chat.conversation_memory import ConversationContext
        st.session_state.conversation_memory = ConversationContext(context_window=10)

    # ensure the input key exists BEFORE creating the widget
    if "chat_input_col3" not in st.session_state:
        st.session_state["chat_input_col3"] = ""

    # Build TF-IDF index once and cache results
    if "chat_tfidf_built" not in st.session_state:
        try:
            # Building TF-IDF index for chat
            start_time = time.time()
            VECT_CHAT, X_CHAT, HISTORY_ROWS_CHAT = build_tf_idf_index(df_history)
            duration_ms = (time.time() - start_time) * 1000
            # TF-IDF index built successfully
        except Exception as e:
            logger.error(f"TF-IDF index build failed: {e}", exc_info=True)
            if config.debug:
                st.warning(f"TF-IDF index build failed: {e}")
            VECT_CHAT, X_CHAT, HISTORY_ROWS_CHAT = None, None, df_history.to_dict(orient="records")
        st.session_state.chat_tfidf_built = True
        st.session_state.VECT_CHAT = VECT_CHAT
        st.session_state.X_CHAT = X_CHAT
        st.session_state.HISTORY_ROWS_CHAT = HISTORY_ROWS_CHAT
    else:
        VECT_CHAT = st.session_state.get("VECT_CHAT")
        X_CHAT = st.session_state.get("X_CHAT")
        HISTORY_ROWS_CHAT = st.session_state.get("HISTORY_ROWS_CHAT")

    # create a placeholder container where the chat pane will be rendered *once* at the end
    chat_container = st.container()

    # -------- helpers (same as before) --------
    def _render_chat_html_and_scroll():
        """Return the HTML that will be rendered inside components.html (includes JS scroll)."""
        pane_height = 300
        messages_html = ""
        if not st.session_state.chat_history:
            messages_html = '<div style="color:#6b7280; padding:6px;">No messages yet. Try: "claim rate for model Sentra" or "recent incidents".</div>'
        else:
            for m in st.session_state.chat_history:
                role = m.get("role", "user")
                ts = m.get("ts", "")
                if role == "user":
                    text = _html.escape(m.get("text", ""))
                    messages_html += (
                        f'<div style="text-align:right; margin-bottom:8px; color:#e6eef8;">'
                        f'<strong>You</strong> <span style="font-size:11px;color:#94a3b8;">{ts}</span>'
                        f'<div style="margin-top:4px;">{text}</div></div>'
                    )
                else:
                    assistant_html = (m.get("text") or "").replace("\n", "<br/>")
                    messages_html += (
                        f'<div style="text-align:left; margin-bottom:8px; color:#cfe9ff;">'
                        f'<strong>Assistant</strong> <span style="font-size:11px;color:#94a3b8;">{ts}</span>'
                        f'<div style="margin-top:4px;">{assistant_html}</div></div>'
                    )

        full_html = f"""
        <div id="chat-pane" style="
            height:{pane_height}px;
            overflow-y:auto;
            padding:8px;
            background:#0b1220;
            border-radius:6px;
            border: 1px solid #334155;
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        ">
            {messages_html}
        </div>

        <script>
        (function() {{
            const el = document.getElementById('chat-pane');
            if (el) {{
                // smooth scroll to bottom
                el.scrollTo({{ top: el.scrollHeight, left: 0, behavior: 'smooth' }});
            }}
        }})();
        </script>
        """
        return full_html, pane_height + 20

    # -------- Voice recorder (OUTSIDE form to avoid component registration issues) --------
    # Components inside forms with clear_on_submit=True lose registration, causing "unregistered ComponentInstance" error
    audio_bytes = None
    try:
        from audiorecorder import audiorecorder as streamlit_audiorecorder
        AUDIO_RECORDER_AVAILABLE = True
    except ImportError as e:
        try:
            from streamlit_audiorecorder import streamlit_audiorecorder
            AUDIO_RECORDER_AVAILABLE = True
        except ImportError:
            AUDIO_RECORDER_AVAILABLE = False
            if config.debug:
                logger.warning(f"streamlit-audiorecorder import failed: {e}")
    
    show_voice_button = AUDIO_RECORDER_AVAILABLE
    # Default for clear action (may be overridden when Clear button exists)
    clear = False
    
    with st.form("chat_form_col3", clear_on_submit=True):
        # Layout: create columns INSIDE the form (required by Streamlit)
        # Input | Send | Clear
        input_col, send_col, clear_col = st.columns([8, 1, 1], gap="small")
        
        with input_col:
            user_q = st.text_input(
                label="Ask something about the data:",
                key="chat_input_col3",
                placeholder="",
                label_visibility="collapsed",
            )
        
        if config.debug or not VOICE_AVAILABLE:
            with st.expander("🔍 Voice Debug Info", expanded=False):
                st.write(f"**AUDIO_RECORDER_AVAILABLE**: {AUDIO_RECORDER_AVAILABLE}")
                st.write(f"**VOICE_AVAILABLE**: {VOICE_AVAILABLE}")
                st.write(f"**voice_enabled (config)**: {config.model.voice_enabled}")
                st.write(f"**voice_service**: {voice_service is not None}")
                st.write(f"**show_voice_button**: {show_voice_button}")
                st.write(f"**AWS Region**: {config.aws.region}")
                st.write(f"**S3 Bucket**: {config.aws.s3_bucket}")
                
                # Show why voice is not available
                if not VOICE_AVAILABLE:
                    if voice_service is None:
                        st.error("**VoiceService creation failed** - Check Streamlit logs for error details")
                        st.info("Try: Check AWS credentials, IAM permissions, and S3 bucket access")
                    elif not config.model.voice_enabled:
                        st.warning("**Voice is disabled in config** - Set `VOICE_ENABLED=true` environment variable")
                if not AUDIO_RECORDER_AVAILABLE:
                    st.error("**Fix**: Install streamlit-audiorecorder in the same Python environment as Streamlit")
                    st.code("""
# Check which Python Streamlit uses:
streamlit --version
python --version

# Install using the same Python:
python -m pip install streamlit-audiorecorder==0.0.6 pydub==0.25.1

# Or if using virtual environment, activate it first:
# On Windows: venv\\Scripts\\activate
# On Mac/Linux: source venv/bin/activate
# Then: pip install streamlit-audiorecorder==0.0.6 pydub==0.25.1
                    """, language="bash")
                if not VOICE_AVAILABLE and config.model.voice_enabled:
                    st.warning("**Fix**: Check AWS credentials and IAM permissions. See AWS_CONSOLE_SETUP_GUIDE.md")
        
        with send_col:
            submitted = st.form_submit_button("→", use_container_width=True, help="Send message")

        # Clear chat button (trash icon)
        with clear_col:
            clear_chat = st.form_submit_button("🗑", use_container_width=True, help="Clear chat")

        # Cursor.ai-style design for chat input and buttons
        st.markdown("""
        <style>
        /* Professional Chat Input - rounded, modern */
        div[data-testid="stForm"] input[data-testid="textInput"] {
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            color: #f1f5f9 !important;
            padding: 10px 16px !important;
            font-size: 14px !important;
            font-weight: 400 !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        }
        
        div[data-testid="stForm"] input[data-testid="textInput"]:focus {
            border-color: #475569 !important;
            box-shadow: 0 0 0 2px rgba(71, 85, 105, 0.2) !important;
            outline: none !important;
        }
        
        div[data-testid="stForm"] input[data-testid="textInput"]:hover {
            border-color: #475569 !important;
        }
        
        div[data-testid="stForm"] input[data-testid="textInput"]::placeholder {
            color: #64748b !important;
            font-style: normal !important;
        }
        
        /* Cursor.ai-style Circular Buttons */
        /* Send button - circular with arrow icon */
        /* Hide submit button when microphone is visible (input is empty) */
        div[data-testid="stForm"] button[kind="formSubmit"]:first-of-type {
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            border: none !important;
            background: #e5e7eb !important;
            color: #1f2937 !important;
            font-size: 18px !important;
            font-weight: 500 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
            margin-top: 0 !important;
        }
        
        /* Send Button - small, professional, circular */
        div[data-testid="stForm"] button[kind="formSubmit"]:first-of-type {
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            border: none !important;
            background: #3b82f6 !important;
            color: white !important;
            font-size: 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
            margin: 0 !important;
        }
        
        div[data-testid="stForm"] button[kind="formSubmit"]:first-of-type:hover {
            background: #2563eb !important;
            transform: scale(1.05) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
        }
        
        div[data-testid="stForm"] button[kind="formSubmit"]:first-of-type:active {
            transform: scale(0.95) !important;
        }
        
        /* Clear button - if present */
        div[data-testid="stForm"] button[kind="formSubmit"]:last-of-type {
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            border: none !important;
            background: #6b7280 !important;
            color: white !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
        }
        
        div[data-testid="stForm"] button[kind="formSubmit"]:last-of-type:hover {
            background: #4b5563 !important;
            transform: scale(1.05) !important;
        }
        
        /* Microphone button - small, professional, circular */
        div[data-testid="stForm"] div[data-testid="element-container"]:has([data-testid="stComponentInstance"]) button,
        div[data-testid="stForm"] div[data-testid="element-container"]:has([data-testid="stComponentInstance"]) > div > div > button {
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            padding: 0 !important;
            border-radius: 50% !important;
            border: none !important;
            background: #6b7280 !important;
            color: white !important;
            font-size: 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1) !important;
            margin: 0 !important;
        }
        
        div[data-testid="stForm"] div[data-testid="element-container"]:has([data-testid="stComponentInstance"]) button:hover,
        div[data-testid="stForm"] div[data-testid="element-container"]:has([data-testid="stComponentInstance"]) > div > div > button:hover {
            background: #4b5563 !important;
            transform: scale(1.05) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Ensure microphone button container aligns properly */
        div[data-testid="stForm"] div[data-testid="element-container"]:has([data-testid="stComponentInstance"]) {
            margin: 0 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Hide default button text, show icon only */
        div[data-testid="stForm"] button[kind="formSubmit"] > p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
        
        /* Form container - minimal styling */
        div[data-testid="stForm"] {
            background: transparent !important;
            border: none !important;
            padding: 0 !important;
            border-radius: 0 !important;
        }
        
        /* Ensure buttons are aligned in the same row */
        div[data-testid="stForm"] > div[data-baseweb="grid"] > div {
            display: flex !important;
            align-items: center !important;
        }
        
        </style>
        <script>
        // Position microphone component aligned to the Clear button at far right
        (function(){
          function positionMic(){
            const form = document.querySelector('[data-testid="stForm"]');
            if(!form) return;
            const submitButtons = form.querySelectorAll('button[kind="formSubmit"]');
            if(submitButtons.length === 0) return;
            // Clear button is expected to be the last form submit button in the row
            const clearBtn = submitButtons[submitButtons.length - 1];
            // Find the mic component relative to our anchor
            const anchor = document.getElementById('voice-mic-anchor');
            let mic = null;
            if(anchor){
              let el = anchor.nextElementSibling;
              while(el && !mic){
                const cand = el.querySelector ? el.querySelector('[data-testid="stComponentInstance"]') : null;
                if(cand){ mic = cand; break; }
                el = el.nextElementSibling;
              }
            }
            if(!mic) return;
            const micContainer = mic.closest('[data-testid="element-container"]');
            if(!micContainer) return;
            const rect = clearBtn.getBoundingClientRect();
            micContainer.style.position = 'fixed';
            micContainer.style.top = (rect.top + rect.height/2 - 18) + 'px';
            // Place immediately to the right of the Clear button
            micContainer.style.left = (rect.right + 12) + 'px';
            micContainer.style.width = '36px';
            micContainer.style.height = '36px';
            micContainer.style.zIndex = '1000';
            const btn = micContainer.querySelector('button');
            if(btn){ btn.style.width='36px'; btn.style.height='36px'; btn.style.borderRadius='50%'; }
            // Create/update label pill
            const labelId = 'voice-mic-label-pill';
            let label = document.getElementById(labelId);
            if(!label){
              label = document.createElement('div');
              label.id = labelId;
              label.textContent = 'Click mic to ask vocally';
              label.style.position = 'fixed';
              label.style.background = '#111827';
              label.style.border = '1px solid #374151';
              label.style.color = '#E5E7EB';
              label.style.padding = '4px 10px';
              label.style.fontSize = '11px';
              label.style.lineHeight = '1';
              label.style.borderRadius = '9999px';
              label.style.boxShadow = '0 1px 2px rgba(0,0,0,0.25)';
              label.style.whiteSpace = 'nowrap';
              document.body.appendChild(label);
            }
            label.style.top = (rect.top + rect.height/2 - 10) + 'px';
            label.style.left = (rect.right - 70) + 'px';
          }
          positionMic();
          setTimeout(positionMic, 50);
          setTimeout(positionMic, 100);
          window.addEventListener('resize', positionMic);
          const obs = new MutationObserver(positionMic);
          obs.observe(document.body, {childList:true, subtree:true});
        })();
        </script>
        """, unsafe_allow_html=True)

    # Clear chat action (handled immediately after form submit)
    clear_chat_clicked = False
    text_submitted = False
    if 'clear_chat' in locals() and clear_chat:
        clear_chat_clicked = True
        st.session_state.chat_history = []
        # Clear any pending audio when clearing chat
        st.session_state["pending_audio_bytes"] = None
        st.session_state["pending_audio_format"] = None
        st.session_state["pending_audio_id"] = None
        try:
            # Reset conversation memory if available
            from chat.conversation_memory import ConversationContext
            st.session_state.conversation_memory = ConversationContext(context_window=10)
        except Exception:
            st.session_state.pop('conversation_memory', None)
    
    # Check if Send button was clicked (text submission)
    if 'submitted' in locals() and submitted:
        text_submitted = True
        # Clear any pending audio when sending text (text takes priority)
        st.session_state["pending_audio_bytes"] = None
        st.session_state["pending_audio_format"] = None
        st.session_state["pending_audio_id"] = None

    # Render microphone OUTSIDE the form (stable) and capture audio reliably
    audio_bytes = None
    if show_voice_button:
        st.markdown('<div id="mic-row-anchor"></div>', unsafe_allow_html=True)
        _sp1, _sp2, _sp3, label_col, button_col = st.columns([7, 1, 1, 2, 1], gap="small")
        with label_col:
            st.markdown(
                '<div style="font-size: 11px; color: #94a3b8; margin-top: 4px; text-align: center;">Click mic to ask vocally</div>',
                unsafe_allow_html=True,
            )
        with button_col:
            try:
                audio_bytes = streamlit_audiorecorder(
                    start_prompt="🎤",
                    stop_prompt="⏹",
                    pause_prompt="",
                    show_visualizer=False,
                    key="voice_recorder_static"
                )
                if audio_bytes and len(audio_bytes) > 0:
                    try:
                        import io, hashlib
                        audio_buffer = io.BytesIO()
                        audio_bytes.export(audio_buffer, format="wav")
                        audio_buffer.seek(0)
                        audio_bytes_data = audio_buffer.read()
                        st.session_state["pending_audio_bytes"] = audio_bytes_data
                        st.session_state["pending_audio_format"] = "wav"
                        audio_hash = hashlib.md5(audio_bytes_data[:1000] if len(audio_bytes_data) > 1000 else audio_bytes_data).hexdigest()[:12]
                        st.session_state["pending_audio_id"] = f"audio_{int(time.time())}_{audio_hash}"
                        logger.info(f"✅ Stored audio in session_state ({len(audio_bytes_data)} bytes), ID: {st.session_state['pending_audio_id']}")
                    except Exception as e:
                        logger.error(f"❌ Failed to store audio in session_state: {e}", exc_info=True)
                        if config.debug:
                            st.error(f"Failed to store audio: {e}")
            except FileNotFoundError as ffmpeg_error:
                if "ffprobe" in str(ffmpeg_error).lower() or "ffmpeg" in str(ffmpeg_error).lower():
                    if config.debug:
                        st.error("⚠️ FFmpeg not installed.")
                    audio_bytes = None
                else:
                    raise
            except Exception as recorder_error:
                logger.warning(f"Voice recording error: {recorder_error}", exc_info=True)
                audio_bytes = None

    # -------- Process audio AFTER form (if recorded) --------
    # Check both current audio_bytes and session_state for pending audio
    # (in case form clear_on_submit cleared the local variable)
    pending_audio_bytes = st.session_state.get("pending_audio_bytes")
    pending_audio_id = st.session_state.get("pending_audio_id")
    
    # Prevent duplicate processing using audio hash/ID
    current_audio_id = None
    if audio_bytes and len(audio_bytes) > 0:
        # Generate ID for current audio
        try:
            import hashlib
            audio_hash = hashlib.md5(str(len(audio_bytes)).encode()).hexdigest()[:8]
            current_audio_id = f"audio_{audio_hash}"
        except:
            current_audio_id = None
    
    if pending_audio_bytes and len(pending_audio_bytes) > 0 and not clear_chat_clicked and not text_submitted:
        logger.info(f"🔍 Found pending audio in session_state ({len(pending_audio_bytes)} bytes), ID: {pending_audio_id}")
        # Check if we already processed this audio
        if pending_audio_id and pending_audio_id == st.session_state.get("_last_processed_audio_id"):
            logger.debug("⏭️ Skipping duplicate audio processing")
            audio_bytes = None
        else:
            # Use pending audio from session_state
            if not audio_bytes or len(audio_bytes) == 0:
                # Reconstruct AudioSegment from stored bytes for processing
                try:
                    from pydub import AudioSegment
                    import io
                    logger.debug(f"🔄 Reconstructing AudioSegment from {len(pending_audio_bytes)} bytes")
                    audio_buffer = io.BytesIO(pending_audio_bytes)
                    audio_bytes = AudioSegment.from_wav(audio_buffer)
                    logger.info(f"✅ Successfully reconstructed AudioSegment ({len(audio_bytes)} frames)")
                except Exception as e:
                    logger.error(f"❌ Failed to reconstruct audio from session_state: {e}", exc_info=True)
                    audio_bytes = None
            else:
                logger.debug(f"✅ Using live audio_bytes ({len(audio_bytes)} frames)")
    
    if audio_bytes and len(audio_bytes) > 0 and not clear_chat_clicked and not text_submitted:
        if not VOICE_AVAILABLE or voice_service is None:
            st.error("⚠️ Voice service not configured. Please set up AWS credentials and IAM permissions.")
            logger.error("Voice recording attempted but voice service is not available")
        else:
            try:
                with st.spinner("🎤 Transcribing audio..."):
                    # Convert AudioSegment to bytes (optimized format for faster processing)
                    import io
                    audio_buffer = io.BytesIO()
                    
                    # Optimize audio: 16kHz mono WAV (smaller file, faster upload/processing)
                    # This reduces file size significantly while maintaining transcription quality
                    try:
                        # Try compressed format first (faster)
                        audio_bytes.export(
                            audio_buffer, 
                            format="wav",
                            parameters=["-ar", "16000", "-ac", "1"]  # 16kHz sample rate, mono channel
                        )
                    except Exception:
                        # Fallback to standard WAV if compression fails
                        audio_bytes.export(audio_buffer, format="wav")
                    
                    audio_buffer.seek(0)
                    audio_bytes_data = audio_buffer.read()
                    
                    logger.debug(f"Audio prepared: {len(audio_bytes_data)} bytes for transcription")
                    
                    transcription_result = voice_service.transcribe_audio_bytes(
                        audio_bytes=audio_bytes_data,
                        language_code=config.model.transcribe_language_code,
                        media_format="wav"
                    )
                    transcribed_text = transcription_result.get("text", "").strip()
                    
                    # Mark this audio as processed to prevent duplicate processing
                    if pending_audio_id:
                        st.session_state["_last_processed_audio_id"] = pending_audio_id
                    elif current_audio_id:
                        st.session_state["_last_processed_audio_id"] = current_audio_id
                    
                    # Clear pending audio from session_state after successful transcription
                    st.session_state["pending_audio_bytes"] = None
                    st.session_state["pending_audio_format"] = None
                    st.session_state["pending_audio_id"] = None
                    
                    if transcribed_text:
                        # Process query immediately (same as Send button) - no rerun needed
                        q = transcribed_text.strip()
                        
                        if q:
                            logger.info(f"Processing voice transcription immediately: '{q[:50]}...'")
                            
                            # Generate submission ID
                            import uuid
                            submission_id = str(uuid.uuid4())
                            ts = datetime.now(timezone.utc).isoformat()
                            
                            # Check for duplicate (same as form submission)
                            last_user_text = None
                            if st.session_state.chat_history:
                                for m in reversed(st.session_state.chat_history):
                                    if m.get("role") == "user":
                                        last_user_text = m.get("text")
                                        break
                            
                            if last_user_text is not None and last_user_text.strip() == q:
                                st.info("This message was already submitted. Showing existing response.")
                            else:
                                # Add user message to chat history
                                st.session_state.chat_history.append({
                                    "role": "user",
                                    "text": q,
                                    "ts": ts,
                                    "id": submission_id
                                })
                                
                                # Generate assistant reply (same as Send button)
                                start_time = time.time()
                                try:
                                    assistant_html = chat_generate_reply(
                                        q,
                                        df_history,
                                        faiss_res,
                                        VECT_CHAT,
                                        X_CHAT,
                                        HISTORY_ROWS_CHAT,
                                        get_bedrock_summary,
                                        top_k=6,
                                        conversation_context=st.session_state.conversation_memory
                                    )
                                    duration_ms = (time.time() - start_time) * 1000
                                    logger.info(f"Generated reply for voice query in {duration_ms:.2f}ms")
                                except Exception as e:
                                    logger.error(f"Error generating reply for voice query '{q[:50]}...': {e}", exc_info=True)
                                    assistant_html = f"<p>Error generating reply: {_html.escape(str(e))}</p>"
                                    duration_ms = (time.time() - start_time) * 1000

                                # Add exchange to conversation memory
                                st.session_state.conversation_memory.add_exchange(
                                    query=q,
                                    response=assistant_html,
                                    exchange_id=submission_id,
                                    handler_used="QueryRouter",
                                    processing_time_ms=duration_ms
                                )
                                
                                # Save conversation memory to session state
                                st.session_state.conversation_memory.save_to_session_state(st.session_state)

                                st.session_state.chat_history.append({
                                    "role":"assistant",
                                    "text":assistant_html,
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "id": submission_id
                                })

                                # persist (best-effort)
                                try:
                                    persist_chat_to_disk(st.session_state.chat_history)
                                    logger.debug("Chat history persisted to disk")
                                except Exception as e:
                                    logger.warning(f"Failed to persist chat history: {e}")
                                    if config.debug:
                                        st.warning(f"Failed to persist chat history: {e}")
                                
                                # Note: Cannot modify st.session_state["chat_input_col3"] after widget instantiation
                                # The form will clear automatically on next submission due to clear_on_submit=True
                                logger.info("Voice query processed and added to chat history - no rerun needed")
                    else:
                        st.warning("Could not transcribe audio. Please try again.")
            except Exception as e:
                logger.error(f"Voice transcription error: {e}", exc_info=True)
                st.error(f"Transcription failed: {str(e)}. Check AWS setup.")

    # -------- handlers (no rendering here) --------
    # Initialize clear to False if not set (e.g., when voice/mic is shown and Clear button absent)
    try:
        _should_clear = bool(clear)
    except NameError:
        _should_clear = False
    if _should_clear:
        st.session_state.chat_history = []
        # do not rerun — simply let the function continue to the final render below
        # the final render will show an empty chat pane
    
    if submitted:
        q = (st.session_state.get("chat_input_col3") or "").strip()
        if not q:
            st.warning("Please type a question first.")
        else:
            # dedupe guard (prevent duplicate appends on accidental double-submit)
            last_user_text = None
            if st.session_state.chat_history:
                for m in reversed(st.session_state.chat_history):
                    if m.get("role") == "user":
                        last_user_text = m.get("text")
                        break

            if last_user_text is not None and last_user_text.strip() == q:
                st.info("This message was already submitted. Showing existing response.")
                # do nothing — final render below will show existing messages
            else:
                import uuid

                submission_id = str(uuid.uuid4())
                ts = datetime.now(timezone.utc).isoformat()

                # append user message
                st.session_state.chat_history.append({"role":"user","text":q,"ts":ts,"id":submission_id})

                # generate assistant reply (synchronous)
                try:
                    # Generating chat reply
                    start_time = time.time()
                    
                    assistant_html = chat_generate_reply(
                        q,
                        df_history,
                        faiss_res,
                        VECT_CHAT,
                        X_CHAT,
                        HISTORY_ROWS_CHAT,
                        get_bedrock_summary,
                        top_k=6,
                        conversation_context=st.session_state.conversation_memory
                    )
                    
                    duration_ms = (time.time() - start_time) * 1000
                    # Chat reply generated successfully
                    
                except Exception as e:
                    logger.error(f"Chat reply generation failed for query '{q[:50]}...': {e}", exc_info=True)
                    assistant_html = f"<p>Error generating reply: {_html.escape(str(e))}</p>"

                # Add exchange to conversation memory
                st.session_state.conversation_memory.add_exchange(
                    query=q,
                    response=assistant_html,
                    exchange_id=submission_id,
                    handler_used="QueryRouter",  # Could be enhanced to track specific handler
                    processing_time_ms=duration_ms
                )
                
                # Save conversation memory to session state
                st.session_state.conversation_memory.save_to_session_state(st.session_state)

                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text":assistant_html,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "id": submission_id
                })

                # persist (best-effort)
                try:
                    persist_chat_to_disk(st.session_state.chat_history)
                    logger.debug("Chat history persisted to disk")
                except Exception as e:
                    logger.warning(f"Failed to persist chat history: {e}")
                    if config.debug:
                        st.warning(f"Failed to persist chat history: {e}")

    # -------- final single render of the chat pane (exactly once per run) --------
    full_html, comp_height = _render_chat_html_and_scroll()
    # render inside the earlier placeholder container so it appears above the form in layout order
    with chat_container:
        components.html(full_html, height=comp_height, scrolling=False)

    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment
def render_col2():
    """
    Render Column 2: Predictive & Prescriptive Analysis + Chat.
    
    This orchestrates the modular components for:
    - Predictive analysis controls and KPIs
    - Prescriptive summary with dealer recommendations
    - Download inference log
    - Chat interface below everything
    """
    # Load model
    model_pipe = load_model(MODEL_PATH)
    
    # Card container for predictive analysis
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # 1. Render controls and get refresh interval
    refresh_interval = render_predictive_controls()
    st_autorefresh(interval=refresh_interval * 1000, key="predictive_autorefresh")

    # 2. Generate prediction
    inf_row, pred_prob = generate_prediction(df_history, model_pipe)
    
    # 3. Log inference
    log_inference(inf_row, pred_prob)
    
    # 4. Get nearest dealers data FIRST (outside any column context to avoid nesting)
    nearest_dealers = None
    if inf_row and pred_prob is not None:
        # Get the data without rendering (to avoid column nesting)
        from helper import fetch_nearest_dealers
        from components_col2 import get_constants
        
        constants = get_constants()
        current_lat = inf_row.get("lat", 38.4405)
        current_lon = inf_row.get("lon", -122.7144)
        
        try:
            nearest_dealers, from_aws = fetch_nearest_dealers(
                current_lat=current_lat,
                current_lon=current_lon,
                place_index_name=constants['PLACE_INDEX_NAME'],
                aws_region=constants['AWS_REGION'],
                fallback_dealers=None,
                text_query="Nissan Service Center",
                top_n=3,
            )
        except Exception as e:
            logger.error(f"fetch_nearest_dealers failed: {e}")
            nearest_dealers = []
    
    # 5. Create side-by-side layout with Predictive Information and Map
    pred_col, map_col = st.columns([1.5, 1], gap="medium")
    
    with pred_col:
        # Render vehicle info and KPI as pure HTML (no nested columns)
        pct_text = f"{pred_prob*100:.1f}%"
        
        # For POC, override with fixed value
        if config.ui.is_poc:
            pct_text = f"{80.8:.1f}%"
        
        # Get threshold from session state
        threshold_pct = st.session_state.get('predictive_threshold_pct', 80)
        
        # Determine KPI color based on threshold
        kpi_color = config.colors.nissan_red if pred_prob * 100 >= threshold_pct else '#10b981'
        
        # Handle POC mode vs regular mode for display values
        if config.ui.is_poc:
            model_display = inf_row['model']
            part_display = inf_row['primary_failed_part']
            mileage_display = "10,200 miles"
            age_display = "6 months"
        else:
            model_display = inf_row['model']
            part_display = inf_row['primary_failed_part']
            mileage_display = f"{inf_row['mileage']:,.1f} mi"
            age_display = f"{inf_row['age']:.1f} yrs"
        
        # Build the HTML for vehicle info and KPI side by side (no HTML comments to avoid display issues)
        vehicle_info_html = f"""<div style="display: flex; gap: 16px; align-items: stretch; margin-top: -24px;">
            <div style="flex: 1;">
                <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 3px;">Model</div>
                <div style="font-size: 14px; font-weight: 700; color: #e6eef8; margin-bottom: 12px;">{model_display}</div>
                <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 3px;">Part</div>
                <div style="font-size: 14px; font-weight: 700; color: #e6eef8;">{part_display}</div>
            </div>
            <div style="flex: 0.6;">
                <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 3px;">Mileage</div>
                <div style="font-size: 14px; font-weight: 700; color: #e6eef8; margin-bottom: 12px;">{mileage_display}</div>
                <div style="font-size: 12px; color: #cbd5e1; margin-bottom: 3px;">Age</div>
                <div style="font-size: 14px; font-weight: 700; color: #e6eef8;">{age_display}</div>
            </div>
            <div style="width: 1px; background: linear-gradient(to bottom, transparent, #334155 20%, #334155 80%, transparent); margin: 0 8px;"></div>
            <div style="flex: 1; display: flex; flex-direction: column; justify-content: center;">
                <div style="font-size: 13px; color: #cbd5e1; margin-bottom: 2px; text-align: center;">Predicted Claim Probability</div>
                <div style="font-size: 34px; font-weight: 700; color: {kpi_color}; text-align: center; line-height: 1;">{pct_text}</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px; text-align: center;">Alert if ≥ {int(threshold_pct)}%</div>
            </div>
        </div>"""
        
        st.markdown(vehicle_info_html, unsafe_allow_html=True)
        st.markdown('<div style="height:2px;"></div>', unsafe_allow_html=True)
        
        # Render prescriptive summary below the predictive information (left side)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        
        if inf_row and pred_prob is not None and nearest_dealers is not None:
            # Render prescriptive summary (this creates its own columns internally)
            render_prescriptive_section(inf_row, pred_prob, render_summary_ui)
        else:
            st.markdown('<div class="card"><div class="card-header">Prescriptive Summary</div>', unsafe_allow_html=True)
            # st.markdown("<div style='padding:12px; color:#94a3b8;'>Waiting for prediction data...</div>", unsafe_allow_html=True)
            # st.markdown('</div>', unsafe_allow_html=True)
    
    with map_col:
        # Render map on the right side of Predictive Information
        if inf_row and pred_prob is not None and nearest_dealers is not None:
            # Render map with dealers and current location
            map_points, dealers_to_plot = build_map_points(inf_row, nearest_dealers, pred_prob)
            render_map_visualization(map_points, dealers_to_plot)
        else:
            # Show placeholder when no data available
            st.markdown('<div class="card"><div class="card-header">Vehicle Location & Nearest Dealers</div>', unsafe_allow_html=True)
            # st.markdown("<div style='padding:12px; color:#94a3b8;'>Waiting for prediction data...</div>", unsafe_allow_html=True)
            # st.markdown('</div>', unsafe_allow_html=True)
        

    # Prescriptive summary is now rendered inside pred_col (left side)

    # st.markdown("</div>", unsafe_allow_html=True)
    
    # Add custom CSS to override Streamlit's default expander spacing and reduce gaps
    # st.markdown("""
    # <style>
    # .streamlit-expander {
    #     margin-top: -20px !important;
    #     margin-bottom: -10px !important;
    # }
    # .streamlit-expander > div {
    #     margin-top: -15px !important;
    #     margin-bottom: -5px !important;
    # }
    # .stPydeckChart {
    #     margin-bottom: -15px !important;
    # }
    # </style>
    # """, unsafe_allow_html=True)

# ------------------------
# Inference page (separate route)
# ------------------------
if page == "inference":
    # Create a compact header with controls on the same row
    st.markdown('<div style="height:1px;"></div>', unsafe_allow_html=True)
    
    # Header row with title and controls
    header_col1, header_col2, header_col3, header_col4 = st.columns([2, 1.5, 1.2, 1], gap="small")
    
    with header_col1:
        st.markdown('<div class="card-header" style="margin-bottom:0; padding:8px 12px;">Real-Time Vehicle Feed</div>', unsafe_allow_html=True)
    
    if not os.path.isfile(LOG_FILE_LOCAL):
        st.markdown("<div style='padding:12px; color:#94a3b8;'>No real-time vehicle information found yet. Predictions will be logged as they run.</div>", unsafe_allow_html=True)
    else:
        df_log = pd.read_csv(LOG_FILE_LOCAL, parse_dates=["timestamp"])
        
        with header_col2:
            min_date = df_log["timestamp"].min().date()
            max_date = df_log["timestamp"].max().date()
            date_range = st.date_input("Date range", value=(min_date, max_date), key="inference_date_range")
        with header_col3:
            text_filter = st.text_input("Filter (model / part)", value="", key="inference_text_filter")
        with header_col4:
            rows_to_show = st.selectbox("Rows", options=[25, 50, 100, 500, 1000], index=1, key="inference_rows_count")

        dr_start, dr_end = date_range
        mask = (df_log["timestamp"].dt.date >= dr_start) & (df_log["timestamp"].dt.date <= dr_end)
        if text_filter.strip():
            t = text_filter.strip().lower()
            mask &= df_log["model"].str.lower().str.contains(t) | df_log["primary_failed_part"].str.lower().str.contains(t)
        df_show = df_log[mask].sort_values("timestamp", ascending=False).head(rows_to_show)

        # Rename columns for better readability and hide pred_prob
        df_display = df_show.copy()
        
        # Define column renaming (handles both old bucket and new continuous formats)
        column_renames = {
            "timestamp": "Event Timestamp",
            "model": "Model",
            "primary_failed_part": "Primary Failed Part",
            "mileage": "Mileage",
            "mileage_bucket": "Mileage",  # Old format (backward compatibility)
            "age": "Age",
            "age_bucket": "Age",          # Old format (backward compatibility)
            "pred_prob_pct": "Predictive %"
        }
        
        # Rename columns that exist
        df_display = df_display.rename(columns=column_renames)
        
        # Hide pred_prob column if it exists
        columns_to_display = [col for col in df_display.columns if col != "pred_prob"]
        df_display = df_display[columns_to_display]

        # Import enhanced table functionality
        from enhanced_inference_table import render_enhanced_inference_table
        
        # Render enhanced table with action buttons
        render_enhanced_inference_table(df_log, date_range, text_filter, rows_to_show)

        btn_col1, btn_col2 = st.columns([1, 1], gap="small")
        with btn_col1:
            csv_bytes = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("Download full log (CSV)", data=csv_bytes, file_name="inference_log.csv", mime="text/csv")
        with btn_col2:
            if st.button("Clear full log"):
                confirm = st.checkbox("Confirm clearing the log (this is permanent).")
                if confirm:
                    try:
                        os.remove(LOG_FILE_LOCAL)
                        st.success("Real-Time Vehicle Feed log cleared.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to delete log: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.stop()
else:
    # Render all three columns
    with col1:
        render_col1()

    with col2:
        render_col2()

        st.markdown('<div style="height:3px; margin:0px 0; background: linear-gradient(90deg, #c3002f, #000000); border-radius:2px;"></div>', unsafe_allow_html=True)
        
        # Chat function below column 2, right adjacent to column 1, stretching till the end
        render_chat_interface()
        
    # 
    # chat_col1, chat_col2 = st.columns([2, 3], gap="medium")
    
    # with chat_col1:
    #     # Empty space to align with column 1
    #     st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # with chat_col2:
    #     # Chat interface in the remaining space
    #     render_chat_interface()

# Footer
st.markdown(
    """
    <hr style="margin-top:20px; margin-bottom:0px; border:none; height:2px; 
            background:linear-gradient(90deg, #c3002f, #000000); border-radius:2px;">
    <div style="text-align:center; font-size:12px; color:#94a3b8; padding:3px 0;">
        © 2025 Tech Mahindra. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True,
)