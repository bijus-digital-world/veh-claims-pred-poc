import os
from click import clear
import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
# from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import re
import html as _html
from datetime import datetime, timezone
import numpy as np  
import math
# import uuid
import streamlit.components.v1 as components


# Imports for chat/RAG
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local helper imports
from helper import (
    load_history_data,
    # compute_rate_per_100,
    MODELS,
    MILEAGE_BUCKETS,
    AGE_BUCKETS,
    NISSAN_HEATMAP_SCALE,
    NISSAN_RED,
    NISSAN_GOLD,
    load_model,
    append_inference_log,
    append_inference_log_s3,
    random_inference_row_from_df,
    # find_nearest_dealers,
    # estimate_repair_cost_range,
    get_bedrock_summary,
    # fetch_dealers_from_aws_location,
    fetch_nearest_dealers,
    reverse_geocode
)

from chat_helper import (
    build_tf_idf_index,
    generate_reply as chat_generate_reply,
    ensure_failures_column
)

try:
    import faiss
    # from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
    HAS_ST_MODEL = True
except Exception as e:
    HAS_FAISS = False
    HAS_ST_MODEL = False
    st.warning("FAISS or sentence-transformers not installed — falling back to TF-IDF. To enable vector store, pip install faiss-cpu sentence-transformers numpy")

# # config: where to persist index + metadata
# VECTOR_DIR = Path("./vector_store")
# VECTOR_DIR.mkdir(exist_ok=True)
# FAISS_INDEX_PATH = VECTOR_DIR / "claims_index.faiss"
# EMB_ARRAY_PATH = VECTOR_DIR / "claim_embs.npy"
# META_PATH = VECTOR_DIR / "claim_meta.npy"   # numpy-serialized list of dicts using np.object_

VECTOR_DIR = Path("./vector_store")
IDX_PATH = VECTOR_DIR / "historical_data_index.faiss"
EMB_PATH = VECTOR_DIR / "historical_data_embs.npy"
META_PATH = VECTOR_DIR / "historical_data_meta.npy"
JSON_META = VECTOR_DIR / "historical_data_meta.json"

# embedding model name 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# Optional config - change as needed
USE_S3 = True
S3_BUCKET = "veh-poc-207567760844-us-east-1"
S3_KEY = "data/vehicle_claims_extended.csv"
MODEL_PATH = "models/claim_rate_model.joblib"
LOG_FILE_LOCAL = "inference_log.csv"
LOG_FILE_S3_KEY = "logs/inference_log.csv"

# Chat log path
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(exist_ok=True)
CHAT_LOG_PATH = LOG_DIR / "chat_history.csv"

LOCATION_PROB_THRESHOLD = 0.5  # threshold to show nearby dealers
PLACE_INDEX_NAME = "NissanPlaceIndex"
AWS_REGION = "us-east-1"

SHOW_REPAIR_COST = False
IS_POC = False

# ------------------------
# Page config and styles
# ------------------------
st.set_page_config(page_title="Nissan - Vehicle Predictive Insights (POC)", page_icon="images/maintenance_icon.svg", layout="wide", initial_sidebar_state="collapsed")
# apply_style is in styles.py; we assume it's already imported/used as before
from styles import apply_style
apply_style()

# ------------------------
# Helper UI functions (added)
# ------------------------
def safe_sorted_unique(series):
    vals = [v for v in pd.unique(series) if pd.notna(v)]
    # cast to str for consistent comparisons and sort case-insensitively
    vals = [str(v) for v in vals]
    return sorted(vals, key=lambda s: s.lower())

@st.cache_resource(show_spinner=True)
def load_persisted_faiss():
    """Load FAISS index, embeddings, and metadata if available on disk."""
    if not IDX_PATH.exists() or not EMB_PATH.exists() or not META_PATH.exists():
        return {
            "available": False,
            "index": None,
            "embs": None,
            "meta": None,
            "d": None,
            "message": "Persisted FAISS files missing. Run build_faiss_index.py first."
        }

    try:
        index = faiss.read_index(str(IDX_PATH))
        embs = np.load(EMB_PATH)
        meta = list(np.load(META_PATH, allow_pickle=True))
        d = embs.shape[1]
        return {
            "available": True,
            "index": index,
            "embs": embs,
            "meta": meta,
            "d": d,
            "message": f"Loaded persisted FAISS index: {len(meta)} vectors · dim={d}"
        }
    except Exception as e:
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

# def load_persisted_faiss():
#     if not IDX_PATH.exists() or not META_PATH.exists() or not EMB_PATH.exists():
#         return {"available": False, "message": "Persisted FAISS files missing. Run build_faiss_index.py first."}
#     try:
#         index = faiss.read_index(str(IDX_PATH))
#         embs = np.load(EMB_PATH)
#         meta = list(np.load(META_PATH, allow_pickle=True))
#         d = embs.shape[1]
#         return {"available": True, "index": index, "embs": embs, "meta": meta, "d": d, "message": f"Loaded index: {len(meta)} vectors, dim={d}"}
#     except Exception as e:
#         return {"available": False, "message": f"Failed to load persisted FAISS: {e}"}

# faiss_res = load_persisted_faiss()
# # then use faiss_res['index'], faiss_res['meta'] in retrieval code (same retrieve_top_k_faiss as before)

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

        if claim_pct >= int(st.session_state.get('predictive_threshold_pct', 80)):
            summary_html = get_bedrock_summary(model_name, part_name, mileage_bucket, age_bucket, claim_pct, 
                                            llm_model_id=llm_model_id, region=region)
        else:
            # Show hardcoded value
            fallback = (
            f"The predicted claim probability is {round(claim_pct,1)}% for {part_name} in {model_name}. "
            "No immediate action recommended; monitor for trend changes."
        )
            summary_html = f"<strong>{risk_token} risk ({pct})</strong>: {_html.escape(fallback)}"

        if not summary_html:
            raise ValueError("Empty summary returned from LLM")
    except Exception:
        fallback = (
            f"The predicted claim probability is {round(claim_pct,1)}% for {part_name} in {model_name}. "
            "No immediate action recommended; monitor for trend changes."
        )
        if claim_pct >= 75:
            risk_token = "High"
        elif claim_pct >= 50:
            risk_token = "Medium"
        else:
            risk_token = "Low"
        pct = f"{round(claim_pct)}%"
        # summary_html = f"<strong>{risk_token} risk ({pct})</strong>: {_html.escape(fallback)}"
        summary_html = f"{_html.escape(fallback)}"

    split_html = re.split(r'\n\s*\n', summary_html, maxsplit=1)
    first_para_html = split_html[0].strip()
    bullets_html = split_html[1].strip() if len(split_html) > 1 else ""

    m = re.search(r'\b(Low|Medium|High)\b', re.sub(r'<[^>]+>', '', first_para_html), flags=re.IGNORECASE)
    risk_token = (m.group(1).capitalize() if m else ("High" if claim_pct>=75 else "Medium" if claim_pct>=50 else "Low"))
    pct_match = re.search(r'\((\d{1,3})%\)', re.sub(r'<[^>]+>', '', first_para_html))
    pct = f"{pct_match.group(1)}%" if pct_match else f"{round(claim_pct)}%"

    if IS_POC:
        summary_html = """<div style='text-align: justify;'>
            Based on recent telemetry data, I have identified a rising trend in engine cooling issues specifically 
            affecting the 2025 Sentra model over the past two weeks. Given your current location, 
            I recommend visiting the <span style='color:#C99700; font-weight:bold;'>
            United Nissan Dealer Service Center</span>, which is the nearest authorized facility.
            </p>
            Further analysis indicates that the supplier <span style='color:#C99700; font-weight:bold;'>Setco Auto Systems</span> 
            has been consistently providing faulty coolant circulation pumps, 
            contributing to the issue. To mitigate future risks, I recommend initiating a logistics change to switch to an 
            alternative supplier, such as <span style='color:#C99700; font-weight:bold;'>Hitachi</span>, 
            to prevent potential mass recalls and ensure continued vehicle reliability.
            </div>
            """
        combined_html = (
            '<div style="display:flex; align-items:center; gap:12px;">'
            f'{_badge_html(risk_token, pct)}'
            f'<div style="flex:1; min-width:0; font-size:1.02rem; line-height:1.35; color:#e6eef8; '
            f'overflow-wrap:break-word; word-break:break-word;">'
            f'{summary_html}'
            f'</div>'
            '</div>'
        )
    else:
        combined_html = (
            '<div style="display:flex; align-items:center; gap:12px;">'
            f'{_badge_html(risk_token, pct)}'
            f'<div style="flex:1; min-width:0; font-size:1.02rem; line-height:1.35; color:#e6eef8; '
            f'overflow-wrap:break-word; word-break:break-word;">'
            f'{first_para_html}'
            f'</div>'
            '</div>'
        )

    st.markdown(combined_html, unsafe_allow_html=True)
    if not IS_POC:
        st.write("")
        if bullets_html:
            # with st.expander("Details / Analyst Evidence", expanded=False):
            st.markdown(bullets_html, unsafe_allow_html=True)
        else:
            # with st.expander("Details / Analyst Evidence", expanded=False):
            st.markdown("<div style='color:#94a3b8;'>No additional details.</div>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_history_cached(use_s3=USE_S3, s3_bucket=S3_BUCKET, s3_key=S3_KEY, local_path="data/vehicle_claims_extended.csv"):
    df_history = load_history_data(use_s3=use_s3, s3_bucket=s3_bucket, s3_key=s3_key, local_path=local_path)
    df_history = ensure_failures_column(df_history)
    return df_history

@st.cache_data(show_spinner=False)
def build_rag_index_cached(df):
    return build_rag_index(df)

# ------------------------
# Reusable utilities for chat
# ------------------------
# helper: convert a history row (pandas Series or dict) to document text
def _row_to_doc(r):
    # same as you used earlier but ensure string
    parts = [
        f"model: {r.get('model','')}",
        f"part: {r.get('primary_failed_part','')}",
        f"mileage_bucket: {r.get('mileage_bucket','')}",
        f"age_bucket: {r.get('age_bucket','')}",
        f"date: {r.get('date','')}",
        f"claims: {int(r.get('claims_count',0))}"
    ]
    return "; ".join(parts)


# cached loader for sentence-transformers model
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name=EMBED_MODEL_NAME):
    if not HAS_ST_MODEL:
        raise RuntimeError("sentence-transformers not available")
    return SentenceTransformer(model_name)


# build (or load) FAISS index and metadata
@st.cache_resource(show_spinner=True)
def build_faiss_index(hist_df, force_rebuild: bool = False):
    """
    Returns: index (faiss.IndexFlatIP), embeddings (np.array NxD), meta (list of dict rows)
    - Builds embedding matrix for each row document and creates FAISS index (inner product similarity on normalized vectors).
    - Persists index and embeddings to disk for subsequent loads.
    """
    # prepare docs
    rows = hist_df.to_dict(orient="records")
    docs = [_row_to_doc(r) for r in rows]

    # If persisted files exist and not forced, try loading
    try:
        if not force_rebuild and HAS_FAISS and FAISS_INDEX_PATH.exists() and EMB_ARRAY_PATH.exists() and META_PATH.exists():
            # load meta and embeddings and index
            meta = list(np.load(META_PATH, allow_pickle=True))
            emb_array = np.load(EMB_ARRAY_PATH)
            # build index from numpy embs
            d = emb_array.shape[1]
            index = faiss.IndexFlatIP(d)
            # ensure normalized embeddings for cosine-like similarity
            faiss.normalize_L2(emb_array)
            index.add(emb_array)
            return index, emb_array, meta
    except Exception:
        # fallback to rebuild
        pass

    # If FAISS or sentence-transformers not installed, raise to allow upper-level fallback
    if not (HAS_FAISS and HAS_ST_MODEL):
        raise RuntimeError("FAISS or sentence-transformers not available for building vector index")

    model = get_embedding_model()
    # compute embeddings in batches to avoid memory spikes
    BATCH = 256
    embs = []
    for i in range(0, len(docs), BATCH):
        batch_docs = docs[i:i+BATCH]
        batch_emb = model.encode(batch_docs, show_progress_bar=False, convert_to_numpy=True)
        embs.append(batch_emb)
    emb_array = np.vstack(embs).astype("float32")

    # normalize (so inner-product == cosine similarity)
    faiss.normalize_L2(emb_array)

    d = emb_array.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product on normalized vectors => cosine similarity
    index.add(emb_array)

    # persist
    try:
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        np.save(EMB_ARRAY_PATH, emb_array)
        np.save(META_PATH, np.array(rows, dtype=object))
    except Exception as e:
        st.warning(f"Failed to persist FAISS index: {e}")

    return index, emb_array, rows

# retrieval: returns top-k rows with similarity score
def retrieve_top_k_faiss(query, index, emb_array, meta_rows, k=6):
    """
    query -> embed -> search -> return list of row dicts with 'score'
    """
    if not (HAS_FAISS and HAS_ST_MODEL):
        return []

    model = get_embedding_model()
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k)  # D: similarities, I: indices
    sims = D[0].tolist()
    idxs = I[0].tolist()
    results = []
    for idx, sim in zip(idxs, sims):
        if idx < 0 or idx >= len(meta_rows):
            continue
        row = dict(meta_rows[idx]) if isinstance(meta_rows[idx], dict) else dict(meta_rows[idx].item() if hasattr(meta_rows[idx],'item') else meta_rows[idx])
        row["score"] = float(sim)
        results.append(row)
    return results

# fallback wrappers that mimic original function signatures
def build_rag_index_faiss(df):
    try:
        idx, embs, rows = build_faiss_index(df, force_rebuild=False)
        return idx, embs, rows
    except Exception as e:
        st.warning(f"FAISS build failed: {e}. Falling back to TF-IDF index.")
        return None, None, df.to_dict(orient='records')


def retrieve_top_k(query, vect, X, rows, k=6):
    """
    Compatibility wrapper: if FAISS index present, use it; else use TF-IDF (existing vect/X).
    We assume earlier code calls retrieve_top_k(query, VECT, X_INDEX, HISTORY_ROWS)
    """
    # If vect is None but we have FAISS saved state, try to use FAISS stored resource
    if HAS_FAISS:
        # attempt to load persisted index (cached resource will return quickly)
        try:
            index, emb_array, meta = build_faiss_index_cached_wrapper(df_history)
            if index is not None:
                return retrieve_top_k_faiss(query, index, emb_array, meta, k=k)
        except Exception:
            pass

    # fallback to TF-IDF behaviour (original)
    if vect is None or X is None or len(rows) == 0:
        return []
    qv = vect.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({**rows[i], "score": float(sims[i])})
    return results

# small cached wrapper to avoid repeated index creation during reruns
@st.cache_resource
def build_faiss_index_cached_wrapper(df):
    # returns index, emb_array, meta_rows or (None, None, [])
    try:
        return build_faiss_index(df, force_rebuild=False)
    except Exception as e:
        return (None, None, [])
    

def build_rag_index(df: pd.DataFrame):
    """Build a simple TF-IDF index over the historical rows. Returns vectorizer and matrix and list of row dicts.
    This is intentionally lightweight (no external vector DB) and fast for small-to-medium datasets."""
    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    if not docs:
        return None, None, []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=6000)
    X = vect.fit_transform(docs)
    rows = df.to_dict(orient="records")
    return vect, X, rows


def retrieve_top_k(query: str, vect, X, rows, k=5):
    """Return top-k rows (as dicts) most similar to query using cosine similarity over TF-IDF."""
    if vect is None or X is None or len(rows) == 0:
        return []
    qv = vect.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    top_idx = sims.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        results.append({**rows[i], "score": float(sims[i])})
    return results


def persist_chat_to_disk(history):
    df = pd.DataFrame(history)
    try:
        df.to_csv(CHAT_LOG_PATH, index=False)
    except Exception as e:
        st.warning(f"Failed to persist chat log: {e}")

# ------------------------
# Load data (S3 fallback to local)
# ------------------------
try:
    # df_history = load_history_data(use_s3=USE_S3, s3_bucket=S3_BUCKET, s3_key=S3_KEY, local_path="data/vehicle_claims.csv")
    df_history = load_history_cached()
    # st.write(len(df_history))
except FileNotFoundError as e:
    st.error(f"❌ Data load failed: {e}")
    st.stop()

REQUIRED_COLS = {"model", "primary_failed_part", "mileage_bucket", "age_bucket", "date", "claims_count", "repairs_count", "recalls_count"}
if not REQUIRED_COLS.issubset(df_history.columns):
    st.error("CSV data missing required columns. Please check vehicle_claims.csv.")
    st.stop()

# Build RAG index once per app run
# VECT, X_INDEX, HISTORY_ROWS = build_rag_index_cached(df_history)

# prefer FAISS — we still keep VECT/X_INDEX for fallback TF-IDF if needed
VECT, X_INDEX, HISTORY_ROWS = None, None, df_history.to_dict(orient='records')
# build faiss cached resource (this takes care of caching)
faiss_idx, faiss_embs, faiss_meta = build_faiss_index_cached_wrapper(df_history)


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
        <a href="?page=inference" style="text-decoration:none; color:inherit;">Inference Log</a>
        <a href="?page=architecture" style="text-decoration:none; color:inherit;">Architecture</a>
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
page = query_params.get("page", ["dashboard"])[0]

# ------------------------
# Layout columns
# ------------------------
col1, col2, col3 = st.columns([2, 1.8, 1.2], gap="medium")
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
              <div class="stat-label">Claims</div>
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

    st.markdown('<div class="chart-header">Failures by Model:</div>', unsafe_allow_html=True)
    if part == "All":
        sev_df = df_history.copy()
    else:
        sev_df = df_history[df_history["primary_failed_part"] == part].copy()

    sev_df["failures_count"] = (
                                    sev_df[["claims_count", "repairs_count", "recalls_count"]]
                                    .fillna(0)
                                    .sum(axis=1)
                                    .astype(int)   
                                )
                                 

    sev_grp = sev_df.groupby("model").agg(incidents=("failures_count", "size"), failures=("failures_count", "sum")).reset_index()
    sev_grp["rate_per_100"] = sev_grp.apply(lambda r: (r["failures"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0, axis=1)
    sev_grp = sev_grp.set_index("model").reindex(MODELS).reset_index()

    if sev_abs:
        y_col = "failures"
        y_label = "Failures"
    else:
        y_col = "rate_per_100"
        y_label = "Failure rate"

    colors = ["#9aa3ad" if (m != st.session_state.model_sel and st.session_state.model_sel != "All") else NISSAN_RED if m == st.session_state.model_sel else "#9aa3ad" for m in sev_grp["model"].tolist()]
    fig_sev = px.bar(sev_grp, x="model", y=y_col, labels={"model": "Model", y_col: y_label}, template="plotly_dark")
    fig_sev.update_traces(marker_color=colors, width=0.5)
    fig_sev.update_layout(margin=dict(l=4, r=4, t=6, b=4), height=220, showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig_sev.update_xaxes(tickfont=dict(size=11, color="#94a3b8"))
    if not sev_abs:
        fig_sev.update_yaxes(tickformat=".1f")
    fig_sev.update_yaxes(tickfont=dict(size=11, color="#94a3b8"))

    max_y = max(sev_grp[y_col].max(), 1)
    for i, row in sev_grp.iterrows():
        ann_text = f"{int(row['failures'])}f / {int(row['incidents'])}i"
        y_pos = row[y_col] + max_y * 0.03
        fig_sev.add_annotation(x=i, y=y_pos, text=ann_text, showarrow=False, font=dict(size=10), align="center", yanchor="bottom", xanchor="center", opacity=0.9)
    st.plotly_chart(fig_sev, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# Column 2: Predictive + Prescriptive
# ------------------------
@st.fragment
def render_col2():
    # Load the model (may be None)
    model_pipe = load_model(MODEL_PATH)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    header_col, slider_col, dropdown_col = st.columns([2, 2.5, 1.25], gap="medium")
    with header_col:
        st.markdown('<div class="card-header" style="height:34px; display:flex; align-items:center;">Predictive Analysis</div>', unsafe_allow_html=True)

    interval_map = {"15s": 15, "30s": 30, "1m": 60, "5m": 300, "15m": 900}
    if "predictive_interval_label" not in st.session_state:
        st.session_state.predictive_interval_label = "15m"
    with slider_col:
        st.select_slider(
            "⏱️",
            options=list(interval_map.keys()),
            # value=st.session_state.predictive_interval_label,
            key="predictive_interval_label",
            format_func=lambda x: f"⏱ {x}",
            label_visibility="collapsed",
            help="Inference data generation",
        )

    threshold_options = [50, 60, 70, 75, 80, 85, 90, 95, 98, 99]
    if "predictive_threshold_pct" not in st.session_state:
        st.session_state.predictive_threshold_pct = 80
    # with dropdown_col:
    #     sel_index = threshold_options.index(st.session_state.predictive_threshold_pct) if st.session_state.predictive_threshold_pct in threshold_options else 4
    #     st.selectbox("⚠️", options=threshold_options, index=sel_index, key="predictive_threshold_pct", label_visibility="collapsed", help="Show actionable summary when predicted claim probability ≥ this value")
    with dropdown_col:
        st.selectbox(
            "⚠️",
            options=threshold_options,
            key="predictive_threshold_pct",
            label_visibility="collapsed",
            help="Show actionable summary when predicted claim probability ≥ this value",
        )
    refresh_interval = interval_map.get(st.session_state.predictive_interval_label, 900)
    st_autorefresh(interval=refresh_interval * 1000, key="predictive_autorefresh")

    inf_row = random_inference_row_from_df(df_history)

    pred_prob = None
    if model_pipe is not None:
        try:
            test_df = pd.DataFrame([{
                "model": inf_row["model"],
                "primary_failed_part": inf_row["primary_failed_part"],
                "mileage_bucket": inf_row["mileage_bucket"],
                "age_bucket": inf_row["age_bucket"],
            }])
            pred_val = model_pipe.predict(test_df)[0]
            pred_prob = float(max(0.0, min(1.0, pred_val)))
        except Exception:
            pred_prob = None

    if pred_prob is None:
        model_risk = {"Leaf": 0.03, "Ariya": 0.04, "Sentra": 0.06}
        mileage_risk = {"0-10k": 0.02, "10-30k": 0.03, "30-60k": 0.06, "60k+": 0.10}
        unique_parts = safe_sorted_unique(df_history["primary_failed_part"])
        part_risk = {p: 0.03 + (idx % 5) * 0.01 for idx, p in enumerate(unique_parts)}
        base = (
            0.5 * model_risk.get(inf_row["model"], 0.04)
            + 0.8 * mileage_risk.get(inf_row["mileage_bucket"], 0.03)
            + 0.6 * part_risk.get(inf_row["primary_failed_part"], 0.03)
        )
        pred_prob = float(max(0.0, min(0.99, base + (0.02 * (0.5 - 0.5)))) )

    pct_text = f"{pred_prob*100:.1f}%"

    try:
        if USE_S3:
            try:
                appended = append_inference_log_s3(inf_row, pred_prob, s3_bucket=S3_BUCKET, s3_key=LOG_FILE_S3_KEY, local_fallback=LOG_FILE_LOCAL)
            except Exception:
                appended = append_inference_log(inf_row, pred_prob, filepath=LOG_FILE_LOCAL)
        else:
            appended = append_inference_log(inf_row, pred_prob, filepath=LOG_FILE_LOCAL)
    except Exception:
        appended = False

    info_l_col, info_r_col, divider_col, kpi_col = st.columns([1, .60, 0.25, 1], gap="small")

    if IS_POC:
        with info_l_col:
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Model</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">Sentra</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Part</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8;">Engine Coolant System</div>""", unsafe_allow_html=True)

        with info_r_col:
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Mileage</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">10,200 miles</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Age</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8;">6 months</div>""", unsafe_allow_html=True)
    else:
        with info_l_col:
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Model</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">{inf_row['model']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Part</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8;">{inf_row['primary_failed_part']}</div>""", unsafe_allow_html=True)

        with info_r_col:
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Mileage</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8; margin-bottom:8px;">{inf_row['mileage_bucket']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div style="font-size:12px; color:#cbd5e1; margin-bottom:4px;">Age</div>
                            <div style="font-weight:700; font-size:14px; color:#e6eef8;">{inf_row['age_bucket']}</div>""", unsafe_allow_html=True)

    with divider_col:
        st.markdown("""
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
            """, unsafe_allow_html=True)

    with kpi_col:
        if IS_POC:
            pct_text = f"{80.8:.1f}%"
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:8px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
                <div style="text-align:center;">
                    <div class="kpi-label">Predicted Claim Probability</div>
                    <div class="kpi-wrap" style="margin-top:6px;">
                    <div class="kpi-num" style="color:{NISSAN_RED}; font-size:34px; line-height:1;">{pct_text}</div>
                    </div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:6px;">
                        Alert if ≥ {int(st.session_state.get('predictive_threshold_pct', 80))}%
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="padding:10px; border-radius:8px; background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));">
                <div style="text-align:center;">
                    <div class="kpi-label">Predicted Claim Probability</div>
                    <div class="kpi-wrap" style="margin-top:6px;">
                    <div class="kpi-num" style="color:{NISSAN_RED}; font-size:34px; line-height:1;">{pct_text}</div>
                    </div>
                    <div style="font-size:11px; color:#94a3b8; margin-top:6px;">
                        Alert if ≥ {int(st.session_state.get('predictive_threshold_pct', 80))}%
                    </div>
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="card-header" style="height:34px; display:flex; align-items:center;">Prescriptive Summary</div>', unsafe_allow_html=True)
    
    # ------------------------
    # Fetch nearest dealers (AWS Location fallback to local)
    # ------------------------
    current_lat = inf_row.get("lat", 38.4405)
    current_lon = inf_row.get("lon", -122.7144)

    try:
        nearest, from_aws = fetch_nearest_dealers(
            current_lat=current_lat,
            current_lon=current_lon,
            place_index_name=PLACE_INDEX_NAME,
            aws_region=AWS_REGION,
            fallback_dealers=None,
            text_query="Nissan Service Center",
            top_n=3,
        )
    except Exception as e:
        st.write("Debug: fetch_nearest_dealers failed:", e)
        nearest = []

    if IS_POC:
        render_summary_ui('Sentra',
                      'Engine Cooling System',
                      '10200 miles',
                      '6 months',
                      80.8)
    else:
        render_summary_ui(inf_row['model'],
                      inf_row['primary_failed_part'],
                      inf_row['mileage_bucket'],
                      inf_row['age_bucket'],
                      pred_prob*100,
                      nearest[0]['name'])

    st.markdown("</div>", unsafe_allow_html=True)

     
    # ------------------------
    # Map card (improved tooltip + legend + readable labels)
    # ------------------------
    st.markdown('<div class="card"><div class="card-header">Current Location of Vehicle & Nearest Dealers</div>', unsafe_allow_html=True)
    
    # Filter dealers within 20 miles (approx 32.1869 km)
    MAX_KM_20_MILES = 32.1869
    nearby_dealers = [d for d in nearest if d.get("distance_km", 99999) <= MAX_KM_20_MILES]
    has_nearby = len(nearby_dealers) > 0

    if not has_nearby:
        # If none within 20 miles, we will still show nearest but flag to the user.
        st.markdown(f"<div style='font-size:14px; color:{NISSAN_GOLD};'>No dealers found within 20 miles. Showing nearest results.</div>", unsafe_allow_html=True)
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

    # Build points (dealers first if we have nearby, else nearest)
    map_points = []
    dealers_to_plot = nearby_dealers if has_nearby else nearest[:3]

    # Dealer points (red)
    if IS_POC:
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
            # short name (first segment of label) for on-map text
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

    # Resolve readable place name for current vehicle (city/town)
    place_name = reverse_geocode(current_lat, current_lon)
    inf_row["place_name"] = place_name or "Current Location"

    # Current vehicle tooltip (green)
    if IS_POC:
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
    else:
        vehicle_tooltip = (
            f"<div style='padding:8px; max-width:260px;'>"
            f"<div style='font-weight:700; font-size:14px; margin-bottom:6px;'>Current Location</div>"
            f"<div style='font-size:13px; color:#d1d5db; margin-bottom:6px;'>{_html.escape(inf_row.get('place_name','Current Location'))}</div>"
            f"<div style='font-size:12px; color:#94a3b8; line-height:1.45;'>"
            f"<b>Vehicle:</b> {_html.escape(str(inf_row.get('model','N/A')))}<br/>"
            f"<b>Part:</b> {_html.escape(str(inf_row.get('primary_failed_part','N/A')))}<br/>"
            f"<b>Mileage:</b> {_html.escape(str(inf_row.get('mileage_bucket','N/A')))}<br/>"
            f"<b>Age:</b> {_html.escape(str(inf_row.get('age_bucket','N/A')))}<br/>"
            f"<b>Claim Prob:</b> {pred_prob*100:.1f}%"
            f"</div></div>"
        )
    # display label: prefer city-like label; truncate to keep map tidy
    display_label = inf_row.get("place_name", "Current Location").split(",")[0]

    if IS_POC:
        current_lat = 36.20
        current_lon = -115.05
    map_points.append({
        "name": "Current Location",
        "short_name": display_label,
        "lat": float(current_lat),
        "lon": float(current_lon),
        "type": "Current",
        "tooltip": vehicle_tooltip
    })

    # Defensive fallback: ensure at least current exists
    if not map_points:
        map_points = [{
            "name": "Current Location",
            "short_name": "Current",
            "lat": float(current_lat),
            "lon": float(current_lon),
            "type": "Current",
            "tooltip": vehicle_tooltip
        }]

    # Helper fields for pydeck
    for p in map_points:
        p["position"] = [p["lon"], p["lat"]]
        # radius meters tuned to col width; display current slightly larger
        p["_radius_m"] = 900 if p.get("type") == "Current" else 520
        # color arrays
        if p.get("type") == "Current":
            p["_fill_color"] = [29, 158, 106, 230]  # green
            p["_line_color"] = [8, 62, 40, 200]
        else:
            p["_fill_color"] = [211, 47, 47, 220]   # red
            p["_line_color"] = [90, 20, 20, 200]

    # Compute bounding center/zoom 
    def _haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def _compute_center_and_zoom(points, viewport_width_px=520):
        lats = [p["lat"] for p in points]
        lons = [p["lon"] for p in points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
        d1 = _haversine_km(min_lat, min_lon, max_lat, max_lon)
        max_dist_km = max(d1, 0.0)
        if max_dist_km < 0.02:
            return center_lat, center_lon, 14
        max_dist_m = max_dist_km * 1000.0
        meters_per_pixel_required = max_dist_m / (viewport_width_px * 0.85)
        lat_rad = math.radians(center_lat)
        meters_per_pixel_at_zoom0 = 156543.03392 * math.cos(lat_rad)
        try:
            zoom = math.log2(meters_per_pixel_at_zoom0 / meters_per_pixel_required)
            zoom = max(3, min(14, int(round(zoom))))
        except Exception:
            zoom = 10
        return center_lat, center_lon, zoom

    center_lat, center_lon, zoom_level = _compute_center_and_zoom(map_points, viewport_width_px=520)

    
    # Render map with styled balloons, outline, labels and HTML tooltip
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
            radius_min_pixels=6,
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
            radius_min_pixels=8,
            radius_max_pixels=90,
        )
        # Text labels: short_name rendered to the right of the marker
        text_layer = pdk.Layer(
            "TextLayer",
            data=map_points,
            get_position="position",
            get_text="short_name",
            get_color=[230, 230, 230],
            get_size=14,
            get_angle=0,
            get_text_anchor= "start",
            get_alignment_baseline= "center",
            billboard=True,
            sizeUnits="pixels",
        )

        tooltip = {
            "html": "{tooltip}",
            "style": {"backgroundColor": "rgba(15,15,15,0.95)", "color": "white", "font-family":"Inter, Arial, sans-serif", "font-size":"13px", "padding":"6px"}
        }

        deck = pdk.Deck(
            layers=[dealers_layer, current_layer, text_layer],
            initial_view_state=pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=zoom_level, pitch=0),
            tooltip=tooltip,
            map_style='dark'
        )
        st.pydeck_chart(deck, use_container_width=True, height=360)

    except Exception as e:
        st.write("Debug: pydeck failed, falling back to st.map — error:", e)
        map_df = pd.DataFrame([{"lat": p["lat"], "lon": p["lon"]} for p in map_points])
        st.map(map_df)

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

    # st.markdown("**Nearest dealers:**")
    with st.expander("Nearest dealers", expanded=False):
        if not dealers_to_plot:
            st.markdown("<div style='color:#94a3b8;'>No dealers within 20 miles.</div>", unsafe_allow_html=True)
        for d in dealers_to_plot:
            st.markdown(f"- **{d.get('name','Nissan Dealer')}** — {d.get('distance_km','N/A')} km ({d.get('eta_min','N/A')} min). Phone: {d.get('phone','N/A')}")

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    # if st.button("📅 Schedule inspection / Contact dealer"):
    #     st.info("Open scheduling modal or handoff to dealer booking flow (todo).")

    if os.path.isfile(LOG_FILE_LOCAL):
        with open(LOG_FILE_LOCAL, "rb") as f:
            st.download_button(label="⬇️ Download Inference Log", data=f, file_name="inference_log.csv", mime="text/csv")

    st.markdown("</div>", unsafe_allow_html=True)


@st.fragment
def render_col3():

    st.markdown('<div class="card"><div class="card-header">Chat with Data</div>', unsafe_allow_html=True)

    # show faiss status 
    try:
        if faiss_res.get("available"):
            pass
        else:
            st.markdown(
                f"<div style='color:#fca5a5; font-size:12px'>FAISS unavailable: {faiss_res.get('message','missing')}</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown("<div style='color:#fca5a5; font-size:12px'>FAISS status unknown</div>", unsafe_allow_html=True)

    # ensure session-state chat history exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ensure the input key exists BEFORE creating the widget
    if "chat_input_col3" not in st.session_state:
        st.session_state["chat_input_col3"] = ""

    # Build TF-IDF index once and cache results
    if "chat_tfidf_built" not in st.session_state:
        try:
            VECT_CHAT, X_CHAT, HISTORY_ROWS_CHAT = build_tf_idf_index(df_history)
        except Exception:
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

    # -------- form (input) --------
    with st.form("chat_form_col3", clear_on_submit=True):
        user_q = st.text_input(
            "Ask something about the data:",
            key="chat_input_col3",
            placeholder="e.g. 'claim rate for model Sentra' or 'show recent incidents'",
            label_visibility="collapsed",
        )
        col_clear, col_send = st.columns([1, 1])
        with col_clear:
            clear = st.form_submit_button("Clear chat", use_container_width=True)
        with col_send:
            submitted = st.form_submit_button("Send", use_container_width=True)

    # -------- handlers (no rendering here) --------
    if clear:
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
                from datetime import datetime, timezone
                import uuid

                submission_id = str(uuid.uuid4())
                ts = datetime.now(timezone.utc).isoformat()

                # append user message
                st.session_state.chat_history.append({"role":"user","text":q,"ts":ts,"id":submission_id})

                # generate assistant reply (synchronous)
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
                    )
                except Exception as e:
                    assistant_html = f"<p>Error generating reply: {_html.escape(str(e))}</p>"

                st.session_state.chat_history.append({
                    "role":"assistant",
                    "text":assistant_html,
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "id": submission_id
                })

                # persist (best-effort)
                try:
                    persist_chat_to_disk(st.session_state.chat_history)
                except Exception:
                    pass

    # -------- final single render of the chat pane (exactly once per run) --------
    full_html, comp_height = _render_chat_html_and_scroll()
    # render inside the earlier placeholder container so it appears above the form in layout order
    with chat_container:
        components.html(full_html, height=comp_height, scrolling=False)

    st.markdown("</div>", unsafe_allow_html=True)


# Render all columns
with col1:
    render_col1()

with col2:
    render_col2()

with col3:
    render_col3()

# ------------------------
# Inference page (separate route)
# ------------------------
if page == "inference":
    st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><div class="card-header">Inference Log</div>', unsafe_allow_html=True)

    if not os.path.isfile(LOG_FILE_LOCAL):
        st.markdown("<div style='padding:12px; color:#94a3b8;'>No inference log found yet. Predictions will be logged as they run.</div>", unsafe_allow_html=True)
    else:
        df_log = pd.read_csv(LOG_FILE_LOCAL, parse_dates=["timestamp"])
        c1, c2, c3 = st.columns([1.5, 1.2, 1], gap="small")
        with c1:
            min_date = df_log["timestamp"].min().date()
            max_date = df_log["timestamp"].max().date()
            date_range = st.date_input("Date range", value=(min_date, max_date), key="inference_date_range")
        with c2:
            text_filter = st.text_input("Filter (model / part)", value="", key="inference_text_filter")
        with c3:
            rows_to_show = st.selectbox("Rows", options=[25, 50, 100, 500, 1000], index=1, key="inference_rows_count")

        dr_start, dr_end = date_range
        mask = (df_log["timestamp"].dt.date >= dr_start) & (df_log["timestamp"].dt.date <= dr_end)
        if text_filter.strip():
            t = text_filter.strip().lower()
            mask &= df_log["model"].str.lower().str.contains(t) | df_log["primary_failed_part"].str.lower().str.contains(t)
        df_show = df_log[mask].sort_values("timestamp", ascending=False).head(rows_to_show)

        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True)

        btn_col1, btn_col2 = st.columns([1, 1], gap="small")
        with btn_col1:
            csv_bytes = df_log.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download full log (CSV)", data=csv_bytes, file_name="inference_log.csv", mime="text/csv")
        with btn_col2:
            if st.button("🗑️ Clear inference log"):
                confirm = st.checkbox("Confirm clearing the log (this is permanent).")
                if confirm:
                    try:
                        os.remove(LOG_FILE_LOCAL)
                        st.success("Inference log cleared.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Failed to delete log: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:left;"><a href="/?page=dashboard" style="text-decoration:none; color:#94a3b8;">⟵ Back to Dashboard</a></div>', unsafe_allow_html=True)
    st.stop()


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
