"""
chat_helper.py — cleaned and patched

Single-file chat helper for "Chat with Data" column.

Key behaviour:
- Default metric is failures_count = claims + repairs + recalls (synthetic).
- Honors explicit requests for claims/repairs/recalls and for cost/time columns if present.
- Retrieval: use FAISS + sentence-transformers if available; otherwise TF-IDF fallback.
- Replies are natural-language summaries (no raw table dumps), with "overall first, then breakdown" pattern.
"""

from typing import Tuple, List, Dict, Optional
import re
import html as _html
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import math
import time

# Import configuration
from config import config

# Logging
from utils.logger import chat_logger as logger

# TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional embedding / faiss
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    HAS_ST = True
    HAS_FAISS = True
except Exception:
    HAS_ST = False
    HAS_FAISS = False

EMBED_MODEL_NAME = config.model.embedding_model_name

# --- add this near other detection helpers (e.g. _detect_count_or_average_intent) ---
def _detect_incident_or_failure_request(text: str) -> Optional[str]:
    """
    Return 'incidents' if text clearly asks for incident count (rows with >=1 metric),
           'failures' if text asks for failures (sum of metric),
           or None if ambiguous.
    This uses many common phrasings so differently-worded questions map to the same intent.
    """
    t = (text or "").lower()

    # explicit incident phrasings (rows with >=1)
    incident_patterns = [
        r"\bincident(s)?\b",
        r"\bincident count\b",
        r"\bnumber of incidents\b",
        r"\bhow many (?:vehicles|cars|records|rows|owners) (?:had|have) (?:at least )?(?:a )?(?:failure|failures)\b",
        r"\bhow many (?:vehicles|records|rows) (?:were )?affected\b",
        r"\bvehicles? affected\b",
        r"\brows with\b.*\b(?:failure|failures)\b",
        r"\bhow many (?:vehicles|records|rows) (?:had an incident|had incidents)\b"
    ]
    for pat in incident_patterns:
        if re.search(pat, t):
            return "incidents"

    # explicit failure phrasings (summed counts)
    failure_patterns = [
        r"\btotal (?:failures|failure count|failure)\b",
        r"\btotal number of failures\b",
        r"\bhow many failures\b",
        r"\bcount of failures\b",
        r"\bwhat is the total failures\b",
        r"\btotal failure(s)?\b"
    ]
    for pat in failure_patterns:
        if re.search(pat, t):
            return "failures"

    # ambiguous short forms: "failure count", "failures" — prefer 'failures'
    if re.search(r"\bfailure(s)?\b|\bfailure count\b|\bfailures count\b|\bfailure total\b", t):
        return "failures"

    return None

def summarize_overall_metric(df: pd.DataFrame, metric_col: str, top_n: int = 6) -> str:
    """
    Canonical overall summary for a metric column. Returns HTML with:
      - total metric
      - total rows
      - incident rows (rows with >=1 metric)
      - average per row and per incident
      - top models by metric
    This replaces older separate summarize_total_metric / compute_count_and_average_html outputs.
    """
    if df is None:
        return "<p>No data available.</p>"
    if metric_col is None:
        return "<p>Could not determine the requested metric.</p>"
    try:
        df2 = df.copy()
        # ensure numeric
        df2[metric_col] = pd.to_numeric(df2.get(metric_col), errors="coerce").fillna(0)
        total_metric = int(df2[metric_col].sum())
        total_rows = int(len(df2))
        incident_mask = df2[metric_col] > 0
        incident_count = int(incident_mask.sum())
        avg_per_row = (total_metric / total_rows) if total_rows > 0 else 0.0
        avg_per_incident = (total_metric / incident_count) if incident_count > 0 else 0.0
        per100 = (total_metric / total_rows) * 100.0 if total_rows > 0 else 0.0
        label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")

        # top models (if available)
        top_models_txt = ""
        if "model" in df2.columns:
            try:
                grp = df2.groupby("model")[metric_col].sum().sort_values(ascending=False).head(top_n)
                lines = [f"{_html.escape(str(idx))} → {int(v)}" for idx, v in grp.items() if pd.notna(idx)]
                top_models_txt = "<br>".join(lines) if lines else ""
            except Exception:
                top_models_txt = ""

        # Plain English summary
        if total_metric == 0:
            summary = f"There are currently no {label} recorded in the dataset."
        elif per100 < 5:
            summary = f"The {label} rate is quite low at {per100:.1f}% across {total_rows} records, with {total_metric} total {label}."
        elif per100 < 15:
            summary = f"We have a moderate {label} rate of {per100:.1f}% across {total_rows} records, totaling {total_metric} {label}."
        else:
            summary = f"There's a concerning {label} rate of {per100:.1f}% across {total_rows} records, with {total_metric} total {label} requiring attention."
        
        # Compose canonical output (includes both sum + incidents + averages + top models)
        html_lines = [
            f"<p>{summary}</p>",
            f"<p><strong>Detailed breakdown for {_html.escape(label)}:</strong></p>",
            "<ul style='margin-top:6px;'>",
            f"<li><strong>Total ({label}):</strong> {total_metric}</li>",
            f"<li><strong>Records:</strong> {total_rows}</li>",
            f"<li><strong>Records with ≥1 {label} (incidents):</strong> {incident_count}</li>",
            f"<li><strong>Average per record:</strong> {avg_per_row:.2f} {'' if metric_col!='failures_count' else 'failures'}</li>",
            f"<li><strong>Average per incident:</strong> {avg_per_incident:.2f} {'' if metric_col!='failures_count' else 'failures'}</li>",
            f"</ul>"
        ]
        if top_models_txt:
            html_lines.append(f"<p><strong>Top models by {label}:</strong><br>{top_models_txt}</p>")

        return "".join(html_lines)
    except Exception as e:
        return f"<p>Could not compute summary due to a data issue: {_html.escape(str(e))}</p>"

def format_total_incidents(df: pd.DataFrame, metric_col: str, top_n: int = 6) -> str:
    """
    Canonical output for 'total incidents' — ALWAYS returns the same format for any
    incident-related phrasing. Incident = number of rows with metric > 0.
    """
    if df is None or metric_col is None:
        return "<p>No data available.</p>"
    try:
        df2 = df.copy()
        df2[metric_col] = pd.to_numeric(df2.get(metric_col), errors="coerce").fillna(0)
        total_rows = int(len(df2))
        incident_count = int((df2[metric_col] > 0).sum())

        # per-cent
        pct = (incident_count / total_rows * 100.0) if total_rows > 0 else 0.0

        # top models by incident rows (count of rows with metric>0)
        top_models_txt = ""
        if "model" in df2.columns and incident_count > 0:
            try:
                grp = df2[df2[metric_col] > 0].groupby("model").size().sort_values(ascending=False).head(top_n)
                lines = [f"{_html.escape(str(idx))} → {int(v)} incidents" for idx, v in grp.items() if pd.notna(idx)]
                top_models_txt = "<br>".join(lines) if lines else ""
            except Exception:
                top_models_txt = ""

        html = (f"<p><strong>Incidents (rows with ≥1):</strong> {incident_count} of {total_rows} records (~{pct:.1f}%).</p>")
        if top_models_txt:
            html += f"<p><strong>Top models by incidents:</strong><br>{top_models_txt}</p>"
        return html
    except Exception as e:
        return f"<p>Could not compute incidents due to a data issue: {_html.escape(str(e))}</p>"

def format_total_failures(df: pd.DataFrame, metric_col: str, top_n: int = 6) -> str:
    # just a thin consistent wrapper to the existing function
    return summarize_overall_metric(df, metric_col, top_n=top_n)

# -------------------------
# COUNT & AVERAGE helpers (add to chat_helper.py)
# -------------------------
def _detect_count_or_average_intent(text: str) -> Tuple[bool, bool]:
    """
    Returns (wants_count, wants_average).
    Recognizes many natural-language variants.
    """
    t = (text or "").lower()
    wants_count = bool(re.search(r"\b(count|how many|total number|total of|how many (?:records|rows)|number of)\b", t))
    wants_average = bool(re.search(r"\b(avg|average|mean|typical|on average|per (?:record|vehicle|vehicle)|average per)\b", t))
    return wants_count, wants_average

def _choose_group_col_from_text(text: str, df: pd.DataFrame) -> Optional[str]:
    """
    Pick a sensible grouping column if the user asked e.g. 'by model' or 'per model'.
    Checks for model, age_bucket, mileage_bucket, dealer, region in that order.
    """
    t = (text or "").lower()
    for cand in ["model", "age_bucket", "mileage_bucket", "dealer", "region"]:
        # handle synonyms like "by model", "per model", or just "model"
        if re.search(rf"\b(by|per)\s+{re.escape(cand.split('_')[0])}\b", t) or _has_word(t, cand.split('_')[0]):
            col = _safe_column(df, [cand])
            if col:
                return col
    # fallback to model if present
    return _safe_column(df, ["model"])

def _format_pct(v: float) -> str:
    try:
        return f"{v:.1f}%"
    except Exception:
        return str(v)

def compute_count_and_average_html(df: pd.DataFrame,
                                   requested_metric: str,
                                   user_text: str,
                                   sample_df: Optional[pd.DataFrame] = None,
                                   top_n_groups: int = 6) -> str:
    """
    Compute totals / counts / averages for requested_metric on df (or sample_df if provided).
    Returns HTML string ready to return from generate_reply.
    Behaviour:
      - If requested_metric resolves to a column, use it;
      - If it is 'failures_count', synthesize if needed.
      - Provide total sum, incident count (rows with metric>0), average per row and per incident.
      - If user asked 'by X' produce top groups by rate_per_100 or average.
    """
    if df is None:
        return "<p>No data available.</p>"

    # Use sample_df (RAG) if provided and non-empty, but still synthesize metric there
    working_df = None
    if sample_df is not None and not sample_df.empty:
        working_df = sample_df.copy()
    else:
        working_df = df.copy()

    # Ensure metric exists or synthesize
    metric = requested_metric or _detect_metric_from_text(user_text)
    metric_col, df_with_metric = _metric_or_fallback_column(working_df, metric)
    if df_with_metric is not None:
        # the resolver returned a modified df (synth failures); prefer that
        working_df = df_with_metric.copy()

    # ensure failures_count is present if requested
    if metric_col == "failures_count" and "failures_count" not in working_df.columns:
        working_df = ensure_failures_column(working_df, out_col="failures_count")
        metric_col = "failures_count"

    if metric_col is None:
        # fallback: try failures_count explicitly
        metric_col, df_with_metric2 = _metric_or_fallback_column(working_df, "failures_count")
        if df_with_metric2 is not None:
            working_df = df_with_metric2.copy()
        if metric_col is None:
            return "<p>Couldn't determine a usable metric from your question (claims, repairs, recalls, or failures). Try rephrasing.</p>"

    # coerce to numeric
    working_df[metric_col] = pd.to_numeric(working_df.get(metric_col), errors="coerce").fillna(0)

    # core aggregates
    total_rows = int(len(working_df))
    total_sum = float(working_df[metric_col].sum())
    incident_mask = working_df[metric_col] > 0
    incident_count = int(incident_mask.sum())
    avg_per_row = (total_sum / total_rows) if total_rows > 0 else 0.0
    avg_per_incident = (total_sum / incident_count) if incident_count > 0 else 0.0

    # Build human-friendly header
    label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
    header_lines = [
        f"<p><strong>Summary for {_html.escape(label)}:</strong></p>",
        f"<ul style='margin-top:6px;'>",
        f"<li><strong>Total ({label}):</strong> {int(total_sum)}</li>",
        f"<li><strong>Rows:</strong> {total_rows}</li>",
        f"<li><strong>Rows with ≥1 {label} (incidents):</strong> {incident_count}</li>",
        f"<li><strong>Average per row:</strong> {avg_per_row:.2f} {'' if metric_col!='failures_count' else 'failures'}</li>",
        f"<li><strong>Average per incident:</strong> {avg_per_incident:.2f} {'' if metric_col!='failures_count' else 'failures'}</li>",
        "</ul>"
    ]

    # If user asked by-group, compute breakdown
    group_col = _choose_group_col_from_text(user_text, working_df)
    group_section = ""
    if group_col and (re.search(r"\b(by|per)\b", user_text.lower()) or _has_word(user_text.lower(), group_col.split("_")[0])):
        try:
            grp = working_df.groupby(group_col).agg(
                total_metric=(metric_col, "sum"),
                rows=(metric_col, "count")
            )
            grp = grp[grp["rows"] > 0].copy()
            grp["rate_per_100"] = (grp["total_metric"] / grp["rows"]) * 100.0
            # top by rate_per_100
            top_by_rate = grp.sort_values("rate_per_100", ascending=False).head(top_n_groups)
            lines = []
            for idx, row in top_by_rate.iterrows():
                lines.append(f"{_html.escape(str(idx))} → total {int(row['total_metric'])}, rate ≈ {row['rate_per_100']:.1f}% ({int(row['rows'])} rows)")
            if lines:
                group_section = "<p><strong>Breakdown by " + _html.escape(group_col) + " (top):</strong><br>" + "<br>".join(lines) + "</p>"
        except Exception:
            group_section = "<p>Could not compute grouped breakdown (data issue).</p>"

    # Compose final HTML
    html = "".join(header_lines) + group_section
    # Suggest phrasing if numbers are suspicious
    if total_sum == 0 and incident_count == 0:
        html += "<p style='color:#d97706;'><em>Note:</em> I found no incidents for the requested metric in the dataset/sample. Check that your dataset contains claims/repairs/recalls or try a different timespan.</p>"
    return html


# -------------------------
# Utilities & indexing
# -------------------------
def _safe_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first existing column name from candidates (case-insensitive), else None."""
    if df is None:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        cl = cand.lower()
        # direct match
        if cl in cols_lower:
            return cols_lower[cl]
        # substring match
        for c_low, c_orig in cols_lower.items():
            if cl in c_low:
                return c_orig
    return None


def _has_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", (text or "").lower()) is not None


def _safe_date_str(dt):
    try:
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(dt)


def _row_to_doc(r: dict) -> str:
    """Compact doc for TF-IDF/embedding indexing (failures-centric)."""
    try:
        claims = int(r.get("claims_count", 0) or 0)
    except Exception:
        claims = 0
    try:
        repairs = int(r.get("repairs_count", 0) or 0)
    except Exception:
        repairs = 0
    try:
        recalls = int(r.get("recalls_count", 0) or 0)
    except Exception:
        recalls = 0
    failures = claims + repairs + recalls
    parts = [
        f"model: {r.get('model','')}",
        f"part: {r.get('primary_failed_part','')}",
        f"mileage_bucket: {r.get('mileage_bucket','')}",
        f"age_bucket: {r.get('age_bucket','')}",
        f"date: {r.get('date','')}",
        f"failures: {failures}",
        f"claims: {claims}",
    ]
    return "; ".join(parts)


def build_tf_idf_index(df: pd.DataFrame, max_features: Optional[int] = None):
    """Build TF-IDF index for retrieval. Uses config default for max_features if not provided."""
    max_features = max_features or config.data.tfidf_max_features
    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    if not docs:
        return None, None, []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=max_features)
    X = vect.fit_transform(docs)
    rows = df.to_dict(orient="records")
    return vect, X, rows


def _embed_query_st_model(query: str, model_name: str = EMBED_MODEL_NAME):
    if not HAS_ST:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True).astype("float32")
    return emb


def retrieve_with_faiss_or_tfidf(query: str,
                                 faiss_res: dict,
                                 tfidf_vect,
                                 tfidf_X,
                                 tfidf_rows,
                                 top_k: Optional[int] = None) -> List[dict]:
    """Try FAISS retrieval if available, otherwise TF-IDF fallback. Uses config default for top_k if not provided."""
    # Apply config default
    top_k = top_k or config.data.rag_top_k
    
    logger.debug(f"Retrieving top-{top_k} results for query: '{query[:50]}...'")
    start_time = time.time()
    
    # FAISS path
    try:
        if faiss_res and faiss_res.get("available") and HAS_ST and HAS_FAISS:
            logger.debug("Using FAISS retrieval")
            index = faiss_res.get("index")
            meta = faiss_res.get("meta", [])
            if index is not None and meta is not None:
                q_emb = _embed_query_st_model(query)
                faiss.normalize_L2(q_emb)
                D, I = index.search(q_emb, top_k)
                sims = D[0].tolist()
                idxs = I[0].tolist()
                results = []
                for idx, sim in zip(idxs, sims):
                    if idx < 0 or idx >= len(meta):
                        continue
                    row = dict(meta[idx]) if isinstance(meta[idx], dict) else dict(meta[idx])
                    row["score"] = float(sim)
                    results.append(row)
                if results:
                    duration_ms = (time.time() - start_time) * 1000
                    logger.info(f"FAISS retrieval returned {len(results)} results in {duration_ms:.2f}ms")
                    return results
    except Exception as e:
        logger.warning(f"FAISS retrieval failed, falling back to TF-IDF: {e}")

    # TF-IDF fallback
    if tfidf_vect is None or tfidf_X is None or not tfidf_rows:
        logger.warning("TF-IDF retrieval unavailable - no index available")
        return []
    
    try:
        logger.debug("Using TF-IDF retrieval fallback")
        qv = tfidf_vect.transform([query])
        sims = cosine_similarity(qv, tfidf_X).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            r = dict(tfidf_rows[i])
            r["score"] = float(sims[i])
            results.append(r)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"TF-IDF retrieval returned {len(results)} results in {duration_ms:.2f}ms")
        return results
    except Exception as e:
        logger.error(f"TF-IDF retrieval failed: {e}", exc_info=True)
        return []


# -------------------------
# Metric helpers
# -------------------------
def ensure_failures_column(df: pd.DataFrame, out_col: str = "failures_count") -> pd.DataFrame:
    """Return a copy of df with a synthetic failures_count = claims + repairs + recalls."""
    df = df.copy()
    def _to_int_col(name: str):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)
        return pd.Series(0, index=df.index, dtype=int)
    claims = _to_int_col("claims_count")
    repairs = _to_int_col("repairs_count")
    recalls = _to_int_col("recalls_count")
    df[out_col] = (claims + repairs + recalls).astype(int)
    return df

def _metric_or_fallback_column(df: pd.DataFrame, requested_metric: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Resolve requested_metric -> actual column name (or create synthetic failures_count).
    Returns (metric_col, df_with_metric_or_None).

    Behavior:
    - If requested_metric maps directly to a column name, return it.
    - Otherwise, only map to a column from an alias group when the user's requested_metric
      clearly belongs to that alias group.
    - If user explicitly asked for 'failures' or 'failures_count', synthesize failures_count if needed.
    """
    if df is None:
        return None, None

    cols_lower = {c.lower(): c for c in df.columns}
    rm = (requested_metric or "").lower().strip()

    # 1) direct exact column name (case-insensitive)
    if rm in cols_lower:
        return cols_lower[rm], None

    # Alias groups: canonical -> synonyms list
    alias_map = {
        "failures_count": ["failures_count", "failures", "failure_count", "failures_counts", "failure"],
        "claims_count": ["claims_count", "claims", "claim_count", "claim_counts", "claim"],
        "repairs_count": ["repairs_count", "repairs", "repair_count", "repair"],
        "recalls_count": ["recalls_count", "recalls", "recall_count", "recall"],
        "claim_cost": ["claim_cost", "warranty_cost", "cost", "amount", "claim_amount"]
    }

    # 2) If requested metric string *matches* any alias exactly (or with underscores/spaces),
    #    map to an existing column from that alias group (if present).
    for canonical, aliases in alias_map.items():
        # Only consider aliases if user explicitly requested one of them
        if not any(rm == a or rm == a.replace("_", " ") for a in aliases):
            continue
        # find if any alias column exists in df (case-insensitive)
        for a in aliases:
            if a in cols_lower:
                return cols_lower[a], None
        # If user explicitly requested failures but none of the alias columns exist,
        # do NOT return here — let the explicit failures synthesis happen below.
        if canonical != "failures_count":
            # for non-failures alias groups (e.g., claims), indicate missing
            return None, None
        # else canonical == "failures_count": fallthrough to synthesis step

    # 3) Explicit failures request -> synthesize failures_count if missing
    if rm in ("failures_count", "failures", "failure", "failure_count", "failure rate"):
        # if the df already has failures_count column, return it
        if "failures_count" in df.columns:
            return "failures_count", None
        # otherwise create a df copy with synthesized failures_count and return it
        df_out = ensure_failures_column(df, out_col="failures_count")
        return "failures_count", df_out

    # 4) Last-ditch substring mapping when requested_metric is clearly one of claim/repair/recall
    if rm in ("claims", "claim", "claims_count", "claim_count"):
        cand = _safe_column(df, ["claims_count", "claims", "claim_count"])
        return (cand, None) if cand else (None, None)
    if rm in ("repairs", "repair", "repairs_count", "repair_count"):
        cand = _safe_column(df, ["repairs_count", "repairs", "repair_count"])
        return (cand, None) if cand else (None, None)
    if rm in ("recalls", "recall", "recalls_count", "recall_count"):
        cand = _safe_column(df, ["recalls_count", "recalls", "recall_count"])
        return (cand, None) if cand else (None, None)

    # 5) Nothing found
    return None, None


def _detect_metric_from_text(text: str) -> str:
    """
    Decide metric from user text. Priority:
      1) If user explicitly mentions 'failure' or 'failure rate' -> failures_count
      2) Explicit 'claim(s)' (but not cost) -> claims_count
      3) 'repair(s)' -> repairs_count
      4) 'recall(s)' -> recalls_count
      5) cost-related -> claim_cost
      6) fallback -> failures_count
    """
    t = (text or "").lower()

    # 1) explicit 'failure' -> choose synthetic failures_count
    if re.search(r"\bfailur(e|es|e?s|e?s rate)?\b", t) or re.search(r"\bfailure rate\b", t):
        return "failures_count"

    # 2) explicit 'claim' (but not cost)
    if re.search(r"\bclaim(s)?\b", t) and not re.search(r"\b(claim cost|warranty cost|cost|amount)\b", t):
        return "claims_count"

    # 3) repairs
    if re.search(r"\brepair(s)?\b", t):
        return "repairs_count"

    # 4) recalls
    if re.search(r"\brecall(s)?\b", t):
        return "recalls_count"

    # 5) cost explicitly
    if re.search(r"\b(claim cost|warranty_cost|warranty cost|cost|amount|claim_amount)\b", t):
        return "claim_cost"

    # 6) fallback default
    return "failures_count"


# -------------------------
# Summaries & analysis
# -------------------------
def summarize_top_failed_parts(df_history: pd.DataFrame, metric: str = "failures_count", top_n: int = 6) -> str:
    if df_history is None or df_history.empty:
        return "<p>No historical data available.</p>"
    part_col = _safe_column(df_history, ["primary_failed_part", "failed_part", "part"])
    if not part_col:
        return "<p>Data does not contain a 'part' field (primary_failed_part). Cannot compute top failed parts.</p>"

    metric_col, df_used = _metric_or_fallback_column(df_history, metric)
    if metric_col is None:
        fallback_col, df_used2 = _metric_or_fallback_column(df_history, "failures_count")
        if fallback_col:
            metric_col = fallback_col
            df_used = df_used2
            notice = "<p><em>Note:</em> Requested metric not available; showing top parts by failures_count instead.</p>"
        else:
            return "<p>Requested metric not available and no suitable fallback found (no claims/repairs/recalls columns).</p>"
    else:
        notice = ""

    try:
        total_metric = pd.to_numeric(df_used[metric_col], errors="coerce").fillna(0).sum()
        total_rows = len(df_used)
        label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
        header = f"<p>Overall {label}: {int(total_metric)} across {total_rows} records.</p>"
        grp = df_used.groupby(part_col).agg(total_metric=(metric_col, "sum"), incidents=(metric_col, "count"))
        grp = grp.sort_values("total_metric", ascending=False).head(top_n)
        lines = []
        for part, row in grp.iterrows():
            lines.append(f"{_html.escape(str(part))} → {int(row['total_metric'])} {label} (incidents: {int(row['incidents'])})")
        return notice + header + "<p>Top failed parts:<br>" + "<br>".join(lines) + "</p>"
    except Exception as e:
        return f"<p>Could not compute top failed parts (error: {_html.escape(str(e))}).</p>"


def summarize_incident_details(df_history: pd.DataFrame, metric: str = "failures_count") -> str:
    if df_history is None or df_history.empty:
        return "<p>No historical data available.</p>"

    metric_col, df_used = _metric_or_fallback_column(df_history, metric)
    if metric_col is None:
        return "<p>Requested metric not available in dataset (no claims/repairs/recalls column).</p>"

    try:
        total_metric = pd.to_numeric(df_used[metric_col], errors="coerce").fillna(0).sum()
        total_rows = len(df_used)
        label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
        header = f"<p>Overall {label}: {int(total_metric)} across {total_rows} records.</p>"
    except Exception:
        return "<p>Unable to aggregate the requested metric (data type issue).</p>"

    time_col = _safe_column(df_used, ["time_to_resolution", "resolution_days", "days_to_resolve"])
    avg_res = None
    if time_col:
        try:
            avg_res = pd.to_numeric(df_used[time_col], errors="coerce").dropna().mean()
        except Exception:
            avg_res = None

    out = header
    if avg_res is not None:
        out += f"<p>Average time to resolution (where available): {avg_res:.1f} days.</p>"
    return out


def _compute_model_trends(df_history: pd.DataFrame,
                          metric_col: str = "failures_count",
                          min_months: int = 6,
                          slope_threshold: float = 0.0,
                          top_n: int = 5):
    """
    Compute per-model monthly slopes for metric_col. Returns list of tuples:
    (model, slope, months_of_data, last_month_value)
    """
    if df_history is None or metric_col is None:
        return []
    df = df_history.copy()
    if metric_col == "failures_count":
        df = ensure_failures_column(df, out_col="failures_count")
    if "date" not in df.columns or metric_col not in df.columns:
        return []
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0).astype(float)

    results = []
    for model, g in df.groupby("model"):
        try:
            g = g.dropna(subset=["date_parsed"])
            if g.empty:
                continue
            monthly = g.set_index("date_parsed").resample("M")[metric_col].sum().sort_index()
            if monthly.dropna().shape[0] < min_months:
                continue
            y = monthly.values.astype(float)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0])
            last_val = float(y[-1]) if len(y) > 0 else 0.0
            results.append((model, slope, len(y), last_val))
        except Exception:
            continue
    results = [r for r in results if r[1] >= slope_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]


# -------------------------
# Missing-column helper
# -------------------------
_COLUMN_SYNONYMS = {
    "region": ["region", "area", "zone", "territory", "city"],
    "dealer": ["dealer", "dealer_name", "dealer id", "dealerid", "service center"],
    "service_center": ["service center", "service_center", "servicecentre", "service center name", "dealer"],
    "claim_cost": ["claim cost", "claim_cost", "cost", "warranty_cost", "claim_amount"],
    "model": ["model", "vehicle model", "variant"],
    "primary_failed_part": ["part", "failed part", "primary_failed_part", "failure part", "pfp"],
    "repairs_count": ["repair", "repairs", "repairs_count", "repair_count"],
    "recalls_count": ["recall", "recalls", "recalls_count", 'recall_count'],
    "time_to_resolution": ["time to resolution", "resolution_time", "time_to_resolution", "days_to_resolve", "resolution_days"],
    
    # NEW: Enhanced columns for comprehensive chat support
    "vin": ["vin", "vehicle identification", "vehicle id", "vehicle number", "vehicle identification number"],
    "supplier_name": ["supplier", "supplier_name", "vendor", "manufacturer", "supplier name"],
    "supplier_id": ["supplier id", "supplier_id", "vendor id", "manufacturer id"],
    "supplier_quality_score": ["quality", "supplier quality", "supplier_quality_score", "quality score", "supplier rating"],
    "defect_rate": ["defect rate", "defect_rate", "failure rate", "quality issue rate", "supplier defect rate"],
    "failure_description": ["failure description", "failure_description", "reason", "root cause", "failure reason", "why failed"],
    "current_lat": ["latitude", "lat", "current_lat", "vehicle latitude"],
    "current_lon": ["longitude", "lon", "current_lon", "vehicle longitude"],
    "city": ["city", "location", "current city", "current location"],
    "dealer_name": ["dealer", "dealer_name", "service center", "nearest dealer", "dealer name"],
    "dealer_distance_km": ["dealer distance", "dealer_distance_km", "distance to dealer", "how far"],
    
    # Manufacturing and diagnostic data
    "manufacturing_date": ["manufacturing date", "manufacturing_date", "manufacture date", "production date", "build date", "when manufactured"],
    "dtc_code": ["dtc", "dtc code", "dtc_code", "diagnostic trouble code", "trouble code", "error code", "fault code", "obd code"],
}


def _requested_missing_columns(user_text: str, df_cols) -> dict:
    """
    Returns a dict mapping requested_key -> (found_bool, matched_column_or_none).
    Example: { 'region': (False, None), 'dealer': (True, 'dealer') }
    Uses proper word-boundary matching.
    """
    found = {}
    txt = (user_text or "").lower()
    cols_lower = {c.lower(): c for c in df_cols}  # map lowercase -> original
    for key, synonyms in _COLUMN_SYNONYMS.items():
        # only check if the user actually mentioned any of the synonyms/phrases
        matched_any = False
        for s in synonyms:
            # word-boundary safe check
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", txt):
                matched_any = True
                break
        if not matched_any:
            # user did not ask about this concept => skip
            continue

        # user asked about this concept — see whether the df contains a matching column
        if key.lower() in cols_lower:
            found[key] = (True, cols_lower[key.lower()])
            continue

        matched_col = None
        for s in synonyms:
            s_low = s.lower()
            for c_low, c_orig in cols_lower.items():
                if s_low in c_low:
                    matched_col = c_orig
                    break
            if matched_col:
                break

        if matched_col:
            found[key] = (True, matched_col)
        else:
            found[key] = (False, None)

    return found


# -------------------------
# Top-level generator (NEW - Using Handler Pattern)
# -------------------------
def generate_reply(user_text: str,
                   df_history: pd.DataFrame,
                   get_bedrock_summary_callable,
                   conversation_context=None) -> str:
    """
    Main entry used by app.py to produce assistant HTML reply.
    
    Uses modular handler pattern for better maintainability.
    """
    from chat.handlers import QueryRouter, QueryContext
    
    ut = (user_text or "").strip()
    logger.info(f"Processing chat query: '{ut[:100]}...'")
    
    # Guard: missing columns requested explicitly
    miss = _requested_missing_columns(ut, df_history.columns)
    if miss:
        missing = [k for k, v in miss.items() if v[0] is False]
        if missing:
            available = ", ".join(sorted(df_history.columns))
            pretty_missing = ", ".join(missing)
            return (f"<p>It looks like you're asking about {pretty_missing}, but your data does not contain that column. "
                    f"Available columns include: <strong>{available}</strong>. Try asking about one of those instead.</p>")
    
    # Create context and use handler pattern
    context = QueryContext(
        query=ut,
        df_history=df_history,
        get_bedrock_summary_callable=get_bedrock_summary_callable,
        conversation_context=conversation_context
    )
    
    # Route to appropriate handler
    router = QueryRouter()
    return router.route(context)

# ---------- helper to parse model/part from user text ----------
def parse_model_part_from_text(df: pd.DataFrame, text: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort: find model & part mentioned in text (case-insensitive substring match)."""
    t = (text or "").lower()
    model = None
    part = None
    try:
        for m in pd.unique(df.get("model", pd.Series(dtype=str))):
            if str(m).lower() in t:
                model = str(m)
                break
    except Exception:
        pass
    try:
        for p in pd.unique(df.get("primary_failed_part", pd.Series(dtype=str))):
            # Case-insensitive substring match for parts
            # Check if query contains part name OR if part name contains query
            part_lower = str(p).lower()
            if part_lower in t or any(word in part_lower for word in t.split() if len(word) > 2):
                part = str(p)
                break
    except Exception:
        pass
    return model, part
