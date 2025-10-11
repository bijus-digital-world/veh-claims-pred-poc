"""
Chat helper for 'Chat with Data' column.
Provides retrieval + reply generation (FAISS-aware, TF-IDF fallback),
parsing utilities and natural-language summarization (no raw row dumps).
Designed to be imported by app.py.

Key behaviours:
- Treats failures (claims + repairs + recalls) as the default metric via a synthetic `failures_count`.
- Honors explicit user requests for `claims`, `repairs`, or `recalls` and (if present) cost columns.
- Uses FAISS (sentence-transformers) if available, otherwise TF-IDF fallback.
- Always presents an overall dataset-level metric first, then a grouped breakdown (by model/age/mileage/dealer) — this pattern is applied consistently across monthly, rate, top-parts and trend queries.
- Does not return raw table rows — replies are natural-language summaries.
"""

from typing import Tuple, List, Dict, Optional
import re
import html as _html
import pandas as pd
import numpy as np

# imports for TF-IDF fallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional sentence-transformers / faiss usage if the app has them available
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    HAS_ST = True
    HAS_FAISS = True
except Exception:
    HAS_ST = False
    HAS_FAISS = False

# default embedding name used elsewhere in app
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# -----------------------
# Synthetic metric helper
# -----------------------
def ensure_failures_column(df: pd.DataFrame, out_col: str = "failures_count") -> pd.DataFrame:
    """
    Ensure df contains `out_col` which is the sum of claims_count + repairs_count + recalls_count
    where available. Works on a copy and returns the df with the new column.
    """
    df = df.copy()

    def _get_col_safe(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)
        return pd.Series(0, index=df.index, dtype=int)

    claims = _get_col_safe("claims_count")
    repairs = _get_col_safe("repairs_count")
    recalls = _get_col_safe("recalls_count")
    df[out_col] = (claims + repairs + recalls).astype(int)
    return df


# ---------- index helpers ----------
def _row_to_doc(r: dict) -> str:
    """Convert a data row to a compact text doc used for TF-IDF / embedding indexing.
    Include failures summary so retrieval reflects failures-centric queries.
    """
    claims = int(r.get('claims_count', 0) or 0)
    repairs = int(r.get('repairs_count', 0) or 0)
    recalls = int(r.get('recalls_count', 0) or 0)
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


def build_tf_idf_index(df: pd.DataFrame, max_features: int = 6000):
    """Build a TF-IDF index for lightweight retrieval (fast for small-medium datasets)."""
    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    if not docs:
        return None, None, []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=max_features)
    X = vect.fit_transform(docs)
    rows = df.to_dict(orient="records")
    return vect, X, rows


def _embed_query_st_model(query: str, model_name: str = EMBED_MODEL_NAME):
    """Return an embedding for query using sentence-transformers (if available)."""
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
                                 top_k: int = 6) -> List[dict]:
    """
    Tries FAISS retrieval (if faiss_res available / valid), otherwise falls back to TF-IDF.
    faiss_res is expected to be the object built in your app.py (keys: available,index,embs,meta,d,message).
    """
    # Try FAISS if available
    try:
        if faiss_res and faiss_res.get("available") and HAS_ST:
            index = faiss_res.get("index")
            embs = faiss_res.get("embs")
            meta = faiss_res.get("meta")
            if index is not None and embs is not None and meta is not None:
                q_emb = _embed_query_st_model(query)
                # normalize for cosine-like inner product
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
                    return results
    except Exception:
        # if anything goes wrong, fallthrough to TF-IDF
        pass

    # TF-IDF fallback
    if tfidf_vect is None or tfidf_X is None or not tfidf_rows:
        return []
    qv = tfidf_vect.transform([query])
    sims = cosine_similarity(qv, tfidf_X).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        results.append({**tfidf_rows[i], "score": float(sims[i])})
    return results


# ---------- small utilities ----------
def parse_model_part_from_text(df: pd.DataFrame, text: str) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort: find model & part mentioned in text (case-insensitive substring match)."""
    t = text.lower()
    model = None
    part = None
    for m in pd.unique(df.get("model", pd.Series(dtype=str))):
        try:
            if str(m).lower() in t:
                model = str(m)
                break
        except Exception:
            continue
    for p in pd.unique(df.get("primary_failed_part", pd.Series(dtype=str))):
        try:
            if str(p).lower() in t:
                part = str(p)
                break
        except Exception:
            continue
    return model, part


def _safe_date_str(dt):
    try:
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(dt)


def build_results_html_table(rows: List[dict], claim_rate: Optional[float] = None, max_rows: int = 30) -> str:
    """Return an HTML fragment summarizing matching rows and optional claim_rate.
    NOTE: kept for compatibility but the chat generate_reply intentionally does not return raw tables.
    """
    if not rows:
        return "<p><em>No matching historical records found.</em></p>"

    out = []
    if claim_rate is not None:
        out.append(f"<p><strong>Sample claim rate:</strong> {claim_rate:.1f}%</p>")

    out.append("<table style='width:100%; border-collapse:collapse; font-size:13px;'>")
    out.append("<thead><tr style='text-align:left; color:#94a3b8;'><th>date</th><th>model</th><th>part</th><th>claims</th><th>mileage</th><th>age</th></tr></thead>")
    out.append("<tbody>")
    for r in rows[:max_rows]:
        date_str = _safe_date_str(r.get("date",""))
        model = _html.escape(str(r.get("model","")))
        part = _html.escape(str(r.get("primary_failed_part","")))
        claims = int(r.get("claims_count", 0)) if r.get("claims_count") is not None else 0
        mileage = _html.escape(str(r.get("mileage_bucket","")))
        age = _html.escape(str(r.get("age_bucket","")))
        out.append(f"<tr style='border-top:1px solid rgba(255,255,255,0.03);'><td style='padding:6px 8px'>{date_str}</td><td style='padding:6px 8px'>{model}</td><td style='padding:6px 8px'>{part}</td><td style='padding:6px 8px'>{claims}</td><td style='padding:6px 8px'>{mileage}</td><td style='padding:6px 8px'>{age}</td></tr>")
    out.append("</tbody></table>")
    if len(rows) > max_rows:
        out.append(f"<div style='color:#94a3b8; margin-top:6px;'>Showing {max_rows} of {len(rows)} matches.</div>")
    return "".join(out)


# ---------- metric detection / synonyms ----------
_COLUMN_SYNONYMS = {
    "region": ["region", "area", "zone", "territory"],
    "dealer": ["dealer", "dealer_name", "dealer id", "dealerid"],
    "service_center": ["service center", "service_center", "servicecentre", "service center name"],
    "claim_cost": ["claim cost", "claim_cost", "cost", "warranty_cost", "claim_amount"],
    "model": ["model", "vehicle model", "variant"],
    "primary_failed_part": ["part", "failed part", "primary_failed_part", "failure part", "pfp"],
    "repairs_count": ["repair", "repairs", "repairs_count"],
    "recalls_count": ["recall", "recalls", "recalls_count"],
    "time_to_resolution": ["time to resolution", "resolution_time", "time_to_resolution", "days_to_resolve", "resolution_days"],
}


def _requested_missing_columns(user_text: str, df_cols) -> dict:
    """
    Returns a dict mapping requested_key -> (found_bool, matched_column_or_none).
    Example: { 'region': (False, None), 'dealer': (True, 'dealer') }
    """
    found = {}
    txt = user_text.lower()
    cols_lower = {c.lower(): c for c in df_cols}  # map lowercase -> original
    for key, synonyms in _COLUMN_SYNONYMS.items():
        matched = False
        for s in synonyms:
            if re.search(r"" + re.escape(s) + r"", txt):
                # user asked about this concept
                # check if any df column matches the synonyms or the key
                # prefer exact column matches (case-insensitive)
                if key.lower() in cols_lower:
                    found[key] = (True, cols_lower[key.lower()])
                else:
                    # find any column whose lowercase appears in synonyms
                    matched_col = None
                    for c_low, c_orig in cols_lower.items():
                        for s2 in synonyms:
                            if s2 in c_low:
                                matched_col = c_orig
                                break
                        if matched_col:
                            break
                    if matched_col:
                        found[key] = (True, matched_col)
                    else:
                        found[key] = (False, None)
                matched = True
                break
        # if phrase not present in user_text, skip
    return found


# ---------- metric intent detection & fallback ----------
def _detect_metric_from_text(text: str) -> str:
    """
    Decide metric from user text.
    Prioritizes explicit requests:
      - exact 'claim rate' or 'current claim rate' -> 'claims_count'
      - explicit 'claims'/'repairs'/'recalls' also respected
    Default: 'failures_count' (synthetic).
    """
    t = (text or "").lower()

    # high-confidence patterns first
    if re.search(r"\bclaim rate\b", t) or re.search(r"\bcurrent claim\b", t) or re.search(r"\bcurrent claim rate\b", t):
        return "claims_count"

    # explicit single-word metric mentions
    if re.search(r"\brecall(s)?\b", t) or re.search(r"\brecalls_count\b", t):
        return "recalls_count"
    if re.search(r"\brepair(s)?\b", t) or re.search(r"\brepairs_count\b", t):
        return "repairs_count"
    # if user says 'claim' together with other words (but not cost), treat as claims
    if re.search(r"\bclaim(s)?\b", t) and not re.search(r"\bclaim cost\b|\bcost\b", t):
        return "claims_count"
    if re.search(r"\bcost\b|claim cost|warranty_cost", t):
        return "claim_cost"

    # fallback: failures (claims + repairs + recalls)
    return "failures_count"


def _safe_column(df: pd.DataFrame, colnames: List[str]) -> Optional[str]:
    """Return the first existing column name from colnames present in df, else None."""
    for c in colnames:
        if c in df.columns:
            return c
    return None


def _metric_or_fallback_column(df: pd.DataFrame, requested_metric: str) -> Tuple[Optional[str], pd.DataFrame]:
    """
    Map a requested metric name to an actual dataframe column name to use.
    If requested_metric == 'failures_count', create/return 'failures_count' (synthetic) and return modified df.
    Otherwise return (colname, df).
    """
    if requested_metric == "failures_count":
        if "failures_count" not in df.columns:
            df_with_failures = ensure_failures_column(df, out_col="failures_count")
            return "failures_count", df_with_failures
        return "failures_count", df

    # existing metric like claims_count/repairs_count/recalls_count or cost
    if requested_metric in df.columns:
        return requested_metric, df

    if requested_metric == "claims_count":
        c = _safe_column(df, ["claims_count", "claim_count", "claims"])
        return (c, df) if c else (None, df)
    if requested_metric == "repairs_count":
        c = _safe_column(df, ["repairs_count", "repairs"])
        return (c, df) if c else (None, df)
    if requested_metric == "recalls_count":
        c = _safe_column(df, ["recalls_count", "recalls"])
        return (c, df) if c else (None, df)
    if requested_metric == "claim_cost":
        c = _safe_column(df, ["claim_cost", "warranty_cost", "cost", "amount", "claim_amount"])
        return (c, df) if c else (None, df)
    return (None, df)


# ---------- top failed parts & incident summaries ----------
def summarize_top_failed_parts(df_history: pd.DataFrame, metric: str = "failures_count", top_n: int = 6) -> str:
    """
    Natural-language summary of top failed parts by the requested metric.
    If claim_cost is requested but not available, informs user and falls back to counts.
    Always begins with an overall metric sentence for context.
    """
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
            notice = ("<p><em>Note:</em> Requested metric not available; showing top parts by failures_count instead.</p>")
        else:
            return "<p>Requested metric not available and no suitable fallback found (no claims/repairs/recalls columns).</p>"
    else:
        notice = ""

    try:
        total_metric = pd.to_numeric(df_used[metric_col], errors="coerce").fillna(0).sum()
        total_rows = len(df_used)
        label = "failures" if metric_col == "failures_count" else metric_col.replace('_',' ')
        header = f"<p>Overall {label}: {int(total_metric)} across {total_rows} records.</p>"

        grp = df_used.groupby(part_col).agg(total_metric=(metric_col, "sum"), incidents=("claims_count" if "claims_count" in df_used.columns else metric_col, "count"))
        grp = grp.sort_values("total_metric", ascending=False).head(top_n)
        lines = []
        for part, row in grp.iterrows():
            if metric_col and any(k in metric_col.lower() for k in ["cost", "amount", "warranty"]):
                lines.append(f"{part} → ${row['total_metric']:.0f} total (incidents: {int(row['incidents'])})")
            else:
                lines.append(f"{part} → {int(row['total_metric'])} {label} (incidents: {int(row['incidents'])})")
        return notice + header + "<p>Top failed parts:<br>" + "<br>".join(lines) + "</p>"
    except Exception as e:
        return f"<p>Could not compute top failed parts (error: {e}).</p>"


def summarize_incident_details(df_history: pd.DataFrame, metric: str = "failures_count") -> str:
    """
    Summarize incident-level totals and (if available) average resolution time.
    Starts with overall totals for context.
    """
    if df_history is None or df_history.empty:
        return "<p>No historical data available.</p>"

    metric_col, df_used = _metric_or_fallback_column(df_history, metric)
    if metric_col is None:
        return "<p>Requested metric not available in dataset (no claims/repairs/recalls column).</p>"

    try:
        total_metric = pd.to_numeric(df_used[metric_col], errors="coerce").fillna(0).sum()
        total_rows = len(df_used)
        label = "failures" if metric_col == "failures_count" else metric_col.replace('_',' ')
        header = f"<p>Overall {label}: {int(total_metric)} across {total_rows} records.</p>"
    except Exception:
        return "<p>Unable to aggregate the requested metric (data type issue).</p>"

    # average time to resolution if available
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


# ---------- model trend computation (metric-agnostic) ----------
def _compute_model_trends(df_history: pd.DataFrame, metric_col: str = "failures_count",
                          min_months: int = 6,
                          slope_threshold: float = 0.0,
                          top_n: int = 5):
    """
    Compute per-model monthly slopes for the requested metric (defaults to failures_count).
    Returns list of tuples: (model, slope, months_of_data, last_month_value)
    """
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


# ---------- top-level generator ----------
"""
chat_helper.py

Provides:
- retrieval helpers (FAISS preferred / TF-IDF fallback)
- data-safe aggregation helpers
- summarizers (top parts, incident details)
- robust generate_reply(...) implementing compute-first + RAG-as-needed behavior
"""

from typing import Tuple, List, Dict, Optional, Any
import re
import html as _html
import pandas as pd
import numpy as np
from dateutil import parser as _date_parse
from datetime import datetime, timezone

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# optional embedding / faiss
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    HAS_ST = True
    HAS_FAISS = True
except Exception:
    HAS_ST = False
    HAS_FAISS = False

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # default embedding if available

# ---------- small utilities ----------
def _has_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", (text or "").lower()) is not None

def _safe_column(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """
    Return first existing column name from candidates (case-insensitive),
    or None if none found.
    """
    if df is None:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        cand_low = cand.lower()
        # direct match
        if cand_low in cols_lower:
            return cols_lower[cand_low]
        # partial match: if candidate text appears inside a column name
        for c_low, c_orig in cols_lower.items():
            if cand_low in c_low:
                return c_orig
    return None

def _safe_date_str(dt):
    try:
        if hasattr(dt, "strftime"):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(dt)

# ---------- text->doc helpers for TF-IDF ----------
def _row_to_doc(r: dict) -> str:
    parts = [
        f"model: {r.get('model','')}",
        f"part: {r.get('primary_failed_part','')}",
        f"mileage_bucket: {r.get('mileage_bucket','')}",
        f"age_bucket: {r.get('age_bucket','')}",
        f"date: {r.get('date','')}",
        f"claims: {r.get('claims_count',0)}",
    ]
    return "; ".join(parts)

def build_tf_idf_index(df: pd.DataFrame, max_features: int = 6000):
    docs = [_row_to_doc(r) for _, r in df.iterrows()]
    if not docs:
        return None, None, []
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=max_features)
    X = vect.fit_transform(docs)
    rows = df.to_dict(orient="records")
    return vect, X, rows

def _embed_query_st_model(query: str, model_name: str = EMBED_MODEL_NAME):
    """Return an embedding for query using sentence-transformers (if available)."""
    if not HAS_ST:
        raise RuntimeError("sentence-transformers not available")
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True).astype("float32")
    return emb

# ---------- retrieval ----------
def retrieve_with_faiss_or_tfidf(query: str,
                                 faiss_res: dict,
                                 tfidf_vect,
                                 tfidf_X,
                                 tfidf_rows,
                                 top_k: int = 6) -> List[dict]:
    """
    Tries FAISS retrieval (if faiss_res available / valid), otherwise falls back to TF-IDF.
    faiss_res is expected to be dict with keys: available, index, embs, meta, d, message
    """
    # Try FAISS if available and configured
    try:
        if faiss_res and faiss_res.get("available") and HAS_ST and HAS_FAISS:
            index = faiss_res.get("index")
            if index is not None:
                q_emb = _embed_query_st_model(query)
                faiss.normalize_L2(q_emb)
                D, I = index.search(q_emb, top_k)
                sims = D[0].tolist()
                idxs = I[0].tolist()
                results = []
                meta = faiss_res.get("meta", [])
                for idx, sim in zip(idxs, sims):
                    if idx < 0 or idx >= len(meta):
                        continue
                    row = dict(meta[idx]) if isinstance(meta[idx], dict) else dict(meta[idx])
                    row["score"] = float(sim)
                    results.append(row)
                if results:
                    return results
    except Exception:
        # fall back to TF-IDF
        pass

    # TF-IDF fallback
    if tfidf_vect is None or tfidf_X is None or not tfidf_rows:
        return []
    try:
        qv = tfidf_vect.transform([query])
        sims = cosine_similarity(qv, tfidf_X).flatten()
        top_idx = sims.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            r = dict(tfidf_rows[i])
            r["score"] = float(sims[i])
            results.append(r)
        return results
    except Exception:
        return []

# ---------- metric helpers ----------
def ensure_failures_column(df: pd.DataFrame, out_col: str = "failures_count") -> pd.DataFrame:
    df = df.copy()
    claims = pd.to_numeric(df.get("claims_count"), errors="coerce").fillna(0)
    repairs = pd.to_numeric(df.get("repairs_count"), errors="coerce").fillna(0)
    recalls = pd.to_numeric(df.get("recalls_count"), errors="coerce").fillna(0)
    df[out_col] = (claims + repairs + recalls).astype(int)
    return df

def _metric_or_fallback_column(df: pd.DataFrame, requested_metric: str) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    """
    Given requested_metric (like 'claims_count' or 'failures_count'), return:
      - metric_col: actual column name that should be used (or None)
      - df_with_metric: possibly modified dataframe (e.g. with failures_count created) or None
    """
    if df is None:
        return None, None
    cols_lower = {c.lower(): c for c in df.columns}

    # exact matches or common aliases
    if requested_metric is None:
        requested_metric = ""

    rm = requested_metric.lower()

    if rm in cols_lower:
        return cols_lower[rm], None

    # claims_count aliases
    for alias in ["claims_count", "claims", "claim_count", "claim_counts"]:
        if alias in cols_lower:
            return cols_lower[alias], None

    # repairs_count
    for alias in ["repairs_count", "repairs", "repair_count"]:
        if alias in cols_lower:
            return cols_lower[alias], None

    # recalls_count
    for alias in ["recalls_count", "recalls", "recall_count"]:
        if alias in cols_lower:
            return cols_lower[alias], None

    # failures_count: create if not present
    if rm == "failures_count" or rm == "failures":
        # create the synthetic failures_count column from claims/repairs/recalls
        df_out = ensure_failures_column(df, out_col="failures_count")
        return "failures_count", df_out

    # final attempt: if user asked for claims but different naming, try substring search
    for c_low, c_orig in cols_lower.items():
        if "claim" in c_low:
            return c_orig, None

    # nothing found
    return None, None

def _detect_metric_from_text(text: str) -> str:
    """
    Decide metric from user text.
    Prioritizes explicit requests:
      - exact 'claim rate' or 'current claim rate' -> 'claims_count'
      - explicit 'claims'/'repairs'/'recalls' also respected
    Default: 'failures_count' (synthetic).
    """
    t = (text or "").lower()

    # high-confidence patterns first
    if re.search(r"\bclaim rate\b", t) or re.search(r"\bcurrent claim\b", t) or re.search(r"\bcurrent claim rate\b", t):
        return "claims_count"

    # explicit single-word metric mentions
    if re.search(r"\brecall(s)?\b", t) or re.search(r"\brecalls_count\b", t):
        return "recalls_count"
    if re.search(r"\brepair(s)?\b", t) or re.search(r"\brepairs_count\b", t):
        return "repairs_count"
    # if user says 'claim' together with other words (but not cost), treat as claims
    if re.search(r"\bclaim(s)?\b", t) and not re.search(r"\bclaim cost\b|\bcost\b", t):
        return "claims_count"
    if re.search(r"\bcost\b|claim cost|warranty_cost", t):
        return "claim_cost"

    # fallback: failures (claims + repairs + recalls)
    return "failures_count"

def _requested_missing_columns(user_text: str, df_cols) -> dict:
    """
    Returns a dict mapping requested_key -> (found_bool, matched_column_or_none).
    Example: { 'region': (False, None), 'dealer': (True, 'dealer') }
    """
    found = {}
    txt = (user_text or "").lower()
    cols_lower = {c.lower(): c for c in df_cols}  # map lowercase -> original
    for key, synonyms in {
        "region": ["region", "area", "zone", "territory"],
        "dealer": ["dealer", "dealer_name", "dealer id", "dealerid"],
        "service_center": ["service center", "service_center", "servicecentre", "service center name"],
        "claim_cost": ["claim cost", "cost", "claim_cost", "warranty_cost"],
        "model": ["model", "vehicle model", "variant"],
        "primary_failed_part": ["part", "failed part", "primary_failed_part", "failure part", "pfp"],
        "repairs_count": ["repair", "repairs", "repairs_count"],
        "recalls_count": ["recall", "recalls", "recalls_count"],
        "time_to_resolution": ["time to resolution", "resolution_time", "time_to_resolution", "days_to_resolve", "resolution_days"],
    }.items():
        matched = False
        for s in synonyms:
            if re.search(r"\b" + re.escape(s) + r"\b", txt):
                # user asked about this concept
                if key.lower() in cols_lower:
                    found[key] = (True, cols_lower[key.lower()])
                else:
                    matched_col = None
                    for c_low, c_orig in cols_lower.items():
                        if s in c_low:
                            matched_col = c_orig
                            break
                    if matched_col:
                        found[key] = (True, matched_col)
                    else:
                        found[key] = (False, None)
                matched = True
                break
        # skip if phrase not present
    return found

# ---------- parse model/part ----------
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
            if str(p).lower() in t:
                part = str(p)
                break
    except Exception:
        pass
    return model, part

# ---------- compute model trends ----------
def _compute_model_trends(df_history: pd.DataFrame,
                          metric_col: str = "claims_count",
                          min_months: int = 6,
                          slope_threshold: float = 0.0,
                          top_n: int = 5):
    """
    Compute per-model monthly slopes for the metric_col.
    Returns list of tuples: (model, slope, months_of_data, last_month_value)
    Only models with >= min_months months of data are considered.
    """
    if df_history is None or metric_col is None:
        return []

    df = df_history.copy()
    if "date" not in df.columns:
        return []

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df[metric_col] = pd.to_numeric(df.get(metric_col, 0), errors="coerce").fillna(0)

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

# ---------- summarizers ----------
def summarize_top_failed_parts(df: pd.DataFrame, metric="failures_count", top_n=6) -> str:
    metric_col = metric if metric in df.columns else _safe_column(df, [metric, "failures_count", "claims_count"])
    if metric_col is None:
        return "<p>Cannot compute top failed parts; metric not available.</p>"
    part_col = _safe_column(df, ["primary_failed_part", "failed_part", "part"])
    if part_col is None:
        return "<p>No part/failure column in the dataset.</p>"
    try:
        grp = df.groupby(part_col)[metric_col].sum().sort_values(ascending=False).head(top_n)
        lines = [f"{_html.escape(str(idx))} → {int(v)}" for idx, v in grp.items()]
        return f"<p><strong>Top failed parts (by {metric_col}):</strong><br>{'<br>'.join(lines)}</p>"
    except Exception as e:
        return f"<p>Could not compute top failed parts: {_html.escape(str(e))}</p>"

def summarize_incident_details(df: pd.DataFrame, metric="failures_count") -> str:
    # Average time to resolution if present
    tcol = _safe_column(df, ["time_to_resolution", "resolution_days", "days_to_resolve"])
    if tcol:
        try:
            avg = float(pd.to_numeric(df.get(tcol), errors="coerce").dropna().mean())
            return f"<p>Average time to resolution: ≈ {avg:.1f} days (based on column '{tcol}').</p>"
        except Exception:
            pass
    # fallback counts by incident type if available
    return "<p>Could not compute incident details — dataset lacks time-to-resolution or incident-specific fields.</p>"

# ---------- top-level generator ----------
def generate_reply(user_text: str,
                   df_history: pd.DataFrame,
                   faiss_res: dict,
                   tfidf_vect,
                   tfidf_X,
                   tfidf_rows,
                   get_bedrock_summary_callable,
                   top_k: int = 6) -> str:
    """
    Entry point used by app.py to produce assistant HTML reply.
    - Focuses on failures (claims+repairs+recalls) by default.
    - Returns only natural-language summaries (no record dumps).
    - Presents an overall metric first, then grouped breakdowns.
    """
    ut = (user_text or "").strip()
    if df_history is None or (hasattr(df_history, "empty") and df_history.empty):
        return "<p>No historical data is loaded. Please upload a dataset.</p>"

    # early guard: detect if user asked about columns that don't exist
    missing_checks = _requested_missing_columns(ut, df_history.columns)
    if missing_checks:
        missing = [k for k, v in missing_checks.items() if v[0] is False]
        if missing:
            available = ", ".join(sorted(df_history.columns))
            pretty_missing = ", ".join(missing)
            return (f"<p>It looks like you're asking about {pretty_missing}, but your dataset does not contain that column. "
                    f"Available columns include: <strong>{available}</strong>. "
                    f"Try asking about one of those (for example, 'failure rate by model' or 'top failed parts').</p>")

    if not ut:
        return "<p>Please ask a question about the historical data (e.g. 'failure rate for model Sentra').</p>"

    # greeting
    if len(ut.split()) <= 3 and any(g in ut.lower() for g in ["hi", "hello", "hey"]):
        return "<p>Hi — I'm the Nissan failures assistant. Ask for monthly failure counts, failure rates by model/age/mileage, or prescriptive guidance (e.g. 'prescribe for model Rogue part Battery').</p>"

    # schema / column list request
    if any(phrase in ut.lower() for phrase in ["column names", "columns", "fields", "schema", "features", "headers"]):
        cols = list(df_history.columns)
        formatted = ", ".join(cols)
        return (f"<p>Your dataset contains the following columns:</p>"
                f"<p style='color:#cfe9ff'>{formatted}</p>"
                f"<p>You can ask questions like:<br>"
                f"<em>• What is the failure rate by model?<br>"
                f"• Which part has most repairs?<br>"
                f"• Show monthly recall trend.</em></p>")

    # timespan / date-range requests
    if re.search(r"\b(how many months|months (?:of )?data|how many years|years (?:of )?data|date range|earliest date|latest date|first date|last date|data from)\b", ut.lower()):
        if "date" not in df_history.columns:
            return "<p>Your dataset does not contain a <strong>'date'</strong> column, so I can't compute the time span.</p>"
        try:
            df_dates = df_history.copy()
            df_dates["date_parsed"] = pd.to_datetime(df_dates["date"], errors="coerce")
            df_dates = df_dates.dropna(subset=["date_parsed"])
            if df_dates.empty:
                return "<p>Could not find any parsable dates in the 'date' column.</p>"
            min_d = df_dates["date_parsed"].min()
            max_d = df_dates["date_parsed"].max()
            months_span = (max_d.year - min_d.year) * 12 + (max_d.month - min_d.month) + 1
            distinct_months = int(df_dates["date_parsed"].dt.to_period("M").nunique())
            rows_with_dates = int(df_dates.shape[0])
            min_s = min_d.strftime("%b %Y")
            max_s = max_d.strftime("%b %Y")
            return (
                f"<p>Dataset date range: <strong>{min_s}</strong> → <strong>{max_s}</strong>.</p>"
                f"<p>This spans <strong>{months_span}</strong> calendar months, with data present in <strong>{distinct_months}</strong> distinct months. "
                f"There are <strong>{rows_with_dates}</strong> records with parsable dates.</p>"
            )
        except Exception as e:
            return f"<p>Couldn't compute date range (error: {_html.escape(str(e))}).</p>"

    # determine metric intent (failures by default)
    requested_metric = _detect_metric_from_text(ut)
    metric_col, df_with_metric = _metric_or_fallback_column(df_history, requested_metric)
    if df_with_metric is not None:
        df_history = df_with_metric

    # If user explicitly asked for claims but not present, inform them
    if requested_metric == "claims_count" and metric_col is None:
        metric_col, _ = _metric_or_fallback_column(df_history, "claims_count")
        if metric_col is None:
            available = ", ".join(sorted(df_history.columns))
            return (f"<p>You asked for the claim rate but this dataset does not contain a 'claims' column. "
                    f"Available columns: <strong>{available}</strong>. I can compute the overall failure rate (claims+repairs+recalls) instead, or you can provide a dataset with a 'claims_count' column.</p>")

    # If user asked about cost but we don't have cost, inform
    if requested_metric == "claim_cost" and (metric_col is None):
        available = ", ".join(sorted(df_history.columns))
        return (f"<p>You asked about claim cost, but this dataset does not contain any cost-like column. "
                f"Available columns include: <strong>{available}</strong>. I can show counts (failures/claims/repairs/recalls) or you can upload a cost column named 'claim_cost' or 'warranty_cost'.</p>")

    # prescriptive shortcut (requires model + part)
    if any(w in ut.lower() for w in ["prescribe", "recommend", "prescriptive", "advice", "action"]):
        model, part = parse_model_part_from_text(df_history, ut)
        if model and part:
            slice_df = df_history[(df_history["model"] == model) & (df_history["primary_failed_part"] == part)]
            if not slice_df.empty:
                mileage_bucket = slice_df.iloc[0].get("mileage_bucket", "")
                age_bucket = slice_df.iloc[0].get("age_bucket", "")
                total_inc = slice_df.shape[0]
                failures_col = "failures_count" if "failures_count" in slice_df.columns else None
                if failures_col is None:
                    slice_df = ensure_failures_column(slice_df, out_col="failures_count")
                    failures_col = "failures_count"
                total_failures = int(slice_df[failures_col].sum())
                pct = (total_failures / total_inc * 100.0) if total_inc > 0 else 0.0
                try:
                    summary_html = get_bedrock_summary_callable(model, part, mileage_bucket, age_bucket, pct)
                    plain = re.sub(r"<[^>]+>", "", summary_html).strip()
                    return f"<p>{_html.escape(plain)}</p>"
                except Exception as e:
                    return f"<p>Could not generate prescriptive summary via Bedrock: {_html.escape(str(e))}</p>"
        return "<p>I can generate a prescriptive summary if you include a model and part (e.g. 'prescribe for model Sentra part Battery').</p>"

    # --- handle explicit TOTAL / OVERALL questions before retrieval ---
    ut_low = ut.lower()
    # total_trigger = re.search(r"\b(total(?: number)?(?: of)?|total count|how many(?: total)?|give me the total|what is the total|total number of)\b", ut_low)
    # wants_failures_word = any(k in ut_low for k in ["failur", "claim", "repair", "recall", "total"])
    
    # skip total branch if query includes "per month" or "monthly"
    if re.search(r"\b(per month|monthly)\b", ut_low):
        total_trigger = None
    else:
        total_trigger = re.search(
            r"\b(total(?: number)?(?: of)?|total count|how many(?: total)?|give me the total|what is the total|total number of)\b",
            ut_low,
        )

    wants_failures_word = any(k in ut_low for k in ["failur", "claim", "repair", "recall", "total"])

    if total_trigger and wants_failures_word:
        if metric_col is None:
            metric_col, df_with_metric = _metric_or_fallback_column(df_history, requested_metric)
            if df_with_metric is not None:
                df_history = df_with_metric
        if metric_col is None:
            available = ", ".join(sorted(df_history.columns))
            return (f"<p>You asked for totals but the requested metric is not available in the dataset. "
                    f"Available columns: <strong>{available}</strong>. Try asking for 'failures', 'claims', 'repairs', or 'recalls'.</p>")
        if metric_col == "failures_count" and "failures_count" not in df_history.columns:
            df_history = ensure_failures_column(df_history, out_col="failures_count")
        try:
            total_metric = int(pd.to_numeric(df_history[metric_col], errors="coerce").fillna(0).sum())
            total_rows = len(df_history)
            per100 = (total_metric / total_rows) * 100.0 if total_rows > 0 else 0.0
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            top_models = []
            if "model" in df_history.columns:
                grp = df_history.groupby("model")[metric_col].sum().sort_values(ascending=False).head(6)
                top_models = [f"{_html.escape(str(idx))} → {int(v)}" for idx, v in grp.items() if pd.notna(idx)]
            top_models_txt = "<br>".join(top_models) if top_models else ""
            return (f"<p>Overall {label}: total = {total_metric} across {total_rows} records "
                    f"(~{per100:.1f} per 100 records).<br>"
                    + (f"<strong>Top models by {label}:</strong><br>{top_models_txt}" if top_models_txt else "") +
                    "</p>")
        except Exception as e:
            return f"<p>Could not compute totals due to a data issue: {_html.escape(str(e))}</p>"


        # ---------- Time-to-resolution / average resolution time ----------
    # Catch a wide set of user phrasings for time-to-resolution / average time to resolve
    if re.search(r"\b(avg|average|mean|median|typical|what is the)\b.*\b(time to (claim )?resolution|resolution time|time to (?:close|resolve)|days to resolve|time to claim)\b", ut.lower()) \
       or re.search(r"\b(time to (claim )?resolution|resolution time|time to resolve|days to resolve|time to claim)\b", ut.lower()):

        # find candidate time-to-resolution column
        time_col = _safe_column(df_history, [
            "time_to_resolution", "resolution_days", "days_to_resolve", "time_to_resolve",
            "resolution_time", "time_to_close", "time_to_claim_resolution", "time_to_claim"
        ])
        if not time_col:
            return ("<p>I can't find a time-to-resolution column in your dataset. "
                    "Look for columns named like <em>time_to_resolution</em>, <em>resolution_days</em>, or <em>days_to_resolve</em>.</p>")

        # coerce numeric and drop missing
        df_tmp = df_history.copy()
        df_tmp[time_col] = pd.to_numeric(df_tmp.get(time_col), errors="coerce")
        df_valid = df_tmp.dropna(subset=[time_col])
        if df_valid.empty:
            return (f"<p>I found column <strong>{_html.escape(time_col)}</strong> but it contains no numeric values I can use to compute averages.</p>")

        # central statistics
        cnt = int(len(df_valid))
        mean_v = float(df_valid[time_col].mean())
        median_v = float(df_valid[time_col].median())
        std_v = float(df_valid[time_col].std(ddof=0)) if cnt > 1 else 0.0
        q25 = float(df_valid[time_col].quantile(0.25))
        q75 = float(df_valid[time_col].quantile(0.75))
        min_v = float(df_valid[time_col].min())
        max_v = float(df_valid[time_col].max())

        # guess unit from column name (helpful but conservative)
        tcol_lower = time_col.lower()
        if "day" in tcol_lower:
            unit = "days"
        elif "hour" in tcol_lower:
            unit = "hours"
        else:
            unit = "units (as recorded in the column)"

        # Build base reply (overall first)
        reply_lines = [
            f"<p>Average time to resolution (based on <strong>{_html.escape(time_col)}</strong>, {cnt} records):</p>",
            f"<ul style='margin-top:6px;'>",
            f"<li><strong>Mean:</strong> {mean_v:.1f} {unit}</li>",
            f"<li><strong>Median:</strong> {median_v:.1f} {unit} (IQR: {q25:.1f}–{q75:.1f})</li>",
            f"<li><strong>Range:</strong> {min_v:.1f} – {max_v:.1f} {unit}</li>",
            f"<li><strong>Std. dev:</strong> {std_v:.1f} {unit}</li>",
            f"</ul>"
        ]

        # If user asked for "by model" / "by part" / "by dealer" include top groups
        if "by model" in ut.lower() or "per model" in ut.lower() or _has_word(ut.lower(), "model"):
            if "model" in df_tmp.columns:
                grp = df_valid.groupby("model")[time_col].agg(["count", "mean"]).sort_values("mean", ascending=False)
                top = grp.head(6)
                if not top.empty:
                    grp_lines = [f"{_html.escape(str(idx))} → mean {row['mean']:.1f} {unit} (n={int(row['count'])})" for idx, row in top.iterrows()]
                    reply_lines.append("<p><strong>Average by model (top):</strong><br>" + "<br>".join(grp_lines) + "</p>")

        elif "by part" in ut.lower() or "by primary_failed_part" in ut.lower() or _has_word(ut.lower(), "part"):
            part_col = _safe_column(df_tmp, ["primary_failed_part", "failed_part", "part"])
            if part_col:
                grp = df_valid.groupby(part_col)[time_col].agg(["count", "mean"]).sort_values("mean", ascending=False)
                top = grp.head(6)
                if not top.empty:
                    grp_lines = [f"{_html.escape(str(idx))} → mean {row['mean']:.1f} {unit} (n={int(row['count'])})" for idx, row in top.iterrows()]
                    reply_lines.append("<p><strong>Average by part (top):</strong><br>" + "<br>".join(grp_lines) + "</p>")

        elif "by dealer" in ut.lower() or _has_word(ut.lower(), "dealer"):
            dealer_col = _safe_column(df_tmp, ["dealer", "dealer_name", "dealer id"])
            if dealer_col:
                grp = df_valid.groupby(dealer_col)[time_col].agg(["count", "mean"]).sort_values("mean", ascending=False)
                top = grp.head(6)
                if not top.empty:
                    grp_lines = [f"{_html.escape(str(idx))} → mean {row['mean']:.1f} {unit} (n={int(row['count'])})" for idx, row in top.iterrows()]
                    reply_lines.append("<p><strong>Average by dealer (top):</strong><br>" + "<br>".join(grp_lines) + "</p>")

        # friendly final hint
        reply_lines.append("<p style='color:#94a3b8; margin-top:6px;'>If you want a different grouping (e.g. 'by model and part') ask: 'average time to resolution by model and part'.</p>")

        return "".join(reply_lines)


    # Retrieval for context (RAG)
    results = retrieve_with_faiss_or_tfidf(ut, faiss_res, tfidf_vect, tfidf_X, tfidf_rows, top_k)

    # sample DataFrame from retrieval results
    sample_df = None
    if results:
        try:
            sample_df = pd.DataFrame(results)
            sample_df["date_parsed"] = pd.to_datetime(sample_df.get("date"), errors="coerce")
            if metric_col and metric_col in sample_df.columns:
                sample_df[metric_col] = pd.to_numeric(sample_df.get(metric_col), errors="coerce").fillna(0)
        except Exception:
            sample_df = None

    text_low = ut.lower()

    # 1) Monthly counts / "per month" requests
    if re.search(r"\b(per month|monthly|claims per month|repairs per month|recalls per month)\b", text_low):
        if metric_col is None:
            if "claim" in text_low:
                metric_col = "claims_count"
            elif "repair" in text_low:
                metric_col = "repairs_count"
            elif "recall" in text_low:
                metric_col = "recalls_count"
            else:
                metric_col = "failures_count"
        if metric_col == "failures_count" and "failures_count" not in df_history.columns:
            df_history = ensure_failures_column(df_history, out_col="failures_count")
        if metric_col not in df_history.columns:
            return "<p>Requested metric not available to compute monthly aggregates. Try 'claims', 'repairs', or 'failures'.</p>"
        try:
            df = df_history.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            if df_month.empty:
                return "<p>Not enough date information to compute monthly aggregates.</p>"
            monthly_vals = df_month.astype(int)
            first_month = monthly_vals.index[0].strftime("%b %Y")
            last_month = monthly_vals.index[-1].strftime("%b %Y")
            total_over_period = int(monthly_vals.sum())
            avg_per_month = int(round(monthly_vals.mean()))
            min_val = int(monthly_vals.min()); max_val = int(monthly_vals.max())
            min_month = monthly_vals.idxmin().strftime("%b %Y")
            max_month = monthly_vals.idxmax().strftime("%b %Y")
            y = monthly_vals.values.astype(float)
            x = np.arange(len(y))
            if len(y) >= 2:
                slope = float(np.polyfit(x, y, 1)[0])
                trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            else:
                slope = 0.0
                trend = "stable"
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            if "by model" in text_low or "per model" in text_low:
                grp = df.dropna(subset=["date_parsed"]).set_index("date_parsed").groupby([pd.Grouper(freq="M"), "model"])[metric_col].sum()
                if grp.empty:
                    return (f"<p>Overall {label} between {first_month} and {last_month}: total = {total_over_period}, "
                            f"average ≈ {avg_per_month} per month. Monthly {label} ranged between {min_val} – {max_val}. "
                            f"The overall trend is {trend} month-over-month.</p>")
                pivot = grp.unstack(fill_value=0)
                avg_by_model = pivot.mean(axis=0).sort_values(ascending=False).head(6)
                lines = [f"{_html.escape(str(idx))} → {val:.1f} /month" for idx, val in avg_by_model.items()]
                return (f"<p>Overall {label} between {first_month} and {last_month}: total = {total_over_period}, "
                        f"average ≈ {avg_per_month} per month. Monthly {label} ranged between {min_val} – {max_val}. "
                        f"The overall trend is {trend} month-over-month.</p>"
                        f"<p><strong>Average per month by model (top):</strong><br>{'<br>'.join(lines)}</p>")
            return (f"<p>Between {first_month} and {last_month}, monthly {label} totals ranged between {min_val} – {max_val} per month. "
                    f"{max_month} showed the highest activity (≈{max_val}) while {min_month} was the lowest (≈{min_val}). "
                    f"Total over the period = {total_over_period}, average ≈ {avg_per_month} per month. The overall trend is {trend} (slope={slope:.2f} {label}/month).</p>")
        except Exception:
            if sample_df is not None and not sample_df.empty and "date_parsed" in sample_df.columns and sample_df["date_parsed"].notna().any():
                sd_month = sample_df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
                if not sd_month.empty:
                    min_val = int(sd_month.min()); max_val = int(sd_month.max()); total_over_period = int(sd_month.sum())
                    return (f"<p>In the matching sample: total = {total_over_period}, monthly {metric_col} ranged between {min_val} – {max_val}. "
                            f"Try the full 'claims per month' query if you want the complete dataset summary.</p>")
            return "<p>Couldn't compute monthly aggregates — ensure the dataset has a parsable 'date' column.</p>"

    # 2) Rate requests
    if "rate" in text_low or "per 100" in text_low or "per 100 records" in text_low:
        if "by model" in text_low or (_has_word(text_low, "model") and not (_has_word(text_low, "age") or _has_word(text_low, "mileage"))):
            group_col = "model"
        elif "age" in text_low or "age bucket" in text_low:
            group_col = "age_bucket"
        elif "mileage" in text_low or "mileage bucket" in text_low:
            group_col = "mileage_bucket"
        elif "dealer" in text_low or "service center" in text_low:
            if _safe_column(df_history, ["dealer"]):
                group_col = _safe_column(df_history, ["dealer"])
            elif _safe_column(df_history, ["service_center"]):
                group_col = _safe_column(df_history, ["service_center"])
            else:
                group_col = None
        else:
            group_col = _safe_column(df_history, ["model"]) or None

        if not group_col:
            return "<p>Could not determine grouping column for rate. Try: 'failure rate by model' or 'failure rate by age bucket'.</p>"
        if group_col not in df_history.columns:
            return f"<p>The dataset does not contain '{group_col}' column. Available columns: {', '.join(sorted(df_history.columns))}.</p>"
        if metric_col is None:
            return "<p>Requested metric not available to compute rates.</p>"
        try:
            df = df_history.copy()
            overall_total = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            overall_rows = len(df)
            overall_per100 = (overall_total / overall_rows) * 100.0 if overall_rows > 0 else 0.0
            label = "failure rate" if metric_col == "failures_count" else metric_col.replace("_", " ")
            grp = df.groupby(group_col).agg(total_metric=(metric_col, "sum"), rows=(metric_col, "count"))
            grp = grp[grp["rows"] > 0]
            grp["rate_per_100"] = (grp["total_metric"] / grp["rows"]) * 100.0
            top_n = grp.sort_values("rate_per_100", ascending=False).head(6)
            lines = [f"{_html.escape(str(idx))} → {row['rate_per_100']:.1f}%" for idx, row in top_n.iterrows()]
            return (f"<p>Overall {label}: total = {int(overall_total)} across {overall_rows} records (~{overall_per100:.1f} per 100 records)." 
                    f"<br><strong>Breakdown by {group_col}:</strong><br>" + "<br>".join(lines) + "</p>")
        except Exception:
            return "<p>Couldn't compute rates for the requested category (data issue).</p>"

    # 3) Trend-ish questions
    if any(tok in text_low for tok in ["trend", "increasing", "rising", "declining", "decreasing"]):

        # --- Case A: user explicitly names a model (e.g. "why is Sentra trending up?")
        mentioned_model = None
        if "model" in df_history.columns:
            for m in df_history["model"].dropna().unique():
                if str(m).lower() in text_low:
                    mentioned_model = str(m)
                    break

        if mentioned_model:
            # compute trend for that specific model
            try:
                df = df_history.copy()
                df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
                df_model = df[df["model"].str.lower() == mentioned_model.lower()]
                if df_model.empty:
                    return f"<p>No data found for model {mentioned_model}.</p>"

                df_month = df_model.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
                if len(df_month) < 3:
                    return f"<p>Not enough months of data for {mentioned_model} to determine a reliable trend.</p>"

                y = df_month.values.astype(float)
                x = np.arange(len(y))
                slope = np.polyfit(x, y, 1)[0]
                trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")

                first_month = df_month.index[0].strftime("%b %Y")
                last_month = df_month.index[-1].strftime("%b %Y")
                max_month = df_month.idxmax().strftime("%b %Y")
                min_month = df_month.idxmin().strftime("%b %Y")
                max_val = int(df_month.max())
                min_val = int(df_month.min())

                label = metric_col.replace("_", " ")
                direction = "upward" if slope > 0 else ("downward" if slope < 0 else "flat")

                return (f"<p><strong>{mentioned_model}</strong> shows a {trend} trend in {label} "
                        f"from {first_month} to {last_month} (slope={slope:.2f} per month).</p>"
                        f"<p>Monthly {label} ranged from {min_val} in {min_month} to {max_val} in {max_month}.</p>"
                        f"<p>This {direction} trend may indicate either an increase in reported issues or higher detection activity for {mentioned_model} in recent months.</p>")
            except Exception as e:
                return f"<p>Couldn't compute trend for {mentioned_model}: {_html.escape(str(e))}</p>"

        # --- Case B: "Which models are rising?" or "overall trend"
        if ("which" in text_low and "model" in text_low) or ("which models" in text_low):
            if metric_col is None:
                return "<p>Requested metric not available to compute model trends.</p>"
            try:
                df = df_history.copy()
                df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
                overall_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
                y_all = overall_month.values
                slope_all = np.polyfit(np.arange(len(y_all)), y_all, 1)[0] if len(y_all) >= 2 else 0.0
                overall_trend = "increasing" if slope_all > 0 else ("decreasing" if slope_all < 0 else "stable")

                model_trends = _compute_model_trends(df, metric_col, min_months=6, slope_threshold=0.0, top_n=6)
                if not model_trends:
                    return f"<p>Overall trend is {overall_trend}. No models show clear upward movement.</p>"

                lines = [f"{m} → slope ≈ {s:.2f} per month (months: {mon}, last={last:.0f})" for (m, s, mon, last) in model_trends]
                return (f"<p>Overall trend: {overall_trend}.<br><strong>Models with strongest rising trends:</strong><br>"
                        + "<br>".join(lines) + "</p>")
            except Exception:
                return "<p>Couldn't determine per-model trends due to data parsing issue.</p>"

        # --- Case C: generic overall trend
        try:
            df = df_history.copy()
            df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")
            df_month = df.dropna(subset=["date_parsed"]).set_index("date_parsed").resample("M")[metric_col].sum()
            if len(df_month) < 3:
                return "<p>Not enough months to determine a reliable overall trend.</p>"
            y = df_month.values
            x = np.arange(len(y))
            slope = np.polyfit(x, y, 1)[0]
            trend = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")
            label = metric_col.replace("_", " ")
            return f"<p>Overall {label} trend is {trend} (slope={slope:.1f} per month).</p>"
        except Exception:
            return "<p>Couldn't determine trend (data parsing issue).</p>"
        
        
    # 4) Top failed parts
    if re.search(r"top failed parts|top parts|top failure|top failures|most frequent parts", text_low):
        return summarize_top_failed_parts(df_history, metric=requested_metric, top_n=6)

    # 5) Incident / failure details
    if re.search(r"incident details|failure details|time to resolution|resolution time|avg resolution", text_low):
        return summarize_incident_details(df_history, metric=requested_metric)

    # 6) Default: concise sample-based summary (no raw rows)
    try:
        if sample_df is None or sample_df.empty:
            df = df_history.copy()
            if metric_col is None:
                return "<p>Couldn't determine the metric to summarize. Try asking about failures, claims, repairs or recalls.</p>"
            total_rows = len(df)
            total_metric = float(pd.to_numeric(df[metric_col], errors="coerce").fillna(0).sum())
            rate_sample = (total_metric / total_rows * 100.0) if total_rows > 0 else 0.0
            top_model = None
            if "model" in df.columns and not df["model"].isna().all():
                modes = df["model"].mode()
                if len(modes) > 0:
                    top_model = modes.iloc[0]
            top_part_val = None
            top_part_col = _safe_column(df, ["primary_failed_part", "failed_part", "part"])
            if top_part_col:
                try:
                    grp = df.groupby(top_part_col)[metric_col].sum()
                    if not grp.empty:
                        top_part_val = grp.idxmax()
                except Exception:
                    top_part_val = None
            label = "failures" if metric_col == "failures_count" else metric_col.replace("_", " ")
            parts = []
            if top_model:
                parts.append(f"Top model overall: {top_model}.")
            if top_part_val:
                parts.append(f"Top failed part overall: {top_part_val}.")
            parts_txt = " ".join(parts)
            return (f"<p>Dataset overview: total records = {total_rows}, total {label} = {int(total_metric)} "
                    f"(approx. {rate_sample:.1f}% per-record). {parts_txt}</p>")
        else:
            sd = sample_df.copy()
            total_sample = len(sd)
            if metric_col and metric_col in sd.columns:
                total_metric = float(pd.to_numeric(sd.get(metric_col, 0), errors="coerce").fillna(0).sum())
            else:
                total_metric = 0.0
            rate_sample = (total_metric / total_sample * 100.0) if total_sample > 0 else 0.0
            top_model = None
            if "model" in sd.columns and not sd["model"].isna().all():
                modes = sd["model"].mode()
                if len(modes) > 0:
                    top_model = modes.iloc[0]
            top_part = None
            if "primary_failed_part" in sd.columns and not sd["primary_failed_part"].isna().all():
                modes_p = sd["primary_failed_part"].mode()
                if len(modes_p) > 0:
                    top_part = modes_p.iloc[0]
            label = "failures" if metric_col == "failures_count" else (metric_col.replace("_", " ") if metric_col else "metric")
            parts = []
            if top_model:
                parts.append(f"Top model in matches: {top_model}.")
            if top_part:
                parts.append(f"Top failed part in matches: {top_part}.")
            parts_txt = " ".join(parts)
            return (f"<p>From the matching sample: sample rows = {total_sample}, total {label} = {int(total_metric)} "
                    f"(sample metric ~ {rate_sample:.1f}%). {parts_txt}</p>")
    except Exception:
        return "<p>Found matches but could not summarize them (unexpected error).</p>"


