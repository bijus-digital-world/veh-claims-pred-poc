# helper.py
"""
Helper utilities for the Nissan POC Streamlit app.
"""
from typing import List, Dict, Tuple, Optional
import math
import os
import csv
from datetime import datetime
import pandas as pd
import joblib
import random
import io
import re
import json
import html as _html
import base64
import boto3
from botocore.exceptions import ClientError

# Nissan Brand Color Palette (kept for consistency)
NISSAN_RED = "#c3002f"
NISSAN_DARK = "#0b0f13"
NISSAN_GRAY = "#94a3b8"
NISSAN_GOLD = "#f59e0b"

# --- stacking palette + plot helper (add to helper.py) ---
# STACK_COLORS = {
#     "claims": "#D24D57",   # muted crimson (claims)
#     "repairs": "#E3B341",  # warm amber/gold (repairs)
#     "recalls": "#4DB6AC",  # teal blue-green (recalls)
# }

# Optional s3fs (not required)
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None  # s3fs may not be available in all environments

MODELS: List[str] = ["Leaf", "Ariya", "Sentra"]
MILEAGE_BUCKETS: List[str] = ["0-10k", "10-30k", "30-60k", "60k+"]
AGE_BUCKETS: List[str] = ["<1yr", "1-3yr", "3-5yr", "5+yr"]
NISSAN_HEATMAP_SCALE = ["#ffffff", "#ffecec", "#ffd1d1", "#ff9b9b", "#ff7070", "#c3002f", "#7a0000", "#000000"]

def load_svg_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_history_data(use_s3: bool = True,
                      s3_bucket: str = "veh-poc-207567760844-us-east-1",
                      s3_key: str = "data/vehicle_claims.csv",
                      local_path: str = "data/vehicle_claims.csv") -> pd.DataFrame:
    """
    Try S3 first (if use_s3 True and s3fs supported), otherwise fallback to local file.
    """
    df = None
    if use_s3 and s3fs is not None:
        s3_path = f"s3://{s3_bucket}/{s3_key}"
        try:
            # storage_options may be required in some environments; remove if you get auth issues
            df = pd.read_csv(s3_path, parse_dates=["date"], storage_options={"anon": False})
            return df
        except Exception:
            # fall back to local
            pass
    if df is None:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Could not find data at S3 or local path ({local_path}).")
        df = pd.read_csv(local_path, parse_dates=["date"])
    return df

def compute_rate_per_100(df: pd.DataFrame, group_cols):
    grouped = df.groupby(group_cols).agg(incidents=("claims_count", "size"),
                                         claims=("claims_count", "sum")).reset_index()
    grouped["rate_per_100"] = grouped.apply(
        lambda r: (r["claims"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0,
        axis=1,
    )
    return grouped

def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception:
        return None

DUP_TOL = 1e-3

def _last_log_row(filepath: str) -> Optional[Dict]:
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            return rows[-1] if rows else None
    except Exception:
        return None

def append_inference_log(inf_row: dict, pred_prob: float, filepath: str = "inference_log.csv") -> bool:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": ts,
        "model": inf_row.get("model"),
        "primary_failed_part": inf_row.get("primary_failed_part"),
        "mileage_bucket": inf_row.get("mileage_bucket"),
        "age_bucket": inf_row.get("age_bucket"),
        "pred_prob": float(round(pred_prob, 6)),
        "pred_prob_pct": float(round(pred_prob * 100.0, 4)),
    }
    last = _last_log_row(filepath)
    if last:
        try:
            same_inputs = (
                str(last.get("model")) == row["model"] and
                str(last.get("primary_failed_part")) == row["primary_failed_part"] and
                str(last.get("mileage_bucket")) == row["mileage_bucket"] and
                str(last.get("age_bucket")) == row["age_bucket"]
            )
            last_prob = float(last.get("pred_prob", 0.0))
            if same_inputs and abs(last_prob - row["pred_prob"]) <= DUP_TOL:
                return False
        except Exception:
            pass
    write_header = not os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return True

def append_inference_log_s3(inf_row: dict, pred_prob: float,
                            s3_bucket: str = "veh-poc-207567760844-us-east-1",
                            s3_key: str = "logs/inference_log.csv",
                            local_fallback: str = "inference_log.csv") -> bool:
    """
    Attempt to append to a CSV stored in S3 using s3fs. If not available or any error occurs,
    fallback to local append.
    """
    if s3fs is None:
        return append_inference_log(inf_row, pred_prob, filepath=local_fallback)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": inf_row.get("model"),
        "primary_failed_part": inf_row.get("primary_failed_part"),
        "mileage_bucket": inf_row.get("mileage_bucket"),
        "age_bucket": inf_row.get("age_bucket"),
        "pred_prob": float(round(pred_prob, 6)),
        "pred_prob_pct": float(round(pred_prob * 100.0, 4)),
    }

    try:
        fs = s3fs.S3FileSystem()
        s3_path = f"{s3_bucket}/{s3_key}"  # do not include s3:// prefix for s3fs.exists
        if fs.exists(s3_path):
            # open in binary and let pandas read it
            with fs.open(s3_path, "rb") as f:
                df = pd.read_csv(f, parse_dates=["timestamp"])
        else:
            df = pd.DataFrame(columns=list(row.keys()))

        if not df.empty:
            last = df.iloc[-1].to_dict()
            same_inputs = (
                str(last.get("model")) == row["model"] and
                str(last.get("primary_failed_part")) == row["primary_failed_part"] and
                str(last.get("mileage_bucket")) == row["mileage_bucket"] and
                str(last.get("age_bucket")) == row["age_bucket"]
            )
            last_prob = float(last.get("pred_prob", 0.0))
            if same_inputs and abs(last_prob - row["pred_prob"]) <= DUP_TOL:
                return False
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        csv_buf = df.to_csv(index=False).encode("utf-8")
        with fs.open(s3_path, "wb") as f:
            f.write(csv_buf)
        return True
    except Exception:
        # fallback to local append if anything goes wrong
        return append_inference_log(inf_row, pred_prob, filepath=local_fallback)

def random_inference_row_from_df(df: pd.DataFrame) -> dict:
    """
    Create a simple POC inference row using observed categorical values in df.
    Produces randomized coordinates across North America (mix of major city seeds +
    uniform continental sampling) so UI/map shows varied locations.
    """
    model = random.choice(df["model"].unique().tolist())
    part = random.choice(df["primary_failed_part"].unique().tolist())
    mileage = random.choice(MILEAGE_BUCKETS)
    age = random.choice(AGE_BUCKETS)

    # Major North American cities (lat, lon) sample pool for more realistic points
    major_cities = [
        {"name": "New York, NY, USA", "lat": 40.7128, "lon": -74.0060},
        {"name": "Los Angeles, CA, USA", "lat": 34.0522, "lon": -118.2437},
        {"name": "Chicago, IL, USA", "lat": 41.8781, "lon": -87.6298},
        {"name": "Houston, TX, USA", "lat": 29.7604, "lon": -95.3698},
        {"name": "Phoenix, AZ, USA", "lat": 33.4484, "lon": -112.0740},
        {"name": "Philadelphia, PA, USA", "lat": 39.9526, "lon": -75.1652},
        {"name": "San Antonio, TX, USA", "lat": 29.4241, "lon": -98.4936},
        {"name": "San Diego, CA, USA", "lat": 32.7157, "lon": -117.1611},
        {"name": "Dallas, TX, USA", "lat": 32.7767, "lon": -96.7970},
        {"name": "San Jose, CA, USA", "lat": 37.3382, "lon": -121.8863},
        {"name": "Austin, TX, USA", "lat": 30.2672, "lon": -97.7431},
        {"name": "Jacksonville, FL, USA", "lat": 30.3322, "lon": -81.6557},
        {"name": "Toronto, ON, Canada", "lat": 43.6532, "lon": -79.3832},
        {"name": "Montreal, QC, Canada", "lat": 45.5017, "lon": -73.5673},
        {"name": "Vancouver, BC, Canada", "lat": 49.2827, "lon": -123.1207},
        {"name": "Calgary, AB, Canada", "lat": 51.0447, "lon": -114.0719},
        {"name": "Mexico City, CDMX, Mexico", "lat": 19.4326, "lon": -99.1332},
        {"name": "Guadalajara, MX", "lat": 20.6597, "lon": -103.3496},
        {"name": "Monterrey, MX", "lat": 25.6866, "lon": -100.3161},
        {"name": "Seattle, WA, USA", "lat": 47.6062, "lon": -122.3321},
        {"name": "Denver, CO, USA", "lat": 39.7392, "lon": -104.9903},
        {"name": "Miami, FL, USA", "lat": 25.7617, "lon": -80.1918},
        {"name": "Boston, MA, USA", "lat": 42.3601, "lon": -71.0589},
        {"name": "Atlanta, GA, USA", "lat": 33.7490, "lon": -84.3880}
    ]

    # Strategy:
    # - with 70% prob pick a real city (so markers cluster at real places)
    # - with 30% prob pick a random point within North America bounding box
    # North America bounding box (approx continental) - lat range covers Mexico to Canada
    NA_LAT_MIN = 15.0    # southern Mexico
    NA_LAT_MAX = 60.0    # southern Canada / Alaska southern boundary
    NA_LON_MIN = -130.0
    NA_LON_MAX = -60.0

    pick_city = random.random() < 0.7
    if pick_city:
        city = random.choice(major_cities)
        lat = float(city["lat"])
        lon = float(city["lon"])
    else:
        lat = random.uniform(NA_LAT_MIN, NA_LAT_MAX)
        lon = random.uniform(NA_LON_MIN, NA_LON_MAX)

    # Return inference row including coordinates
    return {
        "model": model,
        "primary_failed_part": part,
        "mileage_bucket": mileage,
        "age_bucket": age,
        "lat": lat,
        "lon": lon
    }

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bbox_for_radius(lat: float, lon: float, radius_km: float) -> Tuple[float, float, float, float]:
    """
    Returns (min_lat, min_lon, max_lat, max_lon) approximate bounding box for radius_km around lat/lon.
    Useful if you want to restrict queries or perform a first-pass filter.
    """
    # approximate using degrees: 1 deg lat ~= 111 km; lon scaled by cos(lat)
    delta_lat = radius_km / 111.0
    # avoid division by zero at poles
    cos_lat = max(0.0001, math.cos(math.radians(lat)))
    delta_lon = radius_km / (111.320 * cos_lat)
    return (lat - delta_lat, lon - delta_lon, lat + delta_lat, lon + delta_lon)

def find_nearest_dealers(current_lat: float, current_lon: float, dealers: List[Dict], top_n: int = 3) -> List[Dict]:
    rows = []
    for d in dealers:
        if d.get("lat") is None or d.get("lon") is None:
            continue
        dist = haversine_distance_km(current_lat, current_lon, d["lat"], d["lon"])
        eta_minutes = int((dist / 40.0) * 60.0)  # assume average speed 40 km/h
        rows.append({**d, "distance_km": round(dist, 2), "eta_min": max(5, eta_minutes)})
    rows_sorted = sorted(rows, key=lambda r: r["distance_km"])
    return rows_sorted[:top_n]

def estimate_repair_cost_range(model: str, part: str) -> Tuple[int, int]:
    base_by_model = {"Leaf": 100, "Ariya": 150, "Sentra": 120}
    part_factor = {
        "battery": (2000, 5000),
        "brake": (200, 600),
        "suspension": (500, 1200),
        "transmission": (1500, 4000),
        "sensor": (100, 400),
    }
    base = base_by_model.get(model, 120)
    for k, rng in part_factor.items():
        if k in part.lower():
            low, high = rng
            return int(low * 0.9), int(high * 1.1)
    return int(100 + base), int(800 + base)

# --- Reverse geocode current location to get a place name (AWS Location Service) ---
def reverse_geocode(lat, lon, place_index="NissanPlaceIndex", region="us-east-1"):
    try:
        client = boto3.client("location", region_name=region)
        resp = client.search_place_index_for_position(
            IndexName=place_index,
            Position=[lon, lat],
            MaxResults=1
        )
        results = resp.get("Results", [])
        if results and "Place" in results[0]:
            return results[0]["Place"].get("Label", "Unknown Location")
    except Exception as e:
        print("Debug: reverse geocode failed:", e)
    return "Unknown Location"


def fetch_dealers_from_aws_location(current_lat: float, current_lon: float,
                                    place_index_name: Optional[str] = None,
                                    text_query: str = "Nissan Service Center",
                                    max_results: int = 10,
                                    region_name: str = "us-east-1",
                                    filter_countries: Optional[List[str]] = None,
                                    debug: bool = False) -> List[Dict]:
    """
    Query AWS Location Service Place Index for nearby Nissan service centers.
    Returns list of dicts with keys: name, lat, lon, raw (original item).
    On any error or missing config, returns empty list.

    Important:
      - AWS expects BiasPosition as [lon, lat]
      - FilterCountries is ISO-3166 alpha-3 codes, e.g. ["USA"].
    """
    if place_index_name is None:
        return []

    fc = filter_countries or ["USA"]
    try:
        client = boto3.client("location", region_name=region_name)
        # bias position expects [lon, lat] per API
        bias_pos = [current_lon, current_lat]
        if debug:
            print(f"Debug: fetch_dealers_from_aws_location Index={place_index_name} Text={text_query} Bias={bias_pos} Filter={fc}")

        resp = client.search_place_index_for_text(
            IndexName=place_index_name,
            Text=text_query,
            BiasPosition=bias_pos,
            FilterCountries=fc,
            MaxResults=max_results,
        )

        results = []
        for item in resp.get("Results", []):
            place = item.get("Place", {})
            label = place.get("Label", "Nissan Dealer")
            geometry = place.get("Geometry", {})
            # AWS returns Geometry.Point as [lon, lat]
            pt = geometry.get("Point") or []
            if len(pt) >= 2:
                lon, lat = float(pt[0]), float(pt[1])
                results.append({"name": label, "lat": lat, "lon": lon, "raw": place})
        if debug:
            print(f"Debug: fetched {len(results)} dealers from AWS Location (query={text_query})")
        return results
    except ClientError as e:
        if debug:
            print(f"ClientError calling AWS Location: {e}")
        return []
    except Exception as e:
        if debug:
            print(f"Error calling AWS Location: {e}")
        return []

def robust_fetch_dealers(current_lat: float,
                         current_lon: float,
                         place_index_name: Optional[str],
                         aws_region: str = "us-east-1",
                         radius_km: float = 32.19,  # default 20 miles
                         max_results_per_query: int = 3,
                         top_n: int = 3,
                         filter_countries: Optional[List[str]] = None,
                         debug: bool = False) -> Tuple[List[Dict], bool]:
    """
    Robust multi-stage fetch:
      - Tries multiple text queries in order (brand-specific -> broader)
      - Biases by coordinates
      - Requests candidates and applies client-side haversine filtering to ensure results are within radius_km
      - Deduplicates by rounded coordinates
      - If none within radius, returns the nearest remote matches (sorted by computed distance)
    Returns: (list_of_dealer_dicts, from_aws_flag)
    Each dealer dict will include 'name','lat','lon','raw' (if available), 'distance_km', 'eta_min', 'source_query'
    """
    from_aws = False
    if not place_index_name:
        return [], False

    # Prepare candidate queries (specific -> broader). Use reverse geocode city when available.
    place_label = reverse_geocode(current_lat, current_lon, place_index=place_index_name, region=aws_region)
    city = (place_label.split(",")[0].strip()) if place_label and "," in place_label else (place_label if place_label else None)

    queries = [
        "Nissan Service Center",
        "Nissan Dealer",
        f"Nissan Service Center near {city}" if city else None,
        "Nissan Authorized Service Center",
        "Nissan Service & Parts",
        "Nissan maintenance",
        "Nissan repair shop",
    ]
    queries = [q for q in queries if q]

    seen_keys = set()
    nearby_results: List[Dict] = []

    # Try queries in order, stop when we have top_n dealers within radius
    for q in queries:
        try:
            aws_items = fetch_dealers_from_aws_location(
                current_lat=current_lat,
                current_lon=current_lon,
                place_index_name=place_index_name,
                text_query=q,
                max_results=max_results_per_query,
                region_name=aws_region,
                filter_countries=filter_countries,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print("fetch_dealers_from_aws_location failed for query", q, e)
            aws_items = []

        if not aws_items:
            continue

        # mark that we successfully called AWS at least once
        from_aws = True

        # process items: compute distance, dedupe, and keep within radius
        for it in aws_items:
            # key by rounded coords (6 decimals) or fallback to name
            key = None
            try:
                key = f"{round(float(it['lat']),6)}_{round(float(it['lon']),6)}"
            except Exception:
                key = it.get("name", "").lower().strip()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            dist_km = haversine_distance_km(current_lat, current_lon, it["lat"], it["lon"])
            eta_minutes = int((dist_km / 40.0) * 60.0)
            it_enriched = {
                **it,
                "distance_km_remote": it.get("distance_km"),  # if AWS provided any distance metadata
                "distance_km": round(dist_km, 2),
                "eta_min": max(5, eta_minutes),
                "source_query": q
            }
            if dist_km <= radius_km:
                nearby_results.append(it_enriched)

        if len(nearby_results) >= top_n:
            # enough nearby dealers found
            break

    # if we have nearby results, sort & return top_n
    if nearby_results:
        nearby_results = sorted(nearby_results, key=lambda r: r["distance_km"])
        return nearby_results[:top_n], from_aws

    # If none found within radius_km, as a last resort collect a larger superset and return nearest matches
    all_candidates = []
    for q in queries:
        try:
            aws_items = fetch_dealers_from_aws_location(
                current_lat=current_lat,
                current_lon=current_lon,
                place_index_name=place_index_name,
                text_query=q,
                max_results=max_results_per_query,  # fetch more to have a better chance of finding nearest
                region_name=aws_region,
                filter_countries=filter_countries,
                debug=debug,
            )
        except Exception as e:
            if debug:
                print("fetch_dealers_from_aws_location failed for query (second pass)", q, e)
            aws_items = []

        for it in aws_items:
            key = None
            try:
                key = f"{round(float(it['lat']),6)}_{round(float(it['lon']),6)}"
            except Exception:
                key = it.get("name", "").lower().strip()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            dist_km = haversine_distance_km(current_lat, current_lon, it["lat"], it["lon"])
            eta_minutes = int((dist_km / 40.0) * 60.0)
            it_enriched = {
                **it,
                "distance_km_remote": it.get("distance_km"),
                "distance_km": round(dist_km, 2),
                "eta_min": max(5, eta_minutes),
                "source_query": q
            }
            all_candidates.append(it_enriched)

    # sort candidates by computed distance and return top_n (may be far away)
    if all_candidates:
        all_candidates = sorted(all_candidates, key=lambda r: r["distance_km"])
        return all_candidates[:top_n], from_aws

    # nothing from AWS at all -> return empty, from_aws flag indicates whether any AWS call succeeded
    return [], from_aws

def fetch_nearest_dealers(current_lat: float,
                          current_lon: float,
                          place_index_name: Optional[str] = None,
                          aws_region: str = "us-east-1",
                          fallback_dealers: Optional[List[Dict]] = None,
                          text_query: str = "Nissan Service Center",
                          top_n: int = 3,
                          filter_countries: Optional[List[str]] = None,
                          debug: bool = False) -> Tuple[List[Dict], bool]:
    """
    Fetch nearest dealers for a coordinate.
    - Uses robust_fetch_dealers to try multiple queries + client-side filtering when place_index_name is provided.
    - If AWS yields nothing, falls back to provided fallback_dealers or a small local set.
    - Returns (nearest_dealers_list, from_aws_flag)
    Each returned dealer dict includes 'name','lat','lon','distance_km','eta_min' and may include 'raw'.
    """
    try:
        if place_index_name:
            # prefer the robust multi-query approach (tries city biasing, several queries, client-side filtering)
            nearest, from_aws = robust_fetch_dealers(
                current_lat=current_lat,
                current_lon=current_lon,
                place_index_name=place_index_name,
                aws_region=aws_region,
                radius_km=32.19,  # 20 miles by default
                max_results_per_query=3,
                top_n=top_n,
                filter_countries=filter_countries,
                debug=debug,
            )
            if nearest:
                return nearest, from_aws
    except Exception as e:
        if debug:
            print("robust_fetch_dealers failed:", e)
        # fall through to fallback behavior

    # Fallback to provided dealers or local list
    dealers = []
    if fallback_dealers:
        dealers = fallback_dealers
    else:
        dealers = [
            {"name": "Nissan Dealer A", "lat": 38.45, "lon": -122.70},
            {"name": "Nissan Dealer B", "lat": 38.43, "lon": -122.72},
            {"name": "Nissan Dealer C", "lat": 38.42, "lon": -122.71},
            {"name": "Nissan Dealer D", "lat": 38.46, "lon": -122.68},
        ]

    nearest = find_nearest_dealers(current_lat=current_lat, current_lon=current_lon, dealers=dealers, top_n=top_n)
    return nearest, False

# ---- Bedrock helper ----
def get_bedrock_summary(model, part, mileage, age, claim_pct,
                       llm_model_id=None, region="us-east-1"):
    """
    Robust Bedrock caller that selects the correct request schema based on the
    exact model_id being called (detects 'titan' vs 'claude'/ 'anthropic'). 
    Output formatting/styling is unchanged (HTML-safe with <strong> labels and gold Recommended action).
    """
    # --- Setup / context (unchanged) ---
    if claim_pct >= 75:
        risk_label = "High risk"
    elif claim_pct >= 40:
        risk_label = "Medium risk"
    else:
        risk_label = "Low risk"
    pct_display = f"{round(claim_pct)}%"

    bedrock = boto3.client(service_name="bedrock-runtime", region_name=region)

    user_prompt = (
        "Produce a concise, polished analyst-facing summary in EXACT format:\n\n"
        f"1) First line must begin with: \"{risk_label} ({pct_display}):\" followed by one explanatory sentence.\n"
        "2) Blank line.\n"
        "3) Then three bullet lines:\n"
        "- Key observations: short facts.\n"
        "- Potential impact: 1–2 sentences.\n"
        "- Recommended action: one step + short rationale.\n\n"
        "Do not include extra commentary. Output only these four sections.\n\n"
        "Context:\n"
        f"- Vehicle Model: {model}\n"
        f"- Failed Part: {part}\n"
        f"- Mileage Bucket: {mileage}\n"
        f"- Age Bucket: {age}\n"
    )

    # Default model ids
    titan_id = "amazon.titan-text-lite-v1"
    claude_id = "anthropic.claude-3-haiku-20240307-v1:0"

    # Decide which model id to call
    model_id_to_call = llm_model_id.strip() if llm_model_id else claude_id

    # Decide type based ON the model id string we will call (guarantees consistency)
    model_id_lower = model_id_to_call.lower()
    is_titan = "titan" in model_id_lower
    is_claude = ("claude" in model_id_lower) or ("anthropic" in model_id_lower)

    # Safety: if caller passed a friendly name 'titan' or 'claude', normalize to canonical ids
    if llm_model_id:
        if llm_model_id.lower() == "titan" or llm_model_id.lower().startswith("titan"):
            model_id_to_call = titan_id
            is_titan = True
            is_claude = False
        elif llm_model_id.lower() == "claude" or "claude" in llm_model_id.lower() or "anthropic" in llm_model_id.lower():
            model_id_to_call = claude_id
            is_claude = True
            is_titan = False

    # Prepare to call Bedrock with the correct schema
    resp_json = None
    last_diag = []

    try:
        if is_claude:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 320,
                "temperature": 0.18,
                "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
            }
            response = bedrock.invoke_model(
                modelId=model_id_to_call,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw = response["body"].read()
            try:
                resp_json = json.loads(raw)
            except Exception:
                resp_json = {"__raw": raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)}

        elif is_titan:
            titan_body_messages = {
                "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
                "textGenerationConfig": {"maxTokenCount": 320, "temperature": 0.2}
            }
            titan_body_inputtext = {
                "inputText": user_prompt,
                "textGenerationConfig": {"maxTokenCount": 320, "temperature": 0.2}
            }

            titan_attempts = [("messages+textGen", titan_body_messages), ("inputText+textGen", titan_body_inputtext)]

            for name, body in titan_attempts:
                try:
                    response = bedrock.invoke_model(
                        modelId=model_id_to_call,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body),
                    )
                    raw = response["body"].read()
                    try:
                        resp_json = json.loads(raw)
                    except Exception:
                        resp_json = {"__raw": raw.decode() if isinstance(raw, (bytes,bytearray)) else str(raw)}
                    last_diag.append({"attempt": name, "body_preview": body, "success": True})
                    break
                except ClientError as e:
                    err_info = e.response.get("Error", {}) if hasattr(e, "response") else {"Message": str(e)}
                    last_diag.append({"attempt": name, "body_preview": body, "success": False,
                                      "error_code": err_info.get("Code"), "error_message": err_info.get("Message")})
                    continue
        else:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 320,
                "temperature": 0.18,
                "messages": [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
            }
            response = bedrock.invoke_model(
                modelId=model_id_to_call,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
            raw = response["body"].read()
            try:
                resp_json = json.loads(raw)
            except Exception:
                resp_json = {"__raw": raw.decode() if isinstance(raw, (bytes,bytearray)) else str(raw)}

    except ClientError as e:
        err_info = e.response.get("Error", {}) if hasattr(e, "response") else {"Message": str(e)}
        print(err_info)
        raise RuntimeError(f"InvokeModel failed for model_id '{model_id_to_call}': {err_info.get('Message') or str(e)}")

    if resp_json is None:
        raise RuntimeError(
            f"No successful response from model_id '{model_id_to_call}'. Attempts diagnostics: {json.dumps(last_diag, indent=2)}"
        )

    # --- Extract assistant text robustly ---
    assistant_text = ""
    # Titan common shape
    results = resp_json.get("results") or resp_json.get("result") or []
    if isinstance(results, list) and results:
        first = results[0]
        assistant_text = (first.get("outputText") or first.get("output_text")
                          or first.get("output") or first.get("text") or "")

    # Messages-style content
    if not assistant_text:
        messages = resp_json.get("messages") or resp_json.get("message") or []
        if isinstance(messages, list):
            for msg in messages:
                if msg.get("role") in ("assistant", "model", "system"):
                    for c in msg.get("content", []):
                        if isinstance(c, dict) and c.get("type") in ("text", "output_text"):
                            assistant_text += c.get("text", "")
                        elif isinstance(c, str):
                            assistant_text += c
                    if assistant_text:
                        break

    # fallback: top-level 'content'
    if not assistant_text and isinstance(resp_json.get("content"), list):
        for item in resp_json["content"]:
            if isinstance(item, dict) and item.get("type") in ("text", "output_text"):
                assistant_text += item.get("text", "")
            elif isinstance(item, str):
                assistant_text += item

    assistant_text = (assistant_text or "").strip()
    if not assistant_text:
        raise RuntimeError(f"No assistant text extracted from Bedrock response. Raw JSON: {resp_json!r}")

    # --- Normalize, escape, and style assistant text (unchanged output) ---
    assistant_text = re.sub(r'\r\n', '\n', assistant_text)
    assistant_text = re.sub(r'\n{3,}', '\n\n', assistant_text).strip()
    assistant_text = _html.escape(assistant_text)

    # Insert <strong> for the risk label
    prefix_pattern = rf"^{re.escape(risk_label)}\s*\(\s*{re.escape(str(round(claim_pct)))}%\s*\)\s*:"
    assistant_text = re.sub(
        prefix_pattern,
        lambda m: f"<strong>{m.group(0)[:-1]}</strong>:",
        assistant_text,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # Bold other labels
    assistant_text = re.sub(r"-\s*Key observations:", "- <strong>Key observations:</strong>", assistant_text, flags=re.IGNORECASE)
    assistant_text = re.sub(r"-\s*Potential impact:", "- <strong>Potential impact:</strong>", assistant_text, flags=re.IGNORECASE)

    # Highlight Recommended action label
    assistant_text = re.sub(
        r"-\s*Recommended action:",
        '- <strong style="color:gold;">Recommended action:</strong>',
        assistant_text,
        flags=re.IGNORECASE
    )

    # Ensure blank line after first explanatory line
    assistant_text = re.sub(r'(\S)\n([^\-])', r'\1\n\n\2', assistant_text, count=1)

    return assistant_text


# def plot_stacked_bars(df_long, x_col, y_col="count", color_col="type",
#                       category_orders=None, height=240, showlegend=True):
#     """
#     df_long: long-form DataFrame with columns [x_col, color_col, y_col, share_pct (optional)]
#     Returns a Plotly Figure styled for the dashboard.
#     """
#     import plotly.express as px

#     # Use the STACK_COLORS dict defined near the top of helper.py
#     fig = px.bar(
#         df_long,
#         x=x_col,
#         y=y_col,
#         color=color_col,
#         category_orders=category_orders or {},
#         color_discrete_map=STACK_COLORS,
#         template="plotly_dark",
#         labels={x_col: x_col.replace("_", " ").title(), y_col: "Count", color_col: "Failure Type"},
#     )

#     # Thin separators, soft opacity so stacks are readable on dark bg
#     fig.update_traces(marker_line_width=0.35, marker_line_color="rgba(255,255,255,0.12)", opacity=0.88)

#     # Legend centered, single-row, and compact
#     fig.update_layout(
#         barmode="stack",
#         bargap=0.36,
#         height=height,
#         # increase top margin a bit so the raised legend has room
#         margin=dict(l=6, r=6, t=32, b=6),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         showlegend=showlegend,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             # raise the legend so it has horizontal room and stays one row
#             y=1.12,
#             xanchor="center",
#             x=0.1,
#             font=dict(size=11, color="#e6eef8"),
#             bgcolor="rgba(0,0,0,0)",
#             traceorder="normal",
#             itemsizing="constant",  # prevents each item from reserving large padding
#             tracegroupgap=0,        # minimize horizontal gap between groups
#             itemwidth=30,           # tighten marker-text spacing (lower => tighter)
#         ),
#     )

#     # Axis fonts to match the design
#     fig.update_xaxes(tickfont=dict(size=11, color="#94a3b8"))
#     fig.update_yaxes(tickfont=dict(size=11, color="#94a3b8"))

#     # hovertemplate: show count and share% if provided
#     hovertemplate = (
#         "<b>%{x}</b><br>"
#         + "%{fullData.name}: %{y}<br>"
#         + "<extra></extra>"
#     )

#     # Apply hovertemplate to each trace and ensure share_pct (if present) is shown
#     for trace in fig.data:
#         trace.hovertemplate = hovertemplate

#     return fig
