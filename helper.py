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
import numpy as np
import joblib
import random
import io
import re
import json
import html as _html
import base64
import boto3
from botocore.exceptions import ClientError
import time

# Import configuration
from config import config

# Logging
from utils.logger import helper_logger as logger, log_function_call

# Nissan Brand Color Palette (imported from config for consistency)
NISSAN_RED = config.colors.nissan_red
NISSAN_DARK = config.colors.nissan_dark
NISSAN_GRAY = config.colors.nissan_gray
NISSAN_GOLD = config.colors.nissan_gold

# Optional s3fs (not required)
try:
    import s3fs  # type: ignore
except Exception:
    s3fs = None  # s3fs may not be available in all environments

# Data constants (imported from config)
MODELS: List[str] = config.data.models
MILEAGE_BUCKETS: List[str] = config.data.mileage_buckets
AGE_BUCKETS: List[str] = config.data.age_buckets
NISSAN_HEATMAP_SCALE = config.colors.heatmap_scale

def load_svg_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def load_history_data(use_s3: Optional[bool] = None,
                      s3_bucket: Optional[str] = None,
                      s3_key: Optional[str] = None,
                      local_path: Optional[str] = None) -> pd.DataFrame:
    """
    Try S3 first (if use_s3 True and s3fs supported), otherwise fallback to local file.
    Uses config defaults when parameters are not provided.
    """
    # Apply config defaults
    use_s3 = use_s3 if use_s3 is not None else config.aws.use_s3
    s3_bucket = s3_bucket or config.aws.s3_bucket
    s3_key = s3_key or config.paths.s3_data_key
    local_path = local_path or config.paths.local_data_file
    
    # Loading historical data
    
    df = None
    if use_s3 and s3fs is not None:
        s3_path = f"s3://{s3_bucket}/{s3_key}"
        try:
            # Attempting to load from S3
            start_time = time.time()
            
            df = pd.read_csv(s3_path, parse_dates=["date"], storage_options={"anon": False})
            
            duration_ms = (time.time() - start_time) * 1000
            # Loaded from S3 successfully
            return df
        except Exception as e:
            logger.warning(f"S3 load failed, falling back to local file: {e}")
            # fall back to local
            pass
    
    if df is None:
        if not os.path.isfile(local_path):
            logger.error(f"Data file not found at {local_path}")
            raise FileNotFoundError(f"Could not find data at S3 or local path ({local_path}).")
        
        # Loading from local file
        start_time = time.time()
        
        df = pd.read_csv(local_path, parse_dates=["date"])
        
        duration_ms = (time.time() - start_time) * 1000
        # Loaded from local file successfully
    
    return df

# def compute_rate_per_100(df: pd.DataFrame, group_cols):
#     grouped = df.groupby(group_cols).agg(incidents=("claims_count", "size"),
#                                          claims=("claims_count", "sum")).reset_index()
#     grouped["rate_per_100"] = grouped.apply(
#         lambda r: (r["claims"] / r["incidents"] * 100.0) if r["incidents"] > 0 else 0.0,
#         axis=1,
#     )
#     return grouped

def load_model(path: str):
    """Load ML model from disk with logging."""
    try:
        # Loading model
        model = joblib.load(path)
        # Model loaded successfully
        return model
    except Exception as e:
        logger.warning(f"Model load failed from {path}: {e}")
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
        "mileage": float(inf_row.get("mileage", 0)),
        "age": float(inf_row.get("age", 0)),
        "pred_prob": float(round(pred_prob, 6)),
        "pred_prob_pct": float(round(pred_prob * 100.0, 4)),
    }
    # Add latitude and longitude if available (for CSV but not displayed in table)
    if "lat" in inf_row and inf_row.get("lat") is not None:
        row["lat"] = float(inf_row.get("lat"))
    if "lon" in inf_row and inf_row.get("lon") is not None:
        row["lon"] = float(inf_row.get("lon"))
    
    # Check for duplicates and append using pandas (handles column alignment automatically)
    if os.path.isfile(filepath):
        try:
            # Read existing file
            df_existing = pd.read_csv(filepath)
            if not df_existing.empty:
                last = df_existing.iloc[-1].to_dict()
                try:
                    same_inputs = (
                        str(last.get("model")) == str(row["model"]) and
                        str(last.get("primary_failed_part")) == str(row["primary_failed_part"]) and
                        abs(float(last.get("mileage", 0)) - row["mileage"]) < 100 and  # Within 100 miles
                        abs(float(last.get("age", 0)) - row["age"]) < 0.1  # Within 0.1 years
                    )
                    last_prob = float(last.get("pred_prob", 0.0))
                    if same_inputs and abs(last_prob - row["pred_prob"]) <= DUP_TOL:
                        return False
                except Exception:
                    pass
            # Append new row using pandas (handles column alignment automatically)
            df_new = pd.DataFrame([row])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(filepath, index=False)
        except Exception as e:
            # Fallback to simple append if pandas fails
            logger.warning(f"Pandas append failed, using CSV writer: {e}")
            # Get existing columns from file first
            existing_columns = []
            try:
                with open(filepath, mode="r", newline="") as f_read:
                    reader = csv.DictReader(f_read)
                    existing_columns = reader.fieldnames or []
            except Exception:
                pass
            # Merge columns: existing first, then new ones
            all_columns = list(existing_columns) + [k for k in row.keys() if k not in existing_columns]
            # Append row (note: this won't update header if new columns were added)
            with open(filepath, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_columns)
                # Only write row, not header (header already exists)
                writer.writerow(row)
    else:
        # New file - write header and row
        with open(filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    return True

def append_inference_log_s3(inf_row: dict, pred_prob: float,
                            s3_bucket: Optional[str] = None,
                            s3_key: Optional[str] = None,
                            local_fallback: Optional[str] = None) -> bool:
    """
    Attempt to append to a CSV stored in S3 using s3fs. If not available or any error occurs,
    fallback to local append. Uses config defaults when parameters are not provided.
    """
    # Apply config defaults
    s3_bucket = s3_bucket or config.aws.s3_bucket
    s3_key = s3_key or config.paths.inference_log_s3_key
    local_fallback = local_fallback or config.paths.inference_log_local
    
    if s3fs is None:
        return append_inference_log(inf_row, pred_prob, filepath=local_fallback)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": inf_row.get("model"),
        "primary_failed_part": inf_row.get("primary_failed_part"),
        "mileage": float(inf_row.get("mileage", 0)),
        "age": float(inf_row.get("age", 0)),
        "pred_prob": float(round(pred_prob, 6)),
        "pred_prob_pct": float(round(pred_prob * 100.0, 4)),
    }
    # Add latitude and longitude if available (for CSV but not displayed in table)
    if "lat" in inf_row and inf_row.get("lat") is not None:
        row["lat"] = float(inf_row.get("lat"))
    if "lon" in inf_row and inf_row.get("lon") is not None:
        row["lon"] = float(inf_row.get("lon"))

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
                abs(float(last.get("mileage", 0)) - row["mileage"]) < 100 and  # Within 100 miles
                abs(float(last.get("age", 0)) - row["age"]) < 0.1  # Within 0.1 years
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
    Generates continuous mileage and age values instead of buckets.
    """
    model = random.choice(df["model"].unique().tolist())
    part = random.choice(df["primary_failed_part"].unique().tolist())
    
    # Generate continuous values using realistic distributions
    # Log-normal for mileage (most vehicles have lower mileage)
    mileage = float(np.clip(np.random.lognormal(mean=10.0, sigma=0.7), 0, 150000))
    mileage = round(mileage, 1)  # Round to 1 decimal place
    
    # Gamma distribution for age (skewed toward newer vehicles)
    age = float(np.clip(np.random.gamma(shape=2.0, scale=2.5), 0, 15))
    age = round(age, 2)  # Round to 2 decimal places

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
        "mileage": mileage,  # Continuous value (miles)
        "age": age,          # Continuous value (years)
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
def reverse_geocode(lat, lon, place_index: Optional[str] = None, region: Optional[str] = None):
    """Reverse geocode coordinates to place name. Uses config defaults when parameters are not provided."""
    # Apply config defaults
    place_index = place_index or config.aws.place_index_name
    region = region or config.aws.region
    
    logger.debug(f"Reverse geocoding coordinates: lat={lat}, lon={lon}")
    
    try:
        client = boto3.client("location", region_name=region)
        resp = client.search_place_index_for_position(
            IndexName=place_index,
            Position=[lon, lat],
            MaxResults=1
        )
        results = resp.get("Results", [])
        if results and "Place" in results[0]:
            place_name = results[0]["Place"].get("Label", "Unknown Location")
            logger.debug(f"Reverse geocode result: {place_name}")
            return place_name
        logger.debug("No results from reverse geocode")
    except Exception as e:
        logger.warning(f"Reverse geocode failed for lat={lat}, lon={lon}: {e}")
    return "Unknown Location"


def fetch_dealers_from_aws_location(current_lat: float, current_lon: float,
                                    place_index_name: Optional[str] = None,
                                    text_query: str = "Nissan Service Center",
                                    max_results: int = 10,
                                    region_name: Optional[str] = None,
                                    filter_countries: Optional[List[str]] = None,
                                    debug: bool = False) -> List[Dict]:
    """
    Query AWS Location Service Place Index for nearby Nissan service centers.
    Returns list of dicts with keys: name, lat, lon, raw (original item).
    On any error or missing config, returns empty list.
    Uses config defaults when parameters are not provided.

    Important:
      - AWS expects BiasPosition as [lon, lat]
      - FilterCountries is ISO-3166 alpha-3 codes, e.g. ["USA"].
    """
    # Apply config defaults
    place_index_name = place_index_name or config.aws.place_index_name
    region_name = region_name or config.aws.region
    
    if place_index_name is None:
        return []

    fc = filter_countries or ["USA"]
    
    logger.debug(f"Fetching dealers from AWS Location: query='{text_query}', bias=[{current_lat}, {current_lon}]")
    
    try:
        client = boto3.client("location", region_name=region_name)
        # bias position expects [lon, lat] per API
        bias_pos = [current_lon, current_lat]
        
        if debug:
            logger.debug(f"AWS Location params: Index={place_index_name}, Text={text_query}, Bias={bias_pos}, Filter={fc}")

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
        
        # Dealers fetched from AWS Location successfully
        return results
        
    except ClientError as e:
        logger.error(f"AWS Location ClientError: {e}", exc_info=debug)
        return []
    except Exception as e:
        logger.error(f"AWS Location error: {e}", exc_info=debug)
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
            logger.debug(f"Dealer fetch failed for query '{q}': {e}")
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
            logger.debug(f"Dealer fetch failed for query '{q}' (second pass): {e}")
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
        logger.warning(f"Robust fetch dealers failed: {e}", exc_info=debug)
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
    # Generating Bedrock summary
    
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
    
    logger.debug(f"Calling Bedrock API with model_id={model_id_to_call}")
    start_time = time.time()

    try:
        if is_claude:
            logger.debug("Using Claude message format")
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
                duration_ms = (time.time() - start_time) * 1000
                # Bedrock API call successful
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
        error_msg = f"InvokeModel failed for model_id '{model_id_to_call}': {err_info.get('Message') or str(e)}"
        logger.error(f"Bedrock API ClientError: {error_msg}", exc_info=True)
        raise RuntimeError(error_msg)

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


def generate_enhanced_prescriptive_summary(model_name: str, part_name: str, mileage_bucket: str, 
                                         age_bucket: str, claim_pct: float, df_history: pd.DataFrame, 
                                         nearest_dealer: Optional[Dict] = None) -> str:
    """
    Generate a dynamic, data-driven prescriptive summary using real data from the enhanced dataset.
    
    Args:
        model_name: Vehicle model
        part_name: Failed part name
        mileage_bucket: Mileage bucket
        age_bucket: Age bucket
        claim_pct: Claim percentage
        df_history: Historical data DataFrame
        nearest_dealer: Nearest dealer information
    
    Returns:
        HTML-formatted prescriptive summary
    """
    try:
        # Filter data for the specific model and part
        model_data = df_history[
            (df_history['model'] == model_name) & 
            (df_history['primary_failed_part'] == part_name)
        ].copy()
        
        if model_data.empty:
            # Fallback to model-only data
            model_data = df_history[df_history['model'] == model_name].copy()
        
        # Check if we have enhanced data columns
        has_enhanced_data = all(col in df_history.columns for col in ['supplier_name', 'failure_description', 'supplier_quality_score', 'defect_rate'])
        
        if not has_enhanced_data:
            # Return a basic enhanced summary without supplier/failure data
            return generate_basic_enhanced_summary(model_name, part_name, claim_pct, nearest_dealer)
        
        # Get trend information
        trend_info = get_model_trend_info(model_data, model_name, part_name)
        
        # Get supplier information
        supplier_info = get_supplier_analysis(model_data, part_name)
        
        # Get failure reasons
        failure_reasons = get_failure_reasons(model_data, part_name)
        
        # Get dealer information
        dealer_info = get_dealer_recommendation(nearest_dealer)
        
        # Generate the summary with varied language patterns
        summary_parts = []
        
        # 1. Opening statement with trend - varied language patterns
        opening_patterns = [
            "Telemetry analysis reveals",
            "Data analysis indicates",
            "Recent diagnostic data shows",
            "Vehicle monitoring systems have detected",
            "Analytics reveal",
            "Diagnostic insights show",
            "Telemetry monitoring has identified",
            "Vehicle data analysis indicates"
        ]
        
        trend_patterns = {
            'rising': [
                "an escalating pattern of",
                "a concerning uptick in",
                "increasing frequency of",
                "a growing trend in",
                "escalating issues with",
                "rising incidents of"
            ],
            'declining': [
                "a decreasing pattern of",
                "reduced frequency of",
                "a declining trend in",
                "diminishing issues with",
                "fewer incidents of"
            ],
            'stable': [
                "consistent patterns of",
                "ongoing issues with",
                "recurring problems with",
                "persistent challenges with"
            ]
        }
        
        opening = random.choice(opening_patterns)
        
        if trend_info['has_trend']:
            trend_strength = trend_info.get('trend_strength', 'stable')
            if trend_strength in ['strong_rising', 'moderate_rising']:
                trend_desc = random.choice(trend_patterns['rising'])
            elif trend_strength in ['strong_declining', 'moderate_declining']:
                trend_desc = random.choice(trend_patterns['declining'])
            else:  # Stable trend
                trend_desc = random.choice(trend_patterns['stable'])
            
            # Add more context about the trend
            avg_monthly = trend_info.get('avg_monthly', 0)
            latest_count = trend_info.get('latest_count', 0)
            
            summary_parts.append(
                f"{opening} {trend_desc} "
                f"<span style='color:#C99700; font-weight:bold;'>{part_name.lower()}</span> failures "
                f"in the <span style='color:#C99700; font-weight:bold;'>{model_name}</span> model "
                f"over the past {trend_info['months']} months (avg: {avg_monthly:.1f}/month, latest: {latest_count})."
            )
        else:
            summary_parts.append(
                f"{opening} {random.choice(trend_patterns['stable'])} "
                f"<span style='color:#C99700; font-weight:bold;'>{part_name.lower()}</span> failures "
                f"in the <span style='color:#C99700; font-weight:bold;'>{model_name}</span> model."
            )
        
        # 2. Dealer recommendation - only for moderate to high claim likelihood
        if dealer_info and claim_pct >= 40:  # Only recommend dealer visits for moderate+ likelihood
            dealer_patterns = [
                f"Given your current location, I recommend visiting the "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"which is the nearest authorized facility.",
                
                f"For immediate service, the closest option is "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"located just {dealer_info['distance']:.1f} km away.",
                
                f"Your nearest service center is "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"approximately {dealer_info['eta']} minutes from your location.",
                
                f"I suggest heading to "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"the closest authorized service center to your position.",
                
                f"The most convenient service location is "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"which can be reached in about {dealer_info['eta']} minutes."
            ]
            
            summary_parts.append(random.choice(dealer_patterns))
        
        # 3. Supplier analysis - varied language and data-driven insights
        if supplier_info['has_supplier_data']:
            if supplier_info['problematic_supplier']:
                supplier_patterns = [
                    f"Supply chain analysis reveals that "
                    f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['problematic_supplier']}</span> "
                    f"has been delivering substandard {part_name.lower()} components, "
                    f"with a {supplier_info['worst_failure_rate']:.1%} failure rate across {supplier_info['worst_volume']} parts supplied.",
                    
                    f"Quality control data shows "
                    f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['problematic_supplier']}</span> "
                    f"consistently producing faulty {part_name.lower()} parts, "
                    f"with {supplier_info['worst_failure_rate']:.1%} failure rate in a sample of {supplier_info['worst_volume']} components.",
                    
                    f"Supplier performance metrics indicate "
                    f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['problematic_supplier']}</span> "
                    f"has been the source of defective {part_name.lower()} components, "
                    f"showing {supplier_info['worst_failure_rate']:.1%} failure rate across {supplier_info['worst_volume']} parts.",
                    
                    f"Component traceability analysis points to "
                    f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['problematic_supplier']}</span> "
                    f"as the primary source of {part_name.lower()} failures, "
                    f"with {supplier_info['worst_failure_rate']:.1%} failure rate in {supplier_info['worst_volume']} supplied parts."
                ]
                
                summary_parts.append(random.choice(supplier_patterns))
                
                if supplier_info['recommended_supplier']:
                    mitigation_patterns = [
                        f"To address this supply chain risk, consider switching to "
                        f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['recommended_supplier']}</span>, "
                        f"which maintains a quality score of {supplier_info['best_quality_score']:.0f}% and could prevent future recalls.",
                        
                        f"Implementing a supplier transition to "
                        f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['recommended_supplier']}</span> "
                        f"(quality score: {supplier_info['best_quality_score']:.0f}%) would significantly reduce failure rates.",
                        
                        f"A strategic supplier change to "
                        f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['recommended_supplier']}</span> "
                        f"is recommended based on their superior quality metrics ({supplier_info['best_quality_score']:.0f}% score).",
                        
                        f"To prevent mass recalls, initiate procurement discussions with "
                        f"<span style='color:#C99700; font-weight:bold;'>{supplier_info['recommended_supplier']}</span>, "
                        f"which has demonstrated consistent quality performance ({supplier_info['best_quality_score']:.0f}% score)."
                    ]
                    
                    summary_parts.append(random.choice(mitigation_patterns))
        
        # 4. Failure reasons - varied language and data-driven insights
        if failure_reasons:
            top_reason = failure_reasons[0]
            
            failure_patterns = [
                f"Root cause analysis identifies <span style='color:#C99700; font-weight:bold;'>"
                f"{top_reason['description']}</span> as the primary failure mode, "
                f"representing {top_reason['percentage']:.1f}% of all incidents.",
                
                f"Diagnostic data reveals that <span style='color:#C99700; font-weight:bold;'>"
                f"{top_reason['description']}</span> is the leading cause of failures, "
                f"occurring in {top_reason['percentage']:.1f}% of cases.",
                
                f"Failure mode analysis shows <span style='color:#C99700; font-weight:bold;'>"
                f"{top_reason['description']}</span> as the most frequent issue, "
                f"accounting for {top_reason['percentage']:.1f}% of all problems.",
                
                f"Component analysis indicates <span style='color:#C99700; font-weight:bold;'>"
                f"{top_reason['description']}</span> is the dominant failure pattern, "
                f"responsible for {top_reason['percentage']:.1f}% of incidents.",
                
                f"Technical investigation reveals <span style='color:#C99700; font-weight:bold;'>"
                f"{top_reason['description']}</span> as the most prevalent issue, "
                f"occurring in {top_reason['percentage']:.1f}% of failure cases."
            ]
            
            summary_parts.append(random.choice(failure_patterns))
            
            # Add additional failure reasons if available
            if len(failure_reasons) > 1:
                second_reason = failure_reasons[1]
                additional_patterns = [
                    f"Additionally, <span style='color:#C99700; font-weight:bold;'>"
                    f"{second_reason['description']}</span> contributes to {second_reason['percentage']:.1f}% of failures.",
                    
                    f"Secondary analysis shows <span style='color:#C99700; font-weight:bold;'>"
                    f"{second_reason['description']}</span> in {second_reason['percentage']:.1f}% of cases.",
                    
                    f"Other contributing factors include <span style='color:#C99700; font-weight:bold;'>"
                    f"{second_reason['description']}</span> ({second_reason['percentage']:.1f}% of incidents)."
                ]
                
                summary_parts.append(random.choice(additional_patterns))
        
        # 5. Risk assessment - varied language and data-driven insights
        # HIGH % = High likelihood of claim = More urgent, technical recommendations
        if claim_pct >= 75:
            high_likelihood_patterns = [
                f"<span style='color:#ef4444; font-weight:bold;'>High claim likelihood detected:</span> "
                f"The {claim_pct:.1f}% claim probability indicates a high likelihood of component failure "
                f"requiring immediate technical intervention and preventive maintenance protocols.",
                
                f"<span style='color:#ef4444; font-weight:bold;'>Urgent technical action required:</span> "
                f"With a {claim_pct:.1f}% claim probability, immediate diagnostic procedures and "
                f"component replacement protocols should be initiated to prevent system failure.",
                
                f"<span style='color:#ef4444; font-weight:bold;'>Critical maintenance alert:</span> "
                f"The {claim_pct:.1f}% claim probability signals imminent component degradation requiring "
                f"immediate technical assessment and corrective action implementation.",
                
                f"<span style='color:#ef4444; font-weight:bold;'>High-priority technical intervention:</span> "
                f"At {claim_pct:.1f}% claim probability, immediate technical evaluation and "
                f"preventive maintenance procedures are essential to avoid component failure."
            ]
            summary_parts.append(random.choice(high_likelihood_patterns))
            
        elif claim_pct >= 40:
            moderate_likelihood_patterns = [
                f"<span style='color:#f59e0b; font-weight:bold;'>Moderate claim likelihood:</span> "
                f"The {claim_pct:.1f}% claim probability suggests increased monitoring and "
                f"proactive maintenance scheduling to prevent potential component issues.",
                
                f"<span style='color:#f59e0b; font-weight:bold;'>Elevated monitoring recommended:</span> "
                f"With a {claim_pct:.1f}% claim probability, implementing enhanced diagnostic protocols "
                f"and scheduled maintenance intervals is recommended to manage component health.",
                
                f"<span style='color:#f59e0b; font-weight:bold;'>Preventive maintenance advised:</span> "
                f"The {claim_pct:.1f}% claim probability indicates a moderate likelihood requiring "
                f"proactive maintenance scheduling and component condition monitoring.",
                
                f"<span style='color:#f59e0b; font-weight:bold;'>Technical monitoring needed:</span> "
                f"At {claim_pct:.1f}% claim probability, implementing preventive maintenance strategies "
                f"and enhanced monitoring protocols is recommended to prevent component degradation."
            ]
            summary_parts.append(random.choice(moderate_likelihood_patterns))
            
        else:
            low_likelihood_patterns = [
                f"<span style='color:#10b981; font-weight:bold;'>Low claim likelihood:</span> "
                f"With a {claim_pct:.1f}% claim probability, the component appears stable and "
                f"requires only routine monitoring and standard maintenance intervals.",
                
                f"<span style='color:#10b981; font-weight:bold;'>Stable component status:</span> "
                f"The {claim_pct:.1f}% claim probability indicates low likelihood of failure, "
                f"suggesting the component is operating within normal parameters.",
                
                f"<span style='color:#10b981; font-weight:bold;'>Minimal intervention needed:</span> "
                f"At {claim_pct:.1f}% claim probability, the component shows stable performance "
                f"requiring only standard maintenance protocols and routine monitoring."
            ]
            summary_parts.append(random.choice(low_likelihood_patterns))
        
        # Add contextual insights based on data patterns
        if len(summary_parts) > 0:
            # Add contextual closing based on claim likelihood and data availability
            # HIGH % = High likelihood of claim = More technical, urgent recommendations
            if claim_pct >= 75:
                closing_patterns = [
                    "Immediate technical escalation to engineering teams and component specialists is essential.",
                    "This situation requires urgent cross-functional collaboration to implement rapid technical interventions.",
                    "Priority should be given to implementing immediate containment measures and comprehensive root cause analysis.",
                    "Urgent coordination with technical teams and component suppliers is critical to prevent system failures."
                ]
            elif claim_pct >= 40:
                closing_patterns = [
                    "Enhanced monitoring protocols and technical trend analysis should be implemented immediately.",
                    "Consider establishing a technical task force to address the underlying component issues.",
                    "Proactive communication with technical stakeholders and maintenance teams is recommended.",
                    "Implement enhanced diagnostic procedures and preventive maintenance schedules to prevent escalation."
                ]
            else:
                closing_patterns = [
                    "Continue standard monitoring protocols and routine maintenance intervals as scheduled.",
                    "Regular technical data review and component monitoring will help maintain current stable performance.",
                    "Standard operational procedures and maintenance schedules should be sufficient for current component status.",
                    "Maintain current quality control measures and periodic technical reviews as part of routine operations."
                ]
            
            # Only add closing if we have substantial content
            if len(summary_parts) >= 3:
                summary_parts.append(random.choice(closing_patterns))
        
        # Combine all parts into meaningful paragraphs
        full_summary = format_summary_into_paragraphs(summary_parts)
        
        return f"""<div style='text-align: justify; line-height: 1.6;'>
            {full_summary}
        </div>"""
        
    except Exception as e:
        logger.error(f"Failed to generate enhanced prescriptive summary: {e}")
        # Fallback to basic summary - only recommend dealer visit if claim likelihood is moderate or high
        if claim_pct >= 40:
            return f"""<div style='text-align: justify; line-height: 1.6;'>
                <p style='margin-bottom: 12px;'>
                    Based on recent telemetry data, I have identified issues with {part_name.lower()} 
                    affecting the {model_name} model.
                </p>
                <p style='margin-bottom: 12px;'>
                    Given your current location, I recommend visiting the nearest authorized service center 
                    for immediate inspection.
                </p>
            </div>"""
        else:
            return f"""<div style='text-align: justify; line-height: 1.6;'>
                <p style='margin-bottom: 12px;'>
                    Based on recent telemetry data, the {part_name.lower()} components 
                    in the {model_name} model are operating within normal parameters.
                </p>
                <p style='margin-bottom: 12px;'>
                    Continue routine monitoring and standard maintenance protocols as scheduled.
                </p>
            </div>"""


def get_model_trend_info(model_data: pd.DataFrame, model_name: str, part_name: str) -> Dict:
    """Get trend information for the model/part combination with more realistic analysis."""
    try:
        if 'date' not in model_data.columns or model_data.empty:
            return {'has_trend': False, 'slope': 0, 'months': 0}
        
        # Convert date and group by month
        model_data['date_parsed'] = pd.to_datetime(model_data['date'], errors='coerce')
        monthly_data = model_data.dropna(subset=['date_parsed']).set_index('date_parsed').resample('M').size()
        
        if len(monthly_data) < 3:
            return {'has_trend': False, 'slope': 0, 'months': len(monthly_data)}
        
        # Calculate trend slope
        x = np.arange(len(monthly_data))
        y = monthly_data.values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        
        # Calculate more realistic trend metrics
        total_incidents = int(monthly_data.sum())
        avg_monthly = float(monthly_data.mean())
        latest_month = int(monthly_data.iloc[-1]) if len(monthly_data) > 0 else 0
        
        # Determine if trend is meaningful (not just noise)
        # A slope is meaningful if it represents at least 20% change over the period
        meaningful_change = abs(slope) > (avg_monthly * 0.2)
        
        # Calculate trend direction and strength
        if slope > avg_monthly * 0.3:  # Strong rising trend
            trend_strength = 'strong_rising'
        elif slope > avg_monthly * 0.1:  # Moderate rising trend
            trend_strength = 'moderate_rising'
        elif slope < -avg_monthly * 0.3:  # Strong declining trend
            trend_strength = 'strong_declining'
        elif slope < -avg_monthly * 0.1:  # Moderate declining trend
            trend_strength = 'moderate_declining'
        else:  # Stable trend
            trend_strength = 'stable'
        
        return {
            'has_trend': meaningful_change,
            'slope': slope,
            'months': len(monthly_data),
            'latest_count': latest_month,
            'total_incidents': total_incidents,
            'avg_monthly': avg_monthly,
            'trend_strength': trend_strength
        }
    except Exception:
        return {'has_trend': False, 'slope': 0, 'months': 0}


def get_supplier_analysis(model_data: pd.DataFrame, part_name: str) -> Dict:
    """Get supplier analysis for the part with more realistic volume-weighted analysis."""
    try:
        if 'supplier_name' not in model_data.columns or model_data.empty:
            return {'has_supplier_data': False}
        
        # Get supplier performance for this part with volume weighting
        supplier_stats = model_data.groupby('supplier_name').agg({
            'supplier_quality_score': 'first',
            'defect_rate': 'first',
            'claims_count': 'sum',
            'repairs_count': 'sum',
            'recalls_count': 'sum'
        }).reset_index()
        
        if supplier_stats.empty:
            return {'has_supplier_data': False}
        
        # Calculate total failures and volume-weighted metrics
        supplier_stats['total_failures'] = supplier_stats['claims_count'] + supplier_stats['repairs_count'] + supplier_stats['recalls_count']
        supplier_stats['total_records'] = model_data.groupby('supplier_name').size().reset_index(name='record_count')['record_count']
        supplier_stats['failure_rate'] = supplier_stats['total_failures'] / supplier_stats['total_records']
        
        # Filter suppliers with meaningful volume (at least 5 records)
        meaningful_suppliers = supplier_stats[supplier_stats['total_records'] >= 5]
        
        if meaningful_suppliers.empty:
            return {'has_supplier_data': False}
        
        # Find problematic supplier (highest failure rate among suppliers with meaningful volume)
        worst_supplier = meaningful_suppliers.loc[meaningful_suppliers['failure_rate'].idxmax()]
        best_supplier = meaningful_suppliers.loc[meaningful_suppliers['supplier_quality_score'].idxmax()]
        
        # Only flag as problematic if they have both high failure rate AND meaningful volume
        is_problematic = (worst_supplier['failure_rate'] > 0.1 and 
                         worst_supplier['total_records'] >= 10 and 
                         worst_supplier['total_failures'] >= 3)
        
        return {
            'has_supplier_data': True,
            'problematic_supplier': worst_supplier['supplier_name'] if is_problematic else None,
            'recommended_supplier': best_supplier['supplier_name'] if best_supplier['supplier_quality_score'] > 85 else None,
            'worst_defect_rate': float(worst_supplier['defect_rate']),
            'worst_failure_rate': float(worst_supplier['failure_rate']),
            'worst_volume': int(worst_supplier['total_records']),
            'best_quality_score': float(best_supplier['supplier_quality_score'])
        }
    except Exception:
        return {'has_supplier_data': False}


def get_failure_reasons(model_data: pd.DataFrame, part_name: str) -> List[Dict]:
    """Get failure reasons for the part."""
    try:
        if 'failure_description' not in model_data.columns or model_data.empty:
            return []
        
        # Filter to only records with actual failures (claims, repairs, or recalls)
        failure_data = model_data[
            (model_data.get('claims_count', 0) > 0) | 
            (model_data.get('repairs_count', 0) > 0) | 
            (model_data.get('recalls_count', 0) > 0)
        ]
        
        if failure_data.empty:
            return []
        
        # Count failure descriptions from actual failure records only
        failure_counts = failure_data['failure_description'].value_counts()
        total_failures = len(failure_data)
        
        reasons = []
        for desc, count in failure_counts.head(3).items():
            if pd.notna(desc) and desc.strip() and desc != "No failure detected":
                reasons.append({
                    'description': desc,
                    'count': int(count),
                    'percentage': (count / total_failures) * 100
                })
        
        return reasons
    except Exception:
        return []


def format_summary_into_paragraphs(summary_parts: List[str]) -> str:
    """
    Format summary parts into meaningful paragraphs for better readability.
    
    Args:
        summary_parts: List of summary text parts
    
    Returns:
        HTML-formatted summary with proper paragraph breaks
    """
    if not summary_parts:
        return ""
    
    # Group parts into logical paragraphs
    paragraphs = []
    current_paragraph = []
    
    for i, part in enumerate(summary_parts):
        current_paragraph.append(part)
        
        # Determine if we should start a new paragraph
        should_break = False
        
        # Break after opening statement (first part)
        if i == 0:
            should_break = True
        # Break before risk assessment (usually contains color styling)
        elif '<span style=' in part and ('risk' in part.lower() or 'alert' in part.lower() or 'action' in part.lower()):
            should_break = True
        # Break before closing statements (usually the last part)
        elif i == len(summary_parts) - 1 and len(current_paragraph) > 1:
            should_break = True
        # Break after dealer recommendation if we have multiple parts
        elif 'dealer' in part.lower() or 'service center' in part.lower():
            if len(summary_parts) > 3:  # Only break if we have enough content
                should_break = True
        # Break after supplier analysis if we have enough content
        elif 'supplier' in part.lower() or 'quality' in part.lower():
            if len(summary_parts) > 4:
                should_break = True
        
        if should_break and current_paragraph:
            # Join current paragraph and add to paragraphs list
            paragraph_text = " ".join(current_paragraph)
            paragraphs.append(f"<p style='margin-bottom: 12px;'>{paragraph_text}</p>")
            current_paragraph = []
    
    # Add any remaining parts as the last paragraph
    if current_paragraph:
        paragraph_text = " ".join(current_paragraph)
        paragraphs.append(f"<p style='margin-bottom: 12px;'>{paragraph_text}</p>")
    
    return "".join(paragraphs)


def get_dealer_recommendation(nearest_dealer: Optional[Dict]) -> Optional[Dict]:
    """Get dealer recommendation information."""
    if not nearest_dealer:
        return None
    
    # Handle case where nearest_dealer might be a string (fallback)
    if isinstance(nearest_dealer, str):
        return {
            'name': nearest_dealer,
            'distance': 0,
            'eta': 0
        }
    
    return {
        'name': nearest_dealer.get('name', 'Nearest Service Center'),
        'distance': nearest_dealer.get('distance_km', 0),
        'eta': nearest_dealer.get('eta_min', 0)
    }


def generate_basic_enhanced_summary(model_name: str, part_name: str, claim_pct: float, nearest_dealer: Optional[Dict] = None) -> str:
    """
    Generate a basic enhanced summary when full enhanced data is not available.
    This provides a more realistic summary than the simple fallback but without supplier/failure data.
    """
    try:
        # Get dealer information
        dealer_info = get_dealer_recommendation(nearest_dealer)
        
        # Generate varied summary parts
        summary_parts = []
        
        # 1. Opening statement - varied patterns
        opening_patterns = [
            "Telemetry analysis reveals",
            "Data analysis indicates", 
            "Recent diagnostic data shows",
            "Vehicle monitoring systems have detected",
            "Analytics reveal",
            "Diagnostic insights show"
        ]
        
        likelihood_patterns = {
            'high': [
                "high likelihood of failure in",
                "imminent component issues affecting",
                "urgent technical concerns regarding",
                "critical component degradation in"
            ],
            'moderate': [
                "moderate likelihood of issues in",
                "concerning component patterns with",
                "increased failure probability in",
                "elevated component risk affecting"
            ],
            'low': [
                "low likelihood of issues in",
                "stable component performance in",
                "minimal failure probability for",
                "reliable component operation in"
            ]
        }
        
        opening = random.choice(opening_patterns)
        
        if claim_pct >= 75:
            likelihood_desc = random.choice(likelihood_patterns['high'])
        elif claim_pct >= 40:
            likelihood_desc = random.choice(likelihood_patterns['moderate'])
        else:
            likelihood_desc = random.choice(likelihood_patterns['low'])
        
        summary_parts.append(
            f"{opening} {likelihood_desc} "
            f"<span style='color:#C99700; font-weight:bold;'>{part_name.lower()}</span> components "
            f"in the <span style='color:#C99700; font-weight:bold;'>{model_name}</span> model."
        )
        
        # 2. Dealer recommendation - only for moderate to high claim likelihood
        if dealer_info and claim_pct >= 40:  # Only recommend dealer visits for moderate+ likelihood
            dealer_patterns = [
                f"For immediate service, visit "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"located {dealer_info['distance']:.1f} km away.",
                
                f"Your nearest service center is "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"approximately {dealer_info['eta']} minutes from your location.",
                
                f"I recommend heading to "
                f"<span style='color:#C99700; font-weight:bold;'>{dealer_info['name']}</span>, "
                f"the closest authorized service center."
            ]
            summary_parts.append(random.choice(dealer_patterns))
        
        # 3. Risk assessment with varied language
        # HIGH % = High likelihood of claim = More urgent, technical recommendations
        if claim_pct >= 75:
            high_likelihood_patterns = [
                f"<span style='color:#ef4444; font-weight:bold;'>High claim likelihood detected:</span> "
                f"The {claim_pct:.1f}% claim probability indicates a high likelihood of component failure "
                f"requiring immediate technical intervention and preventive maintenance protocols.",
                
                f"<span style='color:#ef4444; font-weight:bold;'>Urgent technical action required:</span> "
                f"With a {claim_pct:.1f}% claim probability, immediate diagnostic procedures and "
                f"component replacement protocols should be initiated to prevent system failure."
            ]
            summary_parts.append(random.choice(high_likelihood_patterns))
            
        elif claim_pct >= 40:
            moderate_likelihood_patterns = [
                f"<span style='color:#f59e0b; font-weight:bold;'>Moderate claim likelihood:</span> "
                f"The {claim_pct:.1f}% claim probability suggests increased monitoring and "
                f"proactive maintenance scheduling to prevent potential component issues.",
                
                f"<span style='color:#f59e0b; font-weight:bold;'>Elevated monitoring recommended:</span> "
                f"With a {claim_pct:.1f}% claim probability, implementing enhanced diagnostic protocols "
                f"and scheduled maintenance intervals is recommended to manage component health."
            ]
            summary_parts.append(random.choice(moderate_likelihood_patterns))
            
        else:
            low_likelihood_patterns = [
                f"<span style='color:#10b981; font-weight:bold;'>Low claim likelihood:</span> "
                f"With a {claim_pct:.1f}% claim probability, the component appears stable and "
                f"requires only routine monitoring and standard maintenance intervals.",
                
                f"<span style='color:#10b981; font-weight:bold;'>Stable component status:</span> "
                f"The {claim_pct:.1f}% claim probability indicates low likelihood of failure, "
                f"suggesting the component is operating within normal parameters."
            ]
            summary_parts.append(random.choice(low_likelihood_patterns))
        
        # 4. Contextual closing
        # HIGH % = High likelihood of claim = More technical, urgent recommendations
        if claim_pct >= 75:
            closing_patterns = [
                "Immediate technical escalation to engineering teams and component specialists is essential.",
                "This situation requires urgent cross-functional collaboration to implement rapid technical interventions.",
                "Priority should be given to implementing immediate containment measures and comprehensive root cause analysis."
            ]
        elif claim_pct >= 40:
            closing_patterns = [
                "Enhanced monitoring protocols and technical trend analysis should be implemented immediately.",
                "Consider establishing a technical task force to address the underlying component issues.",
                "Proactive communication with technical stakeholders and maintenance teams is recommended."
            ]
        else:
            closing_patterns = [
                "Continue standard monitoring protocols and routine maintenance intervals as scheduled.",
                "Regular technical data review and component monitoring will help maintain current stable performance.",
                "Standard operational procedures and maintenance schedules should be sufficient for current component status."
            ]
        
        summary_parts.append(random.choice(closing_patterns))
        
        # Combine all parts into meaningful paragraphs
        full_summary = format_summary_into_paragraphs(summary_parts)
        
        return f"""<div style='text-align: justify; line-height: 1.6;'>
            {full_summary}
        </div>"""
        
    except Exception as e:
        logger.error(f"Failed to generate basic enhanced summary: {e}")
        # Final fallback - only recommend dealer visit if claim likelihood is moderate or high
        if claim_pct >= 40:
            return f"""<div style='text-align: justify; line-height: 1.6;'>
                <p style='margin-bottom: 12px;'>
                    Based on recent telemetry data, I have identified issues with {part_name.lower()} 
                    affecting the {model_name} model.
                </p>
                <p style='margin-bottom: 12px;'>
                    Given your current location, I recommend visiting the nearest authorized service center 
                    for immediate inspection.
                </p>
            </div>"""
        else:
            return f"""<div style='text-align: justify; line-height: 1.6;'>
                <p style='margin-bottom: 12px;'>
                    Based on recent telemetry data, the {part_name.lower()} components 
                    in the {model_name} model are operating within normal parameters.
                </p>
                <p style='margin-bottom: 12px;'>
                    Continue routine monitoring and standard maintenance protocols as scheduled.
                </p>
            </div>"""


