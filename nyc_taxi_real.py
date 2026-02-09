#!/usr/bin/env python3
"""
Load real NYC TLC taxi OD and snap to Manhattan road network.

- TLC parquet: PULocationID, DOLocationID
- Taxi zone lookup: filter Manhattan (Borough == 'Manhattan')
- Taxi zone shapefile: get centroid (lat, lon) per zone
- Snap centroid to nearest OSM road node
"""

import hashlib
import numpy as np
from typing import List, Tuple, Optional, Dict
import os

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import geopandas as gpd
    from shapely.geometry import Point
    _HAS_GEOPANDAS = True
except ImportError:
    _HAS_GEOPANDAS = False

# TLC URLs (may 404/403; we have built-in fallback below)
TAXI_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TAXI_ZONES_ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
NYC_TAXI_ZONES_GEOSON = "https://data.cityofnewyork.us/api/geospatial/d3c5-ddgc?method=export&format=GeoJSON"

# Built-in Manhattan zone centroids (lon, lat) WGS84 — used when shapefile/GeoJSON download fails.
# Approximate positions across Manhattan so each zone snaps to a distinct area.
# Source: standard TLC Manhattan zone list; coords spread over Manhattan bbox.
_MANHATTAN_BBOX = (-74.0479, 40.6829, -73.9067, 40.8820)  # (west, south, east, north)
_MANHATTAN_ZONE_IDS = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
    107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144,
    148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202,
    209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
    244, 246, 249, 261, 262, 263,
]


def _manhattan_zone_centroids_builtin() -> Dict[int, Tuple[float, float]]:
    """Return approximate (lon, lat) for each Manhattan zone so no download is required."""
    w, s, e, n = _MANHATTAN_BBOX
    nrows, ncols = 6, 11  # grid
    out = {}
    for idx, zid in enumerate(_MANHATTAN_ZONE_IDS):
        row, col = idx // ncols, idx % ncols
        lon = w + (e - w) * (col + 0.5) / ncols
        lat = s + (n - s) * (row + 0.5) / nrows
        out[zid] = (lon, lat)
    return out


# Local taxi zones folder (e.g. ./taxi_zones containing taxi_zones.shp)
TAXI_ZONES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "taxi_zones")


def load_zone_centroids_manhattan(
    cache_dir: Optional[str] = None,
    taxi_zones_path: Optional[str] = None,
) -> Dict[int, Tuple[float, float]]:
    """
    Load Manhattan taxi zone centroids: LocationID -> (lon, lat) WGS84.

    Order: cache -> local taxi_zones/ -> download -> built-in.
    """
    cache_file = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "manhattan_zone_centroids.csv")

    if cache_file and os.path.isfile(cache_file) and _HAS_PANDAS:
        df = pd.read_csv(cache_file)
        return {int(row["LocationID"]): (float(row["lon"]), float(row["lat"])) for _, row in df.iterrows()}

    gdf = None
    if _HAS_GEOPANDAS:
        # 1. Local taxi_zones folder (taxi_zones_path or ./taxi_zones)
        local_dir = taxi_zones_path or TAXI_ZONES_DIR
        for name in ("taxi_zones.shp", "taxi_zones.geojson", "NYC_Taxi_Zones.geojson"):
            p = os.path.join(local_dir, name)
            if os.path.isfile(p):
                try:
                    gdf = gpd.read_file(p)
                    break
                except Exception:
                    continue

        # 2. Download if no local file
        if gdf is None:
            for url in [NYC_TAXI_ZONES_GEOSON, TAXI_ZONES_ZIP_URL]:
                try:
                    gdf = gpd.read_file(url)
                    break
                except Exception:
                    continue

    if gdf is not None and len(gdf) > 0:
        gdf = gdf.to_crs("EPSG:4326")
        for col in ("location_i", "LocationID", "OBJECTID"):
            if col in gdf.columns and "LocationID" not in gdf.columns:
                gdf["LocationID"] = gdf[col]
                break
        borough_col = next((c for c in ("borough", "Boro") if c in gdf.columns), None)
        if borough_col:
            manhattan = gdf[gdf[borough_col].astype(str).str.upper() == "MANHATTAN"]
        else:
            manhattan = gdf
        if len(manhattan) == 0:
            manhattan = gdf
        manhattan = manhattan.copy()
        manhattan["centroid"] = manhattan.geometry.centroid
        manhattan["lon"] = manhattan["centroid"].x
        manhattan["lat"] = manhattan["centroid"].y
        centroids = {}
        for _, row in manhattan.iterrows():
            lid = row.get("LocationID", row.get("location_i", row.name))
            try:
                lid = int(lid)
            except (ValueError, TypeError):
                continue
            centroids[lid] = (float(row["lon"]), float(row["lat"]))
        if centroids:
            if cache_file and _HAS_PANDAS:
                pd.DataFrame([
                    {"LocationID": lid, "lon": lon, "lat": lat}
                    for lid, (lon, lat) in centroids.items()
                ]).to_csv(cache_file, index=False)
            return centroids

    # Fallback: built-in approximate centroids (no download)
    return _manhattan_zone_centroids_builtin()


def load_taxi_od_manhattan(
    parquet_path: str,
    n_trips: int = 500,
    seed: int = 42,
    manhattan_only: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Load (PULocationID, DOLocationID) from TLC parquet.
    If manhattan_only, filter to trips where both PU and DO are Manhattan zones.
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas and pyarrow required. pip install pandas pyarrow")

    df = pd.read_parquet(parquet_path, columns=["PULocationID", "DOLocationID"])
    df = df.dropna(subset=["PULocationID", "DOLocationID"])
    df = df.astype({"PULocationID": int, "DOLocationID": int})
    df = df[df["PULocationID"] != df["DOLocationID"]]

    if manhattan_only:
        # Manhattan LocationIDs: 4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243, 244, 246, 249, 261, 262, 263
        manhattan_ids = {
            4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
            107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144,
            148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202,
            209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
            244, 246, 249, 261, 262, 263,
        }
        df = df[df["PULocationID"].isin(manhattan_ids) & df["DOLocationID"].isin(manhattan_ids)]

    rng = np.random.default_rng(seed)
    if len(df) > n_trips:
        df = df.sample(n=n_trips, random_state=rng)
    return df["PULocationID"].tolist(), df["DOLocationID"].tolist()


def load_taxi_od_manhattan_by_time(
    parquet_path: str,
    n_per_period: int = 200,
    seed: int = 42,
    manhattan_only: bool = True,
) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
    """
    Load Manhattan OD pairs binned by time-of-day: peak vs off-peak.
    
    Peak: 7-9am (morning rush), 4-7pm (evening rush)
    Off-peak: 10am-3pm (midday)
    
    Returns: ((peak_pu_ids, peak_do_ids), (offpeak_pu_ids, offpeak_do_ids))
    """
    if not _HAS_PANDAS:
        raise ImportError("pandas and pyarrow required")

    dt_col = None
    for c in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
        try:
            df = pd.read_parquet(parquet_path, columns=[c, "PULocationID", "DOLocationID"])
            if c in df.columns:
                dt_col = c
                break
        except Exception:
            continue
    if dt_col is None:
        raise ValueError("Parquet must have tpep_pickup_datetime or lpep_pickup_datetime")

    df = df.dropna(subset=["PULocationID", "DOLocationID", dt_col])
    df = df.astype({"PULocationID": int, "DOLocationID": int})
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[dt_col])
    df = df[df["PULocationID"] != df["DOLocationID"]]

    if manhattan_only:
        manhattan_ids = {
            4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
            107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142, 143, 144,
            148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194, 202,
            209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
            244, 246, 249, 261, 262, 263,
        }
        df = df[df["PULocationID"].isin(manhattan_ids) & df["DOLocationID"].isin(manhattan_ids)]

    df["hour"] = df[dt_col].dt.hour
    peak_mask = ((df["hour"] >= 7) & (df["hour"] < 9)) | ((df["hour"] >= 16) & (df["hour"] < 19))
    offpeak_mask = (df["hour"] >= 10) & (df["hour"] < 15)
    peak_df = df[peak_mask]
    offpeak_df = df[offpeak_mask]

    rng = np.random.default_rng(seed)
    peak_df = peak_df.sample(n=min(n_per_period, len(peak_df)), random_state=rng) if len(peak_df) > 0 else peak_df
    rng2 = np.random.default_rng(seed + 1)
    offpeak_df = offpeak_df.sample(n=min(n_per_period, len(offpeak_df)), random_state=rng2) if len(offpeak_df) > 0 else offpeak_df

    peak_pu = peak_df["PULocationID"].tolist()
    peak_do = peak_df["DOLocationID"].tolist()
    offpeak_pu = offpeak_df["PULocationID"].tolist()
    offpeak_do = offpeak_df["DOLocationID"].tolist()
    return (peak_pu, peak_do), (offpeak_pu, offpeak_do)


def snap_od_to_road(
    road,  # OSMGraphRoad
    pu_ids: List[int],
    do_ids: List[int],
    zone_centroids: Dict[int, Tuple[float, float]],
) -> Tuple[List[int], List[int]]:
    """
    Snap taxi zone centroids to nearest OSM road nodes.
    Transform zone (lon, lat) to graph CRS, then find nearest node.
    """
    try:
        from pyproj import Transformer
        graph_crs = road.G.graph.get("crs", "EPSG:4326")
        transformer = Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
    except Exception:
        transformer = None

    origins = []
    destinations = []
    for pu, do in zip(pu_ids, do_ids):
        if pu not in zone_centroids or do not in zone_centroids:
            continue
        lon_pu, lat_pu = zone_centroids[pu]
        lon_do, lat_do = zone_centroids[do]
        if transformer:
            x_pu, y_pu = transformer.transform(lon_pu, lat_pu)
            x_do, y_do = transformer.transform(lon_do, lat_do)
        else:
            x_pu, y_pu = lon_pu, lat_pu
            x_do, y_do = lon_do, lat_do
        n_orig = road.nearest_node(x_pu, y_pu)
        n_dest = road.nearest_node(x_do, y_do)
        if n_orig != n_dest:
            origins.append(n_orig)
            destinations.append(n_dest)
    return origins, destinations


def _od_cache_path(parquet_path: str, n_agents: int, seed: int, cache_dir: str) -> str:
    """Cache file path for sampled OD. Keyed by parquet path, n_agents, seed."""
    path_hash = hashlib.md5(os.path.abspath(parquet_path).encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"od_{path_hash}_{n_agents}_{seed}.npz")


def get_real_od(
    road,
    parquet_path: str,
    n_agents: int = 200,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[List[int], List[int]]:
    """
    Load real Manhattan taxi OD and snap to road network.
    Uses cache if available (avoids re-reading full parquet).
    Drops OD pairs whose origin or destination was removed by graph pruning.
    
    Returns (origin_node_ids, destination_node_ids).
    """
    cache_dir = cache_dir or ".manhattan_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = _od_cache_path(parquet_path, n_agents, seed, cache_dir)
    valid = getattr(road, "valid_nodes", lambda: set(road.G.nodes()))()

    def has_path(road, o, d):
        """Check if there is a path from o to d in the graph."""
        if o not in valid or d not in valid:
            return False
        try:
            return road.distance(o, d) != float("inf")
        except Exception:
            return False

    if os.path.isfile(cache_path):
        try:
            data = np.load(cache_path, allow_pickle=False)
            origins = data["origins"].tolist()
            destinations = data["destinations"].tolist()
            n_before = len(origins)
            kept = [(o, d) for o, d in zip(origins, destinations)
                    if o in valid and d in valid and has_path(road, o, d)]
            origins = [o for o, d in kept]
            destinations = [d for o, d in kept]
            n_dropped = n_before - len(origins)
            if n_dropped > 0:
                print(f"  Dropped {n_dropped} OD pairs (invalid or no path)", flush=True)
                try:
                    np.savez_compressed(cache_path, origins=np.array(origins), destinations=np.array(destinations))
                except Exception:
                    pass
            return origins, destinations
        except Exception:
            pass

    zone_centroids = load_zone_centroids_manhattan(cache_dir)
    pu_ids, do_ids = load_taxi_od_manhattan(parquet_path, n_trips=n_agents, seed=seed, manhattan_only=True)
    origins, destinations = snap_od_to_road(road, pu_ids, do_ids, zone_centroids)
    n_before = len(origins)
    kept = [(o, d) for o, d in zip(origins, destinations)
            if o in valid and d in valid and has_path(road, o, d)]
    origins = [o for o, d in kept]
    destinations = [d for o, d in kept]
    n_dropped = n_before - len(origins)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} OD pairs (invalid or no path)", flush=True)

    try:
        np.savez_compressed(cache_path, origins=np.array(origins), destinations=np.array(destinations))
    except Exception:
        pass
    return origins, destinations


def get_real_od_peak_offpeak(
    road,
    parquet_path: str,
    n_per_period: int = 200,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
    """
    Load peak and off-peak OD, snap to road.
    Returns: ((peak_origins, peak_dests), (offpeak_origins, offpeak_dests))
    """
    zone_centroids = load_zone_centroids_manhattan(cache_dir)
    (peak_pu, peak_do), (offpeak_pu, offpeak_do) = load_taxi_od_manhattan_by_time(
        parquet_path, n_per_period=n_per_period, seed=seed, manhattan_only=True
    )
    peak_origins, peak_dests = snap_od_to_road(road, peak_pu, peak_do, zone_centroids)
    offpeak_origins, offpeak_dests = snap_od_to_road(road, offpeak_pu, offpeak_do, zone_centroids)
    return (peak_origins, peak_dests), (offpeak_origins, offpeak_dests)
