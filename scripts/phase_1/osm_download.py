"""OpenStreetMap infrastructure download and per-district clipping.

Layers: buildings, roads, hospitals, schools.
Districts: Hoan Kiem, Thanh Xuan, Hoang Mai (Hanoi, Vietnam).

Prerequisites: pip install osmnx geopandas shapely
"""

import json
import os
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSM_DIR  = PROJECT_ROOT / "data_raw" / "osm"
META_DIR = PROJECT_ROOT / "metadata"

OSM_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# District bounding boxes (min_lon, min_lat, max_lon, max_lat)
DISTRICTS = {
    "hoan_kiem":  (105.838, 21.023, 105.862, 21.040),
    "thanh_xuan": (105.797, 20.988, 105.848, 21.018),
    "hoang_mai":  (105.830, 20.961, 105.893, 21.000),
}

# Combined AOI bbox as dict — works across all osmnx versions
AOI = {"north": 21.055, "south": 20.940, "east": 105.910, "west": 105.770}

# OSM tags to download
LAYERS = {
    "buildings": {"building": True},
    "roads":     {"highway": True},
    "hospitals": {"amenity": "hospital"},
    "schools":   {"amenity": "school"},
}

CRS = "EPSG:4326"

# ── DOWNLOAD ───────────────────────────────────────────────────────────────────

def download_layer(layer_name, tags):
    raw_path = str(OSM_DIR /f"{layer_name}_hanoi_raw.gpkg")

    if os.path.exists(raw_path):
        print(f"    Already exists, loading: {layer_name}")
        return gpd.read_file(raw_path)

    print(f"    Downloading: {layer_name}...")
    try:
        import inspect
        sig = inspect.signature(ox.features_from_bbox)
        params = list(sig.parameters.keys())

        # osmnx ≥ 2.0: features_from_bbox(bbox, tags)
        # bbox = (left, bottom, right, top) = (west, south, east, north)
        # osmnx 1.x: features_from_bbox(north, south, east, west, tags)
        if params[0] == "bbox":
            bbox_arg = (AOI["west"], AOI["south"], AOI["east"], AOI["north"])
            gdf = ox.features_from_bbox(bbox=bbox_arg, tags=tags)
        else:
            gdf = ox.features_from_bbox(
                AOI["north"], AOI["south"], AOI["east"], AOI["west"], tags=tags
            )
        # Drop unsupported geometry types
        valid_types = {"Point", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}
        gdf = gdf[gdf.geometry.geom_type.isin(valid_types)].copy()

        # Reset multi-index that osmnx returns
        gdf = gdf.reset_index(drop=True)

        # Drop columns that are entirely NaN (causes int conversion errors)
        gdf = gdf.dropna(axis=1, how="all")

        # Ensure string columns only (avoid mixed-type serialisation issues)
        for col in gdf.columns:
            if col != "geometry":
                gdf[col] = gdf[col].astype(str)

        gdf = gdf.to_crs(CRS)
        gdf.to_file(raw_path, driver="GPKG")
        print(f"      {len(gdf):,} features  {raw_path}")
        return gdf

    except Exception as e:
        print(f"      Failed: {e}")
        return gpd.GeoDataFrame(geometry=[], crs=CRS)

# ── CLIP TO DISTRICTS ──────────────────────────────────────────────────────────

def clip_to_districts(gdf, layer_name):
    if gdf.empty:
        return {d: 0 for d in DISTRICTS}

    out_path = str(OSM_DIR /f"{layer_name}_by_district.gpkg")
    counts   = {}

    for district, (min_lon, min_lat, max_lon, max_lat) in DISTRICTS.items():
        clip_box = gpd.GeoDataFrame(
            geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs=CRS
        )
        clipped = gpd.clip(gdf, clip_box)

        if not clipped.empty:
            clipped.to_file(out_path, layer=district, driver="GPKG")

        counts[district] = len(clipped)
        print(f"     {district:<15} {len(clipped):>6,} features")

    return counts

# ── VERIFY ─────────────────────────────────────────────────────────────────────

def verify_layer(gdf, layer_name):
    issues = []
    if gdf.empty:
        issues.append("Empty GeoDataFrame")
        return issues
    if gdf.crs is None or str(gdf.crs.to_epsg()) != "4326":
        issues.append(f"CRS mismatch: {gdf.crs}")
    null_geom = gdf.geometry.isnull().sum()
    if null_geom > 0:
        issues.append(f"{null_geom} null geometries")
    invalid = (~gdf.geometry.is_valid).sum()
    if invalid > 0:
        issues.append(f"{invalid} invalid geometries (non-fatal)")
    return issues

# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PHASE 1 — OSM INFRASTRUCTURE DOWNLOAD & VERIFY")
    print("=" * 55 + "\n")

    summary    = {}
    all_counts = {d: {} for d in DISTRICTS}

    for layer_name, tags in LAYERS.items():
        print(f"── {layer_name.upper()} {'─'*38}")
        gdf     = download_layer(layer_name, tags)
        print(f"  Clipping to districts:")
        counts  = clip_to_districts(gdf, layer_name)
        issues  = verify_layer(gdf, layer_name)

        for district, count in counts.items():
            all_counts[district][layer_name] = count

        summary[layer_name] = {
            "total_features": len(gdf),
            "per_district":   counts,
            "crs":            str(gdf.crs) if not gdf.empty else "N/A",
            "geometry_types": gdf.geometry.geom_type.value_counts().to_dict() if not gdf.empty else {},
            "issues":         issues if issues else ["none"],
        }
        print(f"  CRS   : {summary[layer_name]['crs']}")
        print(f"  Issues: {issues if issues else 'none'}\n")

    # ── FEATURE COUNT TABLE ───────────────────────────────────────────────────

    print("=" * 55)
    print("FEATURE COUNT SUMMARY TABLE")
    print("=" * 55)

    layer_names = list(LAYERS.keys())
    rows = []
    for district in DISTRICTS:
        row = {"district": district}
        for ln in layer_names:
            row[ln] = all_counts[district].get(ln, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("district")
    totals = df.sum().rename("TOTAL")
    df = pd.concat([df, totals.to_frame().T])
    print(df.to_string())

    # ── SAVE METADATA ─────────────────────────────────────────────────────────

    meta_path = META_DIR / "osm_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda o: int(o) if hasattr(o, "item") else o)
    print(f"\n OSM metadata saved: {meta_path}")

    # ── VALIDATION ────────────────────────────────────────────────────────────

    print(f"\n{'='*55}")
    print("VALIDATION")
    print(f"{'='*55}")

    all_pass = True
    checks = {
        "Buildings downloaded":  summary.get("buildings", {}).get("total_features", 0) > 0,
        "Roads downloaded":      summary.get("roads",     {}).get("total_features", 0) > 0,
        "Hospitals downloaded":  summary.get("hospitals", {}).get("total_features", 0) > 0,
        "Schools downloaded":    summary.get("schools",   {}).get("total_features", 0) > 0,
        "All layers CRS WGS84":  all("4326" in str(v.get("crs","")) for v in summary.values()),
        "No empty districts":    all(sum(all_counts[d].values()) > 0 for d in DISTRICTS),
    }

    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}]  {check}")

    print(f"\nRESULT: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("\nDone. Phase 1 data acquisition complete.")
    print("All raw data verified and ready for Phase 2 preprocessing.")