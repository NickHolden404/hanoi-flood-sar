"""
scripts/phase_3/cems_aoi03_analysis.py

Focused analysis of EMSR754 AOI03 — the only CEMS AOI overlapping
the Hanoi study area. Fixes the GeometryCollection district error,
produces clean per-district statistics, saves validation-ready files,
and generates the publication figure.

Run after cems_validation_check.py confirms AOI03 has overlap.

Usage:
    python scripts/phase_3/cems_aoi03_analysis.py
"""
from pathlib import Path
import json

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import fiona
from shapely.geometry import box, MultiPolygon, Polygon
from shapely.ops import unary_union

ROOT      = Path(__file__).resolve().parents[2]
CEMS_DIR  = ROOT / "data_raw" / "cems"
OSM_ROOT  = ROOT / "data_raw" / "osm"
OUT_DIR   = ROOT / "data_processed" / "qc_reports"
PROC_DIR  = ROOT / "data_raw" / "cems" / "processed"
CRS       = "EPSG:32648"

OUT_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# AOI03 files — the only ones with Hanoi overlap
_AOI03_DIR = CEMS_DIR / "EMSR754_AOI03_DEL_PRODUCT_v1"
AOI03 = {
    "flood_depth":    _AOI03_DIR / "EMSR754_AOI03_DEL_PRODUCT_floodDepthA_v1.shp",
    "observed_event": _AOI03_DIR / "EMSR754_AOI03_DEL_PRODUCT_observedEventA_v1.shp",
    "aoi_boundary":   _AOI03_DIR / "EMSR754_AOI03_DEL_PRODUCT_areaOfInterestA_v1.shp",
}

DISTRICT_LAYERS = ["hoan_kiem", "thanh_xuan", "hoang_mai"]
DISTRICT_DISPLAY = {
    "hoan_kiem":  "Hoàn Kiếm",
    "thanh_xuan": "Thanh Xuân",
    "hoang_mai":  "Hoàng Mai",
}


# -------------------------------------------------------
# Safe geometry area helper
# -------------------------------------------------------

def safe_area_km2(gdf) -> float:
    """Return total area in km² for a GeoDataFrame, or 0.0 if empty/missing geometry."""
    if gdf is None or gdf.empty or "geometry" not in gdf.columns:
        return 0.0
    return gdf.geometry.area.sum() / 1e6


def safe_len(gdf) -> int:
    """Return len of a GeoDataFrame, or 0 if None/empty."""
    if gdf is None or gdf.empty:
        return 0
    return len(gdf)


# -------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------

def to_valid_polygon(geom):
    """
    Convert any geometry to a valid Polygon or MultiPolygon.
    GeometryCollection from convex_hull of sparse points  bounding box fallback.
    """
    if geom is None:
        return None
    geom = geom.buffer(0)  # fix self-intersections
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    # GeometryCollection — extract polygons or use envelope
    polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")] \
            if hasattr(geom, "geoms") else []
    if polys:
        return unary_union(polys)
    return geom.envelope  # bounding box fallback


def district_geometry(layer_name: str) -> tuple:
    """
    Load district building layer, derive district boundary as
    the envelope (bounding box) of all building footprints.
    More robust than convex_hull for sparse/mixed geometry sets.
    Returns (GeoDataFrame of buildings, Polygon boundary).
    """
    gpkg = OSM_ROOT / "buildings_by_district.gpkg"
    gdf  = gpd.read_file(str(gpkg), layer=layer_name).to_crs(CRS)
    # Use bounding box of all features — avoids GeometryCollection issues
    bounds  = gdf.total_bounds   # [minx, miny, maxx, maxy]
    d_geom  = box(*bounds)
    return gdf, d_geom


# -------------------------------------------------------
# Load and inspect flood layers
# -------------------------------------------------------

def load_aoi03_layers() -> dict:
    layers = {}
    for key, path in AOI03.items():
        if not path.exists():
            print(f"  WARNING: {path.name} not found")
            continue
        gdf = gpd.read_file(str(path)).to_crs(CRS)
        layers[key] = gdf
        print(f"  {key:<20}: {len(gdf):>5} features | "
              f"area={safe_area_km2(gdf):.3f} km²")
        # Print attribute columns for flood_depth
        if key == "flood_depth":
            print(f"    Columns: {list(gdf.columns)}")
            if "dep_wkt" in gdf.columns or any("dep" in c.lower() for c in gdf.columns):
                dep_col = [c for c in gdf.columns if "dep" in c.lower()][0]
                print(f"    Depth values: {gdf[dep_col].value_counts().to_dict()}")
    return layers


# -------------------------------------------------------
# Clip to AOI
# -------------------------------------------------------

def clip_to_aoi(layers: dict) -> tuple:
    aoi_shp = OSM_ROOT / "hanoi_aoi_utm_buffered.shp"
    aoi     = gpd.read_file(str(aoi_shp)).to_crs(CRS)
    aoi_geom = aoi.geometry.union_all()

    clipped = {}
    print(f"\nAOI area: {aoi_geom.area/1e6:.2f} km²")
    for key, gdf in layers.items():
        c = gpd.clip(gdf, aoi_geom)
        clipped[key] = c
        print(f"  {key:<20}: {len(c):>5} features in AOI | "
              f"area={safe_area_km2(c):.3f} km²")
    return clipped, aoi, aoi_geom


# -------------------------------------------------------
# Per-district statistics
# -------------------------------------------------------

def district_stats(clipped: dict) -> list:
    flood_gdf = clipped.get("flood_depth")
    if flood_gdf is None or flood_gdf.empty:
        flood_gdf = clipped.get("observed_event")
    if flood_gdf is None or flood_gdf.empty or "geometry" not in flood_gdf.columns:
        print("No flood layer available for district stats")
        return []

    rows = []
    print(f"\n{'District':<15} {'CEMS flood km²':>15} {'% of flood in district':>24}")
    print("-" * 57)

    total_flood_area = safe_area_km2(flood_gdf)

    for layer_name in DISTRICT_LAYERS:
        dname = DISTRICT_DISPLAY[layer_name]
        try:
            _, d_geom = district_geometry(layer_name)
            # Convert to GeoDataFrame for clip
            d_gdf = gpd.GeoDataFrame(geometry=[d_geom], crs=CRS)
            c = gpd.clip(flood_gdf, d_gdf)
            area_km2 = safe_area_km2(c)
            pct = 100 * area_km2 / total_flood_area if total_flood_area > 0 else 0
            rows.append({
                "district": dname,
                "cems_flood_area_km2": round(area_km2, 4),
                "pct_of_total_cems_flood": round(pct, 2),
                "n_polygons": safe_len(c),
            })
            print(f"  {dname:<13} {area_km2:>14.3f} km²  {pct:>20.1f}%")
        except Exception as e:
            print(f"  {dname}: ERROR — {e}")

    print(f"  {'TOTAL':<13} {total_flood_area:>14.3f} km²  {'100.0':>20}%")
    return rows


# -------------------------------------------------------
# Depth class breakdown (if available)
# -------------------------------------------------------

def depth_breakdown(clipped: dict):
    gdf = clipped.get("flood_depth")
    if gdf is None or gdf.empty:
        return

    dep_cols = [c for c in gdf.columns if any(k in c.lower()
                for k in ("dep", "class", "value", "typ", "cat"))]
    if not dep_cols:
        print("\nNo depth column found in floodDepthA layer")
        return

    dep_col = dep_cols[0]
    print(f"\nDepth class breakdown (column: '{dep_col}'):")
    print("-" * 45)
    for val, grp in gdf.groupby(dep_col):
        area_km2 = safe_area_km2(grp)
        n        = len(grp)
        print(f"  {str(val):<20} {n:>5} polygons  {area_km2:.3f} km²")


# -------------------------------------------------------
# Save validation-ready files
# -------------------------------------------------------

def save_validation_files(clipped: dict, district_rows: list):
    # Primary validation file — flood depth polygons clipped to AOI
    for key in ("flood_depth", "observed_event"):
        gdf = clipped.get(key)
        if gdf is not None and not gdf.empty and "geometry" in gdf.columns:
            out = PROC_DIR / f"EMSR754_AOI03_{key}_AOI_utm48n.gpkg"
            gdf.to_file(str(out), driver="GPKG")
            print(f"  Saved: {out.name}")

    # Merged flood extent (union of flood_depth) — for binary validation mask
    fd = clipped.get("flood_depth")
    if fd is not None and not fd.empty and "geometry" in fd.columns:
        merged = fd.copy()
        merged["flood"] = 1
        dissolved = merged.dissolve(by="flood")[["geometry"]].reset_index(drop=True)
        out = PROC_DIR / "EMSR754_AOI03_flood_extent_dissolved_utm48n.gpkg"
        dissolved.to_file(str(out), driver="GPKG")
        print(f"  Saved: {out.name}")

    # Stats JSON
    if district_rows:
        stats = {
            "source": "EMSR754 AOI03",
            "acquisition": "2024-09-14",
            "sensor": "WorldView-3 0.3m HR2",
            "crs": CRS,
            "districts": district_rows,
        }
        out = PROC_DIR / "cems_aoi03_stats.json"
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Saved: {out.name}")


# -------------------------------------------------------
# Publication figure
# -------------------------------------------------------

def make_figure(clipped: dict, aoi: gpd.GeoDataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax_idx, (key, title) in enumerate([
        ("flood_depth",    "CEMS Flood Depth Classes"),
        ("observed_event", "CEMS Observed Flood Extent"),
    ]):
        ax  = axes[ax_idx]
        gdf = clipped.get(key)
        # Treat missing/no-geometry GDFs as None
        if gdf is not None and (gdf.empty or "geometry" not in gdf.columns):
            gdf = None

        # AOI boundary
        aoi.to_crs(CRS).boundary.plot(
            ax=ax, color="black", linewidth=1.5, zorder=3)

        # District outlines
        dist_colors = {"hoan_kiem": "#E74C3C",
                       "thanh_xuan": "#27AE60",
                       "hoang_mai": "#8E44AD"}
        handles = [mpatches.Patch(color="black", label="Study AOI")]
        for layer_name, dcol in dist_colors.items():
            try:
                _, d_geom = district_geometry(layer_name)
                gpd.GeoDataFrame(geometry=[d_geom], crs=CRS).boundary.plot(
                    ax=ax, color=dcol, linewidth=1.2,
                    linestyle="--", zorder=4)
                handles.append(mpatches.Patch(
                    facecolor="none", edgecolor=dcol, linestyle="--",
                    label=DISTRICT_DISPLAY[layer_name]))
            except Exception:
                pass

        # Flood polygons
        if gdf is not None:
            dep_cols = [c for c in gdf.columns
                        if "dep" in c.lower() or "class" in c.lower()]
            if dep_cols and key == "flood_depth":
                dep_col = dep_cols[0]
                unique_vals = gdf[dep_col].unique()
                colors = plt.cm.Blues(
                    np.linspace(0.3, 0.95, len(unique_vals)))
                for val, col in zip(sorted(unique_vals), colors):
                    sub = gdf[gdf[dep_col] == val]
                    sub.plot(ax=ax, color=col, alpha=0.85,
                             edgecolor="none", zorder=2)
                    handles.append(mpatches.Patch(
                        color=col, alpha=0.85,
                        label=f"Flood depth: {val}"))
            else:
                gdf.plot(ax=ax, color="#2E86C1", alpha=0.7,
                         edgecolor="none", zorder=2)
                handles.append(mpatches.Patch(
                    color="#2E86C1", alpha=0.7, label="Flood extent"))

            area_km2 = safe_area_km2(gdf)
            n_feat   = len(gdf)
        else:
            area_km2 = 0.0
            n_feat   = 0

        ax.set_title(f"{title}\n"
                     f"EMSR754 AOI03 | WorldView-3 | Acquired 14 Sep 2024\n"
                     f"{n_feat} polygons | {area_km2:.3f} km²",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Easting (m, UTM 48N)", fontsize=8)
        ax.set_ylabel("Northing (m, UTM 48N)", fontsize=8)
        ax.legend(handles=handles, fontsize=7, loc="upper right")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Copernicus EMS EMSR754 — Flood Validation Data\n"
        "Typhoon Yagi 2024 | Hanoi Study AOI | EPSG:32648",
        fontsize=12, fontweight="bold", y=1.01)

    fig.tight_layout()
    out = OUT_DIR / "cems_aoi03_validation_figure.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {out}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    print("=" * 60)
    print("CEMS EMSR754 AOI03 — Detailed Analysis")
    print("=" * 60)

    print("\nLoading AOI03 layers...")
    layers = load_aoi03_layers()

    print("\nClipping to study AOI...")
    clipped, aoi, aoi_geom = clip_to_aoi(layers)

    print("\nDepth class breakdown:")
    depth_breakdown(clipped)

    print("\nPer-district flood statistics:")
    district_rows = district_stats(clipped)

    print("\nSaving validation files...")
    save_validation_files(clipped, district_rows)

    print("\nGenerating figure...")
    make_figure(clipped, aoi)

    # Final summary — safe fallback when layers are missing
    flood_gdf = clipped.get("flood_depth")
    if flood_gdf is None or flood_gdf.empty:
        flood_gdf = clipped.get("observed_event")
    flood_area = safe_area_km2(flood_gdf)
    flood_len  = safe_len(flood_gdf)

    print(f"\n{'='*60}")
    print("VALIDATION DATA SUMMARY")
    print(f"{'='*60}")
    print(f"  CEMS activation  : EMSR754 — Flood in Viet Nam")
    print(f"  AOI              : AOI03 (only AOI overlapping Hanoi study area)")
    print(f"  Sensor           : WorldView-3, 0.3m HR2")
    print(f"  Acquisition      : 14 September 2024")
    print(f"  SAR post-event   : 12 September 2024 (2-day offset)")
    print(f"  Flood polygons   : {flood_len} in study AOI")
    print(f"  Flood area (AOI) : {flood_area:.3f} km²")
    print(f"  CEMS AOI coverage: 84.1% of study AOI")
    print(f"\n Validation data ready at: {PROC_DIR}")
    print(f"   Primary file: EMSR754_AOI03_flood_depth_AOI_utm48n.gpkg")
    print(f"   Dissolved mask: EMSR754_AOI03_flood_extent_dissolved_utm48n.gpkg")
    print(f"\n   Use dissolved mask for binary IoU comparison vs model predictions.")
    print(f"   Use depth classes for stratified accuracy analysis by flood severity.")


if __name__ == "__main__":
    main()