"""
scripts/phase_3/cems_validation_check.py

Loads CEMS EMSR754 flood delineation products, checks overlap with
study AOI, reprojects to UTM 48N, and produces a coverage figure.

Run this BEFORE Phase 3 training to confirm validation data is usable.

Usage:
    python scripts/phase_3/cems_validation_check.py
"""
from pathlib import Path
import sys

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.ops import unary_union

ROOT     = Path(__file__).resolve().parents[2]
CEMS_DIR = ROOT / "data_raw" / "cems"
AOI_SHP  = ROOT / "data_raw" / "osm" / "hanoi_aoi_utm_buffered.shp"
OUT_DIR  = ROOT / "data_processed" / "qc_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CRS = "EPSG:32648"


# -------------------------------------------------------
# 1. Auto-detect CEMS vector files
# -------------------------------------------------------
def find_vector_files(cems_dir: Path) -> list:
    """Find all shapefiles and GeoPackages recursively."""
    exts   = {".shp", ".gpkg", ".geojson", ".json"}
    found  = [f for f in cems_dir.rglob("*") if f.suffix.lower() in exts]
    return sorted(found)


def classify_cems_file(path: Path) -> str:
    """
    CEMS file naming convention:
    EMSR754_AOI##_DEL_PRODUCT_v#_vector.shp
    Keywords: DEL=delineation, GRA=grading, OSM=reference
    """
    name = path.stem.upper()
    if "DEL" in name or "DELINEATION" in name:
        return "delineation"
    elif "GRA" in name or "GRADING" in name:
        return "grading"
    elif "OSM" in name or "REF" in name:
        return "reference"
    else:
        return "unknown"


# -------------------------------------------------------
# 2. Load and reproject
# -------------------------------------------------------
def load_cems_layers(files: list) -> dict:
    """
    Load all CEMS vector files, reproject to UTM 48N.
    Returns dict: {filepath_str: GeoDataFrame}
    """
    layers = {}
    for f in files:
        try:
            gdf = gpd.read_file(f)
            if gdf.empty:
                print(f"  SKIP (empty): {f.name}")
                continue
            if gdf.crs is None:
                print(f"  WARNING — no CRS on {f.name}, assuming EPSG:4326")
                gdf = gdf.set_crs("EPSG:4326")
            gdf = gdf.to_crs(TARGET_CRS)
            layers[str(f)] = gdf
            print(f"  Loaded : {f.name}  ({len(gdf)} features, "
                  f"type={classify_cems_file(f)})")
        except Exception as e:
            print(f"  ERROR loading {f.name}: {e}")
    return layers


# -------------------------------------------------------
# 3. Load AOI
# -------------------------------------------------------
def load_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path).to_crs(TARGET_CRS)
    print(f"\nAOI loaded: {aoi_path.name}  ({len(aoi)} features)")
    print(f"  AOI bounds: {aoi.total_bounds.round(0)}")
    return aoi


# -------------------------------------------------------
# 4. Overlap analysis
# -------------------------------------------------------
def analyse_overlap(layers: dict, aoi: gpd.GeoDataFrame) -> dict:
    """
    For each CEMS layer, compute:
    - Total CEMS flood area (km²)
    - Flood area within AOI (km²)
    - % of AOI covered by CEMS extent
    - % of CEMS flood within AOI
    """
    aoi_geom    = aoi.geometry.unary_union
    aoi_area_m2 = aoi_geom.area
    results     = {}

    print(f"\n{'='*60}")
    print(f"AOI area: {aoi_area_m2/1e6:.2f} km²")
    print(f"{'='*60}")

    for path_str, gdf in layers.items():
        fname    = Path(path_str).name
        ftype    = classify_cems_file(Path(path_str))

        # Only analyse flood/delineation layers
        flood_keywords = ["flood", "flooded", "delineation", "water"]
        col_check = " ".join(gdf.columns.tolist() + [fname]).lower()
        is_flood  = (ftype == "delineation" or
                     any(k in col_check for k in flood_keywords))

        # Clip to AOI
        try:
            clipped = gpd.clip(gdf, aoi_geom)
        except Exception:
            clipped = gdf[gdf.intersects(aoi_geom)]

        total_area_m2   = gdf.geometry.area.sum()
        clipped_area_m2 = clipped.geometry.area.sum()
        pct_aoi_covered = 100 * clipped_area_m2 / aoi_area_m2
        pct_in_aoi      = 100 * clipped_area_m2 / max(total_area_m2, 1)

        results[fname] = {
            "type":              ftype,
            "total_features":    len(gdf),
            "clipped_features":  len(clipped),
            "total_area_km2":    round(total_area_m2  / 1e6, 4),
            "aoi_area_km2":      round(clipped_area_m2 / 1e6, 4),
            "pct_aoi_covered":   round(pct_aoi_covered, 2),
            "pct_cems_in_aoi":   round(pct_in_aoi, 2),
            "clipped_gdf":       clipped,
        }

        print(f"\n  File    : {fname}")
        print(f"  Type    : {ftype}")
        print(f"  Features: {len(gdf)} total  {len(clipped)} within AOI")
        print(f"  CEMS flood area (total)  : {total_area_m2/1e6:.3f} km²")
        print(f"  CEMS flood area in AOI   : {clipped_area_m2/1e6:.3f} km²")
        print(f"  % of AOI covered by CEMS : {pct_aoi_covered:.2f}%")
        print(f"  % of CEMS extent in AOI  : {pct_in_aoi:.2f}%")

    return results


# -------------------------------------------------------
# 5. Check district coverage
# -------------------------------------------------------
def check_district_coverage(results: dict, aoi: gpd.GeoDataFrame):
    """
    Load OSM district layers and report CEMS coverage per district.
    """
    gpkg = ROOT / "data_raw" / "osm" / "buildings_by_district.gpkg"
    if not gpkg.exists():
        print("\nDistrict gpkg not found — skipping district breakdown")
        return

    import fiona
    layers_available = fiona.listlayers(str(gpkg))
    print(f"\nDistrict layers found: {layers_available}")

    # Find delineation layer with most AOI coverage
    best = None
    best_area = 0
    for fname, r in results.items():
        if r["type"] == "delineation" and r["aoi_area_km2"] > best_area:
            best_area = r["aoi_area_km2"]
            best      = r["clipped_gdf"]

    if best is None or best.empty:
        print("No delineation layer with AOI coverage found.")
        return

    print(f"\nDistrict-level CEMS flood coverage (best delineation layer):")
    print(f"{'District':<20} {'CEMS flood km²':>16} {'Buildings layer':>16}")
    print("-" * 55)

    for layer_name in layers_available:
        try:
            dist_gdf = gpd.read_file(str(gpkg), layer=layer_name).to_crs(TARGET_CRS)
            dist_geom = dist_gdf.geometry.unary_union
            clipped   = gpd.clip(best, dist_geom)
            area_km2  = clipped.geometry.area.sum() / 1e6
            print(f"  {layer_name:<18} {area_km2:>14.3f} km²")
        except Exception as e:
            print(f"  {layer_name}: ERROR — {e}")


# -------------------------------------------------------
# 6. Figure
# -------------------------------------------------------
def make_coverage_figure(results: dict, aoi: gpd.GeoDataFrame):
    """
    Map showing AOI boundary + all CEMS flood layers colour-coded by type.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # AOI boundary
    aoi.boundary.plot(ax=ax, color="black", linewidth=2, label="Study AOI")

    colors = {"delineation": "#00BFFF", "grading": "#FF6600",
              "reference": "#888888", "unknown": "#CCCCCC"}
    handles = [mpatches.Patch(color="black", label="Study AOI (buffered)")]

    plotted_types = set()
    for path_str, r in results.items():
        gdf   = r["clipped_gdf"]
        ftype = r["type"]
        if gdf.empty:
            continue
        color = colors.get(ftype, "#CCCCCC")
        label = f"CEMS {ftype}" if ftype not in plotted_types else "_nolegend_"
        gdf.plot(ax=ax, color=color, alpha=0.6, edgecolor="none", label=label)
        if ftype not in plotted_types:
            handles.append(mpatches.Patch(color=color, alpha=0.6,
                                          label=f"CEMS {ftype}"))
            plotted_types.add(ftype)

    # Try to overlay district outlines from OSM buildings layer
    gpkg = ROOT / "data_raw" / "osm" / "buildings_by_district.gpkg"
    if gpkg.exists():
        import fiona
        dist_colors = {"hoan_kiem": "red", "thanh_xuan": "green",
                       "hoang_mai": "purple"}
        for lname, dcol in dist_colors.items():
            try:
                d = gpd.read_file(str(gpkg), layer=lname).to_crs(TARGET_CRS)
                hull = d.geometry.unary_union.convex_hull
                gpd.GeoDataFrame(geometry=[hull], crs=TARGET_CRS).boundary.plot(
                    ax=ax, color=dcol, linewidth=1.5, linestyle="--")
                handles.append(mpatches.Patch(
                    facecolor="none", edgecolor=dcol,
                    label=lname.replace("_", " ").title(), linestyle="--"))
            except Exception:
                pass

    ax.set_title("CEMS EMSR754 — Flood Delineation Coverage over Study AOI\n"
                 "Typhoon Yagi 2024 | Acquisition: 14 Sep 2024 | WorldView-3",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Easting (m, UTM 48N)")
    ax.set_ylabel("Northing (m, UTM 48N)")
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    out_path = OUT_DIR / "cems_emsr754_coverage.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCoverage figure saved: {out_path}")


# -------------------------------------------------------
# 7. Save reprojected CEMS for Phase 3 use
# -------------------------------------------------------
def save_reprojected(results: dict):
    """
    Save clipped + reprojected delineation layer as GeoPackage
    for use in Phase 3 validation.
    """
    out_dir = ROOT / "data_raw" / "cems" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path_str, r in results.items():
        if r["type"] == "delineation" and not r["clipped_gdf"].empty:
            fname    = Path(path_str).stem
            out_path = out_dir / f"{fname}_clipped_utm48n.gpkg"
            r["clipped_gdf"].to_file(str(out_path), driver="GPKG")
            print(f"  Saved reprojected: {out_path}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("CEMS EMSR754 Validation Data Check")
    print("=" * 60)

    # Detect files
    files = find_vector_files(CEMS_DIR)
    if not files:
        print(f"\nERROR: No vector files found in {CEMS_DIR}")
        print("Create the directory and place CEMS downloads there:")
        print(f"  {CEMS_DIR}")
        print("\nExpected file types: .shp, .gpkg, .geojson")
        sys.exit(1)

    print(f"\nFound {len(files)} vector file(s):")
    for f in files:
        print(f"  {f.name}")

    # Load AOI
    if not AOI_SHP.exists():
        print(f"\nERROR: AOI shapefile not found: {AOI_SHP}")
        sys.exit(1)
    aoi = load_aoi(AOI_SHP)

    # Load CEMS layers
    print("\nLoading CEMS layers...")
    layers = load_cems_layers(files)
    if not layers:
        print("ERROR: No valid layers loaded.")
        sys.exit(1)

    # Overlap analysis
    results = analyse_overlap(layers, aoi)

    # District breakdown
    check_district_coverage(results, aoi)

    # Figure
    print("\nGenerating coverage figure...")
    make_coverage_figure(results, aoi)

    # Save processed files
    print("\nSaving reprojected delineation layers...")
    save_reprojected(results)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    total_cems_in_aoi = sum(
        r["aoi_area_km2"] for r in results.values()
        if r["type"] == "delineation")
    print(f"Total CEMS delineation area within AOI: {total_cems_in_aoi:.3f} km²")

    has_coverage = any(r["pct_aoi_covered"] > 0 for r in results.values())
    if has_coverage:
        print("\n CEMS data overlaps with study AOI — validation is feasible")
        print("   Reprojected files saved to data_raw/cems/processed/")
        print("   Use these as reference in Phase 3 model evaluation")
    else:
        print("\n  No CEMS overlap with AOI detected")
        print("   Check that the correct EMSR754 AOI products were downloaded")
        print("   The Hanoi delineation may be in a specific sub-AOI product")


if __name__ == "__main__":
    main()