"""
scripts/phase_3/09_exposure_analysis.py

Urban flood exposure analysis — both events, ensemble mean prediction.

Workflow:
  1. Average per-seed probability maps  ensemble mean probability
  2. Threshold at 0.5  ensemble binary flood mask
  3. Vectorize flood mask  flood polygons (min area 100 m²)
  4. Intersect with OSM layers per district  exposure statistics
  5. Save CSV + flood polygon GeoPackage

Inputs:
  outputs/predictions/attn_unet_both_p20/<event>/seed_{0..5}/probability.tif
  data_raw/osm/buildings_by_district.gpkg
  data_raw/osm/roads_by_district.gpkg
  data_raw/osm/hospitals_by_district.gpkg
  data_raw/osm/schools_by_district.gpkg

Outputs:
  outputs/predictions/attn_unet_both_p20/<event>/ensemble_probability.tif
  outputs/predictions/attn_unet_both_p20/<event>/ensemble_binary.tif
  outputs/tables/exposure_<event>_attn_unet_both_p20.csv
  outputs/tables/exposure_<event>_attn_unet_both_p20_flood_polygons.gpkg

Usage:
    python scripts/phase_3/09_exposure_analysis.py
    python scripts/phase_3/09_exposure_analysis.py --event yagi
    python scripts/phase_3/09_exposure_analysis.py --seeds 0 1 2 3 4 5
"""
import argparse
import warnings
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
from shapely.geometry import shape, box
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DISTRICTS   = ["hoan_kiem", "thanh_xuan", "hoang_mai"]
DISTRICT_DISPLAY = {
    "hoan_kiem":  "Hoàn Kiếm",
    "thanh_xuan": "Thanh Xuân",
    "hoang_mai":  "Hoàng Mai",
}

# Road types considered for exposure (excludes footways, paths, etc.)
ROAD_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
}


# ── ensemble helpers ──────────────────────────────────────────────────────────

def build_ensemble(prob_dir: Path, seeds: list[int],
                   out_prob: Path, out_binary: Path,
                   threshold: float = 0.5) -> np.ndarray:
    """Average per-seed probability maps; write ensemble prob + binary TIFs."""
    arrays, profile = [], None
    for s in seeds:
        p = prob_dir / f"seed_{s}" / "probability.tif"
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
        with rasterio.open(str(p)) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    mean_prob = np.mean(arrays, axis=0)

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(str(out_prob), "w", **profile) as dst:
        dst.write(mean_prob, 1)

    binary = (mean_prob >= threshold).astype(np.uint8)
    flood_frac = binary.mean() * 100
    profile.update(dtype=rasterio.uint8, nodata=255)
    with rasterio.open(str(out_binary), "w", **profile) as dst:
        dst.write(binary, 1)

    print(f"  Ensemble probability : {out_prob.name}")
    print(f"  Ensemble binary      : {out_binary.name}  "
          f"(flood={flood_frac:.2f}%, thr={threshold})")
    return binary, profile


# ── vectorize ─────────────────────────────────────────────────────────────────

def vectorize_flood(binary: np.ndarray, profile: dict,
                    min_area_m2: float = 100.0) -> gpd.GeoDataFrame:
    """Convert binary flood raster to polygon GeoDataFrame."""
    geoms = []
    for geom, val in shapes(binary, transform=profile["transform"]):
        if val == 1:
            g = shape(geom)
            if g.area >= min_area_m2:
                geoms.append(g)

    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=profile["crs"])

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=profile["crs"])
    gdf["area_m2"] = gdf.geometry.area
    return gdf


# ── district boundary from building convex hull ───────────────────────────────

def district_boundary(bld_path: str, layer: str,
                       crs_target) -> gpd.GeoDataFrame:
    """Convex hull of building footprints as district proxy boundary."""
    gdf = gpd.read_file(bld_path, layer=layer)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf.to_crs(crs_target)
    hull = gdf.unary_union.convex_hull
    return gpd.GeoDataFrame(geometry=[hull], crs=crs_target)


# ── exposure helpers ──────────────────────────────────────────────────────────

def flood_area_in_district(flood_gdf: gpd.GeoDataFrame,
                            district_gdf: gpd.GeoDataFrame) -> float:
    """Total flood area (km²) within district boundary."""
    clipped = gpd.clip(flood_gdf, district_gdf)
    if clipped.empty:
        return 0.0
    return clipped.geometry.area.sum() / 1e6


def count_flooded_features(feat_gdf: gpd.GeoDataFrame,
                            flood_gdf: gpd.GeoDataFrame,
                            geom_types: list[str]) -> tuple[int, int]:
    """
    Returns (total, flooded) for point/polygon features.
    Uses centroid for polygon features; direct point test for points.
    """
    feat = feat_gdf[feat_gdf.geometry.notna() & ~feat_gdf.geometry.is_empty].copy()
    feat = feat[feat.geom_type.isin(geom_types)]
    if feat.empty:
        return 0, 0

    total = len(feat)
    # Represent each feature as its centroid for flood intersection
    pts = feat.copy()
    pts["geometry"] = feat.geometry.centroid

    flood_union = flood_gdf.geometry.unary_union
    if flood_union is None or flood_union.is_empty:
        return total, 0

    flooded = pts[pts.geometry.within(flood_union)]
    return total, len(flooded)


def road_length_flooded(road_gdf: gpd.GeoDataFrame,
                         flood_gdf: gpd.GeoDataFrame) -> tuple[float, float]:
    """
    Returns (total_km, flooded_km) for road linestrings.
    Clips road lines to flood polygons.
    """
    lines = road_gdf[road_gdf.geometry.notna() &
                     road_gdf.geom_type.isin(["LineString", "MultiLineString"])].copy()
    # Filter to meaningful road types
    if "highway" in lines.columns:
        lines = lines[lines["highway"].isin(ROAD_HIGHWAY_TYPES)]
    if lines.empty:
        return 0.0, 0.0

    total_km = lines.geometry.length.sum() / 1000.0

    flood_union = flood_gdf.geometry.unary_union
    if flood_union is None or flood_union.is_empty:
        return total_km, 0.0

    clipped = gpd.clip(lines, flood_gdf)
    flooded_km = clipped.geometry.length.sum() / 1000.0 if not clipped.empty else 0.0
    return total_km, flooded_km


# ── main exposure computation ─────────────────────────────────────────────────

def compute_exposure(flood_gdf: gpd.GeoDataFrame,
                     osm_root: Path,
                     flood_crs) -> pd.DataFrame:
    """Per-district exposure statistics."""
    bld_path  = str(osm_root / "buildings_by_district.gpkg")
    road_path = str(osm_root / "roads_by_district.gpkg")
    hosp_path = str(osm_root / "hospitals_by_district.gpkg")
    sch_path  = str(osm_root / "schools_by_district.gpkg")

    rows = []
    totals = {k: 0.0 for k in [
        "flood_area_km2", "buildings_total", "buildings_flooded",
        "roads_total_km", "roads_flooded_km",
        "hospitals_total", "hospitals_flooded",
        "schools_total", "schools_flooded",
    ]}

    for dist in DISTRICTS:
        print(f"    Processing district: {DISTRICT_DISPLAY[dist]} ({dist})")

        # District boundary
        dist_gdf = district_boundary(bld_path, dist, flood_crs)

        # Clip flood to district
        flood_dist = gpd.clip(flood_gdf, dist_gdf)

        # Flood area
        fa = flood_dist.geometry.area.sum() / 1e6 if not flood_dist.empty else 0.0

        # Buildings (polygon + multipolygon only — exclude points)
        bld = gpd.read_file(bld_path, layer=dist).to_crs(flood_crs)
        bld_poly = bld[bld.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        bld_total = len(bld_poly)
        if not flood_dist.empty and bld_total > 0:
            flood_union = flood_dist.geometry.unary_union
            bld_centroids = bld_poly.copy()
            bld_centroids["geometry"] = bld_poly.geometry.centroid
            bld_flooded = bld_centroids[bld_centroids.geometry.within(flood_union)]
            n_bld_flooded = len(bld_flooded)
        else:
            n_bld_flooded = 0

        # Roads
        roads = gpd.read_file(road_path, layer=dist).to_crs(flood_crs)
        roads_total_km, roads_flooded_km = road_length_flooded(
            roads, flood_dist if not flood_dist.empty else flood_gdf.iloc[0:0])

        # Hospitals (all geometry types — centroid test)
        hosp = gpd.read_file(hosp_path, layer=dist).to_crs(flood_crs)
        hosp_total, hosp_flooded = count_flooded_features(
            hosp, flood_dist, ["Point", "Polygon", "MultiPolygon"])

        # Schools
        sch = gpd.read_file(sch_path, layer=dist).to_crs(flood_crs)
        sch_total, sch_flooded = count_flooded_features(
            sch, flood_dist, ["Point", "Polygon", "MultiPolygon"])

        row = {
            "district":           DISTRICT_DISPLAY[dist],
            "flood_area_km2":     round(fa, 3),
            "buildings_total":    bld_total,
            "buildings_flooded":  n_bld_flooded,
            "buildings_pct":      round(n_bld_flooded / bld_total * 100, 1)
                                  if bld_total > 0 else 0.0,
            "roads_total_km":     round(roads_total_km, 2),
            "roads_flooded_km":   round(roads_flooded_km, 2),
            "roads_flooded_pct":  round(roads_flooded_km / roads_total_km * 100, 1)
                                  if roads_total_km > 0 else 0.0,
            "hospitals_total":    hosp_total,
            "hospitals_flooded":  hosp_flooded,
            "hospitals_pct":      round(hosp_flooded / hosp_total * 100, 1)
                                  if hosp_total > 0 else 0.0,
            "schools_total":      sch_total,
            "schools_flooded":    sch_flooded,
            "schools_pct":        round(sch_flooded / sch_total * 100, 1)
                                  if sch_total > 0 else 0.0,
            "cems_validated":     dist != "thanh_xuan",  # flag for paper
        }
        rows.append(row)

        for k in totals:
            totals[k] += row[k] if k in row else 0

    # TOTAL row
    tot_bld = totals["buildings_total"]
    tot_bld_f = totals["buildings_flooded"]
    tot_road = totals["roads_total_km"]
    tot_road_f = totals["roads_flooded_km"]
    tot_hosp = totals["hospitals_total"]
    tot_hosp_f = totals["hospitals_flooded"]
    tot_sch = totals["schools_total"]
    tot_sch_f = totals["schools_flooded"]

    rows.append({
        "district":           "TOTAL",
        "flood_area_km2":     round(totals["flood_area_km2"], 3),
        "buildings_total":    int(tot_bld),
        "buildings_flooded":  int(tot_bld_f),
        "buildings_pct":      round(tot_bld_f / tot_bld * 100, 1) if tot_bld > 0 else 0.0,
        "roads_total_km":     round(tot_road, 2),
        "roads_flooded_km":   round(tot_road_f, 2),
        "roads_flooded_pct":  round(tot_road_f / tot_road * 100, 1) if tot_road > 0 else 0.0,
        "hospitals_total":    int(tot_hosp),
        "hospitals_flooded":  int(tot_hosp_f),
        "hospitals_pct":      round(tot_hosp_f / tot_hosp * 100, 1) if tot_hosp > 0 else 0.0,
        "schools_total":      int(tot_sch),
        "schools_flooded":    int(tot_sch_f),
        "schools_pct":        round(tot_sch_f / tot_sch * 100, 1) if tot_sch > 0 else 0.0,
        "cems_validated":     "partial",
    })

    return pd.DataFrame(rows)


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="attn_unet_both_p20")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    p.add_argument("--event", default="all",
                   choices=["yagi", "matmo", "all"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--min_area_m2", type=float, default=100.0)
    return p.parse_args()


def main():
    args = parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    pred_root  = Path(cfg["paths"]["output_root"]) / "predictions" / args.run_name
    table_root = Path(cfg["paths"]["output_root"]) / "tables"
    osm_root   = Path(cfg["paths"]["osm_root"])
    table_root.mkdir(parents=True, exist_ok=True)

    events = cfg["events"] if args.event == "all" else [
        e for e in cfg["events"] if e["name"] == args.event]

    for ev in events:
        name     = ev["name"]
        prob_dir = pred_root / name

        print(f"\n{'='*60}")
        print(f"[{name.upper()}] Flood exposure analysis  "
              f"(ensemble of {len(args.seeds)} seeds)")

        # ── Step 1: build ensemble ────────────────────────────────────────────
        out_prob   = prob_dir / "ensemble_probability.tif"
        out_binary = prob_dir / "ensemble_binary.tif"
        binary_arr, profile = build_ensemble(
            prob_dir, args.seeds, out_prob, out_binary, args.threshold)

        flood_crs = profile["crs"]

        # ── Step 2: vectorize ─────────────────────────────────────────────────
        print(f"  Vectorizing flood mask (min area={args.min_area_m2} m²)...")
        flood_gdf = vectorize_flood(binary_arr, profile, args.min_area_m2)
        print(f"  Flood polygons: {len(flood_gdf):,}  |  "
              f"Total area: {flood_gdf.geometry.area.sum()/1e6:.2f} km²")

        # Save flood polygons
        poly_path = table_root / f"exposure_{name}_{args.run_name}_flood_polygons.gpkg"
        flood_gdf.to_file(str(poly_path), driver="GPKG")
        print(f"  Polygons saved: {poly_path.name}")

        # ── Step 3: exposure analysis ─────────────────────────────────────────
        print(f"\n  Computing per-district exposure...")
        df = compute_exposure(flood_gdf, osm_root, flood_crs)

        # Save CSV
        csv_path = table_root / f"exposure_{name}_{args.run_name}.csv"
        df.to_csv(str(csv_path), index=False)

        # ── Step 4: print results ─────────────────────────────────────────────
        print(f"\n{'EXPOSURE — ' + name.upper():=^70}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))

        tot = df[df["district"] == "TOTAL"].iloc[0]
        print(f"\n  Summary ({name}):")
        print(f"    Flood area           : {tot['flood_area_km2']:.2f} km²")
        print(f"    Buildings flooded    : {int(tot['buildings_flooded']):,} "
              f"/ {int(tot['buildings_total']):,}  ({tot['buildings_pct']:.1f}%)")
        print(f"    Roads flooded        : {tot['roads_flooded_km']:.2f} km "
              f"/ {tot['roads_total_km']:.2f} km  ({tot['roads_flooded_pct']:.1f}%)")
        print(f"    Hospitals at risk    : {int(tot['hospitals_flooded'])} "
              f"/ {int(tot['hospitals_total'])}")
        print(f"    Schools at risk      : {int(tot['schools_flooded'])} "
              f"/ {int(tot['schools_total'])}")

        if name == "matmo":
            print("\n    Matmo: post-event scene 9 days post-peak — "
                  "flood extent is conservative (partial recession).")
        print(f"\n   Saved: {csv_path.name}")

    print(f"\n Exposure analysis complete — tables in {table_root}")


if __name__ == "__main__":
    main()