"""
inference/exposure.py — Urban flood exposure analysis.

Vectorizes a binary flood mask GeoTIFF and intersects it with
OSM infrastructure layers (buildings, roads, hospitals, schools)
from data_raw/osm/*_by_district.gpkg files (Phase 1 outputs).

District boundaries: uses hanoi_aoi_utm_buffered.shp split by
buildings_by_district.gpkg layer names (hoan_kiem, thanh_xuan, hoang_mai).
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


# Layer names inside *_by_district.gpkg (Phase 1 osmnx output)
DISTRICT_LAYERS = ["hoan_kiem", "thanh_xuan", "hoang_mai"]
DISTRICT_DISPLAY = {
    "hoan_kiem":  "Hoàn Kiếm",
    "thanh_xuan": "Thanh Xuân",
    "hoang_mai":  "Hoàng Mai",
}


def vectorize_flood(binary_path: str, min_area_m2: float = 100.0) -> gpd.GeoDataFrame:
    """Convert binary flood mask GeoTIFF  GeoDataFrame of flood polygons."""
    with rasterio.open(binary_path) as src:
        data      = src.read(1)
        transform = src.transform
        crs       = src.crs

    polys = [
        {"geometry": shape(geom), "area_m2": shape(geom).area}
        for geom, val in shapes(data, mask=(data == 1), transform=transform)
        if val == 1 and shape(geom).area >= min_area_m2
    ]

    gdf = gpd.GeoDataFrame(polys, crs=crs)
    print(f"  Flood polygons   : {len(gdf)} features  "
          f"(total area={gdf.geometry.area.sum()/1e6:.2f} km²)")
    return gdf


def _load_district_union(gpkg_path: str, layer: str, crs) -> gpd.GeoDataFrame:
    """Load one layer from a by_district gpkg and reproject."""
    gdf = gpd.read_file(gpkg_path, layer=layer)
    return gdf.to_crs(crs)


def compute_exposure(flood_gdf: gpd.GeoDataFrame, osm_root: str) -> pd.DataFrame:
    """
    Compute per-district flood exposure statistics.

    Parameters
    ----------
    flood_gdf : flood polygon GeoDataFrame (UTM 48N)
    osm_root  : path to data_raw/osm/

    Returns
    -------
    DataFrame with one row per district
    """
    osm_root = Path(osm_root)
    crs      = flood_gdf.crs
    rows     = []

    gpkg_paths = {
        "buildings": osm_root / "buildings_by_district.gpkg",
        "roads":     osm_root / "roads_by_district.gpkg",
        "hospitals": osm_root / "hospitals_by_district.gpkg",
        "schools":   osm_root / "schools_by_district.gpkg",
    }

    # Verify all gpkg files exist
    for name, p in gpkg_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"OSM layer not found: {p}")

    for layer in DISTRICT_LAYERS:
        d_name = DISTRICT_DISPLAY[layer]
        print(f"\n  District: {d_name}")

        # Build district boundary from union of building footprints extent
        # (using buildings layer to derive the district polygon boundary)
        bld_gdf  = _load_district_union(str(gpkg_paths["buildings"]), layer, crs)
        d_geom   = bld_gdf.unary_union.convex_hull

        # Clip flood to district
        d_flood  = gpd.clip(flood_gdf, d_geom)
        flood_area_km2 = d_flood.geometry.area.sum() / 1e6

        row = {
            "district":        d_name,
            "flood_area_km2":  round(flood_area_km2, 4),
        }

        # --- Buildings ---
        total_bld = len(bld_gdf)
        flooded_bld = len(gpd.sjoin(
            bld_gdf, d_flood[["geometry"]],
            how="inner", predicate="intersects"))
        row["buildings_total"]   = total_bld
        row["buildings_flooded"] = flooded_bld
        row["buildings_pct"]     = round(100 * flooded_bld / max(total_bld, 1), 2)
        print(f"    Buildings  : {flooded_bld}/{total_bld} "
              f"({row['buildings_pct']:.1f}%)")

        # --- Roads ---
        rds = _load_district_union(str(gpkg_paths["roads"]), layer, crs)
        total_len_m  = rds.geometry.length.sum()
        flood_rds    = gpd.overlay(rds, d_flood, how="intersection",
                                   keep_geom_type=False)
        flood_len_m  = flood_rds.geometry.length.sum()
        pct_roads    = 100 * flood_len_m / max(total_len_m, 1)
        row["roads_total_km"]  = round(total_len_m / 1000, 3)
        row["roads_flooded_km"] = round(flood_len_m / 1000, 3)
        row["roads_flooded_pct"] = round(pct_roads, 2)
        print(f"    Roads      : {row['roads_flooded_km']:.2f} / "
              f"{row['roads_total_km']:.2f} km ({pct_roads:.1f}%)")

        # --- Hospitals (includes clinics — see Phase 1 note) ---
        hos = _load_district_union(str(gpkg_paths["hospitals"]), layer, crs)
        total_hos   = len(hos)
        flooded_hos = len(gpd.sjoin(
            hos, d_flood[["geometry"]], how="inner", predicate="intersects"))
        row["hospitals_total"]   = total_hos
        row["hospitals_flooded"] = flooded_hos
        row["hospitals_pct"]     = round(100 * flooded_hos / max(total_hos, 1), 2)
        print(f"    Hospitals  : {flooded_hos}/{total_hos}")

        # --- Schools ---
        sch = _load_district_union(str(gpkg_paths["schools"]), layer, crs)
        total_sch   = len(sch)
        flooded_sch = len(gpd.sjoin(
            sch, d_flood[["geometry"]], how="inner", predicate="intersects"))
        row["schools_total"]   = total_sch
        row["schools_flooded"] = flooded_sch
        row["schools_pct"]     = round(100 * flooded_sch / max(total_sch, 1), 2)
        print(f"    Schools    : {flooded_sch}/{total_sch}")

        rows.append(row)

    # Totals row
    df = pd.DataFrame(rows)
    totals = {
        "district":          "TOTAL",
        "flood_area_km2":    round(df["flood_area_km2"].sum(), 4),
        "buildings_total":   df["buildings_total"].sum(),
        "buildings_flooded": df["buildings_flooded"].sum(),
        "buildings_pct":     round(100 * df["buildings_flooded"].sum() /
                                   max(df["buildings_total"].sum(), 1), 2),
        "roads_total_km":    round(df["roads_total_km"].sum(), 3),
        "roads_flooded_km":  round(df["roads_flooded_km"].sum(), 3),
        "roads_flooded_pct": round(100 * df["roads_flooded_km"].sum() /
                                   max(df["roads_total_km"].sum(), 1), 2),
        "hospitals_total":   df["hospitals_total"].sum(),
        "hospitals_flooded": df["hospitals_flooded"].sum(),
        "hospitals_pct":     round(100 * df["hospitals_flooded"].sum() /
                                   max(df["hospitals_total"].sum(), 1), 2),
        "schools_total":     df["schools_total"].sum(),
        "schools_flooded":   df["schools_flooded"].sum(),
        "schools_pct":       round(100 * df["schools_flooded"].sum() /
                                   max(df["schools_total"].sum(), 1), 2),
    }
    df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df