"""Create AOI shapefile directly from the Phase 1 bounding box."""

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / "data_raw" / "osm"
OUT_SHP      = OUT_DIR / "hanoi_aoi.shp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 1 bounding box (WGS84)
LON_MIN, LON_MAX = 105.770, 105.910
LAT_MIN, LAT_MAX = 20.940, 21.055

# 1. Create bounding box in WGS84
bbox_wgs84 = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
gdf_wgs84  = gpd.GeoDataFrame(
    {"name": ["Hanoi_StudyArea"]},
    geometry=[bbox_wgs84],
    crs="EPSG:4326"
)

# 2. Reproject to UTM Zone 48N
gdf_utm = gdf_wgs84.to_crs("EPSG:32648")

# 3. Add 2.5 km buffer
gdf_buffered = gdf_utm.copy()
gdf_buffered["geometry"] = gdf_utm.geometry.buffer(2500)

# 4. Save both versions
out_utm = OUT_DIR / "hanoi_aoi_utm_buffered.shp"
gdf_buffered.to_file(out_utm)
gdf_wgs84.to_file(OUT_SHP)

print(f" WGS84 AOI saved:    {OUT_SHP}")
print(f" UTM buffered saved: {out_utm}")
print(f"\nBounding box (WGS84):")
print(f"  {LON_MIN}–{LON_MAX}°E, {LAT_MIN}–{LAT_MAX}°N")
print(f"\nBuffered extent (UTM 48N):")
print(f"  {gdf_buffered.total_bounds}")