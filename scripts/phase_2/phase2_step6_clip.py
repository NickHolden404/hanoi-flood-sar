"""
Step 6: Clip all speckle-filtered dB rasters to buffered AOI.
  - Load district boundaries (GADM Level 3 — Hoàn Kiếm, Thanh Xuân, Hoàng Mai)
  - Buffer by 2.5 km in UTM Zone 48N
  - Clip all 8 bands to identical extent
  - Verify: same dimensions, same CRS, same transform for pre/post pairs
"""

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.merge import merge
import geopandas as gpd
from pathlib import Path
import logging, json, time
from shapely.geometry import mapping, box
from phase2_config import SCENES, EVENTS, DIRS, DISTRICT_SHP, CRS_UTM, AOI_BUFFER_M

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

TARGET_DISTRICTS = ["Hoàn Kiếm", "Thanh Xuân", "Hoàng Mai"]
# GADM NAME_3 variants (Vietnamese diacritics may differ — adjust if needed)
GADM_DISTRICT_NAMES = [
    "Hoàn Kiếm", "Hoan Kiem",
    "Thanh Xuân", "Thanh Xuan",
    "Hoàng Mai", "Hoang Mai"
]


def load_aoi_polygon():
    gdf = gpd.read_file(DIRS["clipped"].parent.parent / "data_raw/osm/hanoi_aoi_utm_buffered.shp")
    return gdf.geometry.values[0]

    # Filter to target districts — try NAME_3 column (GADM Vietnam L3)
    name_col = None
    for col in ["NAME_3", "NAME_2", "name", "DIST_NAME"]:
        if col in gdf.columns:
            name_col = col
            break
    if name_col is None:
        raise ValueError(f"Cannot find name column. Available: {list(gdf.columns)}")

    mask = gdf[name_col].str.normalize("NFC").isin(
        [n for n in GADM_DISTRICT_NAMES]
    )
    if mask.sum() == 0:
        print(f"  WARNING: No exact match for district names.")
        print(f"  Available: {gdf[name_col].unique()[:20]}")
        print(f"  Falling back to bounding box AOI")
        return None

    districts = gdf[mask].copy()
    print(f"  Matched {len(districts)} district features")

    # Reproject to UTM for metric buffering
    districts_utm = districts.to_crs(CRS_UTM)
    dissolved = districts_utm.dissolve()
    buffered = dissolved.geometry.buffer(AOI_BUFFER_M)
    aoi_geom = buffered.values[0]

    print(f"  AOI buffered extent (UTM): {aoi_geom.bounds}")
    return aoi_geom


def fallback_aoi_polygon():
    """Fallback: bounding-box AOI from phase2_config, reprojected to UTM."""
    import pyproj
    from shapely.ops import transform as shp_transform
    from phase2_config import AOI_WGS84

    bb = box(AOI_WGS84["lon_min"], AOI_WGS84["lat_min"],
             AOI_WGS84["lon_max"], AOI_WGS84["lat_max"])

    proj = pyproj.Transformer.from_crs("EPSG:4326", CRS_UTM, always_xy=True).transform
    bb_utm = shp_transform(proj, bb)

    # Add buffer
    return bb_utm.buffer(AOI_BUFFER_M)


def clip_raster(input_tif: Path, output_tif: Path, aoi_geom):
    """Clip raster to AOI geometry, crop to extent."""
    if output_tif.exists():
        return

    with rasterio.open(input_tif) as src:
        geom_json = [mapping(aoi_geom)]
        out_img, out_transform = rio_mask(src, geom_json, crop=True, nodata=np.nan)
        out_profile = src.profile.copy()
        out_profile.update({
            "height": out_img.shape[1],
            "width":  out_img.shape[2],
            "transform": out_transform,
            "dtype": "float32",
            "compress": "lzw",
            "nodata": np.nan
        })
        with rasterio.open(output_tif, "w", **out_profile) as dst:
            dst.write(out_img)


def verify_alignment(tif_list: list[Path]) -> dict:
    """Verify all rasters share the same transform and dimensions."""
    ref = None
    results = {"aligned": True, "files": []}
    for p in tif_list:
        with rasterio.open(p) as src:
            info = {
                "file": p.name,
                "crs": src.crs.to_epsg(),
                "width": src.width, "height": src.height,
                "transform": list(src.transform)[:6],
                "dtype": src.dtypes[0]
            }
            results["files"].append(info)
            if ref is None:
                ref = info
            else:
                if (info["width"] != ref["width"] or
                    info["height"] != ref["height"] or
                    info["transform"] != ref["transform"]):
                    results["aligned"] = False
    return results


if __name__ == "__main__":
    print("\n=== STEP 6: CLIP TO STUDY AREA ===\n")
    t_start = time.time()

    # Load AOI
    print("Loading district boundaries...")
    aoi = load_aoi_polygon()
    if aoi is None:
        print("Using fallback bounding box AOI")
        aoi = fallback_aoi_polygon()

    # Clip all 8 bands
    print("\nClipping rasters to AOI...")
    all_clipped = []
    for key in SCENES:
        for pol in ["VV", "VH"]:
            src_tif = DIRS["speckle"] / f"{key}_{pol}_db_spk.tif"
            dst_tif = DIRS["clipped"] / f"{key}_{pol}_db_spk_clip.tif"
            print(f"  ⚙  {key} {pol}")
            clip_raster(src_tif, dst_tif, aoi)
            all_clipped.append(dst_tif)
            print(f"   {dst_tif.name}")

    # Verify alignment within each event
    print("\nVerifying spatial alignment...")
    for event_name, event in EVENTS.items():
        pre_key  = event["pre"]
        post_key = event["post"]
        event_tifs = [
            DIRS["clipped"] / f"{pre_key}_VV_db_spk_clip.tif",
            DIRS["clipped"] / f"{pre_key}_VH_db_spk_clip.tif",
            DIRS["clipped"] / f"{post_key}_VV_db_spk_clip.tif",
            DIRS["clipped"] / f"{post_key}_VH_db_spk_clip.tif",
        ]
        result = verify_alignment(event_tifs)
        status = " ALIGNED" if result["aligned"] else " MISALIGNED — CHECK OUTPUT"
        print(f"  {event_name}: {status}")
        if not result["aligned"]:
            logging.error(f"ALIGNMENT FAIL | {event_name} | {result}")

    with open(DIRS["logs"] / "step6_alignment.json", "w") as f:
        json.dump({e: verify_alignment([
            DIRS["clipped"] / f"{EVENTS[e]['pre']}_VV_db_spk_clip.tif"
        ]) for e in EVENTS}, f, indent=2)

    print(f"\n Step 6 complete — {(time.time()-t_start)/60:.1f} min")