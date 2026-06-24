"""SRTM GL1 30 m DEM download, clip, and verify.

Source: OpenTopography REST API. A free demo key is used by default; for higher
quotas register at https://opentopography.org and set OPENTOPO_API_KEY.
"""

import json
import os
from pathlib import Path

import requests
from osgeo import gdal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEM_DIR  = PROJECT_ROOT / "data_raw" / "dem"
META_DIR = PROJECT_ROOT / "metadata"
OPENTOPO_API_KEY = os.environ.get("OPENTOPO_API_KEY", "demoapikeyot2022")

# Bounding box — 3 km buffer around study districts (W, S, E, N)
WEST  = 105.750
SOUTH =  20.930
EAST  = 105.910
NORTH =  21.080

RAW_PATH     = DEM_DIR / "hanoi_srtm30m_raw.tif"
CLIPPED_PATH = DEM_DIR / "hanoi_srtm30m_WGS84.tif"

DEM_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# ── DOWNLOAD ───────────────────────────────────────────────────────────────────

def download_dem():
    if RAW_PATH.exists():
        print(f"  Raw DEM already exists, skipping download.\n")
        return

    print("  Downloading SRTM GL1 30m from OpenTopography...")
    print(f"    Bounding box: W={WEST} S={SOUTH} E={EAST} N={NORTH}\n")

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype":    "SRTMGL1",
        "west":       WEST,
        "south":      SOUTH,
        "east":       EAST,
        "north":      NORTH,
        "outputFormat": "GTiff",
        "API_Key":    OPENTOPO_API_KEY,
    }

    with requests.get(url, params=params, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(RAW_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r    {pct:.1f}%  ({downloaded/1e6:.1f} MB / {total/1e6:.1f} MB)", end="", flush=True)

    print(f"\n  Downloaded: {RAW_PATH}\n")

# ── CLIP ───────────────────────────────────────────────────────────────────────

def clip_dem():
    if CLIPPED_PATH.exists():
        print(f"  Clipped DEM already exists, skipping clip.\n")
        return

    print("  Clipping DEM to study bounding box...")

    gdal.Warp(
        CLIPPED_PATH,
        RAW_PATH,
        outputBounds=(WEST, SOUTH, EAST, NORTH),
        dstSRS="EPSG:4326",
        resampleAlg="bilinear",
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"]
    )

    print(f"  Clipped DEM saved: {CLIPPED_PATH}\n")

# ── VERIFY ─────────────────────────────────────────────────────────────────────

def verify_dem():
    print(" Verifying DEM...")

    ds = gdal.Open(str(CLIPPED_PATH))
    if ds is None:
        print("  ERROR: Cannot open clipped DEM. File may be corrupt.")
        return

    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    stats = band.ComputeStatistics(False)
    min_elev, max_elev, mean_elev, std_elev = stats

    pixel_size_x = abs(gt[1]) * 111320          # degrees  approx metres
    pixel_size_y = abs(gt[5]) * 110540

    meta = {
        "file":            CLIPPED_PATH.name,
        "crs":             "EPSG:4326 (WGS84)" if "WGS_1984" in proj or "WGS84" in proj or "4326" in proj else proj,
        "columns":         cols,
        "rows":            rows,
        "pixel_size_deg":  round(abs(gt[1]), 6),
        "pixel_size_m_approx": f"{pixel_size_x:.1f} x {pixel_size_y:.1f} m",
        "bbox_west":       gt[0],
        "bbox_north":      gt[3],
        "bbox_east":       round(gt[0] + gt[1] * cols, 6),
        "bbox_south":      round(gt[3] + gt[5] * rows, 6),
        "elev_min_m":      round(min_elev,  2),
        "elev_max_m":      round(max_elev,  2),
        "elev_mean_m":     round(mean_elev, 2),
        "elev_std_m":      round(std_elev,  2),
        "nodata_value":    band.GetNoDataValue(),
        "file_size_mb":    round(CLIPPED_PATH.stat().st_size / 1e6, 2),
    }

    ds = None  # close

    # Print report
    print(f"\n{'='*50}")
    print("DEM VERIFICATION REPORT")
    print(f"{'='*50}")
    for k, v in meta.items():
        print(f"  {k:<25} {v}")

    # Sanity checks
    print(f"\n{'='*50}")
    print("SANITY CHECKS")
    print(f"{'='*50}")

    checks = {
        "CRS is WGS84":              "4326" in meta["crs"] or "WGS" in meta["crs"],
        "Elevation min > -10 m":     meta["elev_min_m"] > -10,
        "Elevation max < 500 m":     meta["elev_max_m"] < 500,
        "Elevation mean 5–100 m":    5 < meta["elev_mean_m"] < 100,
        "Pixel size ~30m (0.00028°)": abs(meta["pixel_size_deg"] - 0.000278) < 0.00005,
        "Coverage > 100 cols":       meta["columns"] > 100,
        "Coverage > 100 rows":       meta["rows"] > 100,
    }

    all_pass = True
    for check, result in checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}]  {check}")
        if not result:
            all_pass = False

    print(f"\n{'='*50}")
    print("RESULT:", "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED — review above")
    print(f"{'='*50}\n")

    # Save metadata
    meta_path = META_DIR / "dem_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f" DEM metadata saved: {meta_path}")

# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("PHASE 1 — DEM DOWNLOAD & VERIFICATION")
    print("=" * 50 + "\n")

    download_dem()
    clip_dem()
    verify_dem()

    print("\nDone. Next step: ERA5 rainfall download (era5_download.py)")