# Phase 2 master configuration: Multi-Event SAR Flood Segmentation, Hanoi.
# Override paths via environment variables if your layout differs.

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAR_ROOT     = Path(os.environ.get("SAR_ROOT", PROJECT_ROOT / "data_raw" / "sentinel1"))
SNAP_GPT     = Path(os.environ.get("SNAP_GPT", "gpt"))  # assume on PATH unless overridden

DATA_RAW     = PROJECT_ROOT / "data_raw"
DATA_PROC    = PROJECT_ROOT / "data_processed"
DEM_PATH     = DATA_RAW / "dem" / "hanoi_srtm30m_WGS84.tif"
DISTRICT_SHP = DATA_RAW / "osm" / "hanoi_aoi_utm_buffered.shp"

DIRS = {
    "orbit"     : DATA_PROC / "orbit_corrected",
    "noise"     : DATA_PROC / "noise_removed",
    "calibrated": DATA_PROC / "calibrated",
    "speckle"   : DATA_PROC / "speckle_filtered",
    "terrain"   : DATA_PROC / "terrain_corrected",
    "clipped"   : DATA_PROC / "clipped",
    "stacks"    : DATA_PROC / "final_stacks",
    "normalized": DATA_PROC / "normalized",
    "qc"        : DATA_PROC / "qc_reports",
    "graphs"    : DATA_PROC / "snap_graphs",
    "logs"      : DATA_PROC / "logs",
}

# SAR scene inventory used in the paper.
SCENES = {
    "yagi_pre"  : SAR_ROOT / "S1A_IW_GRDH_1SDV_20240831T225114_20240831T225139_055463_06C40A_A2E8.SAFE",
    "yagi_post" : SAR_ROOT / "S1A_IW_GRDH_1SDV_20240912T225114_20240912T225139_055638_06CAF6_16C8.SAFE",
    "matmo_pre" : SAR_ROOT / "S1A_IW_GRDH_1SDV_20250922T110625_20250922T110650_061100_079DAF_B6AF.SAFE",
    "matmo_post": SAR_ROOT / "S1A_IW_GRDH_1SDV_20251016T110625_20251016T110650_061450_07ABC9_87B2.SAFE",
}

EVENTS = {
    "2024_yagi" : {"pre": "yagi_pre",  "post": "yagi_post"},
    "2025_matmo": {"pre": "matmo_pre", "post": "matmo_post"},
}

# ── Processing parameters ─────────────────────────────────────────────────────
CRS_UTM        = "EPSG:32648"   # UTM Zone 48N / WGS84
PIXEL_SPACING  = 10.0           # metres
AOI_BUFFER_M   = 2500           # 2.5 km buffer (already applied in hanoi_aoi_utm_buffered.shp)
SPECKLE_FILTER = "Refined-Lee"
SPECKLE_WINDOW = 5
NORM_METHOD    = "zscore"       # 'zscore' or 'minmax'

# ── AOI bounding box (WGS84, Phase 1) ────────────────────────────────────────
AOI_WGS84 = {
    "lon_min": 105.770, "lon_max": 105.910,
    "lat_min": 20.940,  "lat_max": 21.055,
}