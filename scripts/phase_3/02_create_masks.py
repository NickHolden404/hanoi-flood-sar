"""
scripts/phase_3/02_create_masks.py

Semi-automated flood mask generation:
  1. Otsu thresholding on VV dB-difference ratio
     — Yagi : Otsu (bimodal distribution, clear flood mode in left tail)
     — Matmo: fixed threshold (9-day temporal gap causes near-symmetric
              distribution; Otsu fires at ~+0.115 dB  ~43% flood,
              physically implausible. Fixed threshold documented as limitation.)
  2. VH dB-difference fixed threshold refinement (< -2.0 dB)
  3. Union of VV and VH detections
  4. Morphological opening (radius=2) + closing (radius=3)
  5. Slope masking (pixels > 5° set to no-flood)
  6. Validity mask applied

Outputs saved to: data_processed/flood_labels/<event>/
  flood_mask.tif     — binary uint8 (0=no flood, 1=flood)
  slope_10m_utm.tif  — slope in degrees at 10m UTM grid
  mask_report.json   — thresholds, flood fraction, QC stats

Usage:
    python scripts/phase_3/02_create_masks.py

After running, visually QC 20-30 tiles per event against the CEMS PDF map
before proceeding to 03_generate_patches.py.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject as warp_reproject
from scipy.ndimage import sobel
from skimage.filters import threshold_otsu
from skimage.morphology import disk, binary_opening, binary_closing
import yaml

ROOT = Path(__file__).resolve().parents[2]


# ── helpers ────────────────────────────────────────────────────────────────

def find_ref_tif(cfg: dict, event: dict) -> Path:
    """Return spatial reference TIF — norm_stack first, raw stack fallback."""
    candidates = [
        Path(cfg["paths"]["norm_stacks_dir"]) / event["norm_stack_tif"],
        Path(cfg["paths"]["stacks_dir"])      / event["stack_tif"],
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No reference TIF for '{event['name']}'. Tried: {candidates}")


def load_meta(tif_path: Path) -> dict:
    with rasterio.open(str(tif_path)) as src:
        return src.meta.copy()


def apply_otsu(arr: np.ndarray) -> tuple:
    """Otsu threshold on finite values. Returns (binary mask, threshold)."""
    valid = arr[np.isfinite(arr)]
    thr   = float(threshold_otsu(valid))
    return (arr < thr).astype(np.uint8), thr


def morphological_clean(mask: np.ndarray,
                        open_r: int, close_r: int) -> np.ndarray:
    mask = binary_opening(mask, disk(open_r))
    mask = binary_closing(mask, disk(close_r))
    return mask.astype(np.uint8)


# ── slope computation ──────────────────────────────────────────────────────

def compute_slope(dem_path: str, ref_tif: Path, out_path: Path):
    """Reproject SRTM DEM to stack grid and compute Sobel slope in degrees."""
    with rasterio.open(str(ref_tif)) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_h, dst_w  = ref.height, ref.width
        meta          = ref.meta.copy()

    with rasterio.open(dem_path) as src:
        dem_arr = np.zeros((dst_h, dst_w), dtype=np.float32)
        warp_reproject(
            source        = rasterio.band(src, 1),
            destination   = dem_arr,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = dst_transform,
            dst_crs       = dst_crs,
            resampling    = Resampling.bilinear,
        )

    # Mask SRTM nodata before slope computation
    dem_arr = np.where(dem_arr > -500, dem_arr, np.nan)
    dem_arr = np.nan_to_num(dem_arr, nan=float(np.nanmean(dem_arr[dem_arr > -500])))

    # Sobel gradient — pixel spacing = 10m
    dx = sobel(dem_arr, axis=1) / (8 * 10.0)
    dy = sobel(dem_arr, axis=0) / (8 * 10.0)
    slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta.update({"count": 1, "dtype": "float32", "nodata": -1})
    with rasterio.open(str(out_path), "w", **meta) as dst:
        dst.write(slope_deg[np.newaxis, ...])

    print(f"  Slope: max={slope_deg.max():.2f}°  "
          f"mean={slope_deg.mean():.2f}°   {out_path.name}")


# ── main mask creation ─────────────────────────────────────────────────────

def create_flood_mask(event: dict, cfg: dict) -> dict:
    lc   = cfg["labeling"]
    name = event["name"]

    # Output directory — data_processed/flood_labels/<event>/
    out_dir = Path(cfg["paths"]["data_processed"]) / "flood_labels" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load raw dB stack (Phase 2 band order: VV_pre, VV_post, VH_pre, VH_post, mask)
    raw_npy = Path(cfg["paths"]["stacks_dir"]) / event["stack_npy"]
    stack   = np.load(str(raw_npy)).astype(np.float32)  # 5 × H × W

    vv_pre   = stack[0];  vv_post = stack[1]
    vh_pre   = stack[2];  vh_post = stack[3]
    validity = stack[4]

    # dB-difference ratios (Post - Pre); negative = backscatter decrease = flood signal
    vv_ratio = np.where(validity == 1, vv_post - vv_pre, np.nan)
    vh_ratio = np.where(validity == 1, vh_post - vh_pre, np.nan)

    # Spatial reference
    ref_tif = find_ref_tif(cfg, event)
    meta    = load_meta(ref_tif)

    # Step 1 — VV dB-difference threshold
    # Both events use fixed thresholds (config-driven):
    #   Yagi : -0.5 dB  — Otsu at -1.05 dB was too conservative vs. visible dark areas
    #   Matmo: -1.0 dB  — loosened from -1.5 dB; combined with morph_open_radius=1
    #                      to recover small real flood patches killed by aggressive opening
    vv_arr      = np.nan_to_num(vv_ratio, nan=0.0)
    thr_key     = f"{name}_vv_fixed_threshold"
    otsu_fallback = name != "matmo"  # keep Otsu as fallback for unknown events
    if thr_key in lc:
        vv_thr  = float(lc[thr_key])
        vv_mask = (vv_arr < vv_thr).astype(np.uint8)
        print(f"  VV fixed threshold ({name:>5}): {vv_thr:+.4f} dB  [config override]")
    elif otsu_fallback:
        vv_mask, vv_thr = apply_otsu(vv_arr)
        print(f"  Otsu VV dB-diff threshold : {vv_thr:+.4f} dB")
    else:
        vv_thr  = -1.5
        vv_mask = (vv_arr < vv_thr).astype(np.uint8)
        print(f"  VV fixed threshold (default): {vv_thr:+.4f} dB")

    # Step 2 — VH fixed threshold
    vh_thr  = float(lc["vh_ratio_threshold"])
    vh_mask = (np.nan_to_num(vh_ratio, nan=0.0) < vh_thr).astype(np.uint8)
    print(f"  VH fixed threshold        : {vh_thr:+.1f} dB")

    # Step 3 — Union
    combined = ((vv_mask == 1) | (vh_mask == 1)).astype(np.uint8)
    print(f"  After union               : {combined.sum():,} flood px "
          f"({combined.mean():.2%})")

    # Step 4 — Morphological cleaning
    combined = morphological_clean(
        combined, lc["morph_open_radius"], lc["morph_close_radius"])
    print(f"  After morphological clean : {combined.sum():,} flood px "
          f"({combined.mean():.2%})")

    # Step 5 — Slope masking
    slope_path = out_dir / "slope_10m_utm.tif"
    if not slope_path.exists():
        print(f"  Computing slope from DEM...")
        compute_slope(cfg["paths"]["dem_raw"], ref_tif, slope_path)

    with rasterio.open(str(slope_path)) as s:
        slope = s.read(1).astype(np.float32)
    combined = np.where(slope > lc["max_slope_deg"], 0, combined).astype(np.uint8)
    steep_removed = int((slope > lc["max_slope_deg"]).sum())
    print(f"  Slope mask (>{lc['max_slope_deg']}°)        : "
          f"{steep_removed:,} px removed")

    # Step 6 — Validity mask
    combined[validity != 1] = 0

    # Final stats
    flood_px   = int(combined.sum())
    total_px   = int(validity.sum())
    flood_frac = flood_px / max(total_px, 1)
    print(f"  Final flood pixels        : {flood_px:,} ({flood_frac:.2%})")

    # Save flood mask GeoTIFF
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "uint8", "nodata": 255})
    mask_path = out_dir / "flood_mask.tif"
    with rasterio.open(str(mask_path), "w", **out_meta) as dst:
        dst.write(combined[np.newaxis, ...])
    print(f"  Mask saved                : {mask_path}")

    # QC report
    report = {
        "event":                      name,
        "vv_threshold_method":        "fixed" if thr_key in lc else "otsu",
        "vv_otsu_db_diff_threshold":  round(vv_thr, 4),
        "vh_fixed_db_diff_threshold": vh_thr,
        "max_slope_deg":              lc["max_slope_deg"],
        "flood_pixels":               flood_px,
        "valid_pixels":               total_px,
        "flood_fraction":             round(flood_frac, 4),
        "flood_area_km2":             round(flood_px * 100 / 1e6, 4),  # 10m px  100m²
        "nodata_fraction":            0.0,  # Phase 2 confirmed 100% valid
        "note": ("Matmo post-event 9 days post-peak — Otsu unusable (symmetric "
                 "distribution, no flood mode). Fixed VV threshold of -1.5 dB used. "
                 "Flood fraction expected conservative (systematic underestimation)."
                 if name == "matmo" else
                 "Yagi post-event 3 days post-peak — labels closest to peak flood extent."),
    }
    with open(out_dir / "mask_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"  QC report saved           : {out_dir / 'mask_report.json'}")

    return report


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    all_reports = {}
    for event in cfg["events"]:
        print(f"\n{'='*55}")
        print(f"  [{event['name'].upper()}] Generating flood mask")
        print(f"{'='*55}")
        report = create_flood_mask(event, cfg)
        all_reports[event["name"]] = report

    print(f"\n{'='*55}")
    print("FLOOD MASK SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Event':<10} {'Flood px':>12} {'Flood %':>10} {'Area km²':>10} "
          f"{'VV thr':>13} {'Method':>8}")
    print(f"  {'-'*66}")
    for name, r in all_reports.items():
        print(f"  {name:<10} {r['flood_pixels']:>12,} "
              f"{r['flood_fraction']:>9.2%} "
              f"{r['flood_area_km2']:>10.3f} "
              f"{r['vv_otsu_db_diff_threshold']:>+13.4f} dB"
              f"  {r['vv_threshold_method']:>8}")

    print(f"\n Masks complete.")
    print(f"   Location: data_processed/flood_labels/")
    print(f"\n  QC REQUIRED before proceeding:")
    print(f"   1. Open CEMS PDF map (data_raw/cems/EMSR754_AOI03_DEL_PRODUCT_v1/Maps/)")
    print(f"   2. Compare Yagi flood_mask.tif visually against CEMS flood polygons")
    print(f"   3. Check 20-30 locations for false positives (building shadow)")
    print(f"      and false negatives (shallow urban flooding)")
    print(f"   4. Record findings before running 03_generate_patches.py")