"""
Step 4b: Apply Refined Lee speckle filter to terrain-corrected linear Sigma0.
Converts to dB AFTER speckle filtering (linear-domain filtering is correct practice).

Processing order: TC_linear  speckle filter (linear)  convert to dB  save

This is the correct sequence because:
  1. Speckle statistics are multiplicative in linear scale
  2. Lee filter is derived assuming multiplicative noise model
  3. dB conversion (log) is applied AFTER filtering
"""

import numpy as np
import rasterio
from pathlib import Path
import logging, json, time
from phase2_config import SCENES, DIRS, SPECKLE_FILTER, SPECKLE_WINDOW

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

def refined_lee_linear(img: np.ndarray, win: int = 5) -> np.ndarray:
    """Refined Lee filter for LINEAR scale SAR intensity."""
    from scipy.ndimage import uniform_filter
    img = img.astype(np.float64)
    nodata_mask = (img <= 0) | np.isnan(img)
    img_safe = np.where(nodata_mask, np.nanmedian(img[~nodata_mask]), img)

    mu = uniform_filter(img_safe, size=win)
    mu_sq = uniform_filter(img_safe ** 2, size=win)
    var_local = np.maximum(mu_sq - mu ** 2, 0)

    ENL = 4.4
    var_speckle = mu ** 2 / ENL
    b = var_local / np.where(var_local + var_speckle > 0, var_local + var_speckle, 1e-12)
    filtered = mu + b * (img_safe - mu)
    filtered = np.where(nodata_mask, np.nan, filtered)
    return np.maximum(filtered, 1e-10).astype(np.float32)


def process_band(scene_key: str, pol: str):
    linear_tif = DIRS["terrain"]  / f"{scene_key}_{pol}_linear.tif"
    out_db_tif = DIRS["speckle"] / f"{scene_key}_{pol}_db_spk.tif"

    if out_db_tif.exists():
        print(f"    {scene_key} {pol}: exists, skip")
        return

    t0 = time.time()
    with rasterio.open(linear_tif) as src:
        profile = src.profile.copy()
        linear = src.read(1)

    # 1. Filter in linear domain
    filtered_linear = refined_lee_linear(linear, win=SPECKLE_WINDOW)

    # 2. Convert to dB
    db = 10.0 * np.log10(np.where(filtered_linear > 0, filtered_linear, 1e-10))
    db = np.where(np.isfinite(db), db, np.nan)

    profile.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(out_db_tif, "w", **profile) as dst:
        dst.write(db[np.newaxis, :, :])

    stats = {
        "min_db": float(np.nanmin(db)), "max_db": float(np.nanmax(db)),
        "mean_db": float(np.nanmean(db)), "std_db": float(np.nanstd(db)),
        "nan_pct": float(np.isnan(db).mean() * 100),
        "elapsed_s": round(time.time() - t0, 1)
    }
    logging.info(f"SPK_POST_TC | {scene_key}_{pol} | {stats}")
    print(f"   {scene_key} {pol}: "
          f"dB range [{stats['min_db']:.1f}, {stats['max_db']:.1f}]  "
          f"std={stats['std_db']:.3f}  ({stats['elapsed_s']}s)")
    return stats


if __name__ == "__main__":
    print("\n=== STEP 4b: SPECKLE FILTER (post terrain correction) ===")
    print(f"    Method: Refined Lee (linear domain)  dB conversion")
    print(f"    Window: {SPECKLE_WINDOW}×{SPECKLE_WINDOW}\n")
    all_stats = {}
    for key in SCENES:
        print(f"\n[{key}]")
        for pol in ["VV", "VH"]:
            s = process_band(key, pol)
            all_stats[f"{key}_{pol}"] = s
    with open(DIRS["logs"] / "step4b_speckle_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print("\n Step 4b complete")