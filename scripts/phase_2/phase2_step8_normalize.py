"""
Step 8: Normalize 4-channel stacks.

Method: Z-score normalization (Standard Score)

Rationale:
  - SAR backscatter in dB is approximately Gaussian per land cover class
  - Z-score preserves relative contrast between pre/post-event changes
    (flood signal) better than min-max, which is dominated by outliers
  - Min-max would compress the dynamic range in dense urban areas where
    bright double-bounce returns are extreme outliers
  - Z-score is standard in SAR-based deep learning (SpaceNet 6, Sen1Floods11)
  - Statistics computed PER CHANNEL (not globally) to preserve
    inter-channel relationships

Output per event:
  - {event}_norm_stack.tif
  - {event}_norm_stack.npy
  - {event}_norm_params.json    statistics for inverse transform
"""

import numpy as np
import rasterio
import json, logging, time
from pathlib import Path
from phase2_config import EVENTS, DIRS, NORM_METHOD

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def zscore_normalize(stack: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Z-score normalize each channel independently.
    stack shape: (C, H, W)
    Returns normalized array and params dict.
    """
    C = stack.shape[0]
    params = {"method": "zscore", "channels": []}
    norm = np.zeros_like(stack, dtype=np.float32)

    for c in range(C):
        band = stack[c]
        mu = float(np.mean(band))
        sigma = float(np.std(band))
        if sigma < 1e-8:
            sigma = 1.0   # prevent division by zero (flat band)
        norm[c] = ((band - mu) / sigma).astype(np.float32)
        params["channels"].append({
            "channel": c + 1,
            "mean": mu,
            "std": sigma,
            "min_raw": float(band.min()),
            "max_raw": float(band.max()),
            "min_norm": float(norm[c].min()),
            "max_norm": float(norm[c].max()),
        })
    return norm, params


def minmax_normalize(stack: np.ndarray) -> tuple[np.ndarray, dict]:
    """Min-max normalize each channel to [0, 1]."""
    C = stack.shape[0]
    params = {"method": "minmax", "channels": []}
    norm = np.zeros_like(stack, dtype=np.float32)

    for c in range(C):
        band = stack[c]
        vmin = float(np.percentile(band, 1))    # robust min (1st percentile)
        vmax = float(np.percentile(band, 99))   # robust max (99th percentile)
        norm[c] = np.clip((band - vmin) / (vmax - vmin + 1e-8), 0, 1)
        params["channels"].append({
            "channel": c + 1,
            "vmin": vmin, "vmax": vmax
        })
    return norm, params


def normalize_event(event_name: str):
    stack_npy = DIRS["stacks"] / f"{event_name}_stack.npy"
    stack_tif = DIRS["stacks"] / f"{event_name}_stack.tif"
    norm_npy   = DIRS["normalized"] / f"{event_name}_norm_stack.npy"
    norm_tif   = DIRS["normalized"] / f"{event_name}_norm_stack.tif"
    params_json = DIRS["normalized"] / f"{event_name}_norm_params.json"

    if norm_tif.exists() and norm_npy.exists():
        print(f"    {event_name}: normalized stack exists, skipping")
        return

    stack = np.load(stack_npy)
    print(f"  Loaded stack: {stack.shape} {stack.dtype}")

    # Normalize
    t0 = time.time()
    if NORM_METHOD == "zscore":
        norm, params = zscore_normalize(stack)
    elif NORM_METHOD == "minmax":
        norm, params = minmax_normalize(stack)
    else:
        raise ValueError(f"Unknown normalization method: {NORM_METHOD}")

    params["event"] = event_name
    params["input_file"] = str(stack_npy)

    # Save NPY
    np.save(norm_npy, norm)

    # Save GeoTIFF (inherit profile from raw stack TIF)
    with rasterio.open(stack_tif) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", compress="lzw")
    with rasterio.open(norm_tif, "w", **profile) as dst:
        dst.write(norm)

    # Save normalization parameters
    with open(params_json, "w") as f:
        json.dump(params, f, indent=2)

    logging.info(f"NORM OK | {event_name} | method={NORM_METHOD} | "
                 f"shape={norm.shape} | elapsed={time.time()-t0:.1f}s")
    print(f"   {event_name}: normalized ({NORM_METHOD}) in {time.time()-t0:.1f}s")
    for ch in params["channels"]:
        print(f"     Ch{ch['channel']}: mean={ch.get('mean', 'n/a'):.3f}  "
              f"std={ch.get('std', 'n/a'):.3f}  "
              f"raw=[{ch.get('min_raw', '?'):.1f}, {ch.get('max_raw', '?'):.1f}] dB")

    return params


if __name__ == "__main__":
    print("\n=== STEP 8: NORMALIZATION ===")
    print(f"    Method: {NORM_METHOD.upper()}")
    print(f"    Rationale: Z-score preserves SAR dB contrast; "
          f"robust to urban double-bounce outliers\n")

    t_start = time.time()
    all_params = {}

    for event_name in EVENTS:
        print(f"\n[{event_name}]")
        p = normalize_event(event_name)
        all_params[event_name] = p

    combined_params = DIRS["normalized"] / "all_normalization_params.json"
    with open(combined_params, "w") as f:
        json.dump(all_params, f, indent=2)

    print(f"\n Step 8 complete — {(time.time()-t_start)/60:.1f} min")
    print(f"   Parameters saved: {combined_params}")