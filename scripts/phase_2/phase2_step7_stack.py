"""
Step 7: Assemble 4-channel raster stacks per event.

Channel order:
  1 — Pre-event  VV (dB, speckle-filtered, terrain-corrected, clipped)
  2 — Post-event VV
  3 — Pre-event  VH
  4 — Post-event VH

Output per event:
  - {event}_stack.tif  (float32 multi-band GeoTIFF)
  - {event}_stack.npy  (C×H×W float32 NumPy array)
  - {event}_stack_meta.json
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
import json, logging, time
from pathlib import Path
from phase2_config import EVENTS, DIRS

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

BAND_DESCRIPTIONS = [
    "Pre-event VV dB",
    "Post-event VV dB",
    "Pre-event VH dB",
    "Post-event VH dB"
]


def load_band(tif_path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return data, profile


def build_stack(event_name: str, pre_key: str, post_key: str):
    tif_out = DIRS["stacks"] / f"{event_name}_stack.tif"
    npy_out = DIRS["stacks"] / f"{event_name}_stack.npy"
    meta_out = DIRS["stacks"] / f"{event_name}_stack_meta.json"

    if tif_out.exists() and npy_out.exists():
        print(f"    {event_name}: stack exists, skipping")
        return

    band_paths = [
        DIRS["clipped"] / f"{pre_key}_VV_db_spk_clip.tif",
        DIRS["clipped"] / f"{post_key}_VV_db_spk_clip.tif",
        DIRS["clipped"] / f"{pre_key}_VH_db_spk_clip.tif",
        DIRS["clipped"] / f"{post_key}_VH_db_spk_clip.tif",
    ]

    # Verify all exist
    for p in band_paths:
        if not p.exists():
            raise FileNotFoundError(f"Band file missing: {p}")

    # Load all bands and reference profile from first band
    bands = []
    ref_profile = None
    for i, p in enumerate(band_paths):
        data, profile = load_band(p)
        if ref_profile is None:
            ref_profile = profile

        # Verify dimensions match
        assert data.shape == bands[0].shape if bands else True, \
            f"Shape mismatch at band {i+1}: {data.shape} vs {bands[0].shape}"

        # Replace NaN with small constant (-30 dB = approx open water)
        nan_count = np.isnan(data).sum()
        if nan_count > 0:
            print(f"    Band {i+1}: replacing {nan_count} NaN values with -30.0 dB")
            data = np.where(np.isnan(data), -30.0, data)

        bands.append(data)

    # Shape: (4, H, W)
    stack = np.stack(bands, axis=0)
    assert stack.dtype == np.float32
    assert not np.any(np.isnan(stack)), "NaN values remain in stack!"

    H, W = stack.shape[1], stack.shape[2]

    # ── Save GeoTIFF ──────────────────────────────────────────────────────
    out_profile = ref_profile.copy()
    out_profile.update({
        "count": 4, "dtype": "float32",
        "compress": "lzw", "nodata": None
    })
    with rasterio.open(tif_out, "w", **out_profile) as dst:
        dst.write(stack)
        for i, desc in enumerate(BAND_DESCRIPTIONS, 1):
            dst.update_tags(i, description=desc)

    # ── Save NumPy ────────────────────────────────────────────────────────
    np.save(npy_out, stack)

    # ── Statistics per band ───────────────────────────────────────────────
    band_stats = []
    for i, desc in enumerate(BAND_DESCRIPTIONS):
        b = stack[i]
        band_stats.append({
            "band": i + 1,
            "description": desc,
            "min": float(b.min()), "max": float(b.max()),
            "mean": float(b.mean()), "std": float(b.std()),
            "shape": [H, W]
        })

    meta = {
        "event": event_name,
        "pre_scene": pre_key, "post_scene": post_key,
        "shape": list(stack.shape),   # [4, H, W]
        "dtype": str(stack.dtype),
        "crs": ref_profile["crs"].to_string(),
        "pixel_size_m": abs(ref_profile["transform"].a),
        "nan_values": int(np.isnan(stack).sum()),
        "band_order": BAND_DESCRIPTIONS,
        "band_stats": band_stats
    }

    with open(meta_out, "w") as f:
        json.dump(meta, f, indent=2)

    logging.info(f"STACK OK | {event_name} | shape={stack.shape} | "
                 f"tif={tif_out} | npy={npy_out}")
    print(f"   {event_name}: stack {stack.shape} saved")
    print(f"     TIF:  {tif_out}")
    print(f"     NPY:  {npy_out}")
    for s in band_stats:
        print(f"     Band {s['band']} ({s['description'][:20]:20s}): "
              f"mean={s['mean']:6.2f}  std={s['std']:5.2f}  "
              f"[{s['min']:6.1f}, {s['max']:5.1f}] dB")
    return meta


if __name__ == "__main__":
    print("\n=== STEP 7: MULTI-CHANNEL STACK ASSEMBLY ===\n")
    t_start = time.time()
    all_meta = {}

    for event_name, event in EVENTS.items():
        print(f"\n[{event_name}]")
        m = build_stack(event_name, event["pre"], event["post"])
        all_meta[event_name] = m

    with open(DIRS["logs"] / "step7_stack_meta.json", "w") as f:
        json.dump(all_meta, f, indent=2)

    print(f"\n Step 7 complete — {(time.time()-t_start)/60:.1f} min")
    print(f"\nFinal stack files:")
    for event_name in EVENTS:
        print(f"  {DIRS['stacks'] / f'{event_name}_stack.tif'}")
        print(f"  {DIRS['stacks'] / f'{event_name}_stack.npy'}")