"""
scripts/phase_3/01_build_stacks.py

Converts Phase 2 normalized stacks (5-channel) into 7-channel model-ready stacks:
  Ch1  VV_pre     Phase 2 Ch1 (already Z-score normalized)
  Ch2  VH_pre     Phase 2 Ch3 (NOTE: Phase 2 band order is VV_pre, VV_post, VH_pre, VH_post)
  Ch3  VV_post    Phase 2 Ch2
  Ch4  VH_post    Phase 2 Ch4
  Ch5  VV_ratio   (VV_post_dB - VV_pre_dB) from raw dB stack, then Z-score
  Ch6  VH_ratio   (VH_post_dB - VH_pre_dB) from raw dB stack, then Z-score
  Ch7  DEM        SRTM resampled to 10m UTM 48N grid, Z-score normalized

Outputs saved to: data_processed/model_stacks/
  <event>_model_stack_7ch.tif
  <event>_model_stack_7ch.npy
  <event>_validity_mask.npy
  <event>_extra_norm_params.json

Usage:
    python scripts/phase_3/01_build_stacks.py
"""
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Phase 2 band indices (0-based) in norm_stack and raw_stack
# Order: [Pre_VV=0, Post_VV=1, Pre_VH=2, Post_VH=3, ValidityMask=4]
BAND_IDX = {
    "VV_pre":   0,
    "VV_post":  1,
    "VH_pre":   2,
    "VH_post":  3,
    "validity": 4,
}


# ── loaders ────────────────────────────────────────────────────────────────

def load_norm_stack(npy_path: Path) -> np.ndarray:
    arr = np.load(str(npy_path))
    assert arr.shape[0] == 5, f"Expected 5 bands, got {arr.shape[0]} in {npy_path}"
    return arr.astype(np.float32)


def load_raw_stack(npy_path: Path) -> np.ndarray:
    return np.load(str(npy_path)).astype(np.float32)


# ── spatial reference TIF ─────────────────────────────────────────────────

def find_ref_tif(cfg: dict, event: dict) -> Path:
    """
    Return a GeoTIFF to use as spatial reference for DEM reprojection.
    Tries norm_stack_tif first, falls back to raw stack_tif.
    Raises FileNotFoundError if neither exists.
    """
    candidates = [
        Path(cfg["paths"]["norm_stacks_dir"]) / event["norm_stack_tif"],
        Path(cfg["paths"]["stacks_dir"])      / event["stack_tif"],
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No reference TIF found for event '{event['name']}'. "
        f"Tried: {[str(c) for c in candidates]}"
    )


# ── ratio computation ──────────────────────────────────────────────────────

def compute_ratio(raw_stack: np.ndarray, pol: str,
                  validity: np.ndarray) -> np.ndarray:
    """
    dB difference: Post_dB - Pre_dB.
    Negative values  backscatter decrease  potential flood signal.
    Invalid pixels set to NaN (replaced with 0.0 after normalization).
    """
    ratio = raw_stack[BAND_IDX[f"{pol}_post"]] - raw_stack[BAND_IDX[f"{pol}_pre"]]
    return np.where(validity == 1, ratio, np.nan)


# ── DEM reprojection ───────────────────────────────────────────────────────

def reproject_dem(dem_path: str, ref_tif: Path) -> np.ndarray:
    """Reproject SRTM DEM (30m WGS84) to match the stack grid (10m UTM 48N)."""
    with rasterio.open(str(ref_tif)) as ref:
        dst_crs       = ref.crs
        dst_transform = ref.transform
        dst_height    = ref.height
        dst_width     = ref.width

    with rasterio.open(dem_path) as src:
        dem_out = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source        = rasterio.band(src, 1),
            destination   = dem_out,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = dst_transform,
            dst_crs       = dst_crs,
            resampling    = Resampling.bilinear,
        )
    return dem_out


# ── normalization ──────────────────────────────────────────────────────────

def zscore_channel(arr: np.ndarray,
                   validity: np.ndarray = None) -> tuple:
    """Z-score over valid/finite pixels. Returns (normalized, mu, sigma)."""
    if validity is not None:
        valid_vals = arr[validity == 1]
    else:
        valid_vals = arr[np.isfinite(arr)]

    mu  = float(np.nanmean(valid_vals))
    sig = float(np.nanstd(valid_vals))
    return (arr - mu) / (sig + 1e-8), mu, sig


def clean(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0.0 and cast to float32."""
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ── main builder ───────────────────────────────────────────────────────────

def build_model_stack(cfg: dict, event: dict, save_dir: Path):
    name = event["name"]
    print(f"\n{'='*55}")
    print(f"  [{name.upper()}] Building 7-channel model stack")

    # Paths
    norm_npy = Path(cfg["paths"]["norm_stacks_dir"]) / event["norm_stack_npy"]
    raw_npy  = Path(cfg["paths"]["stacks_dir"])      / event["stack_npy"]
    ref_tif  = find_ref_tif(cfg, event)

    print(f"  Norm stack : {norm_npy}")
    print(f"  Raw stack  : {raw_npy}")
    print(f"  Ref TIF    : {ref_tif}")

    # Load
    norm_stack = load_norm_stack(norm_npy)
    raw_stack  = load_raw_stack(raw_npy)
    validity   = norm_stack[BAND_IDX["validity"]]

    print(f"  Stack shape: {norm_stack.shape}  "
          f"Valid px: {int(validity.sum()):,} / "
          f"{validity.size:,} "
          f"({100*validity.mean():.1f}%)")

    # Ch 1–4: reorder Phase2 bands to model order
    # Phase 2: [VV_pre, VV_post, VH_pre, VH_post, mask]
    # Model  : [VV_pre, VH_pre,  VV_post, VH_post, ...]
    VV_pre_n  = norm_stack[BAND_IDX["VV_pre"]]
    VH_pre_n  = norm_stack[BAND_IDX["VH_pre"]]
    VV_post_n = norm_stack[BAND_IDX["VV_post"]]
    VH_post_n = norm_stack[BAND_IDX["VH_post"]]

    # Ch 5–6: ratio channels from raw dB, then Z-score
    vv_ratio_raw = compute_ratio(raw_stack, "VV", validity)
    vh_ratio_raw = compute_ratio(raw_stack, "VH", validity)
    vv_ratio_n, vv_mu, vv_sig = zscore_channel(vv_ratio_raw, validity)
    vh_ratio_n, vh_mu, vh_sig = zscore_channel(vh_ratio_raw, validity)

    # Ch 7: DEM — reproject then Z-score
    # Mask nodata (SRTM nodata = -32768; also mask any value < -500m as unphysical)
    dem_raw = reproject_dem(cfg["paths"]["dem_raw"], ref_tif)
    dem_valid_mask = (dem_raw > -500).astype(np.float32)
    dem_raw_masked = np.where(dem_valid_mask == 1, dem_raw, np.nan)
    dem_n, dem_mu, dem_sig = zscore_channel(dem_raw_masked)

    # Verify DEM shape matches stack
    H, W = norm_stack.shape[1], norm_stack.shape[2]
    if dem_raw.shape != (H, W):
        raise ValueError(
            f"DEM shape {dem_raw.shape} does not match stack ({H},{W}). "
            "Check DEM reprojection — ref TIF may not cover the full AOI."
        )

    # Assemble 7-channel stack
    stack_7ch = np.stack([
        clean(VV_pre_n),
        clean(VH_pre_n),
        clean(VV_post_n),
        clean(VH_post_n),
        clean(vv_ratio_n),
        clean(vh_ratio_n),
        clean(dem_n),
    ], axis=0)   # 7 × H × W

    assert stack_7ch.shape == (7, H, W), \
        f"Unexpected stack shape: {stack_7ch.shape}"

    # ── save NPY ──────────────────────────────────────────────────────────
    npy_out = save_dir / f"{name}_model_stack_7ch.npy"
    np.save(str(npy_out), stack_7ch)

    # ── save GeoTIFF ──────────────────────────────────────────────────────
    tif_out = save_dir / f"{name}_model_stack_7ch.tif"
    band_names = ["VV_pre", "VH_pre", "VV_post", "VH_post",
                  "VV_ratio", "VH_ratio", "DEM"]
    with rasterio.open(str(ref_tif)) as ref:
        meta = ref.meta.copy()
    meta.update({"count": 7, "dtype": "float32", "nodata": None})
    with rasterio.open(str(tif_out), "w", **meta) as dst:
        for i, (band, bname) in enumerate(zip(stack_7ch, band_names), 1):
            dst.write(band, i)
            dst.update_tags(i, name=bname)

    # ── save validity mask ────────────────────────────────────────────────
    vmask_out = save_dir / f"{name}_validity_mask.npy"
    np.save(str(vmask_out), validity.astype(np.uint8))

    # ── save normalization params ─────────────────────────────────────────
    params = {
        "event":    name,
        "VV_ratio": {"mean": vv_mu, "std": vv_sig},
        "VH_ratio": {"mean": vh_mu, "std": vh_sig},
        "DEM":      {"mean": dem_mu, "std": dem_sig},
        "note": ("SAR Ch1-4 normalization params in Phase 2 norm_params JSON. "
                 "Ratio and DEM params recorded here for inference reproducibility."),
    }
    params_out = save_dir / f"{name}_extra_norm_params.json"
    with open(str(params_out), "w") as f:
        json.dump(params, f, indent=2)

    # ── print summary ─────────────────────────────────────────────────────
    print(f"  Output shape: {stack_7ch.shape}")
    print(f"  VV_ratio    : μ={vv_mu:+.3f} dB  σ={vv_sig:.3f} dB")
    print(f"  VH_ratio    : μ={vh_mu:+.3f} dB  σ={vh_sig:.3f} dB")
    print(f"  DEM         : μ={dem_mu:.2f} m   σ={dem_sig:.2f} m")
    print(f"  Saved TIF   : {tif_out.name}")
    print(f"  Saved NPY   : {npy_out.name}")
    print(f"  Validity    : {vmask_out.name}")
    print(f"  Norm params : {params_out.name}")

    # Per-channel sanity check
    ch_labels = ["VV_pre", "VH_pre", "VV_post", "VH_post",
                 "VV_ratio", "VH_ratio", "DEM"]
    print(f"\n  Channel stats (should be ~0 mean, ~1 std for normalized channels):")
    print(f"  {'Channel':<12} {'min':>8} {'max':>8} {'mean':>8} {'std':>8}")
    print(f"  {'-'*48}")
    for i, lbl in enumerate(ch_labels):
        ch = stack_7ch[i]
        print(f"  {lbl:<12} {ch.min():>8.3f} {ch.max():>8.3f} "
              f"{ch.mean():>8.4f} {ch.std():>8.4f}")

    return stack_7ch


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Save to data_processed/model_stacks/ (not outputs/) — downstream scripts read here
    save_dir = Path(cfg["paths"]["data_processed"]) / "model_stacks"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {save_dir}")

    for event in cfg["events"]:
        build_model_stack(cfg, event, save_dir)

    print(f"\n{'='*55}")
    print(" All 7-channel model stacks built.")
    print(f"   Location: {save_dir}")
    print("   Next: python scripts/phase_3/02_create_masks.py")