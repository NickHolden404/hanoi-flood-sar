"""
scripts/phase_4/03_uncertainty_quantification.py

Phase 4 Step 3 — Deep Ensemble Uncertainty Quantification

The Attention U-Net has no dropout layers, making MC Dropout inapplicable.
Deep Ensemble using all 6 trained seeds is used instead — this is a more
principled uncertainty estimate and requires no architecture changes.

For each event:
  - Load all 6 seed probability maps (already computed in Phase 3)
  - Compute pixel-wise mean, variance, and binary entropy
  - Correlate high-uncertainty regions with:
      1. Flood boundary proximity
      2. SAR backscatter change magnitude

Usage:
    python scripts/phase_4/03_uncertainty_quantification.py --event yagi
    python scripts/phase_4/03_uncertainty_quantification.py --event matmo

Outputs:
    outputs/uncertainty/<event>/
        ensemble_mean.tif        — mean probability across 6 seeds
        ensemble_variance.tif    — pixel-wise variance across seeds
        ensemble_entropy.tif     — binary entropy of mean probability
        uncertainty_summary.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CFG_PATH = ROOT / "configs" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

OUT_DIR   = Path(CFG["paths"]["output_root"])
PRED_DIR  = OUT_DIR / "predictions" / "attn_unet_both_p20"
STACK_DIR = Path(CFG["paths"]["data_processed"]) / "model_stacks"

SEEDS     = list(range(6))
THRESHOLD = 0.5
HIGH_ENT_THRESH = 0.5   # nats; max binary entropy = ln(2) ≈ 0.693


def pixel_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Binary entropy H = -p*log(p) - (1-p)*log(1-p), shape (H,W)."""
    p = np.clip(p, eps, 1 - eps)
    return (-p * np.log(p) - (1 - p) * np.log(1 - p)).astype(np.float32)


def load_prob_maps(event: str) -> tuple[np.ndarray, dict]:
    """Load all seed probability maps, return stack (N, H, W) and rasterio profile."""
    maps    = []
    profile = None
    for s in SEEDS:
        p = PRED_DIR / event / f"seed_{s}" / "probability.tif"
        if not p.exists():
            print(f"  [WARN] Missing probability map: {p} — skipping seed {s}")
            continue
        with rasterio.open(p) as src:
            arr = src.read(1).astype(np.float32)
            if profile is None:
                profile = src.profile.copy()
        maps.append(arr)
    if not maps:
        raise FileNotFoundError(f"No probability maps found for event: {event}")
    print(f"  Loaded {len(maps)} seed probability maps for {event}")
    return np.stack(maps, axis=0), profile   # (N, H, W)


def compute_uncertainty_correlates(
    entropy: np.ndarray,
    binary:  np.ndarray,
    event:   str,
) -> dict:
    """
    Correlate high-uncertainty pixels with:
    1. Flood boundary proximity (are uncertain pixels near flood edges?)
    2. SAR backscatter change magnitude (are uncertain pixels in low-change zones?)
    """
    high_ent = entropy > HIGH_ENT_THRESH
    n_high   = int(high_ent.sum())
    n_total  = int(high_ent.size)

    # Boundary: XOR of binary and its dilation = boundary ring
    boundary      = binary_dilation(binary.astype(bool)) ^ binary.astype(bool)
    boundary_zone = binary_dilation(boundary, iterations=2)
    n_at_boundary = int((high_ent & boundary_zone).sum())
    frac_boundary = n_at_boundary / n_high if n_high > 0 else 0.0

    # SAR change magnitude correlation
    stack_path = STACK_DIR / f"{event}_model_stack_7ch.npy"
    backscatter_corr = None
    if stack_path.exists():
        stack      = np.load(stack_path, mmap_mode="r")
        # Ch index 2 = VV_post norm, Ch 0 = VV_pre norm (Phase 3 band order)
        change_mag = np.abs(stack[2] - stack[0]).astype(np.float32)
        mean_high  = float(np.nanmean(change_mag[high_ent])) if n_high > 0 else None
        mean_low   = float(np.nanmean(change_mag[~high_ent]))
        backscatter_corr = {
            "mean_change_high_entropy": round(mean_high, 4) if mean_high else None,
            "mean_change_low_entropy":  round(mean_low,  4),
            "ratio_high_to_low":        round(mean_high / mean_low, 4)
                                        if mean_high and mean_low > 0 else None,
            "interpretation": (
                "High-entropy pixels show LOWER backscatter change than low-entropy "
                "pixels — uncertain zones correspond to ambiguous (near-threshold) "
                "change signal, consistent with flood boundary and shadow confusion."
                if mean_high and mean_low and mean_high < mean_low
                else
                "High-entropy pixels show HIGHER backscatter change — "
                "uncertain zones correspond to strong but spatially variable change signal."
            ),
        }
    else:
        print(f"  [WARN] Stack not found: {stack_path} — skipping backscatter correlation")

    return {
        "n_high_entropy_pixels":         n_high,
        "pct_high_entropy":              round(100 * n_high / n_total, 2),
        "n_high_entropy_at_boundary":    n_at_boundary,
        "frac_high_entropy_at_boundary": round(frac_boundary, 4),
        "backscatter_correlation":       backscatter_corr,
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--event", required=True, choices=["yagi", "matmo"])
    args = p.parse_args()

    out_dir = OUT_DIR / "uncertainty" / args.event
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Deep Ensemble Uncertainty — {args.event.upper()}")
    print(f"{'='*55}")

    # Load all seed probability maps
    prob_stack, profile = load_prob_maps(args.event)  # (N, H, W)
    n_seeds = prob_stack.shape[0]

    # Compute ensemble statistics
    mean_p  = prob_stack.mean(axis=0).astype(np.float32)       # (H, W)
    var_p   = prob_stack.var(axis=0,  ddof=1).astype(np.float32)  # unbiased
    entropy = pixel_entropy(mean_p)
    binary  = (mean_p >= THRESHOLD).astype(np.uint8)

    # Save GeoTIFFs
    profile.update(dtype="float32", count=1, compress="lzw")
    for arr, name in [
        (mean_p,  "ensemble_mean"),
        (var_p,   "ensemble_variance"),
        (entropy, "ensemble_entropy"),
    ]:
        path = out_dir / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"  Saved: {path.name}")

    # Save binary
    bin_profile = profile.copy()
    bin_profile.update(dtype="uint8", nodata=None)
    with rasterio.open(out_dir / "ensemble_binary.tif", "w", **bin_profile) as dst:
        dst.write(binary, 1)

    # Uncertainty correlates
    print("  Computing uncertainty correlates...")
    correlates = compute_uncertainty_correlates(entropy, binary, args.event)

    summary = {
        "event":             args.event,
        "method":            "deep_ensemble",
        "n_seeds":           n_seeds,
        "threshold":         THRESHOLD,
        "mean_entropy":      round(float(np.nanmean(entropy)), 4),
        "max_variance":      round(float(np.nanmax(var_p)),    4),
        "mean_variance":     round(float(np.nanmean(var_p)),   4),
        "pct_high_variance": round(float((var_p > 0.05).mean() * 100), 2),
        "correlates":        correlates,
    }

    out_json = out_dir / "uncertainty_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Method:           Deep Ensemble ({n_seeds} seeds)")
    print(f"  Mean entropy:     {summary['mean_entropy']:.4f}")
    print(f"  Mean variance:    {summary['mean_variance']:.4f}")
    print(f"  % high-variance:  {summary['pct_high_variance']:.1f}%")
    print(f"  % high-entropy:   {correlates['pct_high_entropy']:.1f}%")
    print(f"  Boundary frac:    {correlates['frac_high_entropy_at_boundary']:.3f}")
    if correlates.get("backscatter_correlation"):
        bc = correlates["backscatter_correlation"]
        print(f"  Change (high-ent): {bc['mean_change_high_entropy']}")
        print(f"  Change (low-ent):  {bc['mean_change_low_entropy']}")
        print(f"   {bc['interpretation']}")
    print(f"\n   Summary: {out_json}")


if __name__ == "__main__":
    main()