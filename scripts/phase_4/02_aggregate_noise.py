"""
scripts/phase_4/02_aggregate_noise.py

Aggregate label noise sensitivity results across all seeds.
Run after all noise × seed training jobs are complete.

Usage:
    python scripts/phase_4/02_aggregate_noise.py
"""
import json
from pathlib import Path

import numpy as np
import yaml

ROOT     = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "config.yaml"
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

OUT_DIR   = Path(CFG["paths"]["output_root"])
NOISE_DIR = OUT_DIR / "noise_sensitivity"
TABLE_DIR = OUT_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)

NOISE_LEVELS = [5, 10, 15]
SEEDS        = list(range(6))

# Baseline from Phase 3 Attention U-Net (6 seeds, combined training, clean labels)
# Source: Phase 3 report §9.2 — outputs/attn_unet_both_p20/seed_{0..5}/
BASELINE_IOU_MEAN = 0.5746  # mean test IoU, 6 seeds (corrected from attn_unet_both_p20 test_metrics.json; original 0.5913 was ablation full_7ch mean)
BASELINE_IOU_STD  = 0.0295  # std


def load_metric(noise_pct: int, seed: int, key: str = "test_iou") -> float:
    p = NOISE_DIR / f"noise{noise_pct:02d}" / f"seed_{seed}" / "noise_metrics.json"
    if not p.exists():
        print(f"  [MISSING] noise{noise_pct:02d}/seed_{seed}")
        return float("nan")
    with open(p) as f:
        d = json.load(f)
    # test metrics are stored as test_iou, test_dice etc.
    val = d.get(key) or d.get(f"test_{key.replace('test_','')}")
    return float(val) if val is not None else float("nan")


def main():
    summary = {}

    # Baseline row (0% noise — Phase 3 results)
    summary["noise_00pct"] = {
        "noise_pct":  0,
        "noise_frac": 0.0,
        "mean_iou":   round(BASELINE_IOU_MEAN, 4),
        "std_iou":    round(BASELINE_IOU_STD,  4),
        "delta_iou":  0.0,
        "n_seeds":    6,
        "note":       "Phase 3 baseline — clean labels",
    }

    for noise_pct in NOISE_LEVELS:
        ious = [load_metric(noise_pct, s, "test_iou") for s in SEEDS]
        valid = [x for x in ious if not np.isnan(x)]

        if not valid:
            print(f"  [SKIP] noise {noise_pct}% — no results found")
            continue

        mean_iou  = float(np.mean(valid))
        std_iou   = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        delta_iou = mean_iou - BASELINE_IOU_MEAN

        summary[f"noise_{noise_pct:02d}pct"] = {
            "noise_pct":  noise_pct,
            "noise_frac": noise_pct / 100,
            "mean_iou":   round(mean_iou,  4),
            "std_iou":    round(std_iou,   4),
            "delta_iou":  round(delta_iou, 4),
            "n_seeds":    len(valid),
            "per_seed":   [round(x, 4) for x in ious],
        }

    # Print table
    print(f"\n{'='*55}")
    print("  Label Noise Sensitivity — Attention U-Net")
    print(f"{'='*55}")
    print(f"  {'Noise':>7}  {'Mean IoU':>10}  {'Std':>7}  {'ΔIoU':>8}  {'N':>3}")
    print(f"  {'-'*50}")
    for k in sorted(summary.keys()):
        r = summary[k]
        std_str   = f"{r['std_iou']:.4f}"  if r["std_iou"]  else "  —   "
        delta_str = f"{r['delta_iou']:+.4f}" if r["delta_iou"] is not None else "  —   "
        print(f"  {r['noise_pct']:>6}%  {r['mean_iou']:>10.4f}  "
              f"{std_str:>7}  {delta_str:>8}  {r['n_seeds']:>3}")

    # Robustness interpretation
    print(f"\n  Interpretation:")
    keys = sorted([k for k in summary if k != "noise_00pct"])
    if keys:
        max_degradation = min(summary[k]["delta_iou"] for k in keys)
        print(f"  Max IoU degradation at {NOISE_LEVELS[-1]}% noise: {max_degradation:+.4f}")
        if abs(max_degradation) < 0.03:
            verdict = "ROBUST — degradation below 0.03 IoU threshold across all noise levels"
        elif abs(max_degradation) < 0.06:
            verdict = "MODERATE — degradation within acceptable range for pseudo-label regime"
        else:
            verdict = "SENSITIVE — notable degradation; discuss label quality limitations"
        print(f"  Verdict: {verdict}")

    # Save
    out = TABLE_DIR / "noise_sensitivity_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   Saved: {out}")


if __name__ == "__main__":
    main()