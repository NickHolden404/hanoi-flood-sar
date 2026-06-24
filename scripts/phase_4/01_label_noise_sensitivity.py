"""
scripts/phase_4/01_label_noise_sensitivity.py

Phase 4 Step 1.1 — Label Noise Sensitivity Analysis (Critical for Matmo)

Simulates controlled label noise by randomly flipping flood pixels to background
at rates of 5%, 10%, 15%. Trains Attention U-Net under each condition.
Reports IoU degradation vs noise level.

Usage:
    python scripts/phase_4/01_label_noise_sensitivity.py --noise 0.05 --seed 0
    python scripts/phase_4/01_label_noise_sensitivity.py --noise 0.10 --seed 0
    python scripts/phase_4/01_label_noise_sensitivity.py --noise 0.15 --seed 0
    # Repeat for seeds 1–4, then run aggregate

Outputs:
    outputs/noise_sensitivity/noise{pct}/seed_{n}/best.pth + metrics
    outputs/tables/noise_sensitivity_summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.dataset import SARFloodDataset, get_train_transform, get_val_transform
from training.trainer  import Trainer
from models.attention_unet import get_attention_unet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--noise",  type=float, required=True,
                   help="Fraction of flood pixels to flip (0.05, 0.10, 0.15)")
    p.add_argument("--seed",   type=int,   default=0)
    p.add_argument("--config", default=None)
    return p.parse_args()


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class NoisyFloodDataset(SARFloodDataset):
    """Wraps SARFloodDataset to inject controlled label noise at load time."""

    def __init__(self, *args, noise_frac: float = 0.0, noise_rng_seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_frac     = noise_frac
        self.noise_rng      = np.random.default_rng(noise_rng_seed)

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        if self.noise_frac > 0 and self.split == "train":
            # mask is a float32 tensor (0/1). Work on numpy for RNG.
            m_np = mask.numpy().astype(bool)
            flood_idx = np.argwhere(m_np.reshape(-1)).flatten()
            if len(flood_idx) > 0:
                n_flip = max(1, int(len(flood_idx) * self.noise_frac))
                flip_idx = self.noise_rng.choice(flood_idx, n_flip, replace=False)
                m_flat   = m_np.reshape(-1).copy()
                m_flat[flip_idx] = False
                mask = torch.from_numpy(m_flat.reshape(m_np.shape).astype(np.float32))
        return img, mask


def evaluate_on_split(model, patch_dir: Path, split: str, device, cfg: dict) -> dict:
    """Run model on a patch split and return metrics dict."""
    import os
    from training.metrics import SegMetrics
    ds = SARFloodDataset(patch_dir, split=split, transform=get_val_transform())
    nw = 0 if os.name == "nt" else 4
    dl = torch.utils.data.DataLoader(
        ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=nw, pin_memory=True)
    metrics = SegMetrics()
    model.eval()
    with torch.no_grad():
        for imgs, masks in dl:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)          # raw logits — SegMetrics applies sigmoid
            metrics.update(logits, masks)
    return metrics.compute()


def run_noise_experiment(noise_frac: float, seed: int, cfg: dict):
    set_seed(seed)
    noise_pct = int(round(noise_frac * 100))
    out_dir   = (Path(cfg["paths"]["output_root"])
                 / "noise_sensitivity"
                 / f"noise{noise_pct:02d}"
                 / f"seed_{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    patch_dir = Path(cfg["paths"]["data_processed"]) / "patches" / "combined"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = NoisyFloodDataset(
        patch_dir, split="train",
        transform=get_train_transform(),
        noise_frac=noise_frac, noise_rng_seed=seed * 100
    )
    val_ds = SARFloodDataset(patch_dir, split="val", transform=get_val_transform())

    model   = get_attention_unet(cfg, n_channels=7)
    trainer = Trainer(
        model, train_ds, val_ds, cfg,
        run_name=f"noise{noise_pct:02d}_seed{seed}",
        out_dir=out_dir,
    )
    trainer.train()

    # Load best checkpoint and evaluate on test split
    best_ckpt = out_dir / "best.pth"
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model = model.to(device)
    test_metrics = evaluate_on_split(model, patch_dir, "test", device, cfg)

    metrics = {
        "noise_frac":  noise_frac,
        "noise_pct":   noise_pct,
        "seed":        seed,
        "best_val_iou": trainer.best_iou,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    out = out_dir / "noise_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [{noise_pct}% noise, seed {seed}] "
          f"Val IoU: {trainer.best_iou:.4f} | "
          f"Test IoU: {test_metrics.get('iou', float('nan')):.4f}")
    return metrics


def main():
    args = parse_args()
    cfg_path = args.config or ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    run_noise_experiment(args.noise, args.seed, cfg)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation (run after all noise × seed combinations complete)
# python scripts/phase_4/01_label_noise_sensitivity.py --aggregate
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_noise_results(cfg: dict, seeds=range(5)):
    """Aggregate noise sensitivity results across seeds."""
    noise_levels = [0, 5, 10, 15]   # 0 = baseline (load from Phase 3 results)
    out_dir = Path(cfg["paths"]["output_root"]) / "tables"
    out_dir.mkdir(exist_ok=True)

    # Load Phase 3 baseline (0% noise)
    baseline_path = (Path(cfg["paths"]["output_root"])
                     / "attn_unet_both_p20"
                     / "aggregate_results.json")
    baseline_iou = None
    if baseline_path.exists():
        with open(baseline_path) as f:
            b = json.load(f)
        baseline_iou = b.get("test_iou", {}).get("mean")

    summary = {}
    for noise_pct in [5, 10, 15]:
        ious = []
        for s in seeds:
            p = (Path(cfg["paths"]["output_root"])
                 / "noise_sensitivity"
                 / f"noise{noise_pct:02d}"
                 / f"seed_{s}"
                 / "noise_metrics.json")
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                ious.append(d.get("test_iou", float("nan")))
        if ious:
            mean_iou = float(np.nanmean(ious))
            std_iou  = float(np.nanstd(ious))
            delta    = (mean_iou - baseline_iou) if baseline_iou else None
            summary[f"noise_{noise_pct}pct"] = {
                "noise_frac":  noise_pct / 100,
                "mean_iou":    round(mean_iou, 4),
                "std_iou":     round(std_iou,  4),
                "delta_iou":   round(delta, 4) if delta else None,
                "n_seeds":     len(ious),
            }

    if baseline_iou:
        summary["noise_0pct"] = {
            "noise_frac": 0.0,
            "mean_iou":   round(baseline_iou, 4),
            "std_iou":    None,
            "delta_iou":  0.0,
            "n_seeds":    6,
        }

    out = out_dir / "noise_sensitivity_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nLabel Noise Sensitivity Summary:")
    print(f"  {'Noise':>8} {'Mean IoU':>10} {'Std':>8} {'ΔIoU':>10}")
    print("  " + "-" * 40)
    for k in sorted(summary.keys()):
        r = summary[k]
        delta_str = f"{r['delta_iou']:+.4f}" if r["delta_iou"] is not None else "—"
        std_str   = f"{r['std_iou']:.4f}"    if r["std_iou"]   is not None else "—"
        print(f"  {r['noise_frac']*100:>7.0f}%  "
              f"{r['mean_iou']:>10.4f}  {std_str:>8}  {delta_str:>10}")
    print(f"\n   Saved: {out}")
    return summary