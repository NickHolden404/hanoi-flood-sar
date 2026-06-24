"""
scripts/phase_3/07_cross_event.py

Cross-event generalization experiments — one seed at a time.

Scenarios:
  1. Train on Yagi   evaluate on Matmo test patches (cross_yagi_train)
  2. Train on Matmo  evaluate on Yagi test patches  (cross_matmo_train)
  3. Train on both   evaluate on both               (cross_both_train)

For scenarios 1 & 2, the best checkpoint is loaded and evaluated on the
held-out event without fine-tuning, quantifying raw transfer across:
  - Orbit geometry  (descending 37° vs ascending 43°)
  - Flood severity  (Yagi 223mm vs Matmo ~65mm peak rainfall)
  - Temporal gap    (3 days vs 9 days post-peak)

Run once per seed — independently.

Usage:
    python scripts/phase_3/07_cross_event.py --seed 0
    python scripts/phase_3/07_cross_event.py --seed 1
    ...
    python scripts/phase_3/07_cross_event.py --seed 4

Per-seed outputs:
    outputs/cross_<run_name>/seed_<N>/
        best.pth  last.pth  val_metrics.json
        test_metrics.json  train_log.csv  config_used.yaml
    outputs/tables/cross_event_seed_<N>.json    own + cross metrics

Run aggregate_results.py after all seeds complete.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from training.dataset import SARFloodDataset, get_val_transform
from training.metrics import SegMetrics
from models.attention_unet import get_attention_unet
from torch.utils.data import DataLoader


EXPERIMENTS = [
    # (train_event, cross_test_event, run_name)
    ("yagi",  "matmo", "cross_yagi_train"),
    ("matmo", "yagi",  "cross_matmo_train"),
    ("both",  "both",  "cross_both_train"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Cross-event experiments (one seed)")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (0–4)")
    p.add_argument("--config", default=None,
                   help="Path to config.yaml")
    return p.parse_args()


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_run(train_ev: str, run_name: str, seed: int, cfg_path: Path):
    """Launch training subprocess for one cross-event scenario."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase_3" / "train.py"),
        "--model",      "attention_unet",
        "--event",      train_ev,
        "--run_name",   run_name,
        "--seed",       str(seed),
        "--out_subdir", f"seed_{seed}",
    ]
    print(f"  Training: {run_name}/seed_{seed}  (event={train_ev})")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def evaluate_on_event(run_name: str, test_event: str,
                      seed: int, cfg: dict) -> dict:
    """Load best checkpoint, evaluate on test_event test split."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = get_attention_unet(cfg)

    ckpt = (Path(cfg["paths"]["output_root"]) /
            run_name / f"seed_{seed}" / "best.pth")
    if not ckpt.exists():
        print(f"  WARNING: checkpoint not found: {ckpt}")
        return {}

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()

    patch_root = str(Path(cfg["paths"]["data_processed"]) /
                     "patches" / test_event)
    ds = SARFloodDataset(patch_root, "test", transform=get_val_transform())
    dl = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                    shuffle=False, num_workers=2, pin_memory=True)

    metrics = SegMetrics(threshold=cfg["metrics"]["threshold"])
    with torch.no_grad():
        for imgs, msks in dl:
            logits = model(imgs.to(device))
            metrics.update(logits.cpu(), msks)

    m = metrics.compute()
    print(f"  Cross-eval [{run_name}/seed_{seed}  test:{test_event}]  "
          f"IoU={m['iou']:.4f}  Dice={m['dice']:.4f}  F1={m['f1']:.4f}")
    return m


def main():
    args = parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "configs" / "config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["paths"]["output_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_results = {}

    for train_ev, cross_ev, run_name in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  seed={args.seed}  Train:{train_ev}  Cross-test:{cross_ev}")

        # Train
        train_run(train_ev, run_name, args.seed, cfg_path)

        # Own-event test metrics (written by train.py)
        own_path = (Path(cfg["paths"]["output_root"]) /
                    run_name / f"seed_{args.seed}" / "test_metrics.json")
        own_metrics = {}
        if own_path.exists():
            with open(own_path) as f:
                own_metrics = json.load(f).get("metrics", {})

        # Cross-event evaluation (skip for bothboth)
        cross_metrics = {}
        if train_ev != "both":
            cross_metrics = evaluate_on_event(
                run_name, cross_ev, args.seed, cfg)

        seed_results[run_name] = {
            "seed":          args.seed,
            "train_event":   train_ev,
            "cross_test":    cross_ev,
            "own_metrics":   own_metrics,
            "cross_metrics": cross_metrics,
        }

    # Print summary for this seed
    print(f"\n{'CROSS-EVENT  seed='+str(args.seed):=^70}")
    hdr = (f"{'Run':<24} {'Train':>8} {'Own IoU':>10} "
           f"{'Cross IoU':>11} {'ΔIoU':>8}")
    print(hdr)
    print("-" * 70)
    for rn, r in seed_results.items():
        own_iou   = r["own_metrics"].get("iou", float("nan"))
        cross_iou = r["cross_metrics"].get("iou", float("nan"))
        delta = (cross_iou - own_iou
                 if r["cross_metrics"] else float("nan"))
        print(f"{rn:<24} {r['train_event']:>8} {own_iou:>10.4f} "
              f"{cross_iou:>11.4f} {delta:>+8.4f}")

    # Save seed results
    seed_out = out_dir / f"cross_event_seed_{args.seed}.json"
    with open(seed_out, "w") as f:
        json.dump(seed_results, f, indent=2)
    print(f"\nSeed {args.seed} cross-event saved  {seed_out}")
    print("Run aggregate_results.py after all seeds complete.")


if __name__ == "__main__":
    main()