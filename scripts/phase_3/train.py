"""
scripts/phase_3/train.py

Core training entry point — called by 04_train_unet.py,
05_train_attention_unet.py, 06_ablation.py, 07_cross_event.py.

Handles:
  - Deterministic seed setup (Python / NumPy / PyTorch / CUDA)
  - Dataset loading with optional channel subsetting (ablation)
  - Model instantiation
  - Training via Trainer
  - Test evaluation
  - Saving: best.pth, last.pth, val_metrics.json, test_metrics.json,
            train_log.csv, config_used.yaml

Usage (do not call directly — use wrapper scripts):
    python scripts/phase_3/train.py \\
        --model attention_unet \\
        --event both \\
        --run_name attention_unet_both \\
        --seed 0 \\
        --out_subdir seed_0

    # Ablation (channel subset):
    python scripts/phase_3/train.py \\
        --model attention_unet --event both \\
        --run_name ablation_no_dem --seed 2 --out_subdir seed_2 \\
        --channels VV_pre VH_pre VV_post VH_post VV_ratio VH_ratio
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from models.attention_unet import AttentionUNet
from training.dataset import SARFloodDataset, get_train_transform, get_val_transform
from training.trainer import Trainer


# ── reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── channel name  index mapping ──────────────────────────────────────────

ALL_CHANNELS = ["VV_pre", "VH_pre", "VV_post", "VH_post",
                "VV_ratio", "VH_ratio", "DEM"]

def channel_indices(names: list) -> list:
    """Convert channel name list to integer indices into the 7-ch stack."""
    idx = []
    for n in names:
        if n not in ALL_CHANNELS:
            raise ValueError(f"Unknown channel: '{n}'. Valid: {ALL_CHANNELS}")
        idx.append(ALL_CHANNELS.index(n))
    return sorted(idx)


# ── model factory ─────────────────────────────────────────────────────────

def build_model(model_name: str, n_ch: int, cfg: dict) -> torch.nn.Module:
    if model_name == "unet":
        mc = cfg["unet"]
        return UNet(
            in_channels=n_ch,
            encoder_channels=mc["encoder_channels"],
            dropout=mc["dropout"],
        )
    elif model_name == "attention_unet":
        mc = cfg["attention_unet"]
        return AttentionUNet(
            in_channels=n_ch,
            encoder_channels=mc["encoder_channels"],
            dropout=mc["dropout"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── output directory ───────────────────────────────────────────────────────

def make_out_dirs(cfg: dict, run_name: str, out_subdir: str) -> dict:
    """
    Returns paths dict:
      weights_dir : outputs/<run_name>/<out_subdir>/
      logs_dir    : same directory (all outputs co-located per seed)
    """
    base = Path(cfg["paths"]["output_root"]) / run_name / out_subdir
    base.mkdir(parents=True, exist_ok=True)
    return {"base": base}


# ── save helpers ───────────────────────────────────────────────────────────

def save_train_log(history: dict, out_dir: Path):
    """Write epoch-by-epoch training history to train_log.csv."""
    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return
    fieldnames = list(history.keys())
    rows = [{k: history[k][i] for k in fieldnames} for i in range(epochs)]
    with open(out_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch"] + fieldnames)
        w.writeheader()
        for i, row in enumerate(rows, 1):
            w.writerow({"epoch": i, **row})


def save_val_metrics(history: dict, out_dir: Path):
    """Save best val metrics (highest IoU epoch) to val_metrics.json."""
    ious = history.get("val_iou", [])
    if not ious:
        return
    best_ep = int(np.argmax(ious))
    val_m = {
        "best_epoch":  best_ep + 1,
        "best_val_iou": ious[best_ep],
        "best_val_dice": history.get("val_dice", [None])[best_ep],
        "best_val_f1":   history.get("val_f1",   [None])[best_ep],
        "final_epoch":   len(ious),
    }
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_m, f, indent=2)


def save_test_metrics(metrics: dict, out_dir: Path,
                      channels: list, seed: int, run_name: str):
    """Save test metrics to test_metrics.json in standard schema."""
    out = {
        "run_name":  run_name,
        "seed":      seed,
        "channels":  channels,
        "n_channels": len(channels),
        "metrics":   {k: round(float(v), 6) for k, v in metrics.items()},
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)


def save_config(cfg: dict, args: argparse.Namespace, out_dir: Path):
    """Save a snapshot of the config + CLI args used for this run."""
    run_cfg = {
        "cli_args": vars(args),
        "config":   cfg,
    }
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)


# ── patch root resolution ─────────────────────────────────────────────────

def patch_root(cfg: dict, event: str) -> Path:
    """
    Returns the patch directory for the given event.
    event='both'  data_processed/patches/combined/
    event='yagi'  data_processed/patches/yagi/
    """
    event_dir = "combined" if event == "both" else event
    return Path(cfg["paths"]["data_processed"]) / "patches" / event_dir


# ── argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SAR flood segmentation trainer")
    p.add_argument("--model",      required=True,
                   choices=["unet", "attention_unet"])
    p.add_argument("--event",      required=True,
                   choices=["yagi", "matmo", "both"])
    p.add_argument("--run_name",   required=True,
                   help="Experiment name (parent output directory)")
    p.add_argument("--seed",       type=int, required=True,
                   help="Random seed for this run")
    p.add_argument("--out_subdir", default="seed_0",
                   help="Sub-directory under run_name (e.g. seed_0)")
    p.add_argument("--channels",   nargs="+", default=None,
                   help="Channel subset for ablation (default: all 7)")
    p.add_argument("--config",     default=None,
                   help="Path to config.yaml")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Config
    cfg_path = Path(args.config) if args.config \
               else ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Seed — must be set before ANY torch/numpy operations
    set_seed(args.seed)
    print(f"\n[train.py] model={args.model}  event={args.event}  "
          f"seed={args.seed}  run={args.run_name}/{args.out_subdir}")

    # Channels
    ch_names = args.channels if args.channels \
               else cfg["dataset"]["channels"]
    ch_idx   = channel_indices(ch_names)
    n_ch     = len(ch_idx)
    print(f"  Channels ({n_ch}): {ch_names}")

    # Output dirs
    dirs = make_out_dirs(cfg, args.run_name, args.out_subdir)
    out_dir = dirs["base"]

    # Override trainer output dirs via cfg copy
    run_cfg = {**cfg}
    run_cfg["paths"] = {**cfg["paths"]}
    run_cfg["_out_dir"] = str(out_dir)   # Trainer reads this

    # Datasets
    pr = patch_root(cfg, args.event)
    train_ds = SARFloodDataset(str(pr), "train",
                               channel_indices=ch_idx,
                               transform=get_train_transform())
    val_ds   = SARFloodDataset(str(pr), "val",
                               channel_indices=ch_idx,
                               transform=get_val_transform())
    test_ds  = SARFloodDataset(str(pr), "test",
                               channel_indices=ch_idx,
                               transform=get_val_transform())

    print(f"  Patches — train:{len(train_ds)}  "
          f"val:{len(val_ds)}  test:{len(test_ds)}")

    # Model
    model = build_model(args.model, n_ch, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # Trainer — pass out_dir explicitly
    trainer = Trainer(model, train_ds, val_ds, cfg,
                      run_name=args.run_name,
                      out_dir=out_dir)

    # Train
    history = trainer.train()

    # Evaluate on test set
    test_metrics = trainer.evaluate(test_ds)

    # Save all artifacts
    save_train_log(history, out_dir)
    save_val_metrics(history, out_dir)
    save_test_metrics(test_metrics, out_dir, ch_names, args.seed, args.run_name)
    save_config(cfg, args, out_dir)

    # Copy best checkpoint name-tagged for clarity (best.pth already saved by Trainer)
    print(f"\n  All outputs saved to: {out_dir}")
    print(f"  Best val IoU: {max(history.get('val_iou', [0])):.4f}")
    print(f"  Test  IoU:    {test_metrics.get('iou', 0):.4f}")


if __name__ == "__main__":
    main()