"""
scripts/hydro/train_hydro.py

Hydro-aware training entrypoint — mirrors scripts/phase_3/train.py exactly,
with three additions:

  1. Uses BCEDiceHydroLoss instead of the standard BCEDiceLoss from trainer.py.
  2. Adds --lambda_hydro CLI argument (default 0.1).
  3. Passes the input batch (x) to the loss function on every training step.
  4. Writes lambda_hydro into config.json for full reproducibility.
  5. Output pattern: outputs/hydro_loss/lambda_{lambda_hydro}/seed_{seed}/

The Trainer class from training/trainer.py is subclassed (HydroTrainer) so
that the training loop can forward the input tensor to the loss without
touching the read-only trainer.py.

Nothing in train.py, trainer.py, or any model file is modified.
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

# ── project root on sys.path ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet import UNet
from models.attention_unet import AttentionUNet
from training.dataset import SARFloodDataset, get_train_transform, get_val_transform
from training.trainer import Trainer
from training.metrics import SegMetrics
from losses.combined_loss import BCEDiceHydroLoss


# ── reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ── channel helpers (identical to train.py) ────────────────────────────────

ALL_CHANNELS = ["VV_pre", "VH_pre", "VV_post", "VH_post",
                "VV_ratio", "VH_ratio", "DEM"]


def channel_indices(names: list) -> list:
    idx = []
    for n in names:
        if n not in ALL_CHANNELS:
            raise ValueError(f"Unknown channel: '{n}'. Valid: {ALL_CHANNELS}")
        idx.append(ALL_CHANNELS.index(n))
    return sorted(idx)


# ── model factory (identical to train.py) ─────────────────────────────────

def build_model(model_name: str, n_ch: int, cfg: dict) -> torch.nn.Module:
    if model_name == "unet":
        mc = cfg["unet"]
        return UNet(in_channels=n_ch,
                    encoder_channels=mc["encoder_channels"],
                    dropout=mc["dropout"])
    elif model_name == "attention_unet":
        mc = cfg["attention_unet"]
        return AttentionUNet(in_channels=n_ch,
                             encoder_channels=mc["encoder_channels"],
                             dropout=mc["dropout"])
    raise ValueError(f"Unknown model: {model_name}")


# ── output dirs (mirrors train.py make_out_dirs) ───────────────────────────

def make_out_dirs(cfg: dict, run_name: str, out_subdir: str) -> dict:
    base = Path(cfg["paths"]["output_root"]) / run_name / out_subdir
    base.mkdir(parents=True, exist_ok=True)
    return {"base": base}


# ── save helpers (identical to train.py) ──────────────────────────────────

def save_train_log(history: dict, out_dir: Path):
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
    ious = history.get("val_iou", [])
    if not ious:
        return
    best_ep = int(np.argmax(ious))
    val_m = {
        "best_epoch":   best_ep + 1,
        "best_val_iou": ious[best_ep],
        "best_val_dice": history.get("val_dice", [None])[best_ep],
        "best_val_f1":   history.get("val_f1",   [None])[best_ep],
        "final_epoch":   len(ious),
    }
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_m, f, indent=2)


def save_test_metrics(metrics: dict, out_dir: Path,
                      channels: list, seed: int, run_name: str):
    out = {
        "run_name":   run_name,
        "seed":       seed,
        "channels":   channels,
        "n_channels": len(channels),
        "metrics":    {k: round(float(v), 6) for k, v in metrics.items()},
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)


def save_config(cfg: dict, args: argparse.Namespace, out_dir: Path):
    run_cfg = {"cli_args": vars(args), "config": cfg}
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True)


def patch_root(cfg: dict, event: str) -> Path:
    event_dir = "combined" if event == "both" else event
    return Path(cfg["paths"]["data_processed"]) / "patches" / event_dir


# ── hydro-aware trainer ────────────────────────────────────────────────────

class HydroTrainer(Trainer):
    """
    Subclass of Trainer that overrides _run_epoch to pass the input batch
    to BCEDiceHydroLoss.  All other logic (optimizer, scheduler, AMP,
    early stopping, checkpointing) is inherited unchanged.
    """

    def __init__(self, model, train_ds, val_ds, cfg, run_name,
                 out_dir, lambda_hydro: float = 0.1):
        super().__init__(model, train_ds, val_ds, cfg,
                         run_name=run_name, out_dir=out_dir)
        # Replace the criterion set by the parent __init__
        lc = cfg["loss"]
        self.criterion = BCEDiceHydroLoss(
            w_bce=lc["bce_weight"],
            w_dice=lc["dice_weight"],
            lambda_hydro=lambda_hydro,
        )

    def _run_epoch(self, dl, train=True):
        self.model.train(train)
        total_loss = 0.0
        metrics = SegMetrics(threshold=self.cfg["metrics"]["threshold"])

        with torch.set_grad_enabled(train):
            for imgs, msks in dl:
                imgs = imgs.to(self.device)
                msks = msks.to(self.device)

                with torch.amp.autocast("cuda", enabled=self.amp):
                    logits = self.model(imgs)
                    # ── key difference: pass imgs as x_batch ──────────────
                    loss = self.criterion(logits, msks, imgs)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * imgs.size(0)
                metrics.update(logits.detach(), msks.detach())

        return total_loss / len(dl.dataset), metrics.compute()


# ── argument parsing ───────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="SAR flood segmentation trainer with hydrological loss")
    p.add_argument("--model",    required=True,
                   choices=["unet", "attention_unet"])
    p.add_argument("--event",    required=True,
                   choices=["yagi", "matmo", "both"])
    p.add_argument("--run_name", required=True,
                   help="Experiment name (parent output directory)")
    p.add_argument("--seed",     type=int, required=True)
    p.add_argument("--out_subdir", default="seed_0",
                   help="Sub-directory under run_name (e.g. seed_0)")
    p.add_argument("--channels", nargs="+", default=None,
                   help="Channel subset for ablation (default: all 7)")
    p.add_argument("--config",   default=None,
                   help="Path to config.yaml")
    p.add_argument("--lambda_hydro", type=float, default=0.1,
                   help="Weight for hydrological consistency penalty (default: 0.1)")
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    cfg_path = Path(args.config) if args.config \
               else ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(args.seed)
    print(f"\n[train_hydro.py] model={args.model}  event={args.event}  "
          f"seed={args.seed}  lambda_hydro={args.lambda_hydro}  "
          f"run={args.run_name}/{args.out_subdir}")

    ch_names = args.channels if args.channels \
               else cfg["dataset"]["channels"]
    ch_idx   = channel_indices(ch_names)
    n_ch     = len(ch_idx)
    print(f"  Channels ({n_ch}): {ch_names}")

    dirs    = make_out_dirs(cfg, args.run_name, args.out_subdir)
    out_dir = dirs["base"]

    pr       = patch_root(cfg, args.event)
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

    model    = build_model(args.model, n_ch, cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    trainer = HydroTrainer(
        model, train_ds, val_ds, cfg,
        run_name=args.run_name,
        out_dir=out_dir,
        lambda_hydro=args.lambda_hydro,
    )

    history      = trainer.train()
    test_metrics = trainer.evaluate(test_ds)

    save_train_log(history, out_dir)
    save_val_metrics(history, out_dir)
    save_test_metrics(test_metrics, out_dir, ch_names, args.seed, args.run_name)
    save_config(cfg, args, out_dir)

    # ── save lambda_hydro into a dedicated config.json ────────────────────
    hydro_cfg = {
        "lambda_hydro": args.lambda_hydro,
        "model":        args.model,
        "event":        args.event,
        "seed":         args.seed,
        "run_name":     args.run_name,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(hydro_cfg, f, indent=2)

    print(f"\n  All outputs saved to: {out_dir}")
    print(f"  Best val IoU : {max(history.get('val_iou', [0])):.4f}")
    print(f"  Test  IoU    : {test_metrics.get('iou', 0):.4f}")
    print("[Step 3 COMPLETE] train_hydro.py finished.")


if __name__ == "__main__":
    main()