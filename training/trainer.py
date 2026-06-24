"""
training/trainer.py

Trainer — supports UNet & AttentionUNet with AMP, early stopping,
cosine LR scheduling, and full metric logging.

Changes from original:
  - Accepts explicit out_dir (Path) instead of deriving from run_name.
    This enables seed-namespaced output: outputs/<run_name>/seed_N/
  - Saves best.pth and last.pth directly into out_dir.
  - history.json saved into out_dir (co-located with checkpoints).
  - No internal mkdir for weights/ or logs/ sub-trees — caller owns dirs.
"""
import json
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.losses import BCEDiceLoss
from training.metrics import SegMetrics


class Trainer:

    def __init__(self, model, train_ds, val_ds, cfg: dict,
                 run_name: str = "run", out_dir: Path = None):
        self.cfg      = cfg
        self.run_name = run_name
        self.device   = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model    = model.to(self.device)

        # Output directory — must be pre-created by caller (train.py)
        if out_dir is None:
            # Fallback: legacy behaviour without seed subdirs
            out_dir = (Path(cfg["paths"]["output_root"]) /
                       "weights" / run_name)
            out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(out_dir)

        tc = cfg["training"]
        self.epochs   = tc["epochs"]
        self.patience = tc["early_stopping_patience"]
        self.amp      = tc.get("mixed_precision", True)

        # num_workers=0 required on Windows (no fork support)
        nw = 0 if os.name == "nt" else 4

        self.train_dl = DataLoader(
            train_ds, batch_size=tc["batch_size"],
            shuffle=True, num_workers=nw, pin_memory=True)
        self.val_dl = DataLoader(
            val_ds, batch_size=tc["batch_size"],
            shuffle=False, num_workers=nw, pin_memory=True)

        lc = cfg["loss"]
        self.criterion = BCEDiceLoss(lc["bce_weight"], lc["dice_weight"])

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=tc["lr"],
            weight_decay=tc.get("weight_decay", 1e-5))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

        self.history = {
            "train_loss": [], "val_loss": [],
            "val_iou": [], "val_dice": [], "val_f1": [],
        }
        self.best_iou   = 0.0
        self.no_improve = 0

    # ── epoch loop ─────────────────────────────────────────────────────────

    def _run_epoch(self, dl, train=True):
        self.model.train(train)
        total_loss = 0.0
        metrics = SegMetrics(threshold=self.cfg["metrics"]["threshold"])

        with torch.set_grad_enabled(train):
            for imgs, msks in dl:
                imgs = imgs.to(self.device)
                msks = msks.to(self.device)

                with torch.amp.autocast('cuda', enabled=self.amp):
                    logits = self.model(imgs)
                    loss   = self.criterion(logits, msks)

                if train:
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * imgs.size(0)
                metrics.update(logits.detach(), msks.detach())

        return total_loss / len(dl.dataset), metrics.compute()

    # ── training loop ──────────────────────────────────────────────────────

    def train(self) -> dict:
        print(f"\n{'='*60}")
        print(f"  Run   : {self.run_name}")
        print(f"  OutDir: {self.out_dir}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}")

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss, _  = self._run_epoch(self.train_dl, train=True)
            va_loss, vm = self._run_epoch(self.val_dl,   train=False)
            self.scheduler.step()

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            for k in ("iou", "dice", "f1"):
                self.history[f"val_{k}"].append(vm[k])

            elapsed = time.time() - t0
            print(f"Ep {epoch:03d} | tr={tr_loss:.4f} va={va_loss:.4f} "
                  f"IoU={vm['iou']:.4f} Dice={vm['dice']:.4f} "
                  f"F1={vm['f1']:.4f} [{elapsed:.1f}s]")

            if vm["iou"] > self.best_iou:
                self.best_iou   = vm["iou"]
                self.no_improve = 0
                torch.save(self.model.state_dict(),
                           self.out_dir / "best.pth")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        torch.save(self.model.state_dict(), self.out_dir / "last.pth")
        with open(self.out_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        print(f"\n  Best Val IoU: {self.best_iou:.4f}")
        return self.history

    # ── test evaluation ────────────────────────────────────────────────────

    def evaluate(self, test_ds) -> dict:
        """Load best.pth, evaluate on test_ds, return metric dict."""
        dl = DataLoader(
            test_ds, batch_size=self.cfg["training"]["batch_size"],
            shuffle=False, num_workers=0)

        ckpt = self.out_dir / "best.pth"
        if not ckpt.exists():
            print(f"  WARNING: best.pth not found at {ckpt}. Using current weights.")
        else:
            self.model.load_state_dict(
                torch.load(ckpt, map_location=self.device))

        _, metrics = self._run_epoch(dl, train=False)
        print("\n  Test metrics:")
        for k, v in metrics.items():
            print(f"    {k:15s}: {v:.4f}")
        return metrics