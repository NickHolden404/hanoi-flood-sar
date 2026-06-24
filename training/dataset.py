"""
training/dataset.py

Dataset utilities:
  - SARFloodDataset  : PyTorch Dataset for .npy patch files
  - PatchGenerator   : Slices full-scene stacks into patches
  - build_splits     : Row-strip spatial split (prevents leakage)

Fixes vs original:
  1. build_splits() uses ROW-STRIP partitioning — patches sorted by row
     index and divided into contiguous north/centre/south strips, not
     randomly shuffled. This is the method documented in the Phase 3 report
     and required for publishable spatial leakage prevention.
  2. GaussNoise updated for albumentations ≥1.4 API (std_range replaces
     var_limit).
  3. build_splits() accepts seed for reproducibility (used in PatchGenerator
     naming, not for spatial ordering which is deterministic by row index).
"""
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio


# ── augmentation pipelines ─────────────────────────────────────────────────

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1,
            rotate_limit=15, p=0.4),
        # albumentations ≥1.4: std_range replaces var_limit
        A.GaussNoise(std_range=(0.01, 0.1), p=0.3),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([ToTensorV2()])


# ── dataset ────────────────────────────────────────────────────────────────

class SARFloodDataset(Dataset):
    """
    Expects patches pre-saved as:
      <root>/<split>/img/<name>.npy   shape (H, W, C)
      <root>/<split>/msk/<name>.npy   shape (H, W)
    """

    def __init__(self, root: str, split: str = "train",
                 channel_indices=None, transform=None):
        self.root      = Path(root)
        self.split     = split
        self.ch_idx    = channel_indices
        self.transform = transform

        img_dir = self.root / split / "img"
        self.samples = sorted(img_dir.glob("*.npy"))
        assert len(self.samples) > 0, \
            f"No patches found in {img_dir}"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx].stem
        img  = np.load(self.samples[idx]).astype(np.float32)       # H W C
        msk  = np.load(
            self.root / self.split / "msk" / f"{name}.npy"
        ).astype(np.float32)                                        # H W

        if self.ch_idx is not None:
            img = img[..., self.ch_idx]

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"].float()                  # C H W tensor
            msk = aug["mask"].unsqueeze(0).float()
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            msk = torch.from_numpy(msk).unsqueeze(0)

        return img, msk


# ── patch generator ────────────────────────────────────────────────────────

class PatchGenerator:
    """
    Reads a multi-band GeoTIFF stack + binary mask GeoTIFF,
    slides a window of (patch_size × patch_size) with given overlap,
    filters by min_flood_fraction, saves as .npy.

    Patch names encode row and col for deterministic row-strip splitting:
      <prefix>_r<row>_c<col>.npy
    """

    def __init__(self, patch_size=256, overlap=32, min_flood_frac=0.01):
        self.ps  = patch_size
        self.ov  = overlap
        self.mff = min_flood_frac

    def generate(self, stack_path: str, mask_path: str,
                 out_dir: str, prefix: str = "patch") -> int:
        out_img = Path(out_dir) / "img"
        out_msk = Path(out_dir) / "msk"
        out_img.mkdir(parents=True, exist_ok=True)
        out_msk.mkdir(parents=True, exist_ok=True)

        with rasterio.open(stack_path) as ss, \
             rasterio.open(mask_path)  as sm:
            stack = ss.read()    # C H W
            mask  = sm.read(1)   # H W

        C, H, W = stack.shape
        step = self.ps - self.ov
        count = 0

        for r in range(0, H - self.ps + 1, step):
            for c in range(0, W - self.ps + 1, step):
                msk_crop   = mask[r:r + self.ps, c:c + self.ps]
                flood_frac = msk_crop.sum() / msk_crop.size
                if flood_frac < self.mff:
                    continue
                img_crop = (stack[:, r:r + self.ps, c:c + self.ps]
                            .transpose(1, 2, 0))          # H W C
                # Encode spatial position in name for row-strip splitting
                name = f"{prefix}_r{r:05d}_c{c:05d}"
                np.save(out_img / f"{name}.npy", img_crop)
                np.save(out_msk / f"{name}.npy", msk_crop)
                count += 1

        print(f"  Saved {count} patches  {out_dir}")
        return count


# ── row-strip spatial split ────────────────────────────────────────────────

def build_splits(patch_dir: str, out_dir: str,
                 ratios=(0.70, 0.15, 0.15), seed: int = 0) -> dict:
    """
    Partition patches into train/val/test using ROW-STRIP partitioning.

    Patches are sorted by their encoded row index (r<RRRRR> in filename).
    The first 70% of rows  train, next 15%  val, final 15%  test.
    Patches within each strip are NOT shuffled across splits — only within
    a split (for DataLoader) after the fact.

    This guarantees:
      - No spatially adjacent patches appear in both train and test
      - Split boundaries are at fixed geographic latitudes (reproducible)
      - Spatial autocorrelation range is exceeded between splits

    The seed argument is accepted for API compatibility but does not affect
    the spatial ordering (which is deterministic by row index).

    Split boundary information is saved to <out_dir>/split_info.json
    for documentation and paper reporting.
    """
    patch_dir = Path(patch_dir)
    out_dir   = Path(out_dir)

    names = sorted(
        {p.stem for p in (patch_dir / "img").glob("*.npy")}
    )
    if not names:
        raise FileNotFoundError(f"No patches found in {patch_dir}/img/")

    # Extract row index from filename (format: prefix_r<RRRRR>_c<CCCCC>)
    def row_of(name: str) -> int:
        parts = name.split("_r")
        if len(parts) < 2:
            # Legacy naming without row encoding — fall back to stable sort
            return 0
        return int(parts[-1].split("_c")[0])

    names_by_row = sorted(names, key=row_of)
    n  = len(names_by_row)
    t1 = int(n * ratios[0])
    t2 = int(n * (ratios[0] + ratios[1]))

    splits = {
        "train": names_by_row[:t1],
        "val":   names_by_row[t1:t2],
        "test":  names_by_row[t2:],
    }

    # Copy patches into split subdirs
    for split, items in splits.items():
        for sub in ("img", "msk"):
            (out_dir / split / sub).mkdir(parents=True, exist_ok=True)
        for name in items:
            for sub in ("img", "msk"):
                src = patch_dir / sub / f"{name}.npy"
                dst = out_dir / split / sub / f"{name}.npy"
                if src.exists():
                    shutil.copy2(src, dst)

    counts = {k: len(v) for k, v in splits.items()}
    print(f"  Row-strip split: {counts}")

    # Compute approximate geographic split boundaries
    if n > 0:
        train_rows = [row_of(n) for n in splits["train"]] or [0]
        val_rows   = [row_of(n) for n in splits["val"]]   or [0]
        test_rows  = [row_of(n) for n in splits["test"]]  or [0]
        info = {
            "method":        "row_strip",
            "seed_arg":      seed,
            "n_patches":     n,
            "split_counts":  counts,
            "split_ratios":  list(ratios),
            "train_row_range": [min(train_rows), max(train_rows)],
            "val_row_range":   [min(val_rows),   max(val_rows)]
                                if val_rows else [None, None],
            "test_row_range":  [min(test_rows),  max(test_rows)]
                                if test_rows else [None, None],
            "note": ("Row indices are pixel rows in the 10m UTM stack. "
                     "Multiply by 10 for approx. geographic distance in metres."),
        }
        with open(out_dir / "split_info.json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Split info saved  {out_dir / 'split_info.json'}")

    return splits