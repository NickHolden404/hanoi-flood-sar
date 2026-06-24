"""
scripts/phase_3/03_generate_patches.py

Slices 7-channel model stacks + flood masks into 256×256 patches
and applies a stratified 70/15/15 train/val/test split.

Reads from:
  data_processed/model_stacks/<event>_model_stack_7ch.npy   (7 × 1781 × 1963)
  data_processed/flood_labels/<event>/flood_mask.tif

Writes to:
  data_processed/patches/<event>/{train,val,test}/{img,msk}/*.npy
  data_processed/patches/combined/{train,val,test}/{img,msk}/*.npy

Patch filtering: tiles with < 1% flood pixels are excluded to
avoid extreme class imbalance. Spatial non-overlap between splits
is enforced by partitioning row-strips before tiling.

Usage:
    python scripts/phase_3/03_generate_patches.py
    python scripts/phase_3/03_generate_patches.py --patch_size 512
"""
import argparse, json, random, shutil
from pathlib import Path

import numpy as np
import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[2]
SEED = 42  # fixed — not read from config (project.seed key absent)


def make_patches(stack_npy, mask_tif, out_root, event_name,
                 patch_size, overlap, min_flood_frac, split_ratios):

    random.seed(SEED);  np.random.seed(SEED)

    stack = np.load(stack_npy).astype(np.float32)   # 7 × H × W
    _, H, W = stack.shape

    with rasterio.open(mask_tif) as src:
        mask = src.read(1).astype(np.uint8)          # H × W

    assert mask.shape == (H, W), \
        f"Shape mismatch: stack {stack.shape} vs mask {mask.shape}"

    step = patch_size - overlap

    # All valid row/col start positions
    row_starts = list(range(0, H - patch_size + 1, step))
    if row_starts[-1] + patch_size < H:
        row_starts.append(H - patch_size)

    col_starts = list(range(0, W - patch_size + 1, step))
    if col_starts[-1] + patch_size < W:
        col_starts.append(W - patch_size)

    # Collect all valid patches first, then randomly shuffle and split
    all_patches = []
    for r in row_starts:
        for c in col_starts:
            msk_crop = mask[r:r + patch_size, c:c + patch_size]
            if msk_crop.shape != (patch_size, patch_size):
                continue
            ff = float(msk_crop.sum()) / msk_crop.size
            if ff < min_flood_frac:
                continue
            all_patches.append((r, c, ff))

    random.shuffle(all_patches)

    n  = len(all_patches)
    t1 = int(n * split_ratios[0])
    t2 = int(n * (split_ratios[0] + split_ratios[1]))
    split_patches = {
        "train": all_patches[:t1],
        "val":   all_patches[t1:t2],
        "test":  all_patches[t2:],
    }

    counts = {"train": 0, "val": 0, "test": 0}
    flood_fracs = {"train": [], "val": [], "test": []}

    for split, patches in split_patches.items():
        img_dir = Path(out_root) / split / "img"
        msk_dir = Path(out_root) / split / "msk"
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)

        for r, c, ff in patches:
            img_crop = stack[:, r:r + patch_size, c:c + patch_size
                             ].transpose(1, 2, 0)   # H W C

            fname = f"{event_name}_{r:04d}_{c:04d}"
            np.save(img_dir / f"{fname}.npy", img_crop)
            np.save(msk_dir / f"{fname}.npy", mask[r:r + patch_size, c:c + patch_size])
            counts[split] += 1
            flood_fracs[split].append(ff)

    total = sum(counts.values())
    print(f"  [{event_name.upper()}] "
          f"train={counts['train']}  val={counts['val']}  test={counts['test']}  "
          f"total={total}")
    for split in ("train", "val", "test"):
        if flood_fracs[split]:
            print(f"    {split} flood frac — "
                  f"mean={np.mean(flood_fracs[split]):.3f}  "
                  f"min={np.min(flood_fracs[split]):.3f}  "
                  f"max={np.max(flood_fracs[split]):.3f}")
    return counts


def merge_combined(patch_root, events):
    """Copy per-event patches into a unified combined/ directory."""
    for split in ("train", "val", "test"):
        for sub in ("img", "msk"):
            dst = patch_root / "combined" / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            for ev in events:
                src = patch_root / ev / split / sub
                if not src.exists():
                    continue
                for f in sorted(src.glob("*.npy")):
                    shutil.copy2(f, dst / f.name)

    print("\n  [COMBINED]")
    for split in ("train", "val", "test"):
        n = len(list((patch_root / "combined" / split / "img").glob("*.npy")))
        print(f"    {split}: {n} patches")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", type=int, default=None)
    args = parser.parse_args()

    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    ds          = cfg["dataset"]
    patch_size  = args.patch_size or ds["patch_size"]
    overlap     = ds["overlap"]
    min_ff      = ds["min_flood_fraction"]
    split_r     = tuple(ds["split"])
    patch_root  = Path(cfg["paths"]["data_processed"]) / "patches"
    stack_root  = Path(cfg["paths"]["data_processed"]) / "model_stacks"
    label_root  = Path(cfg["paths"]["data_processed"]) / "flood_labels"

    print(f"Patch size={patch_size}  overlap={overlap}  "
          f"min_flood={min_ff:.1%}  split={split_r}")

    event_names = []
    for ev in cfg["events"]:
        name      = ev["name"]
        stack_npy = stack_root / f"{name}_model_stack_7ch.npy"
        mask_tif  = label_root / name / "flood_mask.tif"

        if not stack_npy.exists():
            raise FileNotFoundError(f"Missing model stack: {stack_npy}")
        if not mask_tif.exists():
            raise FileNotFoundError(f"Missing flood mask: {mask_tif}")

        print(f"\n{'='*55}")
        make_patches(
            str(stack_npy), str(mask_tif),
            out_root       = str(patch_root / name),
            event_name     = name,
            patch_size     = patch_size,
            overlap        = overlap,
            min_flood_frac = min_ff,
            split_ratios   = split_r,
        )
        event_names.append(name)

    print(f"\n{'='*55}")
    merge_combined(patch_root, event_names)

    # Save reproducibility record
    meta = {
        "patch_size": patch_size, "overlap": overlap,
        "min_flood_fraction": min_ff,
        "split_ratios": list(split_r), "seed": SEED,
        "events": event_names,
    }
    with open(patch_root / "patch_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n Done. Patches at: {patch_root}")
    print(f"   Config saved: {patch_root / 'patch_config.json'}")


if __name__ == "__main__":
    main()