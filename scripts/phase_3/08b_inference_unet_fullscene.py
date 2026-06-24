"""
scripts/phase_3/08b_inference_unet_fullscene.py

Full-scene inference for baseline U-Net (unet_both, seeds 0-4).
Produces per-seed probability + binary maps for visualisation comparison
against Attention U-Net. No CEMS validation (Attention U-Net is primary model).

Outputs:
  outputs/predictions/unet_both/<event>/seed_{0..4}/probability.tif
  outputs/predictions/unet_both/<event>/seed_{0..4}/binary.tif
  outputs/predictions/unet_both/<event>/ensemble_probability.tif
  outputs/predictions/unet_both/<event>/ensemble_binary.tif

Usage:
    python scripts/phase_3/08b_inference_unet_fullscene.py
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.unet import get_unet
from inference.predict import predict_fullscene, prob_to_binary


def build_ensemble(prob_paths: list[Path], out_prob: Path,
                   out_binary: Path, threshold: float = 0.5):
    arrays, profile = [], None
    for p in prob_paths:
        with rasterio.open(str(p)) as src:
            arrays.append(src.read(1).astype(np.float32))
            if profile is None:
                profile = src.profile.copy()

    mean_prob = np.mean(arrays, axis=0)
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(str(out_prob), "w", **profile) as dst:
        dst.write(mean_prob, 1)

    binary = (mean_prob >= threshold).astype(np.uint8)
    flood_frac = binary.mean() * 100
    profile.update(dtype=rasterio.uint8, nodata=255)
    with rasterio.open(str(out_binary), "w", **profile) as dst:
        dst.write(binary, 1)
    print(f"  Ensemble binary : {out_binary.name}  (flood={flood_frac:.2f}%)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="unet_both")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--event", default="all",
                   choices=["yagi", "matmo", "all"])
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = ROOT / "configs" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack_root = Path(cfg["paths"]["data_processed"]) / "model_stacks"
    pred_root  = Path(cfg["paths"]["output_root"]) / "predictions" / args.run_name
    weights_root = Path(cfg["paths"]["output_root"]) / args.run_name

    events = cfg["events"] if args.event == "all" else [
        e for e in cfg["events"] if e["name"] == args.event]

    for seed in args.seeds:
        ckpt = weights_root / f"seed_{seed}" / "best.pth"
        if not ckpt.exists():
            print(f"  [seed {seed}] Checkpoint not found: {ckpt} — skipping")
            continue

        print(f"\n{'='*55}")
        print(f"  Seed {seed} — {ckpt}")

        model = get_unet(cfg)
        model.load_state_dict(
            torch.load(ckpt, map_location=device, weights_only=True))
        model.to(device).eval()

        for ev in events:
            name       = ev["name"]
            stack_path = str(stack_root / f"{name}_model_stack_7ch.tif")
            out_prob   = str(pred_root / name / f"seed_{seed}" / "probability.tif")
            out_binary = str(pred_root / name / f"seed_{seed}" / "binary.tif")
            Path(out_prob).parent.mkdir(parents=True, exist_ok=True)

            print(f"  [{name.upper()}] Inference...")
            predict_fullscene(model, stack_path, out_prob, cfg, device)
            prob_to_binary(out_prob, out_binary, threshold=args.threshold)

            with rasterio.open(out_binary) as src:
                frac = src.read(1).mean() * 100
            print(f"    flood={frac:.2f}%")

    # Build ensemble per event
    print(f"\n{'='*55}  Building ensembles...")
    for ev in events:
        name = ev["name"]
        prob_paths = [pred_root / name / f"seed_{s}" / "probability.tif"
                      for s in args.seeds
                      if (pred_root / name / f"seed_{s}" / "probability.tif").exists()]
        if prob_paths:
            build_ensemble(prob_paths,
                           pred_root / name / "ensemble_probability.tif",
                           pred_root / name / "ensemble_binary.tif",
                           args.threshold)

    print(f"\n U-Net full-scene inference complete.")
    print(f"   Outputs: {pred_root}")


if __name__ == "__main__":
    main()