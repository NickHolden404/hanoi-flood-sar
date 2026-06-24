"""
scripts/phase_3/08_inference_fullscene.py

Full-scene inference + CEMS external validation (Yagi only).

Runs all seeds of the best Attention U-Net (attn_unet_both_p20) and produces:
  - Per-seed probability and binary maps
  - CEMS IoU / Precision / Recall per seed (Yagi, AOI03 coverage zone)
  - Mean ± std across seeds (headline metrics for paper Abstract)

Checkpoint path convention:
    outputs/<run_name>/seed_<N>/best.pth

CEMS comparison:
  - Raw IoU computed first, no mask modification (headline metric)
  - Recession-zone analysis run separately as supplementary

Usage:
    python scripts/phase_3/08_inference_fullscene.py
    python scripts/phase_3/08_inference_fullscene.py --run_name attn_unet_both_p20 --seeds 0 1 2 3 4 5
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.features import geometry_mask
import torch
import yaml
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.attention_unet import get_attention_unet
from inference.predict import predict_fullscene, prob_to_binary


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_binary_metrics(pred_bin: np.ndarray,
                           ref_bin:  np.ndarray,
                           mask:     np.ndarray | None = None) -> dict:
    """IoU / Precision / Recall for binary uint8 arrays.
    mask: optional boolean array (True = include pixel in evaluation).
    """
    if mask is not None:
        pred_bin = pred_bin[mask]
        ref_bin  = ref_bin[mask]

    pred = pred_bin.astype(bool)
    ref  = ref_bin.astype(bool)

    tp = (pred &  ref).sum()
    fp = (pred & ~ref).sum()
    fn = (~pred & ref).sum()

    iou       = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return {"iou": float(iou), "precision": float(precision),
            "recall": float(recall), "tp": int(tp),
            "fp": int(fp), "fn": int(fn)}


def rasterise_vector(gpkg_path: str, ref_tif: str) -> np.ndarray:
    """Burn vector polygons to a boolean raster matching ref_tif grid."""
    with rasterio.open(ref_tif) as src:
        transform = src.transform
        shape     = (src.height, src.width)
        crs       = src.crs

    gdf = gpd.read_file(gpkg_path).to_crs(crs)
    if gdf.empty:
        return np.zeros(shape, dtype=bool)

    mask = geometry_mask(
        gdf.geometry,
        transform=transform,
        invert=True,           # True = inside polygon = flood
        out_shape=shape)
    return mask


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="attn_unet_both_p20")
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4,5])
    p.add_argument("--event", default="all",
                   choices=["yagi", "matmo", "all"])
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "configs" / "config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stack_root = Path(cfg["paths"]["data_processed"]) / "model_stacks"
    pred_root  = Path(cfg["paths"]["output_root"]) / "predictions" / args.run_name
    out_dir    = Path(cfg["paths"]["output_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    cems_dir   = Path(cfg["paths"]["cems_processed"])
    cems_mask_path = str(cems_dir /
        "EMSR754_AOI03_flood_extent_dissolved_utm48n.gpkg")
    cems_aoi_path = str(cems_dir / "EMSR754_AOI03_DEL_PRODUCT_areaOfInterestA_v1_clipped_utm48n.gpkg")

    events = cfg["events"] if args.event == "all" else [
        e for e in cfg["events"] if e["name"] == args.event]

    # Results container: {event: {seed: metrics}}
    all_results = {ev["name"]: {} for ev in events}

    # ── per-seed loop ─────────────────────────────────────────────────────────
    for seed in args.seeds:
        ckpt = (Path(cfg["paths"]["output_root"]) /
                args.run_name / f"seed_{seed}" / "best.pth")
        if not ckpt.exists():
            print(f"  [seed {seed}] Checkpoint not found: {ckpt} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Seed {seed} — loading {ckpt.name}")

        model = get_attention_unet(cfg)
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

            seed_metrics = {"event": name, "seed": seed}

            # ── CEMS validation (Yagi only) ───────────────────────────────
            if name == "yagi":
                print(f"  [{name.upper()}] CEMS validation...")

                # Load binary prediction
                with rasterio.open(out_binary) as src:
                    pred_arr = src.read(1)

                # Rasterise CEMS flood extent (raw — no modification)
                cems_flood = rasterise_vector(cems_mask_path, out_binary)

                # Rasterise CEMS AOI coverage zone (evaluation boundary)
                cems_aoi   = rasterise_vector(cems_aoi_path, out_binary)

                # ── STEP 1: raw CEMS IoU (headline — no mask modification) ──
                raw = compute_binary_metrics(pred_arr, cems_flood.astype(np.uint8),
                                             mask=cems_aoi)
                seed_metrics["cems_raw"] = raw
                print(f"    CEMS IoU (raw, headline): {raw['iou']:.4f}  "
                      f"Prec={raw['precision']:.4f}  Rec={raw['recall']:.4f}")

                # ── STEP 2: supplementary recession-zone analysis ──────────
                # Flag FPs that fall in pixels with low SRTM elevation
                # (< mean + 0.5*std  plausible recession zone)
                # This contextualises the headline metric but does NOT alter it.
                try:
                    dem_stack = stack_root / f"{name}_model_stack_7ch.tif"
                    with rasterio.open(str(dem_stack)) as src:
                        dem_ch = src.read(7)   # Ch7 = DEM (z-scored)

                    low_elev = dem_ch < (dem_ch[cems_aoi].mean() +
                                         0.5 * dem_ch[cems_aoi].std())
                    recession_zone = low_elev & cems_aoi & ~cems_flood.astype(bool)

                    pred_bool  = pred_arr.astype(bool)
                    fp_total   = int((pred_bool & ~cems_flood & cems_aoi).sum())
                    fp_in_rec  = int((pred_bool & recession_zone).sum())
                    recession_supplementary = {
                        "fp_total_in_aoi":       fp_total,
                        "fp_in_recession_zone":  fp_in_rec,
                        "fraction_fp_in_recession": (fp_in_rec / fp_total
                                                      if fp_total > 0 else 0.0)
                    }
                    seed_metrics["cems_recession_supplementary"] = recession_supplementary
                    print(f"    Recession zone FP fraction: "
                          f"{recession_supplementary['fraction_fp_in_recession']:.3f} "
                          f"({fp_in_rec}/{fp_total} FPs)")
                except Exception as e:
                    print(f"    Recession analysis skipped: {e}")

            all_results[name][seed] = seed_metrics

    # ── aggregate across seeds ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AGGREGATE RESULTS (mean ± std across seeds)")
    print(f"{'='*60}")

    summary = {}
    for ev_name, seed_dict in all_results.items():
        summary[ev_name] = {}
        if not seed_dict:
            continue

        if ev_name == "yagi" and any("cems_raw" in v for v in seed_dict.values()):
            iou_vals  = [v["cems_raw"]["iou"]       for v in seed_dict.values()
                         if "cems_raw" in v]
            prec_vals = [v["cems_raw"]["precision"]  for v in seed_dict.values()
                         if "cems_raw" in v]
            rec_vals  = [v["cems_raw"]["recall"]     for v in seed_dict.values()
                         if "cems_raw" in v]

            cems_agg = {
                "iou_mean":       float(np.mean(iou_vals)),
                "iou_std":        float(np.std(iou_vals)),
                "precision_mean": float(np.mean(prec_vals)),
                "precision_std":  float(np.std(prec_vals)),
                "recall_mean":    float(np.mean(rec_vals)),
                "recall_std":     float(np.std(rec_vals)),
                "n_seeds":        len(iou_vals),
            }
            summary[ev_name]["cems_aggregate"] = cems_agg

            print(f"\n  YAGI — CEMS External Validation (headline):")
            print(f"    IoU       = {cems_agg['iou_mean']:.4f} ± {cems_agg['iou_std']:.4f}")
            print(f"    Precision = {cems_agg['precision_mean']:.4f} ± {cems_agg['precision_std']:.4f}")
            print(f"    Recall    = {cems_agg['recall_mean']:.4f} ± {cems_agg['recall_std']:.4f}")
            print(f"    n_seeds   = {cems_agg['n_seeds']}")
            print(f"\n  *** ABSTRACT HEADLINE: IoU = {cems_agg['iou_mean']:.4f} ± {cems_agg['iou_std']:.4f} ***")

        summary[ev_name]["per_seed"] = {
            str(s): v for s, v in seed_dict.items()}

    # Save full results
    out_path = out_dir / "cems_validation_metrics.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n Results saved  {out_path}")

    # Also save a clean summary for the report
    report_out = out_dir / "cems_validation_summary.json"
    report_summary = {}
    for ev_name, s in summary.items():
        if "cems_aggregate" in s:
            report_summary[ev_name] = s["cems_aggregate"]
    with open(report_out, "w") as f:
        json.dump(report_summary, f, indent=2)
    print(f"   Summary  {report_out}")


if __name__ == "__main__":
    main()