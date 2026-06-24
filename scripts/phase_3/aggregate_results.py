"""
scripts/phase_3/aggregate_results.py

Collects test_metrics.json from all seeds for all experiments and
produces mean ± std tables ready for the paper.

Run after ALL seeds (0–4) have completed for ALL experiments.

Usage:
    python scripts/phase_3/aggregate_results.py
    python scripts/phase_3/aggregate_results.py --seeds 0 1 2  # partial

Outputs:
    outputs/tables/results_main_comparison.json   U-Net vs Attn U-Net
    outputs/tables/results_ablation.json          4 channel configs
    outputs/tables/results_cross_event.json       generalization
    outputs/tables/results_all_summary.csv        one flat CSV for the paper
    (all printed to console as formatted tables)
"""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
METRICS = ["iou", "dice", "f1", "precision", "recall", "pixel_acc"]


# ── helpers ────────────────────────────────────────────────────────────────

def load_cfg() -> dict:
    with open(ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def agg(vals: list) -> dict:
    """Return mean ± std dict for a list of scalar values, ignoring None/nan."""
    clean = [v for v in vals if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "vals": vals}
    return {
        "mean": float(np.mean(clean)),
        "std":  float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "n":    len(clean),
        "vals": vals,
    }


def fmt(a: dict) -> str:
    if np.isnan(a["mean"]):
        return "   —   "
    return f"{a['mean']:.4f}±{a['std']:.4f}"


def load_seed_metrics(run_dir: Path, seed: int,
                      subkey: str = "metrics") -> dict:
    """
    Load test_metrics.json for a given run dir and seed.
    Looks for:  <run_dir>/seed_<N>/test_metrics.json
    Returns metric dict or {} if not found.
    """
    p = run_dir / f"seed_{seed}" / "test_metrics.json"
    if not p.exists():
        return {}
    with open(p) as f:
        data = json.load(f)
    return data.get(subkey, data)  # graceful fallback if key missing


def collect(run_dir: Path, seeds: list, subkey: str = "metrics") -> dict:
    """Collect metric values across seeds  {metric: [val_s0, val_s1, ...]}"""
    per_metric = defaultdict(list)
    for s in seeds:
        m = load_seed_metrics(run_dir, s, subkey)
        for metric in METRICS:
            per_metric[metric].append(m.get(metric, float("nan")))
    return {k: agg(v) for k, v in per_metric.items()}


def print_table(title: str, rows: list, cols: list):
    """rows: list of (label, {metric: agg_dict})"""
    col_w = 16
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    header = f"{'Run':<28}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("-" * (28 + col_w * len(cols)))
    for label, metrics in rows:
        row = f"{label:<28}"
        row += "".join(f"{fmt(metrics.get(c, {'mean':float('nan'),'std':0})):>{col_w}}"
                       for c in cols)
        print(row)


# ── main comparison: U-Net vs Attention U-Net ──────────────────────────────

def results_main(cfg: dict, seeds: list, out_dir: Path) -> dict:
    out_root = Path(cfg["paths"]["output_root"])
    runs = [
        ("U-Net (both)",          out_root / "unet_both"),
        ("Attention U-Net (both)", out_root / "attention_unet_both"),
        ("U-Net (yagi)",          out_root / "unet_yagi"),
        ("Attention U-Net (yagi)", out_root / "attention_unet_yagi"),
        ("U-Net (matmo)",         out_root / "unet_matmo"),
        ("Attention U-Net (matmo)", out_root / "attention_unet_matmo"),
    ]
    results = {}
    rows = []
    for label, run_dir in runs:
        if not run_dir.exists():
            continue
        agg_metrics = collect(run_dir, seeds)
        results[label] = agg_metrics
        rows.append((label, agg_metrics))

    print_table("MAIN COMPARISON — U-Net vs Attention U-Net",
                rows, ["iou", "dice", "f1", "precision", "recall"])

    out = out_dir / "results_main_comparison.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Saved: {out}")
    return results


# ── ablation ───────────────────────────────────────────────────────────────

def results_ablation(cfg: dict, seeds: list, out_dir: Path) -> dict:
    out_root = Path(cfg["paths"]["output_root"])
    exps     = cfg["ablation"]["experiments"]
    results  = {}
    rows     = []

    for exp in exps:
        name     = exp["name"]
        chs      = exp["channels"]
        run_dir  = out_root / f"ablation_{name}"
        if not run_dir.exists():
            print(f"  MISSING: {run_dir} — skipping {name}")
            continue
        agg_m = collect(run_dir, seeds)
        results[name] = {"channels": chs, "n_channels": len(chs),
                         "metrics": agg_m}
        rows.append((f"{name} ({len(chs)}ch)", agg_m))

    print_table("ABLATION STUDY — Channel Configurations (Attention U-Net)",
                rows, ["iou", "dice", "f1", "precision", "recall"])

    out = out_dir / "results_ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Saved: {out}")
    return results


# ── cross-event ────────────────────────────────────────────────────────────

def results_cross_event(cfg: dict, seeds: list, out_dir: Path) -> dict:
    """
    Reads per-seed cross_event_seed_N.json files (written by 07_cross_event.py)
    and aggregates own_metrics + cross_metrics across seeds.
    """
    EXPS = [
        ("cross_yagi_train",  "yagi",  "matmo"),
        ("cross_matmo_train", "matmo", "yagi"),
        ("cross_both_train",  "both",  "both"),
    ]
    results  = {}
    rows_own = []
    rows_crs = []

    for run_name, train_ev, cross_ev in EXPS:
        own_vals   = defaultdict(list)
        cross_vals = defaultdict(list)

        for s in seeds:
            p = out_dir / f"cross_event_seed_{s}.json"
            if not p.exists():
                for m in METRICS:
                    own_vals[m].append(float("nan"))
                    cross_vals[m].append(float("nan"))
                continue
            with open(p) as f:
                data = json.load(f)
            r = data.get(run_name, {})
            for m in METRICS:
                own_vals[m].append(r.get("own_metrics",   {}).get(m, float("nan")))
                cross_vals[m].append(r.get("cross_metrics", {}).get(m, float("nan")))

        own_agg   = {m: agg(own_vals[m])   for m in METRICS}
        cross_agg = {m: agg(cross_vals[m]) for m in METRICS}
        delta_iou = agg([c - o for c, o in
                         zip(cross_vals["iou"], own_vals["iou"])
                         if not (np.isnan(c) or np.isnan(o))])

        results[run_name] = {
            "train_event":   train_ev,
            "cross_test":    cross_ev,
            "own_metrics":   own_agg,
            "cross_metrics": cross_agg,
            "delta_iou":     delta_iou,
        }
        lbl_own   = f"{run_name} (own)"
        lbl_cross = f"{run_name} (cross)"
        rows_own.append((lbl_own,   own_agg))
        rows_crs.append((lbl_cross, cross_agg))

    print_table("CROSS-EVENT GENERALIZATION — own-event test",
                rows_own, ["iou", "dice", "f1"])
    print_table("CROSS-EVENT GENERALIZATION — cross-event test",
                rows_crs, ["iou", "dice", "f1"])

    # ΔIoU summary
    print(f"\n  {'Run':<24} {'ΔIoU (cross−own)':>20}")
    print("  " + "-" * 46)
    for rn, r in results.items():
        d = r["delta_iou"]
        print(f"  {rn:<24} {fmt(d):>20}")

    out = out_dir / "results_cross_event.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n   Saved: {out}")
    return results


# ── flat CSV for paper ─────────────────────────────────────────────────────

def write_summary_csv(main_r: dict, abl_r: dict, cross_r: dict,
                      out_dir: Path):
    rows = []

    def add(section, label, metrics: dict):
        row = {"section": section, "run": label}
        for m in METRICS:
            a = metrics.get(m, {"mean": float("nan"), "std": float("nan")})
            row[f"{m}_mean"] = round(a["mean"], 4)
            row[f"{m}_std"]  = round(a["std"],  4)
        rows.append(row)

    for label, m in main_r.items():
        add("main_comparison", label, m)
    for name, d in abl_r.items():
        add("ablation", name, d["metrics"])
    for rn, d in cross_r.items():
        add("cross_own",   f"{rn}_own",   d["own_metrics"])
        add("cross_event", f"{rn}_cross", d["cross_metrics"])

    out = out_dir / "results_all_summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n   Summary CSV saved: {out}")


# ── entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate multi-seed results")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="Seeds to include (default: all from config)")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    cfg_p   = Path(args.config) if args.config else ROOT / "configs" / "config.yaml"
    cfg     = yaml.safe_load(open(cfg_p))
    seeds   = args.seeds if args.seeds else cfg["experiment"]["seeds"]
    out_dir = Path(cfg["paths"]["output_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Aggregating seeds: {seeds}")
    print(f"Output dir: {out_dir}\n")

    main_r  = results_main(cfg, seeds, out_dir)
    abl_r   = results_ablation(cfg, seeds, out_dir)
    cross_r = results_cross_event(cfg, seeds, out_dir)
    write_summary_csv(main_r, abl_r, cross_r, out_dir)

    print(f"\n{'='*70}")
    print("  Aggregation complete.")
    print(f"  Seeds included : {seeds}  (n={len(seeds)})")
    missing = [s for s in seeds
               if not any((Path(cfg["paths"]["output_root"]) /
                           f"cross_event_seed_{s}.json").exists()
                          for _ in [None])]
    if missing:
        print(f"  WARNING — missing seed files: {missing}")
        print("  Re-run missing seeds before treating results as final.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()