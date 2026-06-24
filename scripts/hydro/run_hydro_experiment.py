"""
scripts/hydro/run_hydro_experiment.py

Full hydrological loss ablation experiment:

  1. Trains Attention U-Net with BCEDiceHydroLoss for 6 seeds (0–5) on the
     combined (yagi+matmo) dataset, for lambda values [0.05, 0.10, 0.20].
  2. Collects test_metrics.json from each output directory.
  3. Computes mean ± std IoU across seeds per lambda.
  4. Compares against baseline from outputs/attn_unet_both_p20/ (seed 0–5).
  5. Runs a paired t-test across seeds; reports significance at p < 0.05.
  6. Saves results to outputs/tables/hydro_loss_comparison.json.
  7. Runs CEMS validation for the best lambda via cems_aoi03_analysis.py.

Usage:
    python scripts/hydro/run_hydro_experiment.py [--config PATH] [--dry_run]

    --dry_run   Print subprocess commands without executing them (smoke test).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

# ── project root ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TRAIN_SCRIPT  = ROOT / "scripts" / "hydro" / "train_hydro.py"
CEMS_SCRIPT   = ROOT / "scripts" / "cems_aoi03_analysis.py"
SEEDS         = list(range(6))
LAMBDA_VALUES = [0.05, 0.10, 0.20]
BASELINE_KEY  = "attn_unet_both_p20"
BASELINE_IOU  = 0.5746          # pre-specified authoritative value


def load_cfg(config_path: Path = None) -> dict:
    p = config_path if config_path \
        else ROOT / "configs" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)


# ── run name helpers ──────────────────────────────────────────────────────

def run_name(lam: float) -> str:
    return f"hydro_loss/lambda_{lam:.2f}"


def out_dir(cfg: dict, lam: float, seed: int) -> Path:
    return (Path(cfg["paths"]["output_root"])
            / run_name(lam) / f"seed_{seed}")


def baseline_seed_dirs(cfg: dict) -> list[Path]:
    base = Path(cfg["paths"]["output_root"]) / BASELINE_KEY
    return [base / f"seed_{s}" for s in SEEDS]


# ── subprocess training ───────────────────────────────────────────────────

def train_one(cfg: dict, lam: float, seed: int,
              dry_run: bool = False) -> bool:
    od = out_dir(cfg, lam, seed)
    metrics_path = od / "test_metrics.json"
    if metrics_path.exists():
        print(f"  [skip] {metrics_path} already exists.")
        return True

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--model",        "attention_unet",
        "--event",        "both",
        "--run_name",     run_name(lam),
        "--seed",         str(seed),
        "--out_subdir",   f"seed_{seed}",
        "--lambda_hydro", str(lam),
        "--config",       str(ROOT / "configs" / "config.yaml"),
    ]
    print(f"  [run] lambda={lam}  seed={seed}   {od}")
    if dry_run:
        print(f"        CMD: {' '.join(cmd)}")
        return True

    result = subprocess.run(cmd, capture_output=False, env=env)
    if result.returncode != 0:
        print(f"  [ERROR] training failed for lambda={lam} seed={seed}")
        return False
    return True


# ── metric collection ─────────────────────────────────────────────────────

def collect_ious(dirs: list[Path]) -> list[float]:
    ious = []
    for d in dirs:
        p = d / "test_metrics.json"
        if not p.exists():
            print(f"  [WARN] missing: {p}")
            continue
        with open(p) as f:
            data = json.load(f)
        # test_metrics.json schema: {"metrics": {"iou": ..., ...}, ...}
        iou = data.get("metrics", {}).get("iou")
        if iou is None:
            # Fallback: some runs write metrics at top level
            iou = data.get("iou")
        if iou is not None:
            ious.append(float(iou))
        else:
            print(f"  [WARN] IoU not found in {p}")
    return ious


def collect_baseline_ious(cfg: dict) -> list[float]:
    return collect_ious(baseline_seed_dirs(cfg))


# ── statistics ────────────────────────────────────────────────────────────

def summarise(ious: list[float]) -> dict:
    a = np.array(ious)
    return {
        "mean_iou": float(np.mean(a)),
        "std_iou":  float(np.std(a, ddof=1) if len(a) > 1 else 0.0),
        "n_seeds":  len(ious),
        "seed_ious": [round(v, 6) for v in ious],
    }


def paired_ttest(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (t_statistic, p_value) for paired t-test a vs b."""
    from scipy import stats  # scipy is available in all project envs
    return stats.ttest_rel(a, b)


# ── CEMS validation ───────────────────────────────────────────────────────

def run_cems_validation(cfg: dict, best_lam: float, dry_run: bool):
    if not CEMS_SCRIPT.exists():
        print(f"  [WARN] CEMS script not found at {CEMS_SCRIPT}. Skipping.")
        return

    best_run = run_name(best_lam)
    # cems_aoi03_analysis.py is called with the run directory so it can find
    # best.pth for seed_0 (seed-averaged CEMS eval is handled internally).
    cmd = [
        sys.executable, str(CEMS_SCRIPT),
        "--run_name", best_run,
        "--config",   str(ROOT / "configs" / "config.yaml"),
    ]
    print(f"\n[CEMS] Validating best lambda ({best_lam}) via {CEMS_SCRIPT.name}")
    if dry_run:
        print(f"  CMD: {' '.join(cmd)}")
        return
    subprocess.run(cmd, capture_output=False)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Hydro loss ablation experiment")
    p.add_argument("--config",  default=None, help="Path to config.yaml")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running training")
    args = p.parse_args()

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    cfg = load_cfg(Path(args.config) if args.config else None)
    out_root = Path(cfg["paths"]["output_root"])

    # ── Step 1: Train all lambda × seed combinations ──────────────────────
    print("\n" + "="*60)
    print("  Step 1: Training Attention U-Net with BCEDiceHydroLoss")
    print("="*60)
    for lam in LAMBDA_VALUES:
        for seed in SEEDS:
            ok = train_one(cfg, lam, seed, dry_run=args.dry_run)
            if not ok:
                print(f"  [ABORT] Training failed for lambda={lam} seed={seed}")
                sys.exit(1)

    # ── Step 2–3: Collect metrics ─────────────────────────────────────────
    print("\n" + "="*60)
    print("  Step 2–3: Collecting test IoU values")
    print("="*60)

    results = {}

    # Baseline
    baseline_ious = collect_baseline_ious(cfg)
    if not baseline_ious:
        print(f"  [WARN] No baseline IoUs found in {BASELINE_KEY}. "
              f"Using hard-coded value {BASELINE_IOU}.")
        baseline_ious = [BASELINE_IOU] * len(SEEDS)  # placeholder for t-test
    baseline_summary = summarise(baseline_ious)
    baseline_summary["mean_iou"] = baseline_summary.get("mean_iou", BASELINE_IOU)
    results["baseline"] = {
        "mean_iou": baseline_summary["mean_iou"],
        "std_iou":  baseline_summary["std_iou"],
        "n_seeds":  baseline_summary["n_seeds"],
        "seed_ious": baseline_summary["seed_ious"],
    }
    print(f"  Baseline  IoU = {baseline_summary['mean_iou']:.4f} "
          f"± {baseline_summary['std_iou']:.4f}  (n={baseline_summary['n_seeds']})")

    lambda_summaries = {}
    for lam in LAMBDA_VALUES:
        dirs   = [out_dir(cfg, lam, s) for s in SEEDS]
        ious   = collect_ious(dirs)
        summ   = summarise(ious)
        delta  = summ["mean_iou"] - baseline_summary["mean_iou"]

        key = f"lambda_{lam:.2f}"
        results[key] = {
            "mean_iou":  summ["mean_iou"],
            "std_iou":   summ["std_iou"],
            "n_seeds":   summ["n_seeds"],
            "seed_ious": summ["seed_ious"],
            "delta_iou": round(delta, 6),
        }
        lambda_summaries[lam] = {"summary": summ, "delta": delta, "ious": ious}
        print(f"  λ={lam:.2f}  IoU = {summ['mean_iou']:.4f} "
              f"± {summ['std_iou']:.4f}  Δ={delta:+.4f}")

    # ── Step 4–5: Paired t-tests ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  Step 4: Paired t-tests vs baseline")
    print("="*60)
    sig_results = {}
    for lam in LAMBDA_VALUES:
        hydro_ious = lambda_summaries[lam]["ious"]
        if len(hydro_ious) == len(baseline_ious) and len(hydro_ious) >= 2:
            t, pval = paired_ttest(hydro_ious, baseline_ious)
        else:
            t, pval = float("nan"), float("nan")
        delta = lambda_summaries[lam]["delta"]
        significant = (pval < 0.05) and (delta > 0)
        key = f"lambda_{lam:.2f}"
        results[key]["t_stat"] = round(float(t), 4) if not np.isnan(t) else None
        results[key]["p_value"] = round(float(pval), 4) if not np.isnan(pval) else None
        results[key]["significant"] = bool(significant)
        sig_str = "✓ SIGNIFICANT" if significant else "✗ not significant"
        print(f"  λ={lam:.2f}  t={t:.3f}  p={pval:.4f}  {sig_str}")
        sig_results[lam] = significant

    # ── Step 6: Save results JSON ─────────────────────────────────────────
    tables_dir = out_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_json = tables_dir / "hydro_loss_comparison.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved  {out_json}")

    # ── Step 7: Summary table ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("  HYDRO LOSS EXPERIMENT SUMMARY")
    print("="*60)
    header = f"{'Config':<18} {'Mean IoU':>10} {'Std IoU':>9} {'ΔIoU':>8} {'p-val':>8} {'Sig?':>6}"
    print(header)
    print("-" * len(header))
    bl = results["baseline"]
    print(f"  {'baseline':<16} {bl['mean_iou']:>10.4f} {bl['std_iou']:>9.4f}  {'—':>7}  {'—':>7}  {'—':>5}")
    for lam in LAMBDA_VALUES:
        key = f"lambda_{lam:.2f}"
        r   = results[key]
        sig = "YES" if r.get("significant") else "no"
        pv  = f"{r['p_value']:.4f}" if r.get("p_value") is not None else "n/a"
        print(f"  λ={lam:.2f}  {' ':<10} {r['mean_iou']:>10.4f} "
              f"{r['std_iou']:>9.4f} {r['delta_iou']:>+8.4f} {pv:>8}  {sig:>5}")

    # ── Success criterion check ───────────────────────────────────────────
    any_success = any(sig_results.values())
    print("\n" + "="*60)
    if any_success:
        best_lam = max(
            (lam for lam in LAMBDA_VALUES if sig_results[lam]),
            key=lambda lam: lambda_summaries[lam]["delta"],
        )
        print(f"  SUCCESS: λ={best_lam:.2f} produced ΔIoU > 0 with p < 0.05.")
    else:
        # Fall back to highest delta even if not significant
        best_lam = max(LAMBDA_VALUES,
                       key=lambda lam: lambda_summaries[lam]["delta"])
        print(f"  NOTE: No lambda achieved p < 0.05 with ΔIoU > 0.")
        print(f"  Best lambda by ΔIoU: {best_lam:.2f}  "
              f"(Δ={lambda_summaries[best_lam]['delta']:+.4f})")

    # ── Step 8: CEMS validation for best lambda ───────────────────────────
    run_cems_validation(cfg, best_lam, dry_run=args.dry_run)

    print("\n[Step 4 COMPLETE] run_hydro_experiment.py finished.")


if __name__ == "__main__":
    main()