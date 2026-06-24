"""
scripts/phase_3/06_ablation.py

Channel ablation study — Attention U-Net, one seed at a time.
Runs 4 channel configurations defined in config.yaml  ablation.experiments.

Run once per seed — independently, without keeping the computer running.

Usage:
    python scripts/phase_3/06_ablation.py --seed 0
    python scripts/phase_3/06_ablation.py --seed 1
    ...
    python scripts/phase_3/06_ablation.py --seed 4

Per-seed outputs land in:
    outputs/ablation_<config_name>/seed_<N>/
        best.pth  last.pth  val_metrics.json
        test_metrics.json  train_log.csv  config_used.yaml

Run aggregate_results.py after all seeds complete to produce mean ± std table.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Ablation study (one seed)")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (0–4)")
    p.add_argument("--config", default=None,
                   help="Path to config.yaml (default: configs/config.yaml)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config) if args.config else ROOT / "configs" / "config.yaml"

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    exps    = cfg["ablation"]["experiments"]
    out_dir = Path(cfg["paths"]["output_root"]) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_results = {}

    for exp in exps:
        name     = exp["name"]
        chs      = exp["channels"]
        run_name = f"ablation_{name}"
        out_subdir = f"seed_{args.seed}"

        print(f"\n{'='*60}")
        print(f"  ABLATION seed={args.seed}  config={name}  "
              f"({len(chs)}ch: {chs})")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "phase_3" / "train.py"),
            "--model",      "attention_unet",
            "--event",      "both",
            "--run_name",   run_name,
            "--seed",       str(args.seed),
            "--out_subdir", out_subdir,
            "--channels",   *chs,
        ]
        subprocess.run(cmd, check=True, cwd=str(ROOT))

        metrics_file = (Path(cfg["paths"]["output_root"]) /
                        run_name / out_subdir / "test_metrics.json")
        if metrics_file.exists():
            with open(metrics_file) as f:
                seed_results[name] = json.load(f)
            m = seed_results[name].get("metrics", {})
            print(f"   IoU={m.get('iou',0):.4f}  Dice={m.get('dice',0):.4f}  "
                  f"F1={m.get('f1',0):.4f}")
        else:
            print(f"  WARNING: metrics not found at {metrics_file}")
            seed_results[name] = {}

    # Save this seed's results
    seed_out = out_dir / f"ablation_seed_{args.seed}.json"
    with open(seed_out, "w") as f:
        json.dump({"seed": args.seed, "results": seed_results}, f, indent=2)
    print(f"\nSeed {args.seed} ablation saved  {seed_out}")
    print("Run aggregate_results.py after all seeds complete.")


if __name__ == "__main__":
    main()