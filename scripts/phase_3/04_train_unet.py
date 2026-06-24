"""
scripts/phase_3/04_train_unet.py

Trains the baseline U-Net for one seed. Run once per seed — independently.

Usage:
    python scripts/phase_3/04_train_unet.py --event both --seed 0
    python scripts/phase_3/04_train_unet.py --event both --seed 1
    ...
    python scripts/phase_3/04_train_unet.py --event both --seed 4

Outputs land in:
    outputs/unet_<event>/seed_<N>/
        best.pth
        last.pth
        val_metrics.json
        test_metrics.json
        train_log.csv
        config_used.yaml
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description="Train baseline U-Net (one seed)")
    p.add_argument("--event", default="both",
                   choices=["yagi", "matmo", "both"],
                   help="Training event split")
    p.add_argument("--seed", type=int, required=True,
                   help="Random seed (0–4 for 5-seed experiment)")
    p.add_argument("--run_name", default=None,
                   help="Override auto-generated run name")
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.run_name or f"unet_{args.event}"
    out_subdir = f"seed_{args.seed}"

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "phase_3" / "train.py"),
        "--model",      "unet",
        "--event",      args.event,
        "--run_name",   run_name,
        "--seed",       str(args.seed),
        "--out_subdir", out_subdir,
    ]
    print(f"[04_train_unet] seed={args.seed}  event={args.event}  "
          f"run={run_name}/{out_subdir}")
    print(f"Command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()