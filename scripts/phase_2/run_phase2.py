"""
run_phase2.py — Execute Phase 2 end-to-end.
Run from hanoi_flood_sar project root with flood_sar env activated.
"""

import subprocess, sys, time
from pathlib import Path

STEPS = [
    ("Init directories",           "phase2_init_dirs.py"),
    ("Step 5: Terrain correction",  "phase2_step5_terrain.py"),   # includes orbit/noise/cal
    ("Step 4b: Speckle filter",     "phase2_step4b_speckle_post_tc.py"),
    ("Step 6: Clip to AOI",         "phase2_step6_clip.py"),
    ("Step 7: Stack assembly",      "phase2_step7_stack.py"),
    ("Step 8: Normalization",       "phase2_step8_normalize.py"),
    ("Step 9: QC",                  "phase2_step9_qc.py"),
    ("Step 10: Final report",       "phase2_step10_final_report.py"),
]

print("=" * 60)
print("PHASE 2: Sentinel-1 SAR Preprocessing Pipeline")
print("Multi-Event Urban Flood Segmentation — Hanoi, Vietnam")
print("=" * 60)
t_global = time.time()

for step_name, script in STEPS:
    print(f"\n{'─'*60}")
    print(f"▶  {step_name}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script], check=False)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n FAILED: {script} (exit code {result.returncode})")
        print(f"   Fix error and re-run from this step.")
        sys.exit(1)
    print(f"   Completed in {elapsed/60:.1f} min")

total = time.time() - t_global
print(f"\n{'='*60}")
print(f" Phase 2 complete — Total time: {total/60:.0f} min")
print(f"{'='*60}")