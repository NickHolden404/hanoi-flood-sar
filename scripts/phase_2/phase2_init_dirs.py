# phase2_init_dirs.py
from phase2_config import DIRS
import logging, json
from datetime import datetime

for name, path in DIRS.items():
    path.mkdir(parents=True, exist_ok=True)
    print(f"   {name:12s}  {path}")

# Initialise processing log
log_path = DIRS["logs"] / "phase2_processing.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Phase 2 initialised")
print(f"\nLog file: {log_path}")