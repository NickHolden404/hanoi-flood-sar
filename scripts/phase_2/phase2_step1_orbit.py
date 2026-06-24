"""
Step 1: Apply precise orbit files to all 4 Sentinel-1 GRD scenes.
Tool: SNAP GPT via subprocess
Output: /data_processed/orbit_corrected/
"""

import subprocess, logging, json, time
from pathlib import Path
from datetime import datetime
from phase2_config import SNAP_GPT, SCENES, DIRS, DIRS

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def build_orbit_graph(input_safe: Path, output_dim: Path) -> Path:
    """Write a SNAP GPT graph XML for orbit correction."""
    graph_xml = f"""<graph id="orbit_graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters>
      <file>{input_safe}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters>
      <file>{output_dim}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
    graph_path = DIRS["graphs"] / f"orbit_{output_dim.stem}.xml"
    graph_path.write_text(graph_xml)
    return graph_path


def apply_orbit(scene_key: str, safe_path: Path):
    """Run orbit correction for a single scene."""
    out_stem = f"{scene_key}_orbit"
    output_dim = DIRS["orbit"] / out_stem

    if (output_dim.with_suffix(".dim")).exists():
        logging.info(f"SKIP {scene_key} — orbit file already exists")
        print(f"    {scene_key}: already processed, skipping")
        return

    graph_path = build_orbit_graph(safe_path, output_dim)

    cmd = [str(SNAP_GPT), str(graph_path), "-c", "4G", "-q", "4"]
    print(f"  ⚙  {scene_key}: applying orbit correction...")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode == 0:
        # Detect orbit type from SNAP stdout
        orbit_type = "Precise" if "Precise" in result.stdout else "Restituted"
        logging.info(
            f"ORBIT OK | {scene_key} | orbit_type={orbit_type} | "
            f"elapsed={elapsed:.1f}s | output={output_dim}.dim"
        )
        print(f"   {scene_key}: orbit applied ({orbit_type}) in {elapsed:.0f}s")
    else:
        logging.error(f"ORBIT FAIL | {scene_key} | stderr: {result.stderr[-500:]}")
        raise RuntimeError(f"Orbit correction failed for {scene_key}.\n{result.stderr[-500:]}")

    return {"scene": scene_key, "orbit_type": orbit_type, "output": str(output_dim) + ".dim"}


if __name__ == "__main__":
    print("\n=== STEP 1: ORBIT FILE APPLICATION ===\n")
    t_start = time.time()
    results = []

    for key, safe_path in SCENES.items():
        if not safe_path.exists():
            raise FileNotFoundError(f"SAFE not found: {safe_path}")
        r = apply_orbit(key, safe_path)
        results.append(r)

    # Save orbit log
    log_json = DIRS["logs"] / "step1_orbit_log.json"
    with open(log_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n Step 1 complete — {len(results)} scenes processed")
    print(f"   Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"   Log saved: {log_json}")