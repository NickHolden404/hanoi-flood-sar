"""
Step 2: Remove thermal noise and border noise from orbit-corrected products.
Tool: SNAP GPT
Output: /data_processed/noise_removed/
"""

import subprocess, logging, json, time
from pathlib import Path
from phase2_config import SNAP_GPT, SCENES, DIRS

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

def build_noise_graph(input_dim: Path, output_dim: Path) -> Path:
    graph_xml = f"""<graph id="noise_graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>{input_dim}</file></parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>
  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources>
    <parameters>
      <file>{output_dim}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
</graph>"""
    graph_path = DIRS["graphs"] / f"noise_{output_dim.stem}.xml"
    graph_path.write_text(graph_xml)
    return graph_path


def remove_noise(scene_key: str):
    input_dim  = DIRS["orbit"]  / f"{scene_key}_orbit.dim"
    output_dim = DIRS["noise"] / f"{scene_key}_noise"

    if not input_dim.exists():
        raise FileNotFoundError(f"Orbit product missing: {input_dim}")

    if (output_dim.with_suffix(".dim")).exists():
        print(f"    {scene_key}: already denoised, skipping")
        return

    graph_path = build_noise_graph(input_dim, output_dim)
    cmd = [str(SNAP_GPT), str(graph_path), "-c", "4G", "-q", "4"]
    print(f"  ⚙  {scene_key}: removing thermal noise...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logging.error(f"NOISE FAIL | {scene_key} | {result.stderr[-500:]}")
        raise RuntimeError(f"Noise removal failed: {scene_key}")

    # Verify: check output .dim exists and data directory is non-empty
    out_data_dir = output_dim.parent / (output_dim.stem + ".data")
    band_files = list(out_data_dir.glob("*.img")) if out_data_dir.exists() else []
    artifact_check = "PASS" if len(band_files) >= 2 else "WARN: fewer bands than expected"

    logging.info(
        f"NOISE OK | {scene_key} | bands={len(band_files)} | "
        f"artifact_check={artifact_check} | elapsed={elapsed:.1f}s"
    )
    print(f"   {scene_key}: denoised in {elapsed:.0f}s ({artifact_check})")


if __name__ == "__main__":
    print("\n=== STEP 2: THERMAL NOISE REMOVAL ===\n")
    t_start = time.time()
    for key in SCENES:
        remove_noise(key)
    print(f"\n Step 2 complete — Total: {(time.time()-t_start)/60:.1f} min")