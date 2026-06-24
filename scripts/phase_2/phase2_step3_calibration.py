"""
Step 3: Radiometric calibration  Sigma0 (linear), then convert to dB.
Tool: SNAP GPT for calibration; rasterio + numpy for dB conversion.
Output: /data_processed/calibrated/
  - {scene}_VV_db.tif
  - {scene}_VH_db.tif
"""

import subprocess, logging, json, time
import numpy as np
import rasterio
from rasterio.enums import Resampling
from pathlib import Path
from phase2_config import SNAP_GPT, SCENES, DIRS

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

def build_calibration_graph(input_dim: Path, output_dim: Path) -> Path:
    graph_xml = f"""<graph id="calibration_graph">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>{input_dim}</file></parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputDNBand>false</outputDNBand>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <file>{output_dim}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
</graph>"""
    graph_path = DIRS["graphs"] / f"cal_{output_dim.stem}.xml"
    graph_path.write_text(graph_xml)
    return graph_path


def linear_to_db(input_tif: Path, output_tif: Path, band_name: str):
    """Convert linear Sigma0 raster to dB scale."""
    with rasterio.open(input_tif) as src:
        profile = src.profile.copy()
        data = src.read(1).astype(np.float32)

    # Replace zeros/negatives with small positive to avoid log(0)
    data = np.where(data <= 0, 1e-10, data)
    db_data = 10.0 * np.log10(data)

    # Replace -inf / inf with NaN sentinel then clip
    db_data = np.where(np.isfinite(db_data), db_data, np.nan)

    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(db_data[np.newaxis, :, :])

    stats = {
        "band": band_name,
        "min_db": float(np.nanmin(db_data)),
        "max_db": float(np.nanmax(db_data)),
        "mean_db": float(np.nanmean(db_data)),
        "std_db": float(np.nanstd(db_data)),
        "pct_nan": float(np.isnan(db_data).mean() * 100),
    }
    logging.info(f"CAL_DB | {band_name} | {stats}")
    print(f"    dB stats: min={stats['min_db']:.1f}  max={stats['max_db']:.1f}  "
          f"mean={stats['mean_db']:.1f}  NaN%={stats['pct_nan']:.2f}")
    return stats


def calibrate(scene_key: str):
    input_dim  = DIRS["noise"]     / f"{scene_key}_noise.dim"
    sigma_dir  = DIRS["calibrated"] / f"{scene_key}_sigma0_linear"
    out_vv_db  = DIRS["calibrated"] / f"{scene_key}_VV_db.tif"
    out_vh_db  = DIRS["calibrated"] / f"{scene_key}_VH_db.tif"

    if out_vv_db.exists() and out_vh_db.exists():
        print(f"    {scene_key}: calibration outputs exist, skipping")
        return

    # ── SNAP calibration to linear sigma0 ─────────────────────────────────
    graph_path = build_calibration_graph(input_dim, sigma_dir)
    cmd = [str(SNAP_GPT), str(graph_path), "-c", "4G", "-q", "4"]
    print(f"  ⚙  {scene_key}: running radiometric calibration...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Calibration GPT failed: {scene_key}\n{result.stderr[-500:]}")
    print(f"    SNAP calibration done in {time.time()-t0:.0f}s")

    # ── Identify output band TIFs from SNAP ────────────────────────────────
    # SNAP BigTIFF output uses band names like Sigma0_VV, Sigma0_VH
    sigma_tif = list(sigma_dir.parent.glob(f"{sigma_dir.stem}*.tif"))
    if not sigma_tif:
        raise FileNotFoundError(f"No calibrated TIF found at {sigma_dir}")

    # SNAP may output multi-band or single-band TIFs depending on version
    # Read band names and split
    with rasterio.open(sigma_tif[0]) as src:
        band_names = [src.descriptions[i] or f"band_{i+1}" for i in range(src.count)]
        print(f"    Output bands: {band_names}")
        for i, bname in enumerate(band_names, 1):
            pol = "VV" if "VV" in bname.upper() else ("VH" if "VH" in bname.upper() else None)
            if pol is None:
                continue
            data = src.read(i).astype(np.float32)
            profile = src.profile.copy()
            profile.update(count=1, dtype="float32")
            # Save linear band temporarily
            tmp_linear = DIRS["calibrated"] / f"{scene_key}_{pol}_linear_tmp.tif"
            with rasterio.open(tmp_linear, "w", **profile) as dst:
                dst.write(data[np.newaxis, :, :])

    # ── Convert to dB ──────────────────────────────────────────────────────
    print(f"  ⚙  {scene_key}: converting to dB...")
    stats_all = []
    for pol, out_path in [("VV", out_vv_db), ("VH", out_vh_db)]:
        tmp = DIRS["calibrated"] / f"{scene_key}_{pol}_linear_tmp.tif"
        if not tmp.exists():
            raise FileNotFoundError(f"Linear temp band missing: {tmp}")
        s = linear_to_db(tmp, out_path, f"{scene_key}_{pol}_db")
        stats_all.append(s)
        tmp.unlink()  # clean up linear temp

    logging.info(f"CALIBRATE OK | {scene_key} | VV_db={out_vv_db} | VH_db={out_vh_db}")
    print(f"   {scene_key}: VV_db + VH_db saved")
    return stats_all


if __name__ == "__main__":
    print("\n=== STEP 3: RADIOMETRIC CALIBRATION ===\n")
    t_start = time.time()
    all_stats = {}
    for key in SCENES:
        print(f"\n[{key}]")
        s = calibrate(key)
        all_stats[key] = s

    with open(DIRS["logs"] / "step3_calibration_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n Step 3 complete — {(time.time()-t_start)/60:.1f} min")