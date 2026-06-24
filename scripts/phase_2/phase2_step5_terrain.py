"""
Step 5: Range-Doppler Terrain Correction.
  - Uses SRTM 30m DEM (local file)
  - Output CRS: UTM Zone 48N (EPSG:32648)
  - Pixel spacing: 10m
  - Resampling: Bilinear
  
Note: Terrain correction requires SNAP GPT because it uses the SAR geometry
metadata (orbit state vectors, timing) from the original .SAFE product.
We attach speckle-filtered dB rasters back to SNAP as external bands OR
run terrain correction in the SNAP preprocessing chain.

Strategy: Re-run Steps 1–5 as a SINGLE chained SNAP GPT graph per scene
to avoid intermediate reprojection artifacts. The speckle filter (Step 4)
is the exception — pure Python Refined Lee is used post-terrain-correction
since SNAP's Refined Lee implementation can over-smooth urban edges.

Revised execution order within SNAP:
  SAFE  Orbit  Noise  Calibrate  TerrainCorrect  export GeoTIFF
Then: apply Python Refined Lee (Step 4 runs AFTER this step on TC output).
"""

import subprocess, logging, json, time
import numpy as np
import rasterio
from pathlib import Path
from phase2_config import SNAP_GPT, SCENES, DIRS, DEM_PATH, CRS_UTM, PIXEL_SPACING

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

def build_full_snap_chain(safe_path: Path, dem_path: Path,
                          output_tif: Path) -> Path:
    """
    Single SNAP GPT graph: Orbit  Noise  Calibrate  TerrainCorrect.
    Output: multi-band GeoTIFF with Sigma0_VV, Sigma0_VH (linear scale).
    dB conversion and speckle filtering applied in Python post-SNAP.
    """
    epsg_num = CRS_UTM.split(":")[1]   # "32648"
    graph_xml = f"""<graph id="full_preprocessing">
  <version>1.0</version>

  <node id="Read">
    <operator>Read</operator>
    <parameters>
      <file>{safe_path}</file>
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

  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
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

  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputGammaBand>false</outputGammaBand>
    </parameters>
  </node>

  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <demName>External DEM</demName>
      <externalDEMFile>{dem_path}</externalDEMFile>
      <externalDEMNoDataValue>-32768.0</externalDEMNoDataValue>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>{PIXEL_SPACING}</pixelSpacingInMeter>
      <mapProjection>EPSG:{epsg_num}</mapProjection>
      <saveDEM>false</saveDEM>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
      <nodataValueAtSea>false</nodataValueAtSea>
    </parameters>
  </node>

  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters>
      <file>{output_tif}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
</graph>"""
    graph_path = DIRS["graphs"] / f"full_chain_{output_tif.stem}.xml"
    graph_path.write_text(graph_xml)
    return graph_path


def run_terrain_correction(scene_key: str, safe_path: Path):
    output_tif = DIRS["terrain"] / f"{scene_key}_tc_sigma0.tif"

    if output_tif.exists():
        print(f"    {scene_key}: terrain-corrected TIF exists, skipping")
        return output_tif

    graph_path = build_full_snap_chain(safe_path, DEM_PATH, output_tif)
    cmd = [str(SNAP_GPT), str(graph_path), "-c", "6G", "-q", "4"]

    print(f"  ⚙  {scene_key}: running full SNAP chain (orbitnoisecalTC)...")
    print(f"      [This is the slowest step — expect 15–40 min per scene]")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        logging.error(f"TC FAIL | {scene_key} | {result.stderr[-800:]}")
        raise RuntimeError(f"Terrain correction failed: {scene_key}")

    # Validation: read output and check CRS + pixel size
    with rasterio.open(output_tif) as src:
        crs_epsg = src.crs.to_epsg()
        px_size = abs(src.transform.a)
        band_count = src.count
        width, height = src.width, src.height

    assert crs_epsg == int(CRS_UTM.split(":")[1]), f"CRS mismatch: {crs_epsg}"
    assert abs(px_size - PIXEL_SPACING) < 0.5, f"Pixel size mismatch: {px_size}"

    logging.info(
        f"TC OK | {scene_key} | CRS=EPSG:{crs_epsg} | px={px_size:.1f}m | "
        f"bands={band_count} | dims={width}×{height} | elapsed={elapsed:.0f}s"
    )
    print(f"   {scene_key}: TC done in {elapsed/60:.1f} min | "
          f"CRS=EPSG:{crs_epsg} | px={px_size:.1f}m | {width}×{height}px")
    return output_tif


def split_and_convert_db(scene_key: str):
    """Split terrain-corrected TIF into per-polarisation dB files."""
    tc_tif = DIRS["terrain"] / f"{scene_key}_tc_sigma0.tif"

    with rasterio.open(tc_tif) as src:
        band_names = [src.descriptions[i] or "" for i in range(src.count)]
        print(f"  Band names in TC output: {band_names}")

        for i, bname in enumerate(band_names, 1):
            pol = "VV" if "VV" in bname.upper() else ("VH" if "VH" in bname.upper() else None)
            if pol is None:
                continue

            out_path = DIRS["terrain"] / f"{scene_key}_{pol}_linear.tif"
            data = src.read(i).astype(np.float32)
            profile = src.profile.copy()
            profile.update(count=1, dtype="float32")

            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data[np.newaxis, :, :])

    print(f"   {scene_key}: split into VV_linear + VH_linear")


if __name__ == "__main__":
    print("\n=== STEP 5: TERRAIN CORRECTION ===\n")
    print("NOTE: Speckle filtering (Step 4) will run AFTER this step")
    print("      on terrain-corrected outputs.\n")
    t_start = time.time()

    for key, safe_path in SCENES.items():
        print(f"\n[{key}]")
        run_terrain_correction(key, safe_path)
        split_and_convert_db(key)

    print(f"\n Step 5 complete — Total: {(time.time()-t_start)/60:.1f} min")
    print(f"\n Now run phase2_step4_speckle_post_tc.py to apply Refined Lee")
    print(f"  on terrain-corrected linear sigma0 rasters.")