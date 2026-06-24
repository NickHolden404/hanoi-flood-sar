"""
Step 10: Final validation checklist + Phase 2 Technical Report generation.
Produces: phase2_technical_report.md
"""

import json, time
import numpy as np
import rasterio
from pathlib import Path
from datetime import datetime
from phase2_config import EVENTS, DIRS, SCENES, CRS_UTM, PIXEL_SPACING, SPECKLE_FILTER, SPECKLE_WINDOW, NORM_METHOD

REPORT_PATH = DIRS["qc"] / "phase2_technical_report.md"


def run_final_validation() -> dict:
    """Run all checklist items and return pass/fail dict."""
    results = {}

    for event_name in EVENTS:
        ev_checks = {}

        # Stack TIF exists and is readable
        stack_tif = DIRS["stacks"] / f"{event_name}_stack.tif"
        stack_npy = DIRS["stacks"] / f"{event_name}_stack.npy"
        norm_tif  = DIRS["normalized"] / f"{event_name}_norm_stack.tif"
        norm_npy  = DIRS["normalized"] / f"{event_name}_norm_stack.npy"
        norm_params = DIRS["normalized"] / f"{event_name}_norm_params.json"

        ev_checks["stack_tif_exists"] = stack_tif.exists()
        ev_checks["stack_npy_exists"] = stack_npy.exists()
        ev_checks["norm_tif_exists"]  = norm_tif.exists()
        ev_checks["norm_npy_exists"]  = norm_npy.exists()
        ev_checks["norm_params_exists"] = norm_params.exists()

        if stack_tif.exists():
            with rasterio.open(stack_tif) as src:
                ev_checks["crs_correct"] = src.crs.to_epsg() == int(CRS_UTM.split(":")[1])
                ev_checks["pixel_size_correct"] = abs(abs(src.transform.a) - PIXEL_SPACING) < 0.5
                ev_checks["band_count_4"] = src.count == 4
                ev_checks["dtype_float32"] = src.dtypes[0] == "float32"

        if stack_npy.exists():
            stack = np.load(stack_npy)
            ev_checks["no_nan_in_stack"] = not np.any(np.isnan(stack))
            ev_checks["no_inf_in_stack"] = not np.any(np.isinf(stack))
            ev_checks["stack_shape_4hw"] = stack.ndim == 3 and stack.shape[0] == 4

        results[event_name] = ev_checks

    # Log file exists
    results["log_file"] = (DIRS["logs"] / "phase2_processing.log").exists()
    results["qc_report"] = (DIRS["qc"] / "phase2_qc_report.md").exists()

    return results


def generate_technical_report(validation: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Count passes
    flat = {k: v for d in [validation.get(e, {}) for e in EVENTS]
            for k, v in d.items()}
    flat.update({k: v for k, v in validation.items() if not isinstance(v, dict)})
    passes = sum(1 for v in flat.values() if v is True)
    total  = sum(1 for v in flat.values() if isinstance(v, bool))

    md = f"""# Phase 2 Technical Report — Sentinel-1 SAR Preprocessing
## Multi-Event Urban Flood Segmentation, Hanoi, Vietnam

**Generated:** {now}  
**Researcher environment:** Windows, Miniconda `flood_sar`  
**Status:** {" COMPLETE" if passes == total else " INCOMPLETE — review failures below"}  
**Validation:** {passes}/{total} checks passed

---

## 1. Processing Workflow
```
C:\\S1\\*.SAFE  (4 scenes)
        │
        ▼  SNAP GPT — chained single graph per scene
  ┌─────────────────────────────────────────────────────────┐
  │  Apply-Orbit-File                                       │
  │                                                        │
  │  ThermalNoiseRemoval + Remove-GRD-Border-Noise          │
  │                                                        │
  │  Calibration  σ⁰ (VV, VH) linear                     │
  │                                                        │
  │  Terrain-Correction (SRTM 30m, UTM 48N, 10m)           │
  └─────────────────────────────────────────────────────────┘
        │
        ▼  Python (rasterio, numpy, scipy)
  Refined Lee Filter (5×5, linear domain)
        
  10 * log10(σ⁰)  dB scale
        
  Clip to buffered AOI (2.5 km, GADM L3 districts)
        
  4-channel stack assembly (Pre VV | Post VV | Pre VH | Post VH)
        
  Z-score normalization (per channel)
        
  QC validation + report
```

---

## 2. Software Versions

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Scripting environment |
| ESA SNAP | 10.x | Orbit, noise, calibration, terrain correction |
| rasterio | 1.3+ | Raster I/O, clipping |
| numpy | 1.24+ | Array operations, statistics |
| scipy | 1.11+ | Speckle filter (uniform_filter) |
| geopandas | 0.14+ | District boundary handling |
| matplotlib | 3.7+ | QC visualisation |
| pyproj | 3.6+ | CRS transformations |

---

## 3. Parameter Table

| Step | Parameter | Value | Justification |
|---|---|---|---|
| Orbit correction | Orbit type | Precise (auto-download) | Maximum geolocation accuracy |
| Noise removal | Border limit | 500 px | Standard Sentinel-1 GRD setting |
| Noise removal | Trim threshold | 0.5 | Default ESA recommendation |
| Calibration | Output | Sigma-naught (σ⁰) | Normalised for incidence angle |
| Calibration | Scale | Linear  dB | dB after speckle filter |
| Speckle filter | Algorithm | Refined Lee | Preserves urban/flood edges |
| Speckle filter | Domain | Linear (multiplicative) | Theoretically correct |
| Speckle filter | Window | 5 × 5 pixels | Balance smoothing vs. resolution |
| Speckle filter | ENL | 4.4 | Sentinel-1 GRD IW standard |
| Terrain correction | DEM | SRTM GL1 30m | Phase 1 acquisition |
| Terrain correction | Algorithm | Range-Doppler | Standard for GRD products |
| Terrain correction | CRS | EPSG:32648 (UTM 48N) | Metric coordinate system |
| Terrain correction | Pixel spacing | 10 m | Native GRD resolution |
| Terrain correction | Resampling | Bilinear | Standard for SAR intensity |
| AOI buffer | Distance | 2,500 m | Capture peri-district flooding |
| Stack channels | Order | Pre VV, Post VV, Pre VH, Post VH | Change-detection convention |
| NaN fill | Value | −30.0 dB | Open water backscatter floor |
| Normalization | Method | Z-score (per channel) | SAR dB ≈ Gaussian; robust to outliers |

---

## 4. File Inventory

### Final Stacks

| File | Shape | Size |
|---|---|---|
| `2024_yagi_stack.tif` | 4 × H × W, float32 | TBD |
| `2024_yagi_stack.npy` | 4 × H × W, float32 | TBD |
| `2025_matmo_stack.tif` | 4 × H × W, float32 | TBD |
| `2025_matmo_stack.npy` | 4 × H × W, float32 | TBD |
| `2024_yagi_norm_stack.tif` | 4 × H × W, float32 | TBD |
| `2024_yagi_norm_stack.npy` | 4 × H × W, float32 | TBD |
| `2025_matmo_norm_stack.tif` | 4 × H × W, float32 | TBD |
| `2025_matmo_norm_stack.npy` | 4 × H × W, float32 | TBD |

### Parameter Files

| File | Contents |
|---|---|
| `2024_yagi_norm_params.json` | Per-channel mean, std for inverse transform |
| `2025_matmo_norm_params.json` | Per-channel mean, std for inverse transform |
| `all_normalization_params.json` | Combined normalization table |

### QC Outputs

| File | Contents |
|---|---|
| `phase2_qc_report.md` | Band statistics, CRS/pixel confirmation |
| `2024_yagi_bands_histograms.png` | 4-band image + histogram panel |
| `2024_yagi_change_detection.png` | ΔdB prepost for VV and VH |
| `2025_matmo_bands_histograms.png` | 4-band image + histogram panel |
| `2025_matmo_change_detection.png` | ΔdB prepost for VV and VH |

---

## 5. Validation Results

"""
    for event_name in EVENTS:
        ev = validation.get(event_name, {})
        md += f"### {event_name}\n\n"
        md += "| Check | Result |\n|---|---|\n"
        for check, result in ev.items():
            icon = "" if result is True else ("" if result is False else "—")
            md += f"| {check} | {icon} |\n"
        md += "\n"

    md += f"""---

## 6. Processing Time Estimate

| Step | Estimated time (4 scenes) |
|---|---|
| SNAP chain (orbitnoisecalTC) | 60–120 min |
| Speckle filtering (Python) | 5–15 min |
| Clipping | 1–3 min |
| Stack assembly | 1–2 min |
| Normalization | < 1 min |
| QC + report | 3–5 min |
| **Total** | **~90–150 min** |

---

## 7. Known Limitations

1. **Matmo temporal gap:** Post-event scene (16 Oct 2025) acquired 9 days after peak. Partial flood recession likely reduces detected inundation extent relative to peak.
2. **DEM resolution:** SRTM 30m in a flat deltaic plain (σ_h = 3.7 m) introduces small but non-zero terrain correction residuals, particularly near river channels.
3. **Single SAR pair per event:** No multi-temporal SAR stack available; temporal averaging for speckle suppression is not possible. Refined Lee filter is the sole speckle mitigation.
4. **Orbit geometry difference between events:** Yagi pair = descending; Matmo pair = ascending. Incidence angles differ. This is the correct choice for internal event consistency but limits direct cross-event comparison of absolute backscatter values. Z-score normalization partially mitigates this.
5. **OSM building incompleteness:** Hoàng Mai under-mapped in OSM. Affects any downstream building-level analysis but not the SAR flood segmentation directly.
6. **SNAP orbit auto-download:** Requires internet access during Step 5. Use restituted orbits as fallback if precise orbits are unavailable (within 3 weeks of acquisition).

---

## 8. Normalization Inverse Transform

To recover original dB values from normalized stacks:
```python
import numpy as np, json
from pathlib import Path

def denormalize(norm_stack, params_json):
    with open(params_json) as f:
        params = json.load(f)
    result = norm_stack.copy()
    for ch in params["channels"]:
        i = ch["channel"] - 1
        result[i] = norm_stack[i] * ch["std"] + ch["mean"]
    return result
```

---

*Phase 2 ends here. Stacks are ready for Phase 3: Pseudo-label generation.*  
*Do NOT proceed to Phase 3 until all Step 10 checks are .*
"""
    return md


if __name__ == "__main__":
    print("\n=== STEP 10: FINAL VALIDATION & TECHNICAL REPORT ===\n")

    validation = run_final_validation()

    for event_name in EVENTS:
        ev = validation.get(event_name, {})
        passes = sum(1 for v in ev.values() if v is True)
        total  = len(ev)
        icon = "" if passes == total else ""
        print(f"  {icon} {event_name}: {passes}/{total} checks passed")
        for k, v in ev.items():
            mark = "" if v else ""
            print(f"      {mark}  {k}")

    report = generate_technical_report(validation)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n Phase 2 Technical Report saved: {REPORT_PATH}")
    print("\n" + "="*60)

    # Final pass/fail summary
    flat = {}
    for e in EVENTS:
        flat.update(validation.get(e, {}))
    all_pass = all(v for v in flat.values() if isinstance(v, bool))

    if all_pass:
        print("  ALL CHECKS PASSED — Phase 2 complete.")
        print("    Stacks are ready for Phase 3 pseudo-label generation.")
    else:
        print("  SOME CHECKS FAILED — review errors before Phase 3.")
    print("="*60)