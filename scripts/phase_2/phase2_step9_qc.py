"""
Step 9: Quality Control — visual inspection, histogram comparison, statistics.
Generates a Markdown QC report + PNG figures.
"""

import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, logging, time
from pathlib import Path
from datetime import datetime
from phase2_config import EVENTS, DIRS, NORM_METHOD, CRS_UTM, PIXEL_SPACING

logging.basicConfig(
    filename=DIRS["logs"] / "phase2_processing.log",
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

BAND_LABELS = ["Pre VV", "Post VV", "Pre VH", "Post VH"]
VMIN_DB, VMAX_DB = -25, 0   # typical SAR dB display range


def load_stack(event_name: str, normalized: bool = False) -> np.ndarray:
    suffix = "norm_stack" if normalized else "stack"
    src_dir = DIRS["normalized"] if normalized else DIRS["stacks"]
    return np.load(src_dir / f"{event_name}_{suffix}.npy")


def plot_false_colour_and_histograms(event_name: str, stack: np.ndarray,
                                      out_path: Path):
    """Generate 4-panel image plot + 4-panel histogram plot."""
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"QC Visual Inspection — {event_name}", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)

    # Row 1: band images
    for i, label in enumerate(BAND_LABELS):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(stack[i], cmap="gray",
                       vmin=VMIN_DB, vmax=VMAX_DB, interpolation="none")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")

    # Row 2: histograms
    colors = ["steelblue", "tomato", "forestgreen", "darkorange"]
    for i, label in enumerate(BAND_LABELS):
        ax = fig.add_subplot(gs[1, i])
        data = stack[i].flatten()
        data = data[np.isfinite(data)]
        ax.hist(data, bins=100, color=colors[i], alpha=0.75, edgecolor="none")
        ax.axvline(np.mean(data), color="black", lw=1.5, linestyle="--",
                   label=f"μ={np.mean(data):.1f}")
        ax.set_xlabel("dB", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(f"{label} histogram", fontsize=10)
        ax.legend(fontsize=8)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def plot_pre_post_difference(event_name: str, stack: np.ndarray, out_path: Path):
    """Plot VV and VH pre–post difference (flood signal)."""
    diff_vv = stack[1] - stack[0]   # post_vv - pre_vv
    diff_vh = stack[3] - stack[2]   # post_vh - pre_vh

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"PrePost Change Detection — {event_name}", fontsize=13)

    for ax, diff, pol in zip(axes, [diff_vv, diff_vh], ["VV", "VH"]):
        vext = max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)))
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vext, vmax=vext)
        ax.set_title(f"Δ{pol} (post − pre) dB", fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ΔdB")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


def check_raster_properties(event_name: str) -> dict:
    """Validate CRS, pixel size, dtype of final stack TIF."""
    tif = DIRS["stacks"] / f"{event_name}_stack.tif"
    checks = {}
    with rasterio.open(tif) as src:
        checks["crs_epsg"] = src.crs.to_epsg()
        checks["pixel_size_m"] = round(abs(src.transform.a), 2)
        checks["width"] = src.width
        checks["height"] = src.height
        checks["band_count"] = src.count
        checks["dtype"] = src.dtypes[0]
        checks["nodata"] = src.nodata

        # Band statistics
        checks["band_stats"] = []
        for i in range(1, src.count + 1):
            d = src.read(i).astype(np.float32)
            checks["band_stats"].append({
                "band": i,
                "min": float(np.nanmin(d)), "max": float(np.nanmax(d)),
                "mean": float(np.nanmean(d)), "std": float(np.nanstd(d)),
                "nan_pct": float(np.isnan(d).mean() * 100),
                "zero_pct": float((d == 0).mean() * 100)
            })

    # Assertions
    assert checks["crs_epsg"] == int(CRS_UTM.split(":")[1]), \
        f"CRS FAIL: {checks['crs_epsg']}"
    assert abs(checks["pixel_size_m"] - PIXEL_SPACING) < 0.5, \
        f"Pixel size FAIL: {checks['pixel_size_m']}"
    assert checks["band_count"] == 4, \
        f"Band count FAIL: {checks['band_count']}"
    assert checks["dtype"] == "float32", \
        f"Dtype FAIL: {checks['dtype']}"

    return checks


def generate_qc_report(all_checks: dict, all_params: dict) -> str:
    """Generate Markdown QC report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = f"""# Phase 2 QC Report — Sentinel-1 SAR Preprocessing
## Multi-Event Urban Flood Segmentation, Hanoi, Vietnam
**Generated:** {now}  
**Normalization method:** {NORM_METHOD.upper()}  
**CRS:** {CRS_UTM}  
**Target pixel spacing:** {PIXEL_SPACING} m

---

## Validation Checklist

| Check | Status |
|---|---|
| Orbit correction applied |  |
| Thermal noise removed |  |
| Border noise removed |  |
| Radiometric calibration (σ⁰) |  |
| Speckle filtering (Refined Lee 5×5, linear domain) |  |
| Terrain correction (SRTM 30m, Range-Doppler) |  |
| All rasters aligned (same extent, CRS, pixel size) |  |
| 4-channel stacks created |  |
| Normalization documented |  |
| No NaN values in final stacks |  |

---

## Raster Properties

"""
    for event_name, checks in all_checks.items():
        md += f"### {event_name}\n\n"
        md += f"| Property | Value |\n|---|---|\n"
        md += f"| CRS | EPSG:{checks['crs_epsg']} |\n"
        md += f"| Pixel size | {checks['pixel_size_m']} m |\n"
        md += f"| Dimensions | {checks['width']} × {checks['height']} px |\n"
        md += f"| Data type | {checks['dtype']} |\n"
        md += f"| Band count | {checks['band_count']} |\n\n"

        md += "**Band statistics (dB):**\n\n"
        md += "| Band | Description | Min | Max | Mean | Std | NaN% |\n"
        md += "|---|---|---|---|---|---|---|\n"
        labels = ["Pre VV", "Post VV", "Pre VH", "Post VH"]
        for i, s in enumerate(checks["band_stats"]):
            md += (f"| {s['band']} | {labels[i]} | {s['min']:.2f} | "
                   f"{s['max']:.2f} | {s['mean']:.2f} | {s['std']:.2f} | "
                   f"{s['nan_pct']:.2f}% |\n")
        md += "\n"

    md += """---

## Normalization Parameters

"""
    for event_name, params in all_params.items():
        if params is None:
            continue
        md += f"### {event_name}\n\n"
        md += f"Method: **{params.get('method', 'unknown').upper()}**\n\n"
        md += "| Channel | Description | Mean (raw dB) | Std (raw dB) |\n"
        md += "|---|---|---|---|\n"
        for i, ch in enumerate(params.get("channels", [])):
            md += (f"| {ch['channel']} | {BAND_LABELS[i]} | "
                   f"{ch.get('mean', 'n/a'):.3f} | {ch.get('std', 'n/a'):.3f} |\n")
        md += "\n"

    md += """---

## Processing Notes

- **Speckle filter:** Refined Lee (Lee 1981; Lee et al. 1999), 5×5 window, applied in **linear scale** before dB conversion
- **Terrain correction:** Range-Doppler algorithm using SRTM GL1 30m DEM; bilinear resampling; 10m output spacing
- **NaN fill value:** −30.0 dB (approximate open water backscatter floor for Sentinel-1 C-band)
- **Matmo post-event gap:** 9 days post-peak rainfall — residual inundation confirmed in Hanoi through mid-October 2025
- **ERA5 note:** Spatial smoothing at 0.25° underestimates station totals; ERA5 used for temporal confirmation only

---

## Known Limitations

1. Matmo post-event scene (16 Oct 2025) is 9 days post-peak — some flood recession may reduce inundation extent vs. peak
2. SRTM 30m DEM introduces slight terrain correction error in flat deltaic areas (height std = 3.7 m)
3. Hoàng Mai OSM building coverage incomplete — peri-urban zones may have unlabeled impervious surfaces
4. Single SAR pair per event — no temporal stack available for change-point filtering
5. C-band SAR penetrates shallow water but not dense vegetation canopy — flooded rice fields may be underdetected

---

*QC report auto-generated by phase2_step9_qc.py*
"""
    return md


if __name__ == "__main__":
    print("\n=== STEP 9: QUALITY CONTROL ===\n")
    t_start = time.time()
    all_checks = {}
    all_params = {}

    for event_name in EVENTS:
        print(f"\n[{event_name}]")

        # Load raw stack for visualization
        stack = load_stack(event_name, normalized=False)
        print(f"  Stack shape: {stack.shape}  dtype: {stack.dtype}")

        # Check for issues
        nan_count = np.isnan(stack).sum()
        inf_count = np.isinf(stack).sum()
        print(f"  NaN: {nan_count}  Inf: {inf_count}")
        if nan_count > 0 or inf_count > 0:
            logging.warning(f"QC WARN | {event_name} | NaN={nan_count} Inf={inf_count}")

        # Generate plots
        print(f"  Generating plots...")
        plot_false_colour_and_histograms(
            event_name, stack,
            DIRS["qc"] / f"{event_name}_bands_histograms.png"
        )
        plot_pre_post_difference(
            event_name, stack,
            DIRS["qc"] / f"{event_name}_change_detection.png"
        )

        # Validate raster properties
        checks = check_raster_properties(event_name)
        all_checks[event_name] = checks
        print(f"   Property checks: CRS=EPSG:{checks['crs_epsg']} | "
              f"px={checks['pixel_size_m']}m | {checks['width']}×{checks['height']}px")

        # Load norm params
        norm_params_path = DIRS["normalized"] / f"{event_name}_norm_params.json"
        if norm_params_path.exists():
            with open(norm_params_path) as f:
                all_params[event_name] = json.load(f)

    # Generate Markdown report
    report_md = generate_qc_report(all_checks, all_params)
    report_path = DIRS["qc"] / "phase2_qc_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    with open(DIRS["logs"] / "step9_qc_checks.json", "w") as f:
        json.dump(all_checks, f, indent=2)

    print(f"\n Step 9 complete — {(time.time()-t_start)/60:.1f} min")
    print(f"   QC report: {report_path}")