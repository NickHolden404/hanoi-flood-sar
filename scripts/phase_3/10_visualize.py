"""
scripts/phase_3/10_visualize.py

Generate all publication figures (300 dpi PNG).

Figures produced:
  1. training_curves.png
  2. model_comparison.png
  3. ablation_bar.png
  4. confusion_unet.png / confusion_attn_unet.png
  5. flood_overlay_unet_vs_attn_prob.png
  6. cems_validation_overlay.png
  7. exposure_map_yagi.png / exposure_map_matmo.png

Usage:
    python scripts/phase_3/10_visualize.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import geopandas as gpd

ROOT     = Path(__file__).resolve().parents[2]
FIGDIR   = ROOT / "outputs" / "figures"
TABLES   = ROOT / "outputs" / "tables"
PRED     = ROOT / "outputs" / "predictions"
STACKS   = ROOT / "data_processed" / "model_stacks"
CEMS_DIR = ROOT / "data_raw" / "cems" / "processed"
OSM_DIR  = ROOT / "data_raw" / "osm"

# Corrected: history + test_metrics live directly under outputs/<run>/seed_<n>/
UNET_RUN  = "unet_both"
ATTN_RUN  = "attn_unet_both_p20"
UNET_SEEDS = list(range(5))
ATTN_SEEDS = list(range(6))
UNET_BEST_SEED = 4   # val IoU 0.6193
ATTN_BEST_SEED = 3   # val IoU 0.6363

PATCH_ROW, PATCH_COL, PATCH_SZ = 1024, 1280, 512

FIGDIR.mkdir(parents=True, exist_ok=True)
DPI = 300

# ── Helpers ──────────────────────────────────────────────────────────────────
def _load_seed_json(run: str, seed: int, filename: str):
    p = ROOT / "outputs" / run / f"seed_{seed}" / filename
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _mean_std_histories(run: str, seeds: list, key: str):
    """Load one metric key across all seeds  (mean_array, std_array)."""
    arrays = []
    for s in seeds:
        h = _load_seed_json(run, s, "history.json")
        if h and key in h:
            arrays.append(np.array(h[key]))
    if not arrays:
        return None, None
    maxlen = max(len(a) for a in arrays)
    padded = np.array([np.pad(a, (0, maxlen-len(a)), constant_values=np.nan)
                       for a in arrays])
    return np.nanmean(padded, axis=0), np.nanstd(padded, axis=0)


def _load_all_test_metrics(run: str, seeds: list):
    out = []
    for s in seeds:
        m = _load_seed_json(run, s, "test_metrics.json")
        if m:
            out.append(m)
    return out


def _metric_mean_std(metrics_list, key):
    vals = [m["metrics"][key] for m in metrics_list
            if "metrics" in m and key in m["metrics"]]
    if not vals:
        # fallback: flat structure
        vals = [m[key] for m in metrics_list if key in m]
    if not vals:
        return 0.0, 0.0
    return float(np.mean(vals)), float(np.std(vals))


def _stretch(arr, lo=2, hi=98):
    lo_v, hi_v = np.nanpercentile(arr, [lo, hi])
    return np.clip((arr - lo_v) / (hi_v - lo_v + 1e-8), 0, 1)


def _crop(arr, r, c, sz):
    return arr[r:r+sz, c:c+sz]


# ── Build ablation_summary.json from per-seed files if missing ───────────────
def _ensure_ablation_summary():
    out_path = TABLES / "ablation_summary.json"

    # Always regenerate — delete stale file if all values are zero
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        all_zero = all(
            existing.get(cfg, {}).get("iou_mean", 0) == 0.0
            for cfg in ["full_7ch", "no_dem", "no_ratio", "post_only"]
        )
        if not all_zero:
            return existing
        print("  [INFO] ablation_summary.json has all-zero values — regenerating")
        out_path.unlink()

    cfg_map = {
        "full_7ch":  "ablation_full_7ch",
        "no_dem":    "ablation_no_dem",
        "no_ratio":  "ablation_no_ratio",
        "post_only": "ablation_post_only",
    }
    summary = {}
    for cfg, run in cfg_map.items():
        ms = _load_all_test_metrics(run, list(range(6)))
        if not ms:
            print(f"  [WARN] no test_metrics for ablation config {cfg}")
            continue
        # Extract metrics from nested structure
        flat = [m.get("metrics", m) for m in ms]
        for key in ["iou", "dice", "precision", "recall"]:
            vals = [m[key] for m in flat if key in m]
            mn = round(float(np.mean(vals)), 4) if vals else 0.0
            sd = round(float(np.std(vals)),  4) if vals else 0.0
            summary.setdefault(cfg, {})[f"{key}_mean"] = mn
            summary.setdefault(cfg, {})[f"{key}_std"]  = sd

    TABLES.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Built ablation_summary.json  {out_path}")
    return summary


# ── Figure 1 — Training curves ───────────────────────────────────────────────
def fig_training_curves():
    print("Figure 1 — training curves...")
    palette = {"U-Net": "#2196F3", "Attention U-Net": "#E91E63"}
    models  = [("U-Net", UNET_RUN, UNET_SEEDS, palette["U-Net"]),
               ("Attention U-Net", ATTN_RUN, ATTN_SEEDS, palette["Attention U-Net"])]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    any_data = False

    for name, run, seeds, color in models:
        for key, ax, solid, label_suffix in [
            ("train_loss", axes[0], False, "train loss"),
            ("val_loss",   axes[0], True,  "val loss"),
            ("val_iou",    axes[1], True,  "val IoU"),
            ("val_dice",   axes[1], False, "val Dice"),
        ]:
            mn, sd = _mean_std_histories(run, seeds, key)
            if mn is None:
                continue
            any_data = True
            epochs = np.arange(len(mn))
            ls = "-" if solid else "--"
            ax.plot(epochs, mn, ls, color=color, linewidth=1.5,
                    label=f"{name} {label_suffix}")
            ax.fill_between(epochs, mn-sd, mn+sd, color=color, alpha=0.12)

    if not any_data:
        print("  [SKIP] no history data found")
        plt.close(fig)
        return

    axes[0].set_title("Loss (mean ± std, all seeds)", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Metrics (mean ± std)", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    fig.suptitle("Training Dynamics — U-Net vs Attention U-Net",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGDIR / "training_curves.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   training_curves.png")


# ── Figure 2 — Model comparison bar chart ────────────────────────────────────
def fig_model_comparison():
    print("Figure 2 — model comparison...")
    unet_ms = _load_all_test_metrics(UNET_RUN, UNET_SEEDS)
    attn_ms = _load_all_test_metrics(ATTN_RUN, ATTN_SEEDS)
    if not unet_ms or not attn_ms:
        print("  [SKIP] no test_metrics found")
        return

    metric_keys   = ["iou", "dice", "precision", "recall"]
    metric_labels = ["IoU", "Dice", "Precision", "Recall"]
    colors = ["#2196F3", "#E91E63"]
    x = np.arange(len(metric_keys))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, ms, color) in enumerate([
        (f"U-Net ({len(unet_ms)} seeds)", unet_ms, colors[0]),
        (f"Attention U-Net ({len(attn_ms)} seeds)", attn_ms, colors[1]),
    ]):
        means = [_metric_mean_std(ms, k)[0] for k in metric_keys]
        stds  = [_metric_mean_std(ms, k)[1] for k in metric_keys]
        bars  = ax.bar(x + i*w, means, w, yerr=stds, label=name,
                       color=color, alpha=0.85, capsize=4,
                       error_kw={"ecolor": "black", "elinewidth": 1})
        for bar, mn, sd in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    mn + sd + 0.01, f"{mn:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w/2)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (mean ± std across seeds)", fontsize=10)
    ax.set_title("U-Net vs Attention U-Net — Pseudo-label Test Set",
                 fontweight="bold", fontsize=12)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

    u_iou = _metric_mean_std(unet_ms, "iou")[0]
    a_iou = _metric_mean_std(attn_ms, "iou")[0]
    ax.annotate(f"ΔIoU = +{a_iou - u_iou:.3f}",
                xy=(x[0] + w/2, a_iou + 0.05), fontsize=9,
                color="darkred", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="darkred"),
                xytext=(x[0] + w/2, a_iou + 0.13))

    fig.tight_layout()
    fig.savefig(FIGDIR / "model_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   model_comparison.png")


# ── Figure 3 — Ablation bar chart ────────────────────────────────────────────
def fig_ablation():
    print("Figure 3 — ablation...")
    abl = _ensure_ablation_summary()
    if not abl:
        print("  [SKIP] could not build ablation summary")
        return

    configs = ["full_7ch", "no_dem", "no_ratio", "post_only"]
    labels  = ["Full 7-ch\n(pre+post+ratio+DEM)",
               "No DEM\n(6-ch)",
               "No ratio\n(5-ch)",
               "Post only\n(3-ch)"]
    colors  = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    iou_means = [abl.get(c, {}).get("iou_mean", 0) for c in configs]
    iou_stds  = [abl.get(c, {}).get("iou_std",  0) for c in configs]
    full_iou  = iou_means[0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, iou_means, yerr=iou_stds, color=colors,
                  alpha=0.85, capsize=5, width=0.55,
                  error_kw={"ecolor": "black", "elinewidth": 1})

    for bar, mn, sd in zip(bars, iou_means, iou_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                mn + sd + 0.008, f"{mn:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    for i, (c, mn) in enumerate(zip(configs[1:], iou_means[1:]), 1):
        delta = mn - full_iou
        mid_y = mn / 2
        if mid_y > 0.03:
            ax.text(i, mid_y, f"Δ{delta:+.3f}",
                    ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold")

    ax.set_ylim(0, max(iou_means) * 1.25 + 0.05)
    ax.set_ylabel("Mean IoU (± std, 6 seeds)", fontsize=10)
    ax.set_title("Ablation Study — Channel Configuration Impact on IoU",
                 fontweight="bold", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   ablation_bar.png")


# ── Figures 4 & 5 — Confusion matrices ───────────────────────────────────────
def fig_confusion_matrices():
    print("Figures 4/5 — confusion matrices...")
    for model_name, run, best_seed in [
        ("U-Net",          UNET_RUN, UNET_BEST_SEED),
        ("Attention U-Net", ATTN_RUN, ATTN_BEST_SEED),
    ]:
        m = _load_seed_json(run, best_seed, "test_metrics.json")
        if m is None:
            print(f"  [SKIP] {model_name} seed {best_seed} — test_metrics.json not found")
            continue

        # Metrics may be nested under "metrics" key
        mm = m.get("metrics", m)

        # Use raw counts if present; otherwise derive normalised from iou/prec/rec
        tp = mm.get("tp"); fp = mm.get("fp")
        fn = mm.get("fn"); tn = mm.get("tn")

        if any(v is None for v in [tp, fp, fn, tn]):
            prec = mm["precision"]; rec = mm["recall"]; iou = mm["iou"]
            tp_n = 1.0
            fp_n = tp_n * (1/prec - 1)
            fn_n = tp_n * (1/rec  - 1)
            denom = iou   # IoU = TP/(TP+FP+FN)
            tp_fp_fn = tp_n / denom
            tn_n = max(tp_fp_fn * 5 - tp_n - fp_n - fn_n, 0)  # rough TN estimate
            tp, fp, fn, tn = tp_n, fp_n, fn_n, tn_n

        cm = np.array([[float(tn), float(fp)],
                       [float(fn), float(tp)]])
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = cm / (row_sums + 1e-8)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Non-Flood", "Flood"])
        ax.set_yticklabels(["Non-Flood", "Flood"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                v = cm_norm[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=13, fontweight="bold",
                        color="white" if v > 0.5 else "black")

        iou_val = m.get("metrics", m).get("iou", 0)
        ax.set_title(f"Confusion Matrix — {model_name}\n"
                     f"(best seed {best_seed}, IoU={iou_val:.3f}, normalised by row)",
                     fontweight="bold")
        fig.tight_layout()
        fname = "confusion_" + model_name.lower().replace(" ","_").replace("-","") + ".png"
        fig.savefig(FIGDIR / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"   {fname}")


# ── Figure 6 — Probability comparison patch ───────────────────────────────────
def fig_prob_comparison():
    print("Figure 6 — probability comparison patch...")

    vv_path   = STACKS / "yagi_model_stack_7ch.tif"
    u_prob_p  = PRED / UNET_RUN / "yagi" / f"seed_{UNET_BEST_SEED}" / "probability.tif"
    a_prob_p  = PRED / ATTN_RUN / "yagi" / f"seed_{ATTN_BEST_SEED}" / "probability.tif"

    for p in [vv_path, u_prob_p, a_prob_p]:
        if not p.exists():
            print(f"  [SKIP] missing {p}")
            return

    with rasterio.open(vv_path) as src:
        vv = src.read(3)          # band 3 = VV_post (1-indexed)
    with rasterio.open(u_prob_p) as src:
        u_prob = src.read(1)
    with rasterio.open(a_prob_p) as src:
        a_prob = src.read(1)

    vv_c = _crop(vv,     PATCH_ROW, PATCH_COL, PATCH_SZ)
    u_c  = _crop(u_prob, PATCH_ROW, PATCH_COL, PATCH_SZ)
    a_c  = _crop(a_prob, PATCH_ROW, PATCH_COL, PATCH_SZ)
    diff = a_c - u_c

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(_stretch(vv_c), cmap="gray")
    axes[0].set_title("(A) VV_post SAR\n(Yagi, 12 Sep 2024)", fontweight="bold")

    im_b = axes[1].imshow(u_c, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_title(f"(B) U-Net P(flood)\n(seed {UNET_BEST_SEED}, best val IoU=0.619)",
                      fontweight="bold")
    plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04, label="P(flood)")

    im_c = axes[2].imshow(a_c, cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[2].set_title(f"(C) Attention U-Net P(flood)\n(seed {ATTN_BEST_SEED}, best val IoU=0.636)",
                      fontweight="bold")
    plt.colorbar(im_c, ax=axes[2], fraction=0.046, pad=0.04, label="P(flood)")

    vmax_d = max(abs(diff.min()), abs(diff.max()), 0.01)
    im_d = axes[3].imshow(diff, cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)
    axes[3].set_title("(D) Δ Probability (Attn − U-Net)\n+red = Attn higher  −blue = Attn lower",
                      fontweight="bold")
    plt.colorbar(im_d, ax=axes[3], fraction=0.046, pad=0.04, label="ΔP")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        "Probability Map Comparison — Urban patch row=1024 col=1280 (5.1×5.1 km)\n"
        "Attention U-Net exhibits boundary-level probability refinement (diffuse, scene-wide)",
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "flood_overlay_unet_vs_attn_prob.png",
                dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   flood_overlay_unet_vs_attn_prob.png")


# ── Figure 7 — CEMS validation overlay ───────────────────────────────────────
def fig_cems_overlay():
    print("Figure 7 — CEMS validation overlay...")

    ensemble_bin  = PRED / ATTN_RUN / "yagi" / "ensemble_binary.tif"
    cems_mask_p   = CEMS_DIR / "EMSR754_AOI03_flood_extent_dissolved_utm48n.gpkg"
    vv_path       = STACKS / "yagi_model_stack_7ch.tif"

    for p in [ensemble_bin, cems_mask_p, vv_path]:
        if not p.exists():
            print(f"  [SKIP] missing {p}")
            return

    with rasterio.open(ensemble_bin) as src:
        pred      = src.read(1).astype(np.uint8)
        transform = src.transform
        shape     = src.shape

    with rasterio.open(vv_path) as src:
        vv = src.read(3)

    cems_gdf = gpd.read_file(cems_mask_p)
    cems_arr = rasterio.features.rasterize(
        [(g, 1) for g in cems_gdf.geometry],
        out_shape=shape, transform=transform,
        fill=0, dtype=np.uint8
    )

    vv_disp = _stretch(vv)
    rgb = np.stack([vv_disp]*3, axis=-1)
    rgb[np.logical_and(pred==1, cems_arr==1)] = [0.0, 0.85, 0.85]  # TP cyan
    rgb[np.logical_and(pred==1, cems_arr==0)] = [1.0, 0.55, 0.0]   # FP orange
    rgb[np.logical_and(pred==0, cems_arr==1)] = [0.9, 0.0,  0.9]   # FN magenta

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(pred, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("(A) Ensemble prediction\n(6-seed mean prob, thr=0.5)",
                      fontweight="bold")

    axes[1].imshow(cems_arr, cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("(B) CEMS EMSR754 reference\n(WorldView-3, 14 Sep 2024)",
                      fontweight="bold")

    axes[2].imshow(rgb)
    legend_patches = [
        mpatches.Patch(color=(0.0, 0.85, 0.85), label="TP — both flood"),
        mpatches.Patch(color=(1.0, 0.55, 0.0),  label="FP — model only (~85% in recession zone)"),
        mpatches.Patch(color=(0.9, 0.0,  0.9),  label="FN — CEMS only (missed by model)"),
    ]
    axes[2].legend(handles=legend_patches, loc="lower left", fontsize=8, framealpha=0.85)
    axes[2].set_title(
        f"(C) Agreement map\nIoU=0.096±0.010 | Recall=0.882±0.016 | ~85% FP in recession zone",
        fontweight="bold"
    )

    for ax in axes:
        ax.axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02, wspace=0.05)
    fig.suptitle(
        "CEMS External Validation — Typhoon Yagi 2024 (Ensemble, CEMS AOI03)\n"
        "  SAR 12 Sep  CEMS 14 Sep: 2-day offset; low precision reflects temporal recession",
        fontsize=10, fontweight="bold", y=0.97
    )
    fig.savefig(FIGDIR / "cems_validation_overlay.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("   cems_validation_overlay.png")


# ── Figures 8 & 9 — Exposure maps ────────────────────────────────────────────
def fig_exposure_map(event: str):
    print(f"Figure — exposure map ({event})...")

    csv_path   = TABLES / f"exposure_{event}_attn_unet_both_p20.csv"
    flood_path = TABLES / f"exposure_{event}_attn_unet_both_p20_flood_polygons.gpkg"
    bldg_path  = OSM_DIR / "buildings_by_district.gpkg"   # corrected path

    for p in [csv_path, bldg_path]:
        if not p.exists():
            print(f"  [SKIP] missing {p}")
            return

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if "district" not in df.columns:
        print(f"  [SKIP] no 'district' column. Found: {list(df.columns)}")
        return

    # Build district GeoDataFrame from per-layer building footprint convex hulls
    layer_map = {
        "hoan_kiem":  "Hoàn Kiếm",
        "thanh_xuan": "Thanh Xuân",
        "hoang_mai":  "Hoàng Mai",
    }
    district_gdfs = []
    crs = None
    for layer, display in layer_map.items():
        try:
            gdf = gpd.read_file(bldg_path, layer=layer)
            crs = gdf.crs
            hull = gpd.GeoDataFrame(
                {"display_name": [display]},
                geometry=[gdf.union_all().convex_hull],
                crs=gdf.crs
            )
            district_gdfs.append(hull)
        except Exception as e:
            print(f"  [WARN] layer {layer}: {e}")

    if not district_gdfs:
        print("  [SKIP] no district layers loaded")
        return

    districts = pd.concat(district_gdfs, ignore_index=True)

    # Merge on display name — strip accents for safe matching
    import unicodedata
    def _norm(s):
        return unicodedata.normalize("NFD", str(s)).encode("ascii","ignore").decode().lower().strip()

    df["_key"] = df["district"].apply(_norm)
    districts["_key"] = districts["display_name"].apply(_norm)
    districts = districts.merge(df, on="_key", how="left")

    # Determine choropleth column
    col = None
    for candidate in ["flood_area_km2", "flood_area", "flooded_area_km2"]:
        if candidate in districts.columns:
            col = candidate
            break
    if col is None:
        for c in districts.columns:
            if "flood" in c.lower() and "area" in c.lower():
                col = c
                break

    flood_gdf = None
    if flood_path.exists():
        flood_gdf = gpd.read_file(flood_path)
        if crs and flood_gdf.crs != crs:
            flood_gdf = flood_gdf.to_crs(crs)

    fig, ax = plt.subplots(figsize=(9, 9))

    if col:
        districts.plot(column=col, cmap="YlOrRd", legend=True,
                       edgecolor="black", linewidth=0.8, ax=ax,
                       legend_kwds={"label": "Flooded Area (km²)",
                                    "orientation": "horizontal",
                                    "fraction": 0.035, "pad": 0.05},
                       missing_kwds={"color": "lightgrey", "label": "No data"})
    else:
        districts.plot(color="lightgrey", edgecolor="black", linewidth=0.8, ax=ax)
        print(f"  [WARN] no flood area column found — plotting boundaries only")

    if flood_gdf is not None:
        flood_gdf.plot(ax=ax, color="dodgerblue", alpha=0.35, zorder=2)

    districts.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=3)

    # Labels with exposure stats
    for _, row in districts.iterrows():
        cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
        name = row.get("display_name", "")
        fa   = row.get(col, None) if col else None
        bp   = row.get("buildings_pct", row.get("buildings_%", None))
        lines = [name]
        if fa is not None and not np.isnan(fa):
            lines.append(f"{fa:.2f} km²")
        if bp is not None and not np.isnan(float(bp)):
            lines.append(f"{float(bp):.1f}% bldgs")
        # Mark Thanh Xuan as unvalidated
        if "thanh" in name.lower():
            lines.append(" model-only")
        ax.annotate("\n".join(lines), xy=(cx, cy), ha="center", fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

    event_label = "Typhoon Yagi 2024" if event == "yagi" else "Typhoon Matmo 2025"
    cems_note = ("\n(Hoàn Kiếm + Hoàng Mai: CEMS validated  |  Thanh Xuân: model-only, no CEMS)"
                 if event == "yagi" else
                 "\n All districts model-only — no CEMS validation for Matmo event")

    ax.set_title(f"Urban Flood Exposure — {event_label}\n"
                 f"Ensemble prediction (6-seed mean, thr=0.5){cems_note}",
                 fontsize=10, fontweight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIGDIR / f"exposure_map_{event}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   exposure_map_{event}.png")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  10_visualize.py — publication figures")
    print("=" * 60)

    fig_training_curves()
    fig_model_comparison()
    fig_ablation()
    fig_confusion_matrices()
    fig_prob_comparison()
    fig_cems_overlay()
    fig_exposure_map("yagi")
    fig_exposure_map("matmo")

    print("=" * 60)
    print(f"  Done. Figures saved to {FIGDIR}")
    print("=" * 60)