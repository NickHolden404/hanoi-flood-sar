"""ERA5 hourly total precipitation download and extraction.

Source: ECMWF Copernicus Climate Data Store (CDS).
Events: Typhoon Yagi (September 2024) and Typhoon Matmo (October 2025).

Prerequisites:
  1. Free CDS account: https://cds.climate.copernicus.eu
  2. pip install cdsapi xarray numpy pandas matplotlib
  3. Place a .cdsapirc file in your home directory with your API key:
       url: https://cds.climate.copernicus.eu/api
       key: YOUR-API-KEY
"""

import json
import os
from pathlib import Path

import cdsapi
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERA5_DIR = PROJECT_ROOT / "data_raw" / "era5"
META_DIR = PROJECT_ROOT / "metadata"

ERA5_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

# Hanoi grid cell centre (nearest ERA5 point ~0.25° grid)
HANOI_LAT = 21.00
HANOI_LON = 105.75  # nearest 0.25° grid point to city centre

# ERA5 area: N/W/S/E (must align to 0.25° grid)
AREA = [21.25, 105.50, 20.75, 106.00]  # covers 3x3 grid cells over Hanoi

# Download windows
EVENTS = {
    "yagi_2024": {
        "label":       "Typhoon Yagi 2024",
        "year":        "2024",
        "months":      ["08", "09"],
        "days_aug":    [str(d).zfill(2) for d in range(30, 32)],  # Aug 30-31
        "days_sep":    [str(d).zfill(2) for d in range(1,  16)],  # Sep 1-15
        "peak_date":   "2024-09-09",
        "nc_file":     "era5_tp_yagi_2024_aug30_sep15.nc",
        "csv_file":    "era5_tp_yagi_2024_timeseries.csv",
        "plot_file":   "era5_tp_yagi_2024_rainfall_curve.png",
    },
    "matmo_2025": {
        "label":       "Typhoon Matmo 2025",
        "year":        "2025",
        "months":      ["09", "10"],
        "days_sep":    [str(d).zfill(2) for d in range(26, 31)],  # Sep 26-30
        "days_oct":    [str(d).zfill(2) for d in range(1,  16)],  # Oct 1-15
        "peak_date":   "2025-10-07",
        "nc_file":     "era5_tp_matmo_2025_sep26_oct15.nc",
        "csv_file":    "era5_tp_matmo_2025_timeseries.csv",
        "plot_file":   "era5_tp_matmo_2025_rainfall_curve.png",
    },
}

# ── DOWNLOAD ───────────────────────────────────────────────────────────────────

def download_era5(event_key):
    ev   = EVENTS[event_key]
    path = str(ERA5_DIR /ev["nc_file"])

    if os.path.exists(path):
        print(f"  Already exists, skipping: {ev['nc_file']}\n")
        return path

    print(f"  Downloading ERA5: {ev['label']}")
    print(f"     {path}\n")

    c = cdsapi.Client()

    # Build day list depending on which months are spanned
    if event_key == "yagi_2024":
        requests_list = [
            {   # August portion
                "year": ev["year"], "month": "08",
                "day":  ev["days_aug"],
            },
            {   # September portion
                "year": ev["year"], "month": "09",
                "day":  ev["days_sep"],
            },
        ]
    else:  # matmo_2025
        requests_list = [
            {   # September portion
                "year": ev["year"], "month": "09",
                "day":  ev["days_sep"],
            },
            {   # October portion
                "year": ev["year"], "month": "10",
                "day":  ev["days_oct"],
            },
        ]

    # Download each month chunk then merge
    tmp_files = []
    for i, req in enumerate(requests_list):
        tmp = path.replace(".nc", f"_tmp{i}.nc")
        tmp_files.append(tmp)
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable":     "total_precipitation",
                "year":         req["year"],
                "month":        req["month"],
                "day":          req["day"],
                "time":         [f"{h:02d}:00" for h in range(24)],
                "area":         AREA,
                "format":       "netcdf",
            },
            tmp
        )

    # Merge chunks into single file (without dask)
    print("  Merging monthly chunks...")
    datasets = [xr.open_dataset(f) for f in tmp_files]
    ds = xr.concat(datasets, dim="valid_time")
    ds = ds.sortby("valid_time")
    ds.to_netcdf(path)
    ds.close()
    for d in datasets:
        d.close()

    for tmp in tmp_files:
        os.remove(tmp)

    print(f"  Saved: {path}\n")
    return path

# ── EXTRACT TIME SERIES ────────────────────────────────────────────────────────

def extract_timeseries(event_key):
    ev      = EVENTS[event_key]
    nc_path = str(ERA5_DIR /ev["nc_file"])
    csv_out = str(ERA5_DIR /ev["csv_file"])

    print(f" Extracting time series: {ev['label']}")

    ds  = xr.open_dataset(nc_path)
    var = "tp" if "tp" in ds else list(ds.data_vars)[0]

    # Extract nearest grid point to Hanoi centre
    ts = ds[var].sel(
        latitude=HANOI_LAT,
        longitude=HANOI_LON,
        method="nearest"
    )

    df = ts.to_dataframe(name="precip_m").reset_index()
    # CDS API v2 uses 'valid_time', older versions use 'time' — handle both
    time_col = "valid_time" if "valid_time" in df.columns else "time"
    df = df.rename(columns={time_col: "time"})
    df = df[["time", "precip_m"]].copy()
    df["precip_mm_hr"] = (df["precip_m"] * 1000).clip(lower=0)
    df["precip_mm_3hr"] = df["precip_mm_hr"].rolling(3).sum()
    df["precip_mm_24hr"] = df["precip_mm_hr"].rolling(24).sum()
    df["date"] = pd.to_datetime(df["time"]).dt.date

    # Daily accumulation
    daily = df.groupby("date")["precip_mm_hr"].sum().reset_index()
    daily.columns = ["date", "precip_mm_day"]

    # Peak stats
    peak_hr_idx  = df["precip_mm_hr"].idxmax()
    peak_24h_idx = df["precip_mm_24hr"].idxmax()

    peak_hr_time  = df.loc[peak_hr_idx,  "time"]
    peak_hr_val   = df.loc[peak_hr_idx,  "precip_mm_hr"]
    peak_24h_time = df.loc[peak_24h_idx, "time"]
    peak_24h_val  = df.loc[peak_24h_idx, "precip_mm_24hr"]

    # 3-day accumulation around confirmed peak date
    peak_date   = pd.Timestamp(ev["peak_date"])
    mask_3day   = (
        (pd.to_datetime(df["time"]) >= peak_date - pd.Timedelta(days=1)) &
        (pd.to_datetime(df["time"]) <= peak_date + pd.Timedelta(days=1))
    )
    accum_3day  = df.loc[mask_3day, "precip_mm_hr"].sum()

    print(f"  Peak hourly intensity : {peak_hr_val:.2f} mm/hr  @ {peak_hr_time}")
    print(f"  Peak 24hr accumulation: {peak_24h_val:.2f} mm    @ {peak_24h_time}")
    print(f"  3-day accum (±1d peak): {accum_3day:.2f} mm\n")

    df.to_csv(csv_out, index=False)
    print(f"  CSV saved: {csv_out}")
    ds.close()

    return df, daily, {
        "event":              ev["label"],
        "confirmed_peak_date": ev["peak_date"],
        "peak_hourly_mm_hr":  round(float(peak_hr_val),  2),
        "peak_hourly_time":   str(peak_hr_time),
        "peak_24hr_mm":       round(float(peak_24h_val), 2),
        "peak_24hr_time":     str(peak_24h_time),
        "accum_3day_mm":      round(float(accum_3day),   2),
    }

# ── PLOT ───────────────────────────────────────────────────────────────────────

def plot_rainfall(event_key, df, daily):
    ev       = EVENTS[event_key]
    plot_out = str(ERA5_DIR /ev["plot_file"])
    peak_dt  = pd.Timestamp(ev["peak_date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    fig.suptitle(f"ERA5 Rainfall — {ev['label']}", fontsize=13, fontweight="bold")

    # Top: hourly
    t = pd.to_datetime(df["time"])
    ax1.bar(t, df["precip_mm_hr"], width=0.04, color="#2196F3", alpha=0.7, label="Hourly (mm/hr)")
    ax1.plot(t, df["precip_mm_24hr"], color="#F44336", linewidth=1.5, label="24hr rolling sum (mm)")
    ax1.axvline(peak_dt, color="orange", linestyle="--", linewidth=1.5, label=f"Confirmed peak: {ev['peak_date']}")
    ax1.set_ylabel("Precipitation (mm)")
    ax1.legend(fontsize=9)
    ax1.set_title("Hourly Intensity + 24hr Rolling Accumulation")
    ax1.grid(axis="y", alpha=0.3)

    # Bottom: daily
    daily["date"] = pd.to_datetime(daily["date"])
    ax2.bar(daily["date"], daily["precip_mm_day"], color="#4CAF50", alpha=0.8, width=0.8)
    ax2.axvline(peak_dt, color="orange", linestyle="--", linewidth=1.5, label=f"Confirmed peak: {ev['peak_date']}")
    ax2.set_ylabel("Daily Accumulation (mm/day)")
    ax2.set_title("Daily Accumulation")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_out}\n")

# ── MAIN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PHASE 1 — ERA5 RAINFALL DOWNLOAD & VERIFICATION")
    print("=" * 55 + "\n")

    all_meta = {}

    for key in ["yagi_2024", "matmo_2025"]:
        print(f"{'─'*55}")
        print(f"EVENT: {EVENTS[key]['label']}")
        print(f"{'─'*55}")

        download_era5(key)
        df, daily, stats = extract_timeseries(key)
        plot_rainfall(key, df, daily)
        all_meta[key] = stats

    # Save combined metadata
    meta_path = META_DIR / "era5_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f" ERA5 metadata saved: {meta_path}")

    # Final summary
    print(f"\n{'='*55}")
    print("ERA5 SUMMARY")
    print(f"{'='*55}")
    for key, s in all_meta.items():
        print(f"\n  {s['event']}")
        print(f"    Confirmed peak date : {s['confirmed_peak_date']}")
        print(f"    Peak hourly         : {s['peak_hourly_mm_hr']} mm/hr  @ {s['peak_hourly_time']}")
        print(f"    Peak 24hr accum     : {s['peak_24hr_mm']} mm  @ {s['peak_24hr_time']}")
        print(f"    3-day accumulation  : {s['accum_3day_mm']} mm")

    print(f"\nDone. Next step: OSM infrastructure download (osm_download.py)")