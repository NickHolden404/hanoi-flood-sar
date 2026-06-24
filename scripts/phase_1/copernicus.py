"""Sentinel-1 GRD scene search and download from the Copernicus Data Space Ecosystem.

Requires a free CDSE account: https://dataspace.copernicus.eu
Set credentials via environment variables before running:
    CDSE_USERNAME, CDSE_PASSWORD
"""

import json
import os
from datetime import datetime
from pathlib import Path

import requests

USERNAME = os.environ.get("CDSE_USERNAME")
PASSWORD = os.environ.get("CDSE_PASSWORD")
if not USERNAME or not PASSWORD:
    raise SystemExit(
        "CDSE_USERNAME and CDSE_PASSWORD environment variables must be set. "
        "Register at https://dataspace.copernicus.eu"
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR  = PROJECT_ROOT / "data_raw" / "sentinel1"
YAGI_DIR  = BASE_DIR / "2024_yagi"
MATMO_DIR = BASE_DIR / "2025_matmo"
META_DIR  = PROJECT_ROOT / "metadata"

AOI = "POLYGON((105.780 20.960, 105.880 20.960, 105.880 21.060, 105.780 21.060, 105.780 20.960))"

for d in (YAGI_DIR, MATMO_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)


def get_token(username, password):
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id":  "cdse-public",
            "username":   username,
            "password":   password,
        },
    )
    r.raise_for_status()
    return r.json()["access_token"]


def search_s1(date_start, date_end, label):
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-1' "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{AOI}') "
            f"and ContentDate/Start ge {date_start}T00:00:00.000Z "
            f"and ContentDate/Start le {date_end}T23:59:59.000Z "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'GRD') "
            f"and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode' and att/OData.CSC.StringAttribute/Value eq 'IW')"
        ),
        "$orderby": "ContentDate/Start asc",
        "$top": 20,
    }
    r = requests.get(url, params=params)
    results = r.json().get("value", [])

    print(f"=== {label} ({date_start} to {date_end}) ===")
    if not results:
        print("  No results found.\n")
        return []
    for p in results:
        size_gb = p.get("ContentLength", 0) / 1e9
        print(f"  Name: {p['Name']}")
        print(f"  Date: {p['ContentDate']['Start']}")
        print(f"  Size: {size_gb:.2f} GB")
        print(f"  ID:   {p['Id']}")
        print()
    return results


def download_scene(product_id, product_name, out_dir, username, password):
    # Tokens expire in ~10 min and each scene takes 15-30 min, so re-auth per scene.
    out_path = out_dir / (product_name + ".zip")

    if out_path.exists():
        size_gb = out_path.stat().st_size / 1e9
        print(f"Already exists ({size_gb:.2f} GB), skipping: {product_name}\n")
        return out_path

    token = get_token(username, password)
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    print(f"Downloading: {product_name}")
    print(f"  -> {out_path}")

    with session.get(url, stream=True, timeout=3600, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:.1f}%  {downloaded/1e9:.2f} / {total/1e9:.2f} GB", end="", flush=True)

    print(f"\nDone: {product_name}\n")
    return out_path


def save_metadata(all_results, filename):
    path = META_DIR / filename
    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Metadata saved: {path}")


if __name__ == "__main__":
    print("PHASE 1 - SENTINEL-1 SCENE SEARCH\n")

    all_results = {
        "yagi_pre":      search_s1("2024-08-25", "2024-09-02", "YAGI PRE-EVENT"),
        "yagi_post":     search_s1("2024-09-10", "2024-09-12", "YAGI POST-EVENT"),
        "yagi_bkup":     search_s1("2024-09-13", "2024-09-15", "YAGI BACKUP POST"),
        "matmo_pre":     search_s1("2025-09-21", "2025-09-29", "MATMO PRE-EVENT"),
        "matmo_post":    search_s1("2025-10-08", "2025-10-10", "MATMO POST-EVENT"),
        "matmo_bkup":    search_s1("2025-10-11", "2025-10-13", "MATMO BACKUP POST"),
        "matmo_post_w1": search_s1("2025-10-04", "2025-10-07", "MATMO POST WIDE (Oct 4-7)"),
        "matmo_post_w2": search_s1("2025-10-14", "2025-10-20", "MATMO LATE POST (Oct 14-20)"),
    }

    save_metadata(all_results, "sentinel1_search_results.json")

    # Scenes used in the paper. Yagi: descending pair (pass time 22:51 UTC).
    # Matmo: ascending pair (pass time 11:06 UTC). Mismatched-orbit backups skipped.
    scenes_to_download = [
        ("cdfcdbf2-b343-4b4a-9d16-5b8768cecfde",
         "S1A_IW_GRDH_1SDV_20240831T225114_20240831T225139_055463_06C40A_A2E8",
         YAGI_DIR),
        ("417316b8-6237-44c2-929f-577171f83d4d",
         "S1A_IW_GRDH_1SDV_20240912T225114_20240912T225139_055638_06CAF6_16C8",
         YAGI_DIR),
        ("5fa3e411-a951-45aa-9017-3dee68e49725",
         "S1A_IW_GRDH_1SDV_20250922T110625_20250922T110650_061100_079DAF_B6AF",
         MATMO_DIR),
        ("0f72fc11-a9a5-425a-a1e5-a8d1b80762f1",
         "S1A_IW_GRDH_1SDV_20251016T110625_20251016T110650_061450_07ABC9_87B2",
         MATMO_DIR),
    ]

    print("\nPHASE 1 - SENTINEL-1 SCENE DOWNLOAD\n")
    for i, (pid, name, out_dir) in enumerate(scenes_to_download, 1):
        print(f"Scene {i}/{len(scenes_to_download)}")
        download_scene(pid, name, out_dir, USERNAME, PASSWORD)

    print("\nDOWNLOAD COMPLETE")
    for d in (YAGI_DIR, MATMO_DIR):
        files = sorted(f for f in os.listdir(d) if f.endswith(".zip"))
        total_gb = sum((d / f).stat().st_size for f in files) / 1e9
        print(f"\n{d}")
        for f in files:
            gb = (d / f).stat().st_size / 1e9
            print(f"  {f}  ({gb:.2f} GB)")
        print(f"  Subtotal: {total_gb:.2f} GB")

    print("\nDone:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
