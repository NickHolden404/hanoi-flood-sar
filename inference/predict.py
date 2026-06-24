"""
inference/predict.py — Full-scene sliding-window inference.

Reads a 7-channel model stack (output of 01_build_stacks.py),
tiles it through the model with overlap-averaging,
and writes a float32 probability GeoTIFF + binary flood mask GeoTIFF.
"""
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F


def predict_fullscene(model, stack_path: str, out_prob_path: str,
                      cfg: dict, device=None) -> np.ndarray:
    """
    Parameters
    ----------
    model         : loaded model with weights, in eval mode
    stack_path    : path to 7-channel model stack GeoTIFF
    out_prob_path : output path for probability GeoTIFF (float32, 0–1)
    cfg           : config dict
    device        : torch device (auto-detected if None)

    Returns
    -------
    prob_map : H×W float32 numpy array
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ps   = cfg["dataset"]["patch_size"]    # 256
    ov   = cfg["dataset"]["overlap"]       # 128
    step = ps - ov

    model.to(device).eval()

    with rasterio.open(stack_path) as src:
        stack = src.read().astype(np.float32)   # 7 × H × W
        meta  = src.meta.copy()

    _, H, W = stack.shape
    prob_map = np.zeros((H, W), dtype=np.float32)
    cnt_map  = np.zeros((H, W), dtype=np.float32)

    # Include a final tile anchored to the far edge to ensure full coverage
    rows = list(range(0, H - ps, step)) + [H - ps]
    cols = list(range(0, W - ps, step)) + [W - ps]

    with torch.no_grad():
        for r in rows:
            for c in cols:
                tile  = stack[:, r:r+ps, c:c+ps]
                t     = torch.from_numpy(tile).unsqueeze(0).to(device)
                logit = model(t)
                prob  = torch.sigmoid(logit).squeeze().cpu().numpy()
                prob_map[r:r+ps, c:c+ps] += prob
                cnt_map [r:r+ps, c:c+ps] += 1

    prob_map /= np.maximum(cnt_map, 1)

    Path(out_prob_path).parent.mkdir(parents=True, exist_ok=True)
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32", "nodata": -1})
    with rasterio.open(out_prob_path, "w", **out_meta) as dst:
        dst.write(prob_map[np.newaxis, ...])

    print(f"  Probability map  : {out_prob_path}")
    return prob_map


def prob_to_binary(prob_path: str, out_binary_path: str,
                   threshold: float = 0.5) -> np.ndarray:
    """Convert probability GeoTIFF  binary flood mask GeoTIFF (uint8)."""
    with rasterio.open(prob_path) as src:
        prob = src.read(1)
        meta = src.meta.copy()

    binary = (prob >= threshold).astype(np.uint8)
    meta.update({"dtype": "uint8", "nodata": 255})
    with rasterio.open(out_binary_path, "w", **meta) as dst:
        dst.write(binary[np.newaxis, ...])

    flood_frac = binary.mean()
    print(f"  Binary mask      : {out_binary_path}  "
          f"(flood={flood_frac:.2%}, thr={threshold})")
    return binary