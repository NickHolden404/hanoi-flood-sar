"""
losses/hydro_loss.py

HydrologicalConsistencyLoss — penalises flood predictions at pixels that are
locally elevated relative to their cardinal neighbours, which is physically
implausible (water cannot pool on local high ground).

Physical motivation
-------------------
If pred[i,j] is high (predicted flood) AND dem[i,j] > mean(cardinal neighbours)
+ threshold, the prediction violates basic hydrostatic principles.  We apply a
soft, fully-differentiable version: penalty = pred[i,j] * elevation_excess_mask,
where the mask is a sigmoid-smoothed indicator rather than a hard binary gate.

Usage
-----
    loss_fn = HydrologicalConsistencyLoss(threshold=2.0)
    hydro_loss = loss_fn(pred, x_batch)   # pred: (B,1,H,W), x: (B,7,H,W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class HydrologicalConsistencyLoss(nn.Module):
    """
    Penalise flood predictions that are locally inconsistent with the DEM.

    Parameters
    ----------
    threshold : float
        Elevation difference (metres) above mean of cardinal neighbours above
        which a flood prediction is considered physically implausible.
        Default: 2.0 m (one SRTM pixel elevation step — soft threshold).
    """

    def __init__(self, threshold: float = 2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred: Tensor, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        pred : Tensor  (B, 1, H, W)  sigmoid output in [0, 1]
        x    : Tensor  (B, 7, H, W)  full input batch; DEM is channel index 6

        Returns
        -------
        Tensor  scalar loss value
        """
        # ── extract DEM channel ──────────────────────────────────────────
        dem = x[:, 6:7, :, :]          # (B, 1, H, W)

        # ── reflect-pad by 1 to handle borders ───────────────────────────
        # pad order: (left, right, top, bottom)
        dem_pad = F.pad(dem, (1, 1, 1, 1), mode="reflect")  # (B,1,H+2,W+2)

        # ── cardinal neighbours (sliced from padded tensor) ───────────────
        north = dem_pad[:, :, :-2, 1:-1]   # row i-1
        south = dem_pad[:, :, 2:,  1:-1]   # row i+1
        west  = dem_pad[:, :, 1:-1, :-2]   # col j-1
        east  = dem_pad[:, :, 1:-1, 2:]    # col j+1

        mean_nbr = (north + south + west + east) / 4.0  # (B, 1, H, W)

        # ── elevation excess: how much higher is [i,j] than its neighbours?
        excess = dem - mean_nbr - self.threshold   # positive  implausible

        # ── soft mask: sigmoid so gradient flows through ──────────────────
        # Values well above threshold  ~1.0; below threshold  ~0.0
        implausible = torch.sigmoid(excess)        # (B, 1, H, W)

        # ── penalty: weighted by prediction confidence ────────────────────
        # pred already in [0,1]; this is the soft version of
        # "penalise pred[i,j] if it predicts flood AND terrain is elevated"
        penalty = pred * implausible               # (B, 1, H, W)

        return penalty.mean()


if __name__ == "__main__":
    # ── quick smoke test ─────────────────────────────────────────────────
    B, H, W = 2, 256, 256
    pred_t = torch.rand(B, 1, H, W)
    x_t    = torch.rand(B, 7, H, W)
    # Make DEM channel (idx 6) span realistic Hanoi elevations (0–20 m)
    x_t[:, 6, :, :] = torch.rand(B, H, W) * 20.0

    loss_fn = HydrologicalConsistencyLoss(threshold=2.0)
    val = loss_fn(pred_t, x_t)
    assert val.ndim == 0 and val.item() >= 0, "Unexpected output shape/sign"
    print(f"[Step 1 COMPLETE] HydrologicalConsistencyLoss smoke test passed. "
          f"Loss = {val.item():.6f}")