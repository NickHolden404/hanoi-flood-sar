"""
losses/combined_loss.py

BCEDiceHydroLoss — extends the existing BCE+Dice loss with a hydrological
consistency penalty term:

    L_total = w_bce * L_BCE + w_dice * L_Dice + lambda_hydro * L_Hydro

The hydrological term is imported from losses/hydro_loss.py and penalises
flood predictions at pixels that are locally elevated relative to their DEM
neighbours.  See hydro_loss.py for physical motivation.

The BCE and Dice components replicate the behaviour of models/losses.py
(BCEDiceLoss) exactly — same smoothing, same validity-mask contract — so
the two loss modules can be compared on equal terms.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Ensure project root is importable regardless of working directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from losses.hydro_loss import HydrologicalConsistencyLoss


# ── helper: Dice loss (mirrors models/losses.py exactly) ─────────────────

class _DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter   = (probs * targets).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)


# ── main combined loss ────────────────────────────────────────────────────

class BCEDiceHydroLoss(nn.Module):
    """
    Combined loss = w_bce * BCE + w_dice * Dice + lambda_hydro * HydroLoss.

    Parameters
    ----------
    w_bce        : float  weight for BCE term          (default 0.5)
    w_dice       : float  weight for Dice term         (default 0.5)
    lambda_hydro : float  weight for hydro penalty     (default 0.1)
    hydro_threshold : float  elevation threshold in m  (default 2.0)
    """

    def __init__(
        self,
        w_bce: float = 0.5,
        w_dice: float = 0.5,
        lambda_hydro: float = 0.1,
        hydro_threshold: float = 2.0,
    ):
        super().__init__()
        self.w_bce        = w_bce
        self.w_dice       = w_dice
        self.lambda_hydro = lambda_hydro

        self._bce   = nn.BCEWithLogitsLoss(reduction="none")
        self._dice  = _DiceLoss(smooth=1.0)
        self._hydro = HydrologicalConsistencyLoss(threshold=hydro_threshold)

    def forward(
        self,
        logits:  Tensor,   # (B, 1, H, W) — raw model output before sigmoid
        targets: Tensor,   # (B, 1, H, W) — binary flood mask
        x_batch: Tensor,   # (B, 7, H, W) — full input stack (DEM at ch 6)
        validity_mask: Tensor = None,  # (B,1,H,W) float — 1=valid, kept for
                                       # API compatibility with BCEDiceLoss
    ) -> Tensor:
        """
        Parameters
        ----------
        logits       : raw model output (before sigmoid)
        targets      : binary segmentation mask in {0, 1}
        x_batch      : full 7-channel input tensor; DEM must be channel 6
        validity_mask: optional nodata mask (1=valid); zero-weights nodata px

        Returns
        -------
        Tensor  scalar total loss
        """
        # ── BCE ───────────────────────────────────────────────────────────
        bce_map = self._bce(logits, targets)
        if validity_mask is not None:
            bce_map  = bce_map * validity_mask
            n_valid  = validity_mask.sum().clamp(min=1)
            bce_loss = bce_map.sum() / n_valid
        else:
            bce_loss = bce_map.mean()

        # ── Dice ──────────────────────────────────────────────────────────
        if validity_mask is not None:
            v = validity_mask.expand_as(logits)
            dice_loss = self._dice(logits * v, targets * v)
        else:
            dice_loss = self._dice(logits, targets)

        # ── Hydro ─────────────────────────────────────────────────────────
        pred = torch.sigmoid(logits)
        hydro_loss = self._hydro(pred, x_batch)

        # ── Combine ───────────────────────────────────────────────────────
        total = (self.w_bce * bce_loss
                 + self.w_dice * dice_loss
                 + self.lambda_hydro * hydro_loss)
        return total


if __name__ == "__main__":
    B, H, W = 2, 256, 256
    logits  = torch.randn(B, 1, H, W)
    targets = (torch.rand(B, 1, H, W) > 0.8).float()
    x       = torch.rand(B, 7, H, W)
    x[:, 6, :, :] = torch.rand(B, H, W) * 20.0   # realistic DEM

    loss_fn = BCEDiceHydroLoss(w_bce=0.5, w_dice=0.5, lambda_hydro=0.1)
    val = loss_fn(logits, targets, x)
    assert val.ndim == 0 and val.item() > 0
    print(f"[Step 2 COMPLETE] BCEDiceHydroLoss smoke test passed. "
          f"Loss = {val.item():.6f}")