"""
models/losses.py — BCE + Dice composite loss with validity mask support.
Phase 2 confirmed 100% valid pixels for both events; mask retained for robustness.
"""
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter   = (probs * targets).sum()
        return 1 - (2. * inter + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce    = nn.BCEWithLogitsLoss(reduction="none")
        self.dice   = DiceLoss()
        self.bce_w  = bce_w
        self.dice_w = dice_w

    def forward(self, logits, targets, validity_mask=None):
        """
        validity_mask : (B, 1, H, W) float — 1=valid, 0=nodata (Phase 2 Ch5).
        When None, all pixels are treated as valid.
        """
        bce_map = self.bce(logits, targets)
        if validity_mask is not None:
            bce_map  = bce_map * validity_mask
            n_valid  = validity_mask.sum().clamp(min=1)
            bce_loss = bce_map.sum() / n_valid
            v        = validity_mask.expand_as(logits)
            dice_loss = self.dice(logits * v, targets * v)
        else:
            bce_loss  = bce_map.mean()
            dice_loss = self.dice(logits, targets)

        return self.bce_w * bce_loss + self.dice_w * dice_loss