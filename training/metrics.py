"""
training/metrics.py — Accumulating segmentation metrics.
Tracks TP/FP/FN/TN across batches then computes final scores.
"""
import torch


class SegMetrics:
    def __init__(self, threshold=0.5, eps=1e-7):
        self.thr = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor,
               validity_mask: torch.Tensor = None):
        preds = (torch.sigmoid(logits) > self.thr).long()
        t     = targets.long()
        if validity_mask is not None:
            v     = validity_mask.long().expand_as(preds)
            preds = preds * v
            t     = t * v
        self.tp += (preds * t).sum().item()
        self.fp += (preds * (1 - t)).sum().item()
        self.fn += ((1 - preds) * t).sum().item()
        self.tn += ((1 - preds) * (1 - t)).sum().item()

    def compute(self) -> dict:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        e = self.eps
        precision = tp / (tp + fp + e)
        recall    = tp / (tp + fn + e)
        f1        = 2 * precision * recall / (precision + recall + e)
        iou       = tp / (tp + fp + fn + e)
        dice      = 2 * tp / (2 * tp + fp + fn + e)
        pixel_acc = (tp + tn) / (tp + fp + fn + tn + e)
        return dict(iou=iou, dice=dice, precision=precision,
                    recall=recall, f1=f1, pixel_acc=pixel_acc)