"""Loss functions and batch-level Mixup/CutMix."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = "ce", **kw):
    weight = kw.get("weight")
    if kind == "ce":
        return nn.CrossEntropyLoss(weight=weight)
    if kind == "ls":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=float(kw.get("smoothing", .1)))
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), kw.get("alpha", weight))
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted requires weight=<tensor>")
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unknown loss: {kind!r}")


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = .1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        log_pt = F.log_softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - log_pt.exp()) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or not len(counts) or torch.any(counts <= 0):
        raise ValueError("counts must be a non-empty vector of positive values")
    if beta == 0:
        weights = counts.reciprocal()
    elif 0 < beta < 1:
        weights = (1 - beta) / (1 - torch.pow(beta, counts))
    else:
        raise ValueError("beta must be 0 or in (0, 1)")
    return (weights / weights.mean()).float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        mixed = x * lam + x[perm] * (1 - lam)
    elif mode == "cutmix":
        _, _, h, w = x.shape
        ratio = math.sqrt(1 - lam)
        cut_w, cut_h = int(w * ratio), int(h * ratio)
        cx, cy = int(torch.randint(w, (1,), device=x.device)), int(torch.randint(h, (1,), device=x.device))
        x1, x2, y1, y2 = max(cx-cut_w//2, 0), min(cx+cut_w//2, w), max(cy-cut_h//2, 0), min(cy+cut_h//2, h)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - (x2-x1)*(y2-y1)/float(w*h)
    else:
        raise ValueError(f"Unknown mix mode: {mode!r}")
    return mixed, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
