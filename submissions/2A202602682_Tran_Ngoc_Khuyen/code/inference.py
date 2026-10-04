"""Inference, TTA, ensembling, calibration, and Conv-BN fusion."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def predict_logits(model, loader, device, view=None):
    model.eval(); names, labels, outputs = [], [], []
    with torch.inference_mode():
        for images, target, filenames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(device_type=torch.device(device).type, enabled=torch.device(device).type == "cuda"):
                logits = model(images)
            outputs.append(logits.float().cpu()); labels.append(target.cpu()); names.extend(filenames)
    return names, torch.cat(labels).numpy(), torch.cat(outputs).numpy()


def view_identity(x): return x


def view_hflip(x): return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    if crop <= 0 or crop > min(x.shape[-2:]):
        raise ValueError("crop must fit within the input")
    h, w = x.shape[-2:]; positions = [(0, 0), (0, w-crop), (h-crop, 0), (h-crop, w-crop), ((h-crop)//2, (w-crop)//2)]
    return [x[..., y:y+crop, left:left+crop] for y, left in positions]


def views_multiscale(x, sizes):
    sizes = list(sizes)
    if not sizes or any(int(s) <= 0 for s in sizes):
        raise ValueError("sizes must contain positive integers")
    return [F.interpolate(x, size=(int(s), int(s)), mode="bilinear", align_corners=False, antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    if not logits_per_view:
        raise ValueError("At least one view is required")
    tensors = [torch.as_tensor(x, dtype=torch.float64) for x in logits_per_view]
    if any(x.shape != tensors[0].shape for x in tensors):
        raise ValueError("All views must have the same shape")
    if space == "prob": probs = torch.stack([x.softmax(1) for x in tensors]).mean(0)
    elif space == "logit": probs = torch.stack(tensors).mean(0).softmax(1)
    else: raise ValueError("space must be 'prob' or 'logit'")
    return probs.numpy()


def ensemble_probs(list_of_probs):
    if not list_of_probs: raise ValueError("At least one model is required")
    arrays = [np.asarray(x, dtype=np.float64) for x in list_of_probs]
    if any(x.shape != arrays[0].shape for x in arrays): raise ValueError("Probability arrays have different shapes")
    result = np.mean(arrays, axis=0)
    if np.any(result < 0) or np.any(~np.isfinite(result)): raise ValueError("Invalid probabilities")
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    if logits.ndim != 2 or len(logits) != len(labels): raise ValueError("Invalid logits/labels")
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=.1, max_iter=100, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(); loss = F.cross_entropy(logits / log_t.exp().clamp(1e-3, 1e3), labels); loss.backward(); return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(1e-3, 1e3))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0: raise ValueError("T must be finite and positive")
    return torch.as_tensor(logits, dtype=torch.float64).div(T).softmax(1).numpy()


def _fuse_sequential(module):
    names = list(module._modules)
    for left, right in zip(names, names[1:]):
        conv, bn = module._modules[left], module._modules[right]
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            module._modules[left] = torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
            module._modules[right] = nn.Identity()
    for child in module.children(): _fuse_sequential(child)


def fuse_conv_bn(model):
    fused = copy.deepcopy(model).eval()
    _fuse_sequential(fused)
    return fused
