"""Correctly synchronized inference latency benchmarks."""
from __future__ import annotations
import time
import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if warmup < 0 or iters < 1: raise ValueError("warmup >= 0 and iters >= 1 are required")
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        if sync: sync()
        start = time.perf_counter(); fn()
        if sync: sync()
        times.append((time.perf_counter() - start) * 1000)
    values = np.asarray(times)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "mean": float(values.mean()), "n": iters}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    import torch
    if dtype not in {"fp32", "amp", "fp16"}: raise ValueError("dtype must be fp32, amp, or fp16")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    model = model.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16": model = model.half(); x = x.half()
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dtype == "amp"):
            model(x)
    stats = bench(forward, warmup, iters, torch.cuda.synchronize if dev.type == "cuda" else None)
    stats.update({"gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
                  "dtype": dtype, "batch": batch_size, "img_size": img_size,
                  "images_per_s": batch_size / (stats["p50"] / 1000), "torch": torch.__version__})
    return stats


def tta_latency(model, k_views: int, **kw) -> dict:
    import torch
    if k_views < 1: raise ValueError("k_views must be positive")
    dev = torch.device(kw.get("device", "cuda")); dtype = kw.get("dtype", "fp32")
    model = model.to(dev).eval(); x = torch.randn(kw.get("batch_size", 1), 3, kw.get("img_size", 224), kw.get("img_size", 224), device=dev)
    if dtype == "fp16": model = model.half(); x = x.half()
    def forward():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dtype == "amp"):
            for _ in range(k_views): model(x)
    stats = bench(forward, kw.get("warmup", 10), kw.get("iters", 100), torch.cuda.synchronize if dev.type == "cuda" else None)
    stats.update({"k_views": k_views, "dtype": dtype, "batch": x.shape[0], "img_size": x.shape[-1],
                  "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
                  "images_per_s": x.shape[0] / (stats["p50"] / 1000), "torch": torch.__version__})
    return stats
