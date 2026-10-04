"""Validation-only inference study and correctly synchronized latency benchmark."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as lab_eval
import benchmark, dataset, inference, model as model_utils


def load_run(exp_id: str, seed: int = 0, device="cuda"):
    run_dir = ROOT / "runs" / exp_id / f"seed{seed}"
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    net = model_utils.build_model(cfg["backbone"], pretrained=False, init="finetune",
                                  drop_rate=cfg["drop_rate"])
    saved = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(saved["model"]); return net.to(device).eval(), cfg


def metrics_row(exp_id, method, checkpoint, k, y, probs, latency=None):
    m = lab_eval.compute_metrics(y, probs.argmax(1), probs)
    latency = latency or {}
    return {"exp_id": exp_id, "method": method, "checkpoint": checkpoint, "k": k,
            "macro_f1_val": m["macro_f1"], "top1_val": m["top1"], "ece_val": m["ece"],
            "p50_ms": latency.get("p50"), "p95_ms": latency.get("p95"),
            "p99_ms": latency.get("p99"), "images_per_s": latency.get("images_per_s")}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda": raise RuntimeError("Inference benchmark requires CUDA on this machine")
    selected = "C01"
    model, cfg = load_run(selected, device=device)
    _, val_df, _ = dataset.load_split(ROOT / cfg["labels_dir"], cfg["fold"])
    loader224 = dataset.make_loader(val_df, ROOT / cfg["images_dir"],
        dataset.build_transforms(False, 224), cfg["batch_size"], False, None, cfg["num_workers"])
    names, y, logits0 = inference.predict_logits(model, loader224, device)
    names_flip, y_flip, logits_flip = inference.predict_logits(model, loader224, device, inference.view_hflip)
    if names != names_flip or not np.array_equal(y, y_flip): raise RuntimeError("TTA order mismatch")
    p0 = inference.apply_temperature(logits0, 1.0)
    prob_tta = inference.aggregate_views([logits0, logits_flip], "prob")
    logit_tta = inference.aggregate_views([logits0, logits_flip], "logit")
    temperature = inference.fit_temperature(logits0, y)
    calibrated = inference.apply_temperature(logits0, temperature)

    loader256 = dataset.make_loader(val_df, ROOT / cfg["images_dir"],
        dataset.build_transforms(False, 256), cfg["batch_size"], False, None, cfg["num_workers"])
    names256, y256, logits256 = inference.predict_logits(model, loader256, device)
    if names != names256 or not np.array_equal(y, y256): raise RuntimeError("Resolution order mismatch")
    p256 = inference.apply_temperature(logits256, 1.0)
    temperature256 = inference.fit_temperature(logits256, y256)
    calibrated256 = inference.apply_temperature(logits256, temperature256)

    deit = lab_eval.read_pred(str(ROOT / "predictions" / "B04_seed0_val.csv"))
    index = {name: i for i, name in enumerate(deit.filenames)}
    deit_probs = np.stack([deit.probs[index[name]] for name in names])
    ensemble = inference.ensemble_probs([p0, deit_probs])

    latency_rows = []
    for dtype in ("fp32", "amp", "fp16"):
        fresh, _ = load_run(selected, device=device)
        result = benchmark.latency_report(fresh, 1, 224, dtype=dtype, device=device, warmup=10, iters=100)
        result.update({"configuration": "C01 ConvNeXt-Tiny I00", "fused_bn": False})
        latency_rows.append(result)
    fresh, _ = load_run(selected, device=device)
    tta_time = benchmark.tta_latency(fresh, 2, batch_size=1, img_size=224, dtype="amp",
                                     device=device, warmup=10, iters=100)
    tta_time.update({"configuration": "C01 ConvNeXt-Tiny I01 TTA x2", "fused_bn": False})
    latency_rows.append(tta_time)
    fresh, _ = load_run(selected, device=device)
    resolution256_time = benchmark.latency_report(fresh, 1, 256, dtype="fp32", device=device,
                                                  warmup=10, iters=100)
    resolution256_time.update({"configuration": "C01 ConvNeXt-Tiny I03 resolution256",
                               "fused_bn": False})
    latency_rows.append(resolution256_time)
    base_amp = next(x for x in latency_rows if x["dtype"] == "amp" and x["configuration"].endswith("I00"))
    methods = [
        ("I00", "1-view 224", 1, p0, base_amp),
        ("I01", "horizontal-flip TTA, mean probability", 2, prob_tta, tta_time),
        ("I02", "horizontal-flip TTA, mean logits", 2, logit_tta, tta_time),
        ("I03", "test resolution 256", 1, p256, resolution256_time),
        ("I04", f"temperature scaling T={temperature:.6f}", 1, calibrated, base_amp),
        ("I05", "probability ensemble ConvNeXt+DeiT", 2, ensemble, None),
        ("I06", "AMP/FP16 deployment (predictions unchanged)", 1, p0, base_amp),
        ("I07", f"resolution 256 + temperature T={temperature256:.6f}", 1, calibrated256, resolution256_time),
    ]
    rows = []
    for exp_id, method, k, probs, timing in methods:
        rows.append(metrics_row(exp_id, method, "C01_seed0", k, y, probs, timing))
        lab_eval.save_predictions(ROOT / "predictions" / f"{exp_id}_seed0_val.csv", names, y, probs)
    baseline_p50 = rows[0]["p50_ms"]
    for row in rows:
        row["relative_cost"] = row["p50_ms"] / baseline_p50 if row["p50_ms"] else None
    out = ROOT / "inference_out"; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "inference.csv", index=False)
    pd.DataFrame(latency_rows).to_csv(out / "latency.csv", index=False)
    (out / "temperature.json").write_text(json.dumps({"temperature_224": temperature,
        "temperature_256": temperature256}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(latency_rows).to_string(index=False))


if __name__ == "__main__": main()
