"""Collect real run artifacts into the required results.xlsx workbook."""
from __future__ import annotations
import json
import sys
import glob
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import eval as lab_eval

SHEETS = {
    "Backbones": ["exp_id", "backbone", "pretrained_tag", "params_m", "gmac", "img_size", "epochs", "seed", "macro_f1_val", "top1_val", "train_seconds_per_epoch", "latency_batch1_ms", "notes"],
    "Training": ["exp_id", "backbone", "axis", "change_from_T00", "seed", "macro_f1_val", "top1_val", "delta_macro_f1", "f1_chinee_apple", "f1_snake_weed", "notes"],
    "Inference": ["exp_id", "method", "checkpoint", "k", "macro_f1_val", "top1_val", "ece_val", "p50_ms", "p95_ms", "p99_ms", "images_per_s", "relative_cost"],
    "Final": ["exp_id", "configuration", "seed", "macro_f1_val", "macro_f1_test", "top1_test", "ece_test", "mean", "std"],
    "PerClass": ["configuration", "class", "support", "precision", "recall", "f1"],
    "Latency": ["configuration", "gpu", "dtype", "batch", "fused_bn", "p50_ms", "p95_ms", "p99_ms", "images_per_s"],
    "Summary": ["rank", "exp_id", "configuration", "macro_f1_val", "top1_val", "p95_ms", "notes"],
}


def collect_runs(root=Path("runs")):
    rows = []
    for path in root.glob("*/seed*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        row = {**config, **summary}
        pred_path = Path("predictions") / f"{row['exp_id']}_seed{row['seed']}_val.csv"
        if pred_path.exists():
            pred = lab_eval.read_pred(str(pred_path))
            metrics = lab_eval.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
            row.update({"balanced_acc_val": metrics["balanced_acc"], "ece_val": metrics["ece"],
                        "f1_chinee_apple": metrics["f1"][0], "f1_snake_weed": metrics["f1"][7]})
        rows.append(row)
    return rows


def main():
    rows = collect_runs(); backbone_rows = [r for r in rows if r["exp_id"].startswith("B")]
    training_rows = [r for r in rows if r["exp_id"].startswith(("T", "C"))]
    axes = {
        "T00": ("Baseline", "Công thức nền"), "T01": ("A", "scratch thay finetune"),
        "T02": ("A", "frozen thay finetune"), "T03": ("B", "color thay basic"),
        "T04": ("B", "RandAugment thay basic"), "T05": ("B", "Mixup alpha=0.4"),
        "T06": ("B", "CutMix alpha=1.0"), "T07": ("C", "label smoothing=0.1"),
        "T08": ("C", "focal gamma=2"), "T09": ("C", "class-balanced CE beta=0.9999"),
        "T10": ("D", "balanced sampler"), "T11": ("F", "EMA decay=0.999"),
        "C01": ("Combination", "ConvNeXt + RandAugment + balanced sampler"),
    }
    baseline = next((r["macro_f1_val"] for r in training_rows if r["exp_id"] == "T00"), None)
    pretrained_tags = {"resnet50": "a1_in1k", "resnext50_32x4d": "a1h_in1k",
                       "convnext_tiny": "in12k_ft_in1k", "deit_small_patch16_224": "fb_in1k",
                       "efficientnet_b0": "ra_in1k"}
    latency_file = Path("inference_out/backbone_latency.json")
    backbone_latency = {row["exp_id"]: row for row in json.loads(latency_file.read_text())} if latency_file.exists() else {}
    for row in backbone_rows:
        row["pretrained_tag"] = pretrained_tags.get(row["backbone"], "timm default")
        row["latency_batch1_ms"] = backbone_latency.get(row["exp_id"], {}).get("p50")
        row["notes"] = "Latency batch-1 FP32 p50; 10 warmup + 100 iterations"
    for row in training_rows:
        row["axis"], row["change_from_T00"] = axes.get(row["exp_id"], ("", ""))
        row["delta_macro_f1"] = row["macro_f1_val"] - baseline if baseline is not None else None
        row["notes"] = "1 seed; chưa dùng để kết luận vượt nhiễu"
    frames = {name: pd.DataFrame(columns=columns) for name, columns in SHEETS.items()}
    if backbone_rows: frames["Backbones"] = pd.DataFrame(backbone_rows).reindex(columns=SHEETS["Backbones"])
    if training_rows: frames["Training"] = pd.DataFrame(training_rows).reindex(columns=SHEETS["Training"])
    inference_path = Path("inference_out/inference.csv")
    if inference_path.exists():
        frames["Inference"] = pd.read_csv(inference_path).reindex(columns=SHEETS["Inference"])
    latency_path = Path("inference_out/latency.csv")
    if latency_path.exists():
        latency = pd.read_csv(latency_path)
        latency["p50_ms"], latency["p95_ms"], latency["p99_ms"] = latency["p50"], latency["p95"], latency["p99"]
        frames["Latency"] = latency.reindex(columns=SHEETS["Latency"])

    final_rows, per_class_rows = [], []
    for tag in ("T00", "F01"):
        files = sorted(glob.glob(f"predictions/{tag}_seed*_test.csv"))
        if len(files) >= 3:
            metrics = []
            for file in files:
                pred = lab_eval.read_pred(file); m = lab_eval.compute_metrics(pred.y_true, pred.y_pred, pred.probs)
                metrics.append(m)
                val_file = Path("predictions") / f"{tag}_seed{pred.seed}_val.csv"
                val_f1 = None
                if val_file.exists():
                    vp = lab_eval.read_pred(str(val_file)); val_f1 = lab_eval.compute_metrics(vp.y_true, vp.y_pred, vp.probs)["macro_f1"]
                final_rows.append({"exp_id": tag, "configuration": "Baseline ResNet50 I00" if tag == "T00" else "Final ConvNeXt C01 + I07",
                    "seed": pred.seed, "macro_f1_val": val_f1, "macro_f1_test": m["macro_f1"],
                    "top1_test": m["top1"], "ece_test": m["ece"]})
            f1_mean, f1_std = lab_eval.mean_std([m["macro_f1"] for m in metrics])
            top_mean, top_std = lab_eval.mean_std([m["top1"] for m in metrics])
            final_rows.append({"exp_id": f"{tag}-mean", "configuration": "Tổng hợp 3 seed", "seed": "mean±std",
                "macro_f1_test": f1_mean, "top1_test": top_mean,
                "mean": f"F1 {f1_mean:.4f} ± {f1_std:.4f}; top1 {top_mean:.4f} ± {top_std:.4f}"})
            for i, class_name in enumerate(lab_eval.CLASS_NAMES):
                per_class_rows.append({"configuration": tag, "class": class_name,
                    "support": int(metrics[0]["support"][i]),
                    "precision": float(np.mean([m["precision"][i] for m in metrics])),
                    "recall": float(np.mean([m["recall"][i] for m in metrics])),
                    "f1": float(np.mean([m["f1"][i] for m in metrics]))})
    if final_rows: frames["Final"] = pd.DataFrame(final_rows).reindex(columns=SHEETS["Final"])
    if per_class_rows: frames["PerClass"] = pd.DataFrame(per_class_rows).reindex(columns=SHEETS["PerClass"])

    ranked = sorted(rows, key=lambda row: row.get("macro_f1_val", -1), reverse=True)[:10]
    frames["Summary"] = pd.DataFrame([{
        "rank": i, "exp_id": row["exp_id"],
        "configuration": f"{row['backbone']} | aug={row['aug']} | loss={row['loss']}",
        "macro_f1_val": row["macro_f1_val"], "top1_val": row["top1_val"],
        "p95_ms": None, "notes": "Validation; latency chưa đo",
    } for i, row in enumerate(ranked, 1)], columns=SHEETS["Summary"])
    with pd.ExcelWriter("results.xlsx", engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.book[name]; sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(40, max(12, max(len(str(c.value or "")) for c in column) + 2))
    print(f"Wrote results.xlsx from {len(rows)} real runs")


if __name__ == "__main__": main()
