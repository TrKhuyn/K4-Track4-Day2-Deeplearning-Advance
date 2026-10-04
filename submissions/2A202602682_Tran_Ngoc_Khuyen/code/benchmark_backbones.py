"""Measure comparable batch-1 FP32 latency for all screened backbones."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark, model as model_utils


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    rows = []
    for exp_id in ("B01", "B02", "B03", "B04", "B05"):
        run = ROOT / "runs" / exp_id / "seed0"
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        net = model_utils.build_model(cfg["backbone"], pretrained=False, init="finetune")
        net.load_state_dict(torch.load(run / "best.pt", map_location="cuda", weights_only=False)["model"])
        result = benchmark.latency_report(net, 1, 224, "fp32", "cuda", warmup=10, iters=100)
        result.update({"exp_id": exp_id, "backbone": cfg["backbone"]}); rows.append(result)
        print(exp_id, result)
    out = ROOT / "inference_out/backbone_latency.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__": main()
