"""Fit one temperature on final validation logits and calibrate saved test logits.

This performs no test forward pass: it only transforms logits already produced once
by the locked final training runs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval as lab_eval
import inference


def read_npz(path):
    data = np.load(path); return data["Filename"].tolist(), data["y_true"], data["logits"]


def main():
    vals = [read_npz(ROOT / "runs" / "F01" / f"seed{s}" / "val_logits.npz") for s in (0, 1, 2)]
    temperature = inference.fit_temperature(np.concatenate([x[2] for x in vals]),
                                            np.concatenate([x[1] for x in vals]))
    print(f"Locked validation temperature: {temperature:.8f}")
    for seed, (names, labels, logits) in enumerate(vals):
        lab_eval.save_predictions(ROOT / "predictions" / f"F01_seed{seed}_val.csv",
                                  names, labels, inference.apply_temperature(logits, temperature))
        test_names, test_labels, test_logits = read_npz(ROOT / "runs" / "F01" / f"seed{seed}" / "test_logits.npz")
        lab_eval.save_predictions(ROOT / "predictions" / f"F01uncal_seed{seed}_test.csv",
                                  test_names, test_labels, inference.apply_temperature(test_logits, 1.0))
        lab_eval.save_predictions(ROOT / "predictions" / f"F01_seed{seed}_test.csv",
                                  test_names, test_labels, inference.apply_temperature(test_logits, temperature))
    (ROOT / "inference_out" / "final_temperature.txt").write_text(str(temperature), encoding="utf-8")


if __name__ == "__main__": main()
