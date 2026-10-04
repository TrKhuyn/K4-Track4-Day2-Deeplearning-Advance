"""Create final confusion matrix and representative hard-class error montage."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eval as lab_eval


def main():
    out = ROOT / "curves"; out.mkdir(exist_ok=True)
    cm = np.zeros((9, 9), dtype=np.int64)
    for seed in (0, 1, 2):
        seed_pred = pd.read_csv(ROOT / f"predictions/F01_seed{seed}_test.csv")
        np.add.at(cm, (seed_pred.y_true.astype(int), seed_pred.y_pred.astype(int)), 1)
    fig, ax = plt.subplots(figsize=(10, 8)); im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(9), lab_eval.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(9), lab_eval.CLASS_NAMES); ax.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thật",
        title="F01 — ma trận nhầm lẫn cộng qua 3 seed")
    for i in range(9):
        for j in range(9): ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                                    color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "F01_confusion_matrix.png", dpi=180); plt.close(fig)

    pred = pd.read_csv(ROOT / "predictions/F01_seed0_test.csv")
    hard = pred[((pred.y_true == 0) & (pred.y_pred == 7)) | ((pred.y_true == 7) & (pred.y_pred == 0))]
    if hard.empty:
        hard = pred[(pred.y_true.isin([0, 7])) & (pred.y_true != pred.y_pred)]
    hard = hard.head(12)
    cols = min(4, max(1, len(hard))); rows = max(1, int(np.ceil(len(hard) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 4.0 * rows), squeeze=False)
    axes = list(axes.flat)
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, hard.iterrows()):
        ax.imshow(Image.open(ROOT / "data/images" / row.Filename).convert("RGB")); ax.axis("off")
        ax.set_title(f"true: {lab_eval.CLASS_NAMES[int(row.y_true)]}\npred: {lab_eval.CLASS_NAMES[int(row.y_pred)]}", fontsize=9)
    fig.suptitle("Ví dụ lỗi ở hai lớp khó (F01 seed 0)"); fig.tight_layout()
    fig.savefig(out / "F01_hard_class_errors.png", dpi=160); plt.close(fig)
    print("confusion_total", int(cm.sum()), "hard_examples", len(hard))


if __name__ == "__main__": main()
