"""Generate required EDA/pipeline evidence without using test for selection."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset, losses, model as model_utils


def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    out = ROOT / "eda"; out.mkdir(exist_ok=True)
    train, val, test = dataset.load_split(ROOT / "data/labels", 0)
    stats = dataset.check_split(train, val, test, ROOT / "data/images")
    counts = pd.DataFrame(stats["per_class"]).astype(int)
    counts.index = dataset.CLASS_NAMES
    ax = counts.plot.bar(figsize=(12, 5)); ax.set(ylabel="Số ảnh", title="DeepWeeds fold 0 — phân bố lớp")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(out / "class_distribution.png", dpi=160); plt.close()

    all_labels = pd.concat([train, val, test], ignore_index=True)
    fig, axes = plt.subplots(9, 3, figsize=(9, 25))
    for label in range(9):
        rows = all_labels[all_labels.Label == label].sample(3, random_state=label)
        for col, (_, row) in enumerate(rows.iterrows()):
            axes[label, col].imshow(Image.open(ROOT / "data/images" / row.Filename).convert("RGB"))
            axes[label, col].set_title(dataset.CLASS_NAMES[label]); axes[label, col].axis("off")
    plt.tight_layout(); plt.savefig(out / "samples_3_per_class.png", dpi=140); plt.close()

    transform = dataset.build_transforms(True, 224, "randaug")
    rows = train.sample(8, random_state=0); fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    mean = torch.tensor(dataset.IMAGENET_MEAN)[:, None, None]; std = torch.tensor(dataset.IMAGENET_STD)[:, None, None]
    for ax, (_, row) in zip(axes.flat, rows.iterrows()):
        image = transform(Image.open(ROOT / "data/images" / row.Filename).convert("RGB"))
        ax.imshow(((image * std + mean).clamp(0, 1)).permute(1, 2, 0)); ax.set_title(dataset.CLASS_NAMES[int(row.Label)]); ax.axis("off")
    plt.tight_layout(); plt.savefig(out / "augmentation_samples.png", dpi=160); plt.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_utils.build_model("efficientnet_b0", pretrained=False, init="scratch").to(device)
    x = torch.randn(64, 3, 224, 224, device=device); y = torch.arange(64, device=device) % 9
    with torch.no_grad():
        logits = net(x); initial_loss = float(torch.nn.functional.cross_entropy(logits, y))
        uniform_loss = float(torch.nn.functional.cross_entropy(torch.zeros_like(logits), y))
        focal_diff = float(abs(losses.FocalLoss(0)(logits, y) - torch.nn.functional.cross_entropy(logits, y)))
    # Required debugging check: a model must be able to memorize one tiny,
    # fixed batch.  Random tensors are intentional: this tests optimization,
    # not generalization or dataset quality.
    tiny_x, tiny_y = x[:8], y[:8]
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
    overfit_loss = None; overfit_steps = 0
    net.train()
    for step in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(net(tiny_x), tiny_y)
        loss.backward(); optimizer.step()
        overfit_loss = float(loss.detach()); overfit_steps = step
        if overfit_loss < 0.02: break
    evidence = {"split": stats, "initial_ce_random_network": initial_loss,
                "uniform_head_ce": uniform_loss, "ln_9": float(np.log(9)),
                "focal_gamma0_ce_abs_diff": focal_diff, "overfit_batch_final_loss": overfit_loss,
                "overfit_batch_steps": overfit_steps,
                "note": "Ảnh augmentation và 3 mẫu/lớp nằm cùng thư mục."}
    (out / "pipeline_checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__": main()
