"""Single, reproducible training pipeline for all DeepWeeds experiments."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
import types
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import eval as lab_eval
import dataset
import losses
import model as model_utils


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    eval_img_size: int | None = None
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    return torch.optim.AdamW(model_utils.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total = max(1, int(cfg.epochs * steps_per_epoch)); warmup = int(cfg.warmup_epochs * steps_per_epoch)
    def factor(step):
        if warmup and step < warmup: return max((step + 1) / warmup, 1e-8)
        progress = (step - warmup) / max(1, total - warmup)
        return .5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    def __init__(self, model, decay: float):
        if not 0 < decay < 1: raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay; self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source = model.state_dict(); target = self.module.state_dict()
        for name, value in target.items():
            incoming = source[name].detach()
            if value.is_floating_point(): value.mul_(self.decay).add_(incoming, alpha=1-self.decay)
            else: value.copy_(incoming)

    @torch.no_grad()
    def copy_to(self, model) -> None:
        model.load_state_dict(self.module.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train(); model_utils.set_frozen_backbone_eval(model)
    total_loss = 0.0; seen = 0; device_type = torch.device(device).type
    for images, target, _ in loader:
        images = images.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
        targets = target
        if cfg.mix: images, targets = losses.mix_batch(images, target, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, enabled=cfg.amp and device_type == "cuda"):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, target)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if ema: ema.update(model)
        total_loss += float(loss.detach()) * len(images); seen += len(images)
    return {"train_loss": total_loss / max(seen, 1), "lr": max(g["lr"] for g in optimizer.param_groups)}


def evaluate(model, loader, criterion, device):
    model.eval(); filenames, labels, logits_all = [], [], []; total_loss = 0.0; seen = 0
    with torch.inference_mode():
        for images, target, names in loader:
            images = images.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            logits = model(images); loss = criterion(logits, target)
            filenames.extend(names); labels.append(target.cpu()); logits_all.append(logits.float().cpu())
            total_loss += float(loss) * len(images); seen += len(images)
    return filenames, torch.cat(labels).numpy(), torch.cat(logits_all).numpy(), total_loss / max(seen, 1)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history); path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(frame.epoch, frame.train_loss, marker="o", label="train")
    axes[0].plot(frame.epoch, frame.val_loss, marker="o", label="val")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss"); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(frame.epoch, frame.macro_f1_val, marker="o", label="macro-F1 val")
    axes[1].plot(frame.epoch, frame.top1_val, marker="o", label="top-1 val")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation metrics"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)


def _criterion(cfg, train_df, device):
    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = train_df["Label"].value_counts().reindex(range(dataset.NUM_CLASSES)).to_numpy()
        weight = losses.class_weights(counts, 0.0 if cfg.class_weight_beta is None else cfg.class_weight_beta).to(device)
    return losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                  gamma=cfg.focal_gamma, weight=weight)


def run(cfg: Config) -> dict:
    set_seed(cfg.seed); output = run_dir(cfg); output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_stats = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    (output / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    train_loader = dataset.make_loader(train_df, cfg.images_dir, dataset.build_transforms(True, cfg.img_size, cfg.aug),
                                       cfg.batch_size, True, cfg.sampler, cfg.num_workers)
    eval_size = cfg.img_size if cfg.eval_img_size is None else cfg.eval_img_size
    val_loader = dataset.make_loader(val_df, cfg.images_dir, dataset.build_transforms(False, eval_size),
                                     cfg.batch_size, False, None, cfg.num_workers)
    test_loader = dataset.make_loader(test_df, cfg.images_dir, dataset.build_transforms(False, eval_size),
                                      cfg.batch_size, False, None, cfg.num_workers) if cfg.save_test_predictions else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model_utils.build_model(cfg.backbone, init=cfg.init, drop_rate=cfg.drop_rate).to(device)
    criterion = _criterion(cfg, train_df, device); optimizer = build_optimizer(net, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(net, cfg.ema_decay) if cfg.ema_decay is not None else None
    checkpoint = output / "best.pt"; history_path = output / "history.csv"
    if checkpoint.exists() and history_path.exists() and len(pd.read_csv(history_path)) >= cfg.epochs:
        history = pd.read_csv(history_path).to_dict("records")
        best = max(history, key=lambda row: row["macro_f1_val"])
        best_score, best_epoch = best["macro_f1_val"], int(best["epoch"])
        print(f"Reusing completed training in {output}; running post-processing only.")
    else:
        best_score = -math.inf; best_epoch = 0; history = []
        for epoch in range(1, cfg.epochs + 1):
            started = time.perf_counter()
            row = train_one_epoch(net, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
            eval_net = ema.module if ema else net
            _, y_true, logits, val_loss = evaluate(eval_net, val_loader, criterion, device)
            probs = torch.from_numpy(logits).softmax(1).numpy(); metrics = lab_eval.compute_metrics(y_true, probs.argmax(1), probs)
            row.update({"epoch": epoch, "val_loss": val_loss, "macro_f1_val": metrics["macro_f1"],
                        "top1_val": metrics["top1"], "epoch_seconds": time.perf_counter() - started})
            history.append(row); print(json.dumps(row))
            if row["macro_f1_val"] > best_score:
                best_score = row["macro_f1_val"]; best_epoch = epoch
                torch.save({"model": eval_net.state_dict(), "epoch": epoch, "macro_f1_val": best_score, "config": asdict(cfg)}, checkpoint)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    plot_curves(history, Path("curves") / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png", f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed}")
    saved = torch.load(checkpoint, map_location=device, weights_only=False); net.load_state_dict(saved["model"])
    names, y_true, logits, _ = evaluate(net, val_loader, criterion, device)
    np.savez_compressed(output / "val_logits.npz", Filename=np.asarray(names), y_true=y_true, logits=logits)
    val_probs = torch.from_numpy(logits).softmax(1).numpy(); lab_eval.save_predictions(pred_path(cfg, "val"), names, y_true, val_probs)
    val_metrics = lab_eval.compute_metrics(y_true, val_probs.argmax(1), val_probs)
    if test_loader is not None:
        names, y_true, logits, _ = evaluate(net, test_loader, criterion, device)
        np.savez_compressed(output / "test_logits.npz", Filename=np.asarray(names), y_true=y_true, logits=logits)
        lab_eval.save_predictions(pred_path(cfg, "test"), names, y_true, torch.from_numpy(logits).softmax(1).numpy())
    summary = {"exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best_epoch,
               "macro_f1_val": float(val_metrics["macro_f1"]), "top1_val": float(val_metrics["top1"]),
               "params_m": model_utils.count_params(net), "gmac": model_utils.count_gmacs(net, cfg.img_size),
               "train_seconds_per_epoch": float(np.mean([x["epoch_seconds"] for x in history]))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _convert(value: str, annotation):
    args = get_args(annotation); allows_none = type(None) in args
    if value.lower() in {"none", "null"}:
        if allows_none: return None
        raise ValueError("None is not allowed")
    base = next((x for x in args if x is not type(None)), annotation) if args else annotation
    if base is bool:
        if value.lower() not in {"true", "false", "1", "0", "yes", "no"}: raise ValueError(f"Invalid bool: {value}")
        return value.lower() in {"true", "1", "yes"}
    return base(value) if base in {str, int, float} else value


def parse_overrides(pairs: list[str]) -> dict:
    hints = get_type_hints(Config); valid = {field.name for field in fields(Config)}; result = {}
    for pair in pairs:
        if "=" not in pair: raise ValueError(f"Override must be KEY=VALUE: {pair!r}")
        key, value = pair.split("=", 1)
        if key not in valid: raise KeyError(f"Unknown Config field: {key}")
        result[key] = _convert(value, hints[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(); print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__": main()
