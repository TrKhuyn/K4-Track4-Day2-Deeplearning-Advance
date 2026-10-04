"""Model construction, parameter groups, and model complexity utilities."""
from __future__ import annotations
import copy
import torch
from torch import nn

SUGGESTED_BACKBONES = {"resnet50": "resnet50", "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny", "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224", "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100"}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install timm with `pip install timm` to build models") from exc
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Unknown init mode: {init!r}")
    resolved = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(resolved, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    model._deepweeds_init = init
    model._deepweeds_name = resolved
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_parameter_ids(model):
    classifier = model.get_classifier()
    modules = list(classifier) if isinstance(classifier, (tuple, list)) else [classifier]
    return {id(p) for module in modules if isinstance(module, nn.Module) for p in module.parameters()}


def freeze_backbone(model) -> None:
    head_ids = _head_parameter_ids(model)
    if not head_ids:
        raise ValueError("Model classifier has no parameters")
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._deepweeds_frozen = True
    set_frozen_backbone_eval(model)


def set_frozen_backbone_eval(model) -> None:
    if not getattr(model, "_deepweeds_frozen", False):
        return
    head_ids = _head_parameter_ids(model)
    for module in model.modules():
        params = list(module.parameters(recurse=False))
        if params and all(id(p) not in head_ids for p in params):
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = _head_parameter_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head": []}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            groups["head"].append(parameter)
        elif parameter.ndim <= 1:
            groups["backbone_no_decay"].append(parameter)
        else:
            groups["backbone_decay"].append(parameter)
    result = []
    if groups["backbone_decay"]:
        result.append({"params": groups["backbone_decay"], "lr": lr_backbone, "weight_decay": weight_decay})
    if groups["backbone_no_decay"]:
        result.append({"params": groups["backbone_no_decay"], "lr": lr_backbone, "weight_decay": 0.0})
    if groups["head"]:
        result.append({"params": groups["head"], "lr": lr_head, "weight_decay": weight_decay})
    return result


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        clone = copy.deepcopy(model).eval()
        macs = FlopCountAnalysis(clone, torch.zeros(1, 3, img_size, img_size, device=device)).total()
        return float(macs / 1e9)
    except ImportError:
        try:
            from thop import profile
            device = next(model.parameters()).device
            macs, _ = profile(copy.deepcopy(model).eval(), inputs=(torch.zeros(1, 3, img_size, img_size, device=device),), verbose=False)
            return float(macs / 1e9)
        except ImportError:
            return float("nan")
