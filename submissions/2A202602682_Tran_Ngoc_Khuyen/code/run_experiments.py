"""Run the required Lab 2 experiment matrix without ever selecting on test.

Examples:
  python code/run_experiments.py backbones
  python code/run_experiments.py training
  python code/run_experiments.py combine
  python code/run_experiments.py final --confirm-test
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from train import Config, run

BACKBONES = [
    ("B01", "resnet50"), ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"), ("B04", "deit_small_patch16_224"),
    ("B05", "efficientnet_b0"),
]

# Controlled ablations: each row differs from T00 in one named factor.
TRAINING = [
    Config(exp_id="T00"),
    Config(exp_id="T01", init="scratch"),
    Config(exp_id="T02", init="frozen"),
    Config(exp_id="T03", aug="color"),
    Config(exp_id="T04", aug="randaug"),
    Config(exp_id="T05", mix="mixup", mix_alpha=.4),
    Config(exp_id="T06", mix="cutmix", mix_alpha=1.0),
    Config(exp_id="T07", loss="ls", label_smoothing=.1),
    Config(exp_id="T08", loss="focal", focal_gamma=2.0),
    Config(exp_id="T09", loss="ce_weighted", class_weight_beta=.9999),
    Config(exp_id="T10", sampler="balanced"),
    Config(exp_id="T11", ema_decay=.999),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("backbones", "training", "combine", "final"))
    parser.add_argument("--confirm-test", action="store_true", help="Required for the one-time final test run")
    parser.add_argument("--final-backbone", default="resnet50", help="Choose only after validation experiments")
    args = parser.parse_args()
    if args.stage == "backbones":
        configs = [Config(exp_id=exp_id, backbone=name) for exp_id, name in BACKBONES]
    elif args.stage == "training":
        configs = TRAINING
    elif args.stage == "combine":
        # Combination of the two strongest ResNet ablations, transferred to
        # the winning backbone.  This is validation-only.
        configs = [Config(exp_id="C01", backbone="convnext_tiny", aug="randaug",
                          sampler="balanced")]
    else:
        if not args.confirm_test:
            parser.error("Final stage reads test. Re-run with --confirm-test only after locking the configuration on val.")
        # Locked after validation: F01 uses the winning backbone/recipe.  Keep
        # T00 as the required ResNet-50 baseline with the same three seeds.
        # Inference/calibration is applied later without re-opening test.
        configs = [Config(exp_id="T00", backbone="resnet50", seed=seed,
                          save_test_predictions=True) for seed in (0, 1, 2)]
        configs += [Config(exp_id="F01", backbone=args.final_backbone, aug="randaug",
                           sampler="balanced", eval_img_size=256, seed=seed,
                           save_test_predictions=True) for seed in (0, 1, 2)]
    for cfg in configs:
        print(run(cfg))


if __name__ == "__main__":
    main()
