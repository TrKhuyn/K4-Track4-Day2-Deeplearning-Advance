"""DeepWeeds data loading, split validation, transforms, and DataLoaders."""
from __future__ import annotations

import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
               "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# The official subset CSVs contain only Filename and Label.  Species is
# available in labels.csv and is useful for EDA, but is not part of the split
# contract and must therefore remain optional here.
REQUIRED_COLUMNS = ("Filename", "Label")


def load_split(labels_dir: str | Path, fold: int = 0):
    labels_dir = Path(labels_dir)
    if fold not in range(5):
        raise ValueError("fold must be in [0, 4]")
    frames = []
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split file: {path}")
        df = pd.read_csv(path)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        if df["Filename"].duplicated().any():
            raise ValueError(f"Duplicate Filename in {path}")
        labels = pd.to_numeric(df["Label"], errors="raise")
        if not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"Labels outside [0, {NUM_CLASSES - 1}] in {path}")
        frames.append(df.reset_index(drop=True))
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")
    splits = {"train": train_df, "val": val_df, "test": test_df}
    names = {key: set(df["Filename"].astype(str)) for key, df in splits.items()}
    overlap = {"train_val": names["train"] & names["val"],
               "train_test": names["train"] & names["test"],
               "val_test": names["val"] & names["test"]}
    if any(overlap.values()):
        raise ValueError("Split overlap detected: " + str({k: sorted(v)[:10] for k, v in overlap.items() if v}))
    union = set().union(*names.values())
    if len(union) != 17_509:
        raise ValueError(f"Expected 17,509 unique images, found {len(union):,}")
    missing = [name for name in sorted(union) if not (images_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CSV images are missing; first: {missing[:10]}")
    result = {
        "n": {key: len(df) for key, df in splits.items()},
        "per_class": {key: {str(i): int(df["Label"].value_counts().get(i, 0))
                            for i in range(NUM_CLASSES)} for key, df in splits.items()},
        "overlap": {key: len(value) for key, value in overlap.items()},
        "union": len(union), "missing_files": 0,
    }
    print(pd.DataFrame(result["per_class"]).rename_axis("Label"))
    print("split sizes:", result["n"], "overlap:", result["overlap"], "union:", result["union"])
    return result


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.Resize(max(img_size, round(img_size / .875))),
                                   transforms.CenterCrop(img_size), *norm])
    base = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
    extras = {"basic": [], "color": [transforms.ColorJitter(.3, .3, .3, .1)],
              "trivial": [transforms.TrivialAugmentWide()],
              "randaug": [transforms.RandAugment(num_ops=2, magnitude=9)]}
    if aug in ("none", None):
        base = [transforms.Resize(max(img_size, round(img_size / .875))), transforms.CenterCrop(img_size)]
        extra = []
    elif aug in extras:
        extra = extras[aug]
    else:
        raise ValueError(f"Unknown augmentation: {aug!r}")
    return transforms.Compose([*base, *extra, *norm])


class DeepWeedsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        with Image.open(self.images_dir / filename) as image:
            image = image.convert("RGB")
            image = self.transform(image) if self.transform else image.copy()
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    data = DeepWeedsDataset(df, images_dir, transform)
    torch_sampler = None
    if sampler == "balanced":
        counts = df["Label"].value_counts()
        weights = df["Label"].map(lambda label: 1.0 / counts.loc[label]).to_numpy()
        torch_sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), True)
    elif sampler not in (None, "none"):
        raise ValueError(f"Unknown sampler: {sampler!r}")
    return DataLoader(data, batch_size=batch_size, shuffle=train and torch_sampler is None,
                      sampler=torch_sampler, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=train and len(data) >= batch_size,
                      worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
