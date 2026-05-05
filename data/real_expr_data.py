import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset

from config import n_id


EXP_TO_IDX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

ID_TO_IDX = {
    "aia": 0,
    "bonnie": 1,
    "jules": 2,
    "malcolm": 3,
    "mery": 4,
    "ray": 5,
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class ExprSample:
    image_path: str
    id_label: int
    exp_label: int


def _infer_identity_from_name(path: str) -> Optional[int]:
    basename = os.path.basename(path).lower()
    for identity, idx in ID_TO_IDX.items():
        if identity in basename:
            return idx
    return None


def discover_expr_samples(
    root: str,
    split: str,
    default_id: int = 0,
    parse_id_from_filename: bool = False,
) -> List[ExprSample]:
    split_root = os.path.join(root, split)
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    samples: List[ExprSample] = []
    for exp_name, exp_idx in EXP_TO_IDX.items():
        exp_dir = os.path.join(split_root, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        for dirpath, _, files in os.walk(exp_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                image_path = os.path.join(dirpath, fname)
                if parse_id_from_filename:
                    parsed = _infer_identity_from_name(image_path)
                    id_label = parsed if parsed is not None else default_id
                else:
                    id_label = default_id
                samples.append(
                    ExprSample(
                        image_path=image_path,
                        id_label=max(0, min(id_label, n_id - 1)),
                        exp_label=exp_idx,
                    )
                )
    if not samples:
        raise RuntimeError(f"No samples found under {split_root}.")
    return samples


class RealExpressionDataset(Dataset):
    def __init__(self, samples: Sequence[ExprSample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, sample.id_label, sample.exp_label


class MixedExpressionDataset(Dataset):
    """
    Mix two datasets with weighted random sampling.
    The resulting length follows `base_len`, usually the real dataset size.
    """

    def __init__(
        self,
        primary_dataset: Dataset,
        secondary_dataset: Dataset,
        primary_weight: float = 0.7,
        base_len: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        if not (0.0 <= primary_weight <= 1.0):
            raise ValueError("primary_weight must be in [0, 1]")
        self.primary_dataset = primary_dataset
        self.secondary_dataset = secondary_dataset
        self.primary_weight = primary_weight
        self.base_len = (
            base_len
            if base_len is not None
            else max(len(primary_dataset), len(secondary_dataset))
        )
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return self.base_len

    def __getitem__(self, idx: int):
        use_primary = self.rng.random() < self.primary_weight
        if use_primary:
            sample_idx = self.rng.randrange(len(self.primary_dataset))
            return self.primary_dataset[sample_idx]
        sample_idx = self.rng.randrange(len(self.secondary_dataset))
        return self.secondary_dataset[sample_idx]
