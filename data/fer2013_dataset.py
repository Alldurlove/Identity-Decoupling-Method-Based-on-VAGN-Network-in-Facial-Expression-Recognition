import csv
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
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

FER_FOLDER_TO_EXP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "joy",
    "neutral": "neutral",
    "sad": "sadness",
    "surprise": "surprise",
}

# FER2013 CSV label mapping:
# 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
FER_CSV_LABEL_TO_EXP = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "sadness",
    5: "surprise",
    6: "neutral",
}

CSV_USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "val",
    "PrivateTest": "test",
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class FerSample:
    path: Optional[str]
    pixels: Optional[str]
    exp_label: int
    split: str
    pseudo_id: int


def make_pseudo_id(key: str) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_id


def detect_data_format(data_root: str) -> str:
    csv_path = os.path.join(data_root, "fer2013.csv")
    if os.path.isfile(csv_path):
        return "csv"
    return "folder"


def parse_csv_samples(data_root: str) -> List[FerSample]:
    csv_path = os.path.join(data_root, "fer2013.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"FER2013 csv not found: {csv_path}")

    samples: List[FerSample] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            emotion = int(row["emotion"])
            usage = row["Usage"]
            split = CSV_USAGE_TO_SPLIT.get(usage)
            if split is None:
                continue
            exp_name = FER_CSV_LABEL_TO_EXP.get(emotion)
            if exp_name is None:
                continue
            key = f"csv:{idx}:{usage}:{emotion}"
            samples.append(
                FerSample(
                    path=None,
                    pixels=row["pixels"],
                    exp_label=EXP_TO_IDX[exp_name],
                    split=split,
                    pseudo_id=make_pseudo_id(key),
                )
            )
    return samples


def parse_folder_samples(data_root: str) -> List[FerSample]:
    samples: List[FerSample] = []
    for split in ("train", "val", "test"):
        split_root = os.path.join(data_root, split)
        if not os.path.isdir(split_root):
            continue
        for folder_name in sorted(os.listdir(split_root)):
            folder_path = os.path.join(split_root, folder_name)
            if not os.path.isdir(folder_path):
                continue
            mapped_exp = FER_FOLDER_TO_EXP.get(folder_name.lower())
            if mapped_exp is None:
                continue
            exp_label = EXP_TO_IDX[mapped_exp]
            for img_name in sorted(os.listdir(folder_path)):
                if not img_name.lower().endswith(IMAGE_EXTS):
                    continue
                img_path = os.path.join(folder_path, img_name)
                key = f"folder:{img_path}"
                samples.append(
                    FerSample(
                        path=img_path,
                        pixels=None,
                        exp_label=exp_label,
                        split=split,
                        pseudo_id=make_pseudo_id(key),
                    )
                )
    return samples


class FER2013Dataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        transform,
        data_format: str = "auto",
    ) -> None:
        self.transform = transform
        self.split = split
        if data_format == "auto":
            data_format = detect_data_format(data_root)
        if data_format not in ("csv", "folder"):
            raise ValueError("data_format must be one of: auto, csv, folder")

        if data_format == "csv":
            all_samples = parse_csv_samples(data_root)
        else:
            all_samples = parse_folder_samples(data_root)

        self.samples = [s for s in all_samples if s.split == split]
        if not self.samples:
            raise RuntimeError(
                f"No FER2013 samples found for split='{split}' under {data_root} (format={data_format})."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _decode_pixels(self, pixels: str) -> Image.Image:
        arr = np.fromstring(pixels, dtype=np.uint8, sep=" ")
        if arr.size != 48 * 48:
            raise ValueError(f"Unexpected pixels length: {arr.size}")
        img = arr.reshape(48, 48)
        return Image.fromarray(img, mode="L").convert("RGB")

    def __getitem__(self, idx: int) -> Tuple[object, int, int]:
        sample = self.samples[idx]
        if sample.path is not None:
            image = Image.open(sample.path).convert("RGB")
        else:
            image = self._decode_pixels(sample.pixels or "")
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, sample.pseudo_id, sample.exp_label
