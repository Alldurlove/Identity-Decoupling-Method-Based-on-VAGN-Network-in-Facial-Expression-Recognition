import os
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


ID_TO_IDX = {
    "aia": 0,
    "bonnie": 1,
    "jules": 2,
    "malcolm": 3,
    "mery": 4,
    "ray": 5,
}

EXP_TO_IDX = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6,
}

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class FERGMultiLabelDataset(Dataset):
    def __init__(self, dataroot: str, transform=None):
        self.dataroot = dataroot
        self.transform = transform
        self.samples: List[Tuple[str, int, int]] = []
        self._discover()

    def _discover(self) -> None:
        for id_name in sorted(os.listdir(self.dataroot)):
            id_path = os.path.join(self.dataroot, id_name)
            if not os.path.isdir(id_path):
                continue
            id_key = id_name.lower()
            if id_key not in ID_TO_IDX:
                continue
            for exp_folder_name in sorted(os.listdir(id_path)):
                exp_path = os.path.join(id_path, exp_folder_name)
                if not os.path.isdir(exp_path):
                    continue
                exp_key = exp_folder_name.split("_")[-1].lower()
                if exp_key not in EXP_TO_IDX:
                    continue
                for img_name in sorted(os.listdir(exp_path)):
                    if not img_name.lower().endswith(IMAGE_EXTS):
                        continue
                    img_full_path = os.path.join(exp_path, img_name)
                    self.samples.append(
                        (img_full_path, ID_TO_IDX[id_key], EXP_TO_IDX[exp_key])
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, id_label, exp_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, id_label, exp_label
