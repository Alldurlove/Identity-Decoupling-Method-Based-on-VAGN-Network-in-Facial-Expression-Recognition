import argparse
import csv
import hashlib
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torchvision.transforms as transforms
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import image_size, n_id, nc, nz
from models.VAE.Decoder import PPRL_VGAN_Decoder
from models.VAE.Encoder import PPRL_VGAN_Encoder
from models.generator import PPRL_VGAN_Generator


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


@dataclass
class Sample:
    image_path: str
    source_id: int
    source_exp: int
    rel_path: str


def discover_samples(dataroot: str) -> List[Sample]:
    samples: List[Sample] = []
    for id_name in sorted(os.listdir(dataroot)):
        id_path = os.path.join(dataroot, id_name)
        if not os.path.isdir(id_path):
            continue
        id_key = id_name.lower()
        if id_key not in ID_TO_IDX:
            continue

        for exp_folder in sorted(os.listdir(id_path)):
            exp_path = os.path.join(id_path, exp_folder)
            if not os.path.isdir(exp_path):
                continue
            exp_key = exp_folder.split("_")[-1].lower()
            if exp_key not in EXP_TO_IDX:
                continue

            for img_name in sorted(os.listdir(exp_path)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                img_path = os.path.join(exp_path, img_name)
                rel_path = os.path.join(id_name, exp_folder, img_name)
                samples.append(
                    Sample(
                        image_path=img_path,
                        source_id=ID_TO_IDX[id_key],
                        source_exp=EXP_TO_IDX[exp_key],
                        rel_path=rel_path,
                    )
                )
    return samples


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_generator(checkpoint: str, device: torch.device) -> PPRL_VGAN_Generator:
    encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
    decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
    net_g = PPRL_VGAN_Generator(encoder, decoder)
    state_dict = torch.load(checkpoint, map_location=device)
    net_g.load_state_dict(state_dict)
    net_g.to(device)
    net_g.eval()
    return net_g


def to_pil_image(tensor: torch.Tensor) -> Image.Image:
    t = tensor.detach().cpu()
    t = (t + 1.0) / 2.0
    t = torch.clamp(t, 0.0, 1.0)
    img = (t.numpy().transpose(1, 2, 0) * 255).astype("uint8")
    return Image.fromarray(img, mode="RGB")


def deterministic_split(rel_path: str) -> str:
    digest = hashlib.md5(rel_path.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "val"
    return "test"


def choose_target_id(source_id: int, rng: random.Random) -> int:
    candidates = [idx for idx in range(n_id) if idx != source_id]
    return rng.choice(candidates)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_generation(
    dataroot: str,
    output_root: str,
    checkpoint: str,
    seed: int,
) -> str:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    samples = discover_samples(dataroot)
    transform = build_transform()
    net_g = load_generator(checkpoint, device)

    os.makedirs(output_root, exist_ok=True)
    image_root = os.path.join(output_root, "images")
    os.makedirs(image_root, exist_ok=True)
    metadata_path = os.path.join(output_root, "metadata.csv")

    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "anonymized_path",
                "source_path",
                "source_id",
                "source_exp",
                "target_id",
            ],
        )
        writer.writeheader()

        with torch.no_grad():
            for sample in samples:
                target_id = choose_target_id(sample.source_id, rng)
                source_img = Image.open(sample.image_path).convert("RGB")
                source_tensor = transform(source_img).unsqueeze(0).to(device)
                c = torch.nn.functional.one_hot(
                    torch.tensor([target_id], dtype=torch.long), num_classes=n_id
                ).float().to(device)
                fake_img, _, _ = net_g(source_tensor, c)
                fake_pil = to_pil_image(fake_img[0])

                out_rel = sample.rel_path.rsplit(".", 1)[0] + f"_to_{target_id}.png"
                out_abs = os.path.join(image_root, out_rel)
                ensure_parent(out_abs)
                fake_pil.save(out_abs)

                writer.writerow(
                    {
                        "split": deterministic_split(sample.rel_path),
                        "anonymized_path": out_abs,
                        "source_path": sample.image_path,
                        "source_id": sample.source_id,
                        "source_exp": sample.source_exp,
                        "target_id": target_id,
                    }
                )

    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate anonymized images for privacy attack experiments."
    )
    parser.add_argument("--dataroot", type=str, required=True, help="FERG dataset root.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained generator checkpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="attack_data/anonymized",
        help="Output directory for anonymized images and metadata.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata_file = run_generation(
        dataroot=args.dataroot,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        seed=args.seed,
    )
    print(f"Anonymized dataset generated. Metadata: {metadata_file}")
