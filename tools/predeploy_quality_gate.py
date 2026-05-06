import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence

import cv2
import numpy as np
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

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class Metrics:
    sharpness: float
    contrast: float


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def list_images(root: str) -> List[str]:
    paths: List[str] = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def sample_images(paths: Sequence[str], max_samples: int, seed: int) -> List[str]:
    if len(paths) <= max_samples:
        return list(paths)
    rng = random.Random(seed)
    return rng.sample(list(paths), max_samples)


def load_generator(checkpoint: str, device: torch.device) -> PPRL_VGAN_Generator:
    encoder = PPRL_VGAN_Encoder(nc=nc, nz=nz)
    decoder = PPRL_VGAN_Decoder(nz=nz, n_id=n_id, nc=nc)
    net_g = PPRL_VGAN_Generator(encoder, decoder)
    state_dict = torch.load(checkpoint, map_location=device)
    net_g.load_state_dict(state_dict)
    net_g.to(device)
    net_g.eval()
    return net_g


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    x = tensor.detach().cpu()
    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    img = (x.numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img


def compute_metrics(rgb: np.ndarray) -> Metrics:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    return Metrics(sharpness=sharpness, contrast=contrast)


def infer_metrics_for_ckpt(
    checkpoint: str,
    image_paths: Sequence[str],
    target_id: int,
    device: torch.device,
) -> Dict[str, float]:
    net_g = load_generator(checkpoint, device=device)
    transform = build_transform()
    metrics: List[Metrics] = []
    c = torch.nn.functional.one_hot(
        torch.tensor([target_id], dtype=torch.long), num_classes=n_id
    ).float().to(device)

    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)
            fake, _, _ = net_g(x, c)
            rgb = tensor_to_rgb(fake[0])
            metrics.append(compute_metrics(rgb))

    if not metrics:
        return {"sharpness_mean": 0.0, "contrast_mean": 0.0}
    return {
        "sharpness_mean": float(np.mean([m.sharpness for m in metrics])),
        "contrast_mean": float(np.mean([m.contrast for m in metrics])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-deploy quality gate: compare visual quality of two checkpoints."
    )
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-sharpness-ratio",
        type=float,
        default=0.85,
        help="Candidate sharpness must be >= baseline * this ratio.",
    )
    parser.add_argument(
        "--min-contrast-ratio",
        type=float,
        default=0.85,
        help="Candidate contrast must be >= baseline * this ratio.",
    )
    parser.add_argument("--output-json", default="quality_gate_report.json")
    args = parser.parse_args()

    image_paths = list_images(args.image_root)
    if not image_paths:
        raise RuntimeError(f"No images found under: {args.image_root}")
    sampled = sample_images(image_paths, max_samples=args.max_samples, seed=args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline = infer_metrics_for_ckpt(
        args.baseline_ckpt, sampled, args.target_id, device=device
    )
    candidate = infer_metrics_for_ckpt(
        args.candidate_ckpt, sampled, args.target_id, device=device
    )

    sharpness_ratio = candidate["sharpness_mean"] / max(1e-8, baseline["sharpness_mean"])
    contrast_ratio = candidate["contrast_mean"] / max(1e-8, baseline["contrast_mean"])
    passed = (
        sharpness_ratio >= args.min_sharpness_ratio
        and contrast_ratio >= args.min_contrast_ratio
    )

    report = {
        "device": str(device),
        "num_images": len(sampled),
        "baseline_ckpt": args.baseline_ckpt,
        "candidate_ckpt": args.candidate_ckpt,
        "baseline": baseline,
        "candidate": candidate,
        "ratios": {
            "sharpness_ratio": sharpness_ratio,
            "contrast_ratio": contrast_ratio,
        },
        "thresholds": {
            "min_sharpness_ratio": args.min_sharpness_ratio,
            "min_contrast_ratio": args.min_contrast_ratio,
        },
        "pass": passed,
        "recommendation": "deploy" if passed else "reject_candidate",
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Quality gate report saved to {args.output_json}")

    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
