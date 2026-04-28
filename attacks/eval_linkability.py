import argparse
import json
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from attacks.train_source_id_attacker import SmallIdAttacker, build_transform, load_split_samples


class PathOnlyDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


def compute_auc(scores: List[float], labels: List[int]) -> float:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    pos_count = 0
    neg_count = 0
    for idx, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += idx
            pos_count += 1
        else:
            neg_count += 1
    if pos_count == 0 or neg_count == 0:
        return 0.5
    u = rank_sum - pos_count * (pos_count + 1) / 2.0
    return float(u / (pos_count * neg_count))


def compute_eer(scores: List[float], labels: List[int]) -> float:
    thresholds = sorted(set(scores))
    best_gap = 1.0
    best_eer = 1.0
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fpr = fp / max(1, (fp + tn))
        fnr = fn / max(1, (fn + tp))
        gap = abs(fpr - fnr)
        if gap < best_gap:
            best_gap = gap
            best_eer = (fpr + fnr) / 2.0
    return float(best_eer)


def sample_pairs_by_identity(
    identity_to_indices: Dict[int, List[int]],
    num_pos: int,
    num_neg: int,
    rng: random.Random,
) -> List[Tuple[int, int, int]]:
    pairs: List[Tuple[int, int, int]] = []
    valid_ids = [k for k, v in identity_to_indices.items() if len(v) >= 2]
    if valid_ids:
        for _ in range(num_pos):
            sid = rng.choice(valid_ids)
            i, j = rng.sample(identity_to_indices[sid], k=2)
            pairs.append((i, j, 1))

    all_ids = list(identity_to_indices.keys())
    if len(all_ids) >= 2:
        for _ in range(num_neg):
            sid_a, sid_b = rng.sample(all_ids, k=2)
            i = rng.choice(identity_to_indices[sid_a])
            j = rng.choice(identity_to_indices[sid_b])
            pairs.append((i, j, 0))
    return pairs


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.attacker_ckpt, map_location=device)
    model = SmallIdAttacker(num_classes=ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    split_samples = load_split_samples(args.metadata_csv, args.split, args.image_column)
    image_paths = [s.image_path for s in split_samples]
    source_ids = [s.source_id for s in split_samples]
    if len(split_samples) < 2:
        raise ValueError("Not enough samples for linkability evaluation.")

    dataset = PathOnlyDataset(image_paths, build_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            _, emb = model(images)
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb.cpu())
    emb_mat = torch.cat(embeddings, dim=0)

    identity_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx, sid in enumerate(source_ids):
        identity_to_indices[sid].append(idx)

    rng = random.Random(args.seed)
    pairs = sample_pairs_by_identity(
        identity_to_indices=identity_to_indices,
        num_pos=args.num_pos_pairs,
        num_neg=args.num_neg_pairs,
        rng=rng,
    )
    if not pairs:
        raise ValueError("Failed to sample pairs; check metadata class balance.")

    scores: List[float] = []
    labels: List[int] = []
    for i, j, label in pairs:
        s = torch.dot(emb_mat[i], emb_mat[j]).item()
        scores.append(s)
        labels.append(label)

    auc = compute_auc(scores, labels)
    eer = compute_eer(scores, labels)
    result = {
        "split": args.split,
        "num_pairs": len(pairs),
        "num_pos_pairs": sum(1 for _, _, y in pairs if y == 1),
        "num_neg_pairs": sum(1 for _, _, y in pairs if y == 0),
        "auc": auc,
        "eer": eer,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate linkability attack with cosine similarity over attacker embeddings."
    )
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--attacker-ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image-column", type=str, default="anonymized_path")
    parser.add_argument("--num-pos-pairs", type=int, default=5000)
    parser.add_argument("--num-neg-pairs", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", type=str, default="attack_results/linkability_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
