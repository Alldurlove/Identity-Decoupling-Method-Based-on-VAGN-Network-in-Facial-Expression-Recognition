import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import image_size


NUM_EXP_CLASSES = 7


@dataclass
class AttackSample:
    image_path: str
    source_exp: int


class CsvExpressionDataset(Dataset):
    def __init__(self, samples: Sequence[AttackSample], transform: transforms.Compose):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        return self.transform(image), torch.tensor(sample.source_exp, dtype=torch.long)


class SmallExpressionAttacker(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features(x).flatten(1)
        emb = self.embedding(feat)
        logits = self.classifier(emb)
        return logits, emb


def load_split_samples(metadata_csv: str, split: str, image_column: str) -> List[AttackSample]:
    out: List[AttackSample] = []
    with open(metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] != split:
                continue
            out.append(
                AttackSample(
                    image_path=row[image_column],
                    source_exp=int(row["source_exp"]),
                )
            )
    return out


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def compute_macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    f1_values: List[float] = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (labels == cls)).sum().item()
        fp = ((preds == cls) & (labels != cls)).sum().item()
        fn = ((preds != cls) & (labels == cls)).sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        f1_values.append(f1)
    return float(sum(f1_values) / len(f1_values))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_count += labels.size(0)
            all_preds.append(torch.argmax(logits, dim=1).cpu())
            all_labels.append(labels.cpu())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    macro_f1 = compute_macro_f1(preds, labels, num_classes=num_classes)
    return {
        "loss": total_loss / max(1, total_count),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_count += labels.size(0)
    return total_loss / max(1, total_count)


def save_training_curves(history: Dict[str, List[float]], output_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip curve plot because matplotlib is unavailable: {exc}")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Expression Attacker Loss Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.plot(epochs, history["val_macro_f1"], label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = build_transform()

    train_samples = load_split_samples(args.metadata_csv, "train", args.image_column)
    val_samples = load_split_samples(args.metadata_csv, "val", args.image_column)
    test_samples = load_split_samples(args.metadata_csv, "test", args.image_column)

    train_loader = DataLoader(
        CsvExpressionDataset(train_samples, transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        CsvExpressionDataset(val_samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        CsvExpressionDataset(test_samples, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = SmallExpressionAttacker(num_classes=NUM_EXP_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt = os.path.join(args.output_dir, "best_expression_attacker.pth")
    best_val_acc = -1.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, device, criterion, NUM_EXP_CLASSES)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        print(
            f"[Epoch {epoch + 1}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_column": args.image_column,
                    "num_classes": NUM_EXP_CLASSES,
                },
                best_ckpt,
            )

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion, NUM_EXP_CLASSES)

    report_path = os.path.join(args.output_dir, "expression_report.json")
    history_path = os.path.join(args.output_dir, "expression_train_history.json")
    curves_path = os.path.join(args.output_dir, "expression_training_curves.png")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    save_training_curves(history, curves_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata_csv": args.metadata_csv,
                "image_column": args.image_column,
                "num_train": len(train_samples),
                "num_val": len(val_samples),
                "num_test": len(test_samples),
                "test_metrics": test_metrics,
                "history_path": history_path,
                "curves_path": curves_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Expression evaluation saved to {report_path}")
    print(
        "Final expression metrics: "
        f"acc={test_metrics['acc']:.4f}, macro_f1={test_metrics['macro_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train expression classifier on anonymized outputs."
    )
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument(
        "--image-column",
        type=str,
        default="anonymized_path",
        help="CSV column containing image path. Use source_path for raw-image baseline.",
    )
    parser.add_argument("--output-dir", type=str, default="attack_results/expression")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
