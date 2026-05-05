import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


EXPRESSIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


@dataclass
class FileSample:
    source_path: str
    expression: str
    relative_name: str


def parse_split_ratio(raw: str) -> Tuple[int, int, int]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError("--split-ratio must be like 70,15,15")
    train, val, test = (int(x.strip()) for x in parts)
    if train <= 0 or val < 0 or test < 0:
        raise ValueError("split values must be non-negative and train > 0")
    if train + val + test != 100:
        raise ValueError("split values must sum to 100")
    return train, val, test


def discover_expression_dirs(source_root: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for exp in EXPRESSIONS:
        exp_dir = os.path.join(source_root, exp)
        if os.path.isdir(exp_dir):
            out[exp] = exp_dir
    return out


def list_samples(expression_dirs: Dict[str, str]) -> List[FileSample]:
    samples: List[FileSample] = []
    for exp, exp_dir in expression_dirs.items():
        for root, _, files in os.walk(exp_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(IMAGE_EXTS):
                    continue
                src = os.path.join(root, fname)
                rel = os.path.relpath(src, exp_dir)
                samples.append(FileSample(source_path=src, expression=exp, relative_name=rel))
    return samples


def deterministic_split(key: str, split_ratio: Tuple[int, int, int]) -> str:
    train_pct, val_pct, _ = split_ratio
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def ensure_dirs(output_root: str) -> None:
    for split in ("train", "val", "test"):
        for exp in EXPRESSIONS:
            os.makedirs(os.path.join(output_root, split, exp), exist_ok=True)


def safe_copy(src: str, dst: str, overwrite: bool) -> None:
    parent = os.path.dirname(dst)
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(dst) and not overwrite:
        return
    shutil.copy2(src, dst)


def build_destination_name(relative_name: str) -> str:
    # Flatten nested directories to avoid hidden class leakage via subfolder names.
    flat = relative_name.replace("\\", "__").replace("/", "__")
    return flat


def run(args: argparse.Namespace) -> str:
    split_ratio = parse_split_ratio(args.split_ratio)
    expression_dirs = discover_expression_dirs(args.source_root)
    if not expression_dirs:
        raise RuntimeError(
            "No expression folders found. Expected subdirs like "
            f"{', '.join(EXPRESSIONS)} under {args.source_root}"
        )

    missing = [exp for exp in EXPRESSIONS if exp not in expression_dirs]
    if missing:
        print(f"Warning: missing expressions in source root: {missing}")

    samples = list_samples(expression_dirs)
    if not samples:
        raise RuntimeError("No image files found in expression folders.")

    ensure_dirs(args.output_root)
    split_counter: Counter = Counter()
    expr_split_counter: Counter = Counter()
    index_rows: List[Dict[str, str]] = []

    for sample in samples:
        split = deterministic_split(
            key=f"{sample.expression}/{sample.relative_name}",
            split_ratio=split_ratio,
        )
        dst_name = build_destination_name(sample.relative_name)
        dst_path = os.path.join(args.output_root, split, sample.expression, dst_name)
        safe_copy(sample.source_path, dst_path, overwrite=args.overwrite)
        split_counter[split] += 1
        expr_split_counter[(split, sample.expression)] += 1
        index_rows.append(
            {
                "split": split,
                "expression": sample.expression,
                "source_path": sample.source_path,
                "prepared_path": dst_path,
            }
        )

    report = {
        "source_root": args.source_root,
        "output_root": args.output_root,
        "split_ratio": args.split_ratio,
        "num_total": len(samples),
        "split_counts": {k: split_counter.get(k, 0) for k in ("train", "val", "test")},
        "split_expression_counts": {
            f"{split}/{exp}": expr_split_counter.get((split, exp), 0)
            for split in ("train", "val", "test")
            for exp in EXPRESSIONS
        },
        "missing_expressions": missing,
    }

    os.makedirs(args.output_root, exist_ok=True)
    index_path = os.path.join(args.output_root, "index.json")
    report_path = os.path.join(args.output_root, "prepare_report.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_rows, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare real-face expression dataset into train/val/test folders."
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Root folder containing expression subfolders (anger, disgust, ...).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="real_data",
        help="Prepared dataset root with train/val/test/<expression>/ layout.",
    )
    parser.add_argument(
        "--split-ratio",
        type=str,
        default="70,15,15",
        help="Train/val/test percentages, must sum to 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_file = run(args)
    print(f"Prepared dataset report: {report_file}")
