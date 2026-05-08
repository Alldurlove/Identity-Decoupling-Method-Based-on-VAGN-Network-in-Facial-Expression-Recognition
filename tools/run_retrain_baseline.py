import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_train_command(cfg: Dict, python_bin: str, epochs_override: int = 0) -> List[str]:
    train = cfg["train"]
    epochs = epochs_override if epochs_override > 0 else train["epochs"]
    command = [
        python_bin,
        train["script"],
        "--data-root",
        train["data_root"],
        "--data-format",
        train.get("data_format", "auto"),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(train["batch_size"]),
        "--num-workers",
        str(train["num_workers"]),
        "--lr",
        str(train["lr"]),
        "--g-updates",
        str(train.get("g_updates", 2)),
        "--seed",
        str(cfg.get("seed", 42)),
        "--save-dir",
        train["save_dir"],
    ]
    if "lambda_recon" in train:
        command.extend(["--lambda-recon", str(train["lambda_recon"])])
    if "lambda_edge" in train:
        command.extend(["--lambda-edge", str(train["lambda_edge"])])
    if "stage1_epochs" in train:
        command.extend(["--stage1-epochs", str(train["stage1_epochs"])])
    if "stage2_id_warmup_epochs" in train:
        command.extend(["--stage2-id-warmup-epochs", str(train["stage2_id_warmup_epochs"])])
    return command


def save_manifest(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible real-data retraining baseline from a JSON config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrain_real_baseline.json",
        help="Path to experiment config JSON.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=sys.executable,
        help="Python interpreter to run training command.",
    )
    parser.add_argument(
        "--epochs-override",
        type=int,
        default=0,
        help="Override epochs for quick sanity run (0 = use config).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and write manifest without executing training.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    command = build_train_command(cfg, python_bin=args.python_bin, epochs_override=args.epochs_override)
    cmd_str = " ".join(shlex.quote(c) for c in command)

    save_dir = cfg["train"]["save_dir"]
    manifest_path = os.path.join(save_dir, "run_manifest.json")
    payload = {
        "experiment_name": cfg.get("experiment_name", "retrain_real_baseline"),
        "config_path": os.path.abspath(args.config),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python_bin": args.python_bin,
        "epochs_override": args.epochs_override,
        "dry_run": args.dry_run,
        "command": command,
        "command_pretty": cmd_str,
    }
    save_manifest(manifest_path, payload)
    print(f"Manifest saved: {manifest_path}")
    print(f"Train command: {cmd_str}")

    if args.dry_run:
        print("Dry run enabled, skip execution.")
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
