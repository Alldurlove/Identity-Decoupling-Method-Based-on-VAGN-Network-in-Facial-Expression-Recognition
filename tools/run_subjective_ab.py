import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple


def run_shell(command: str) -> str:
    result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def parse_pairs(values: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for value in values:
        if "::" not in value:
            raise ValueError(f"Invalid model pair '{value}'. Use format label::/path/to/ckpt.pth")
        label, path = value.split("::", 1)
        out.append((label.strip(), path.strip()))
    return out


def poll_health(url: str, retries: int = 15, sleep_seconds: float = 0.5) -> str:
    last_error = ""
    for _ in range(retries):
        proc = subprocess.run(
            f'curl -s "{url}"',
            shell=True,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        last_error = proc.stderr.strip() or f"curl_exit={proc.returncode}"
        time.sleep(sleep_seconds)
    raise RuntimeError(f"Health check failed after retries: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Switch through model checkpoints for subjective A/B testing."
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec format: label::/abs/path/to/checkpoint.pth. Repeat multiple times.",
    )
    parser.add_argument("--project-root", default="/home/ubuntu/VGAN-Project")
    parser.add_argument("--health-url", default="http://127.0.0.1:8000/api/health")
    parser.add_argument("--sleep-seconds", type=int, default=0, help="Optional wait after each switch.")
    parser.add_argument("--output-json", default="attack_results/subjective_ab_switch_log.json")
    args = parser.parse_args()

    pairs = parse_pairs(args.model)
    logs: List[Dict] = []

    for label, checkpoint in pairs:
        deploy_cmd = f'bash tools/deploy_best_checkpoint.sh "{checkpoint}"'
        deploy_proc = subprocess.run(
            deploy_cmd,
            shell=True,
            check=False,
            cwd=args.project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if deploy_proc.returncode not in (0, 7):
            raise RuntimeError(
                f"Deploy command failed for {label} ({checkpoint}) with code {deploy_proc.returncode}:\n"
                f"{deploy_proc.stdout}"
            )

        health_raw = poll_health(args.health_url)
        try:
            health_json = json.loads(health_raw)
        except json.JSONDecodeError:
            health_json = {"raw": health_raw}

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "checkpoint": checkpoint,
            "health": health_json,
        }
        logs.append(record)
        print(f"[{label}] switched to {checkpoint}")
        print(json.dumps(health_json, ensure_ascii=False))

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
    print(f"Switch log saved: {args.output_json}")


if __name__ == "__main__":
    main()
