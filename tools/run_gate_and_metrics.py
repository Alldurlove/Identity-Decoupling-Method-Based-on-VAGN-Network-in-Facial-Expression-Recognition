import argparse
import json
import os
import subprocess
import sys
from typing import Dict


def run_cmd(command: str) -> None:
    print(f"[RUN] {command}")
    subprocess.run(command, shell=True, check=True)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run quality gate + expression metric evaluation for a candidate checkpoint."
    )
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--candidate-ckpt", required=True)
    parser.add_argument("--image-root", required=True, help="Validation image root used by quality gate.")
    parser.add_argument("--quality-output-json", default="quality_gate_report.json")
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--min-sharpness-ratio", type=float, default=0.85)
    parser.add_argument("--min-contrast-ratio", type=float, default=0.85)

    parser.add_argument("--metadata-csv", default="")
    parser.add_argument("--dataroot", default="", help="Required if metadata-csv is empty.")
    parser.add_argument("--anonymized-output-root", default="attack_data/anonymized_eval")
    parser.add_argument("--expr-output-dir", default="attack_results/expression_eval")
    parser.add_argument("--expr-epochs", type=int, default=20)
    parser.add_argument("--summary-json", default="attack_results/gate_metrics_summary.json")
    args = parser.parse_args()

    run_cmd(
        f'{args.python_bin} tools/predeploy_quality_gate.py '
        f'--baseline-ckpt "{args.baseline_ckpt}" '
        f'--candidate-ckpt "{args.candidate_ckpt}" '
        f'--image-root "{args.image_root}" '
        f'--target-id {args.target_id} '
        f'--max-samples {args.max_samples} '
        f'--min-sharpness-ratio {args.min_sharpness_ratio} '
        f'--min-contrast-ratio {args.min_contrast_ratio} '
        f'--output-json "{args.quality_output_json}"'
    )

    metadata_csv = args.metadata_csv
    if not metadata_csv:
        if not args.dataroot:
            raise ValueError("Provide either --metadata-csv or --dataroot.")
        run_cmd(
            f'{args.python_bin} attacks/generate_anonymized_dataset.py '
            f'--dataroot "{args.dataroot}" '
            f'--checkpoint "{args.candidate_ckpt}" '
            f'--output-root "{args.anonymized_output_root}"'
        )
        metadata_csv = os.path.join(args.anonymized_output_root, "metadata.csv")

    run_cmd(
        f'{args.python_bin} attacks/train_expression_attacker.py '
        f'--metadata-csv "{metadata_csv}" '
        f'--image-column anonymized_path '
        f'--output-dir "{args.expr_output_dir}" '
        f'--epochs {args.expr_epochs}'
    )

    quality = load_json(args.quality_output_json)
    expression = load_json(os.path.join(args.expr_output_dir, "expression_report.json"))

    summary = {
        "baseline_ckpt": args.baseline_ckpt,
        "candidate_ckpt": args.candidate_ckpt,
        "quality_gate": quality,
        "expression_metrics": expression.get("test_metrics", {}),
        "recommendation": "deploy" if quality.get("pass", False) else "reject",
        "artifacts": {
            "quality_output_json": args.quality_output_json,
            "expression_report_json": os.path.join(args.expr_output_dir, "expression_report.json"),
            "metadata_csv": metadata_csv,
        },
    }
    os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
    with open(args.summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {args.summary_json}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
