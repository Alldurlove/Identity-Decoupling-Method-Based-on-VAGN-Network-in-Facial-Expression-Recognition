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


def evaluate_single_checkpoint(
    tag: str,
    checkpoint: str,
    dataroot: str,
    output_root: str,
    python_bin: str,
    source_epochs: int,
    expr_epochs: int,
) -> Dict[str, str]:
    run_dir = os.path.join(output_root, tag)
    data_dir = os.path.join(run_dir, "anonymized")
    source_dir = os.path.join(run_dir, "source_id")
    expr_dir = os.path.join(run_dir, "expression")
    link_path = os.path.join(run_dir, "linkability_report.json")
    os.makedirs(run_dir, exist_ok=True)

    run_cmd(
        f'{python_bin} attacks/generate_anonymized_dataset.py '
        f'--dataroot "{dataroot}" --checkpoint "{checkpoint}" --output-root "{data_dir}"'
    )
    metadata_csv = os.path.join(data_dir, "metadata.csv")

    run_cmd(
        f'{python_bin} attacks/train_source_id_attacker.py '
        f'--metadata-csv "{metadata_csv}" --image-column anonymized_path '
        f'--output-dir "{source_dir}" --epochs {source_epochs}'
    )
    attacker_ckpt = os.path.join(source_dir, "best_source_id_attacker.pth")

    run_cmd(
        f'{python_bin} attacks/eval_linkability.py '
        f'--metadata-csv "{metadata_csv}" --attacker-ckpt "{attacker_ckpt}" '
        f'--image-column anonymized_path --output-json "{link_path}"'
    )

    run_cmd(
        f'{python_bin} attacks/train_expression_attacker.py '
        f'--metadata-csv "{metadata_csv}" --image-column anonymized_path '
        f'--output-dir "{expr_dir}" --epochs {expr_epochs}'
    )

    return {
        "metadata_csv": metadata_csv,
        "source_id_report": os.path.join(source_dir, "source_id_attack_report.json"),
        "linkability_report": link_path,
        "expression_report": os.path.join(expr_dir, "expression_report.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run privacy + expression comparison for baseline and finetuned checkpoints."
    )
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--finetuned-ckpt", required=True)
    parser.add_argument("--output-root", default="attack_results/checkpoint_comparison")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--source-epochs", type=int, default=20)
    parser.add_argument("--expr-epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    baseline_paths = evaluate_single_checkpoint(
        tag="baseline",
        checkpoint=args.baseline_ckpt,
        dataroot=args.dataroot,
        output_root=args.output_root,
        python_bin=args.python_bin,
        source_epochs=args.source_epochs,
        expr_epochs=args.expr_epochs,
    )
    finetuned_paths = evaluate_single_checkpoint(
        tag="finetuned",
        checkpoint=args.finetuned_ckpt,
        dataroot=args.dataroot,
        output_root=args.output_root,
        python_bin=args.python_bin,
        source_epochs=args.source_epochs,
        expr_epochs=args.expr_epochs,
    )

    baseline_source = load_json(baseline_paths["source_id_report"])
    finetuned_source = load_json(finetuned_paths["source_id_report"])
    baseline_link = load_json(baseline_paths["linkability_report"])
    finetuned_link = load_json(finetuned_paths["linkability_report"])
    baseline_expr = load_json(baseline_paths["expression_report"])
    finetuned_expr = load_json(finetuned_paths["expression_report"])

    summary = {
        "baseline_checkpoint": args.baseline_ckpt,
        "finetuned_checkpoint": args.finetuned_ckpt,
        "baseline": {
            "source_id_test": baseline_source.get("test_metrics", {}),
            "linkability": baseline_link,
            "expression_test": baseline_expr.get("test_metrics", {}),
        },
        "finetuned": {
            "source_id_test": finetuned_source.get("test_metrics", {}),
            "linkability": finetuned_link,
            "expression_test": finetuned_expr.get("test_metrics", {}),
        },
    }

    summary_path = os.path.join(args.output_root, "comparison_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Comparison summary saved to {summary_path}")


if __name__ == "__main__":
    main()
