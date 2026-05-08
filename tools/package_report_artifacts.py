import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List


DEFAULT_CANDIDATES = [
    "loss_curve.png",
    "metric_curve.png",
    "train_history.json",
    "train_summary.json",
    "quality_gate_report.json",
    "quality_gate_report_recover.json",
    "quality_gate_report_recover_v2.json",
    "quality_gate_report_real_from_scratch.json",
    "expression_report.json",
    "source_id_attack_report.json",
    "linkability_report.json",
]


def collect_files(search_roots: List[str], explicit: List[str]) -> List[str]:
    found: List[str] = []
    for p in explicit:
        if os.path.isfile(p):
            found.append(os.path.abspath(p))

    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name in DEFAULT_CANDIDATES:
                    found.append(os.path.abspath(os.path.join(dirpath, name)))

    # Keep stable order and dedupe.
    deduped = sorted(set(found))
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package experiment artifacts into a report-ready folder."
    )
    parser.add_argument("--output-dir", default="report_pack")
    parser.add_argument(
        "--search-root",
        action="append",
        default=["checkpoints", "attack_results", "."],
        help="Directory to search for known artifact names (repeatable).",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Explicit file path to include (repeatable).",
    )
    parser.add_argument("--summary-md", default="report_pack/REPORT_SUMMARY.md")
    args = parser.parse_args()

    files = collect_files(args.search_root, args.file)
    os.makedirs(args.output_dir, exist_ok=True)

    copied: List[Dict[str, str]] = []
    for src in files:
        rel_name = os.path.basename(src)
        dst = os.path.join(args.output_dir, rel_name)
        stem, ext = os.path.splitext(rel_name)
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(args.output_dir, f"{stem}_{i}{ext}")
            i += 1
        shutil.copy2(src, dst)
        copied.append({"source": src, "packaged": os.path.abspath(dst)})

    manifest_path = os.path.join(args.output_dir, "artifact_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "output_dir": os.path.abspath(args.output_dir),
                "artifact_count": len(copied),
                "artifacts": copied,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_lines = [
        "# Report Artifact Summary",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Artifact count: {len(copied)}",
        f"- Manifest: `{manifest_path}`",
        "",
        "## Included Files",
    ]
    if copied:
        summary_lines.extend([f"- `{item['packaged']}`" for item in copied])
    else:
        summary_lines.append("- No artifacts found. Add explicit files via `--file`.")

    os.makedirs(os.path.dirname(args.summary_md), exist_ok=True)
    with open(args.summary_md, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    print(f"Packaged {len(copied)} artifacts into {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {args.summary_md}")


if __name__ == "__main__":
    main()
