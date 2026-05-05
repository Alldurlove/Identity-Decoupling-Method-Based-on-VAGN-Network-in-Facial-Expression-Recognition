import argparse
import base64
import json
import time
from io import BytesIO
from typing import Dict, List

import requests
from PIL import Image


def make_data_uri(gray: int = 120, width: int = 320, height: int = 240) -> str:
    image = Image.new("RGB", (width, height), (gray, gray, gray))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=70)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def infer_once(base_url: str, data_uri: str, target_id: int) -> Dict:
    response = requests.post(
        f"{base_url}/api/infer",
        json={"image_base64": data_uri, "target_id": target_id},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-deploy runtime smoke test for web_app.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--output-json", default="runtime_verify_report.json")
    args = parser.parse_args()

    health = requests.get(f"{args.base_url}/api/health", timeout=10)
    health.raise_for_status()
    health_data = health.json()

    data_uri = make_data_uri()
    latencies_ms: List[float] = []
    output_sizes: List[int] = []
    modes: List[str] = []
    for _ in range(args.rounds):
        start = time.time()
        result = infer_once(args.base_url, data_uri, args.target_id)
        latencies_ms.append((time.time() - start) * 1000.0)
        modes.append(result.get("mode", "unknown"))
        output_sizes.append(len(result.get("image_base64", "")))

    report = {
        "health": health_data,
        "infer_rounds": args.rounds,
        "latency_ms": {
            "avg": sum(latencies_ms) / max(1, len(latencies_ms)),
            "min": min(latencies_ms) if latencies_ms else None,
            "max": max(latencies_ms) if latencies_ms else None,
        },
        "modes": sorted(set(modes)),
        "output_size": {
            "avg": sum(output_sizes) / max(1, len(output_sizes)),
            "min": min(output_sizes) if output_sizes else None,
            "max": max(output_sizes) if output_sizes else None,
        },
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, ensure_ascii=False))
    print(f"Runtime verification report saved to {args.output_json}")


if __name__ == "__main__":
    main()
