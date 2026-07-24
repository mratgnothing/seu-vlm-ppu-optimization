from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "file",
    "timestamp",
    "backend",
    "dataset",
    "sample_count",
    "accuracy",
    "avg_ttft_ms",
    "avg_throughput_tokens_per_sec",
    "validation_passed",
    "benchmark_elapsed_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSON files")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "file": path.name,
                "timestamp": payload.get("timestamp"),
                "backend": payload.get("backend"),
                "dataset": Path(payload.get("dataset_path", "")).name,
                "sample_count": payload.get("sample_count"),
                "accuracy": payload.get("accuracy", {}).get("score"),
                "avg_ttft_ms": payload.get("performance", {}).get("avg_ttft_ms"),
                "avg_throughput_tokens_per_sec": payload.get("performance", {}).get(
                    "avg_throughput_tokens_per_sec"
                ),
                "validation_passed": payload.get("public_validation", {}).get("passed"),
                "benchmark_elapsed_seconds": payload.get("timing", {}).get(
                    "benchmark_elapsed_seconds"
                ),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

