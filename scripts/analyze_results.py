from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a public benchmark result")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    total = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    valid_ttfts = [float(record["ttft_ms"]) for record in records if record["ttft_ms"]]
    valid_throughputs = [
        float(record["throughput_tokens_per_sec"])
        for record in records
        if record["throughput_tokens_per_sec"]
    ]
    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else None,
        "validation_failures": sum(bool(record["validation_errors"]) for record in records),
        "avg_ttft_ms": round(sum(valid_ttfts) / len(valid_ttfts), 3) if valid_ttfts else None,
        "avg_throughput_tokens_per_sec": (
            round(sum(valid_throughputs) / len(valid_throughputs), 3)
            if valid_throughputs
            else None
        ),
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    answers = {str(record["question_id"]): record for record in payload["answers"]}

    samples: dict[str, dict[str, str]] = {}
    with args.dataset.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            samples[str(row["index"])] = row

    joined: list[dict[str, object]] = []
    for sample_id, result in answers.items():
        sample = samples[sample_id]
        joined.append(
            {
                **result,
                "question": sample.get("question"),
                "reference_answer": sample.get("answer"),
                "category": sample.get("category") or "unknown",
                "subcategory": sample.get("l2-category") or "unknown",
            }
        )

    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_subcategory: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in joined:
        by_category[str(record["category"])].append(record)
        by_subcategory[str(record["subcategory"])].append(record)

    report = {
        "result_path": str(args.result.resolve()),
        "dataset_path": str(args.dataset.resolve()),
        "backend": payload.get("backend"),
        "overall": aggregate(joined),
        "by_category": {
            key: aggregate(value) for key, value in sorted(by_category.items())
        },
        "by_subcategory": {
            key: aggregate(value) for key, value in sorted(by_subcategory.items())
        },
        "failures": [
            {
                "question_id": record["question_id"],
                "question": record["question"],
                "reference_answer": record["reference_answer"],
                "parsed_answer": record["parsed_answer"],
                "validation_errors": record["validation_errors"],
                "category": record["category"],
                "subcategory": record["subcategory"],
            }
            for record in joined
            if not record["correct"] or record["validation_errors"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print(f"Failure records: {len(report['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

