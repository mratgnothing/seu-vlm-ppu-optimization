from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image


REQUIRED_COLUMNS = {"index", "question", "A", "B", "C", "D", "answer", "image"}


def audit_dataset(path: Path, decode_limit: int) -> dict[str, object]:
    row_count = 0
    duplicate_ids: list[str] = []
    invalid_answers: list[str] = []
    invalid_images: list[dict[str, str]] = []
    answer_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    columns: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        missing_columns = sorted(REQUIRED_COLUMNS - set(columns))
        if missing_columns:
            raise ValueError(f"{path.name}: missing columns: {missing_columns}")

        for row in reader:
            row_count += 1
            sample_id = str(row["index"])
            if sample_id in seen_ids:
                duplicate_ids.append(sample_id)
            seen_ids.add(sample_id)

            answer = (row.get("answer") or "").strip().upper()
            answer_counts[answer] += 1
            if answer not in {"A", "B", "C", "D"}:
                invalid_answers.append(sample_id)

            if row_count <= decode_limit:
                try:
                    image_bytes = base64.b64decode(row["image"], validate=True)
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.verify()
                except Exception as exc:
                    invalid_images.append(
                        {"sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"}
                    )

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "columns": columns,
        "row_count": row_count,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "answer_counts": dict(sorted(answer_counts.items())),
        "invalid_answer_ids": invalid_answers,
        "decoded_image_samples": min(row_count, decode_limit),
        "invalid_images": invalid_images,
        "passed": not duplicate_ids and not invalid_answers and not invalid_images,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit public MMBench TSV files")
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--decode-limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = [audit_dataset(path, args.decode_limit) for path in args.datasets]
    payload = {"datasets": reports, "passed": all(report["passed"] for report in reports)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

