#!/usr/bin/env python3
"""Compare public benchmark outputs sample-by-sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_answers(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, payload["answers"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    reference, ref_answers = load_answers(args.reference)
    candidate, cand_answers = load_answers(args.candidate)
    if len(ref_answers) != len(cand_answers):
        raise ValueError("sample count differs")

    fields = ("parsed_answer", "correct", "token_count")
    mismatch_counts = {
        field: sum(a[field] != b[field] for a, b in zip(ref_answers, cand_answers))
        for field in fields
    }
    ids_match = all(
        a["question_id"] == b["question_id"]
        for a, b in zip(ref_answers, cand_answers)
    )
    summary = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "sample_count": len(ref_answers),
        "question_ids_match": ids_match,
        "mismatch_counts": mismatch_counts,
        "reference_performance": reference["performance"],
        "candidate_performance": candidate["performance"],
        "reference_accuracy": reference["accuracy"],
        "candidate_accuracy": candidate["accuracy"],
        "candidate_first_meta": cand_answers[0]["meta"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if ids_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
