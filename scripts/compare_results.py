#!/usr/bin/env python3
"""Compare two benchmark result files with sample-level consistency checks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized results")
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--optimized", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _answer_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["question_id"]): record
        for record in payload.get("answers", [])
    }


def compare_payloads(
    baseline: dict[str, Any],
    optimized: dict[str, Any],
) -> dict[str, Any]:
    baseline_answers = _answer_map(baseline)
    optimized_answers = _answer_map(optimized)
    baseline_ids = set(baseline_answers)
    optimized_ids = set(optimized_answers)
    common_ids = sorted(baseline_ids & optimized_ids)

    baseline_ttft = float(baseline["performance"]["avg_ttft_ms"])
    optimized_ttft = float(optimized["performance"]["avg_ttft_ms"])
    baseline_throughput = float(
        baseline["performance"]["avg_throughput_tokens_per_sec"]
    )
    optimized_throughput = float(
        optimized["performance"]["avg_throughput_tokens_per_sec"]
    )
    baseline_accuracy = float(baseline["accuracy"]["score"])
    optimized_accuracy = float(optimized["accuracy"]["score"])

    ttft_ratios = [
        float(optimized_answers[sample_id]["ttft_ms"])
        / float(baseline_answers[sample_id]["ttft_ms"])
        for sample_id in common_ids
        if float(baseline_answers[sample_id]["ttft_ms"]) > 0
    ]
    throughput_ratios = [
        float(optimized_answers[sample_id]["throughput_tokens_per_sec"])
        / float(baseline_answers[sample_id]["throughput_tokens_per_sec"])
        for sample_id in common_ids
        if float(baseline_answers[sample_id]["throughput_tokens_per_sec"]) > 0
    ]
    changed_answers = [
        {
            "question_id": sample_id,
            "baseline": baseline_answers[sample_id].get("parsed_answer"),
            "optimized": optimized_answers[sample_id].get("parsed_answer"),
        }
        for sample_id in common_ids
        if baseline_answers[sample_id].get("parsed_answer")
        != optimized_answers[sample_id].get("parsed_answer")
    ]
    token_count_changes = [
        {
            "question_id": sample_id,
            "baseline": baseline_answers[sample_id].get("token_count"),
            "optimized": optimized_answers[sample_id].get("token_count"),
        }
        for sample_id in common_ids
        if baseline_answers[sample_id].get("token_count")
        != optimized_answers[sample_id].get("token_count")
    ]

    return {
        "sample_contract": {
            "baseline_count": len(baseline_ids),
            "optimized_count": len(optimized_ids),
            "same_question_ids": baseline_ids == optimized_ids,
            "missing_from_optimized": sorted(baseline_ids - optimized_ids),
            "new_in_optimized": sorted(optimized_ids - baseline_ids),
            "changed_parsed_answers": changed_answers,
            "token_count_changes": token_count_changes,
        },
        "accuracy": {
            "baseline": baseline_accuracy,
            "optimized": optimized_accuracy,
            "absolute_change": optimized_accuracy - baseline_accuracy,
        },
        "ttft_ms": {
            "baseline_mean": baseline_ttft,
            "optimized_mean": optimized_ttft,
            "improvement_rate": (
                1.0 - optimized_ttft / baseline_ttft
                if baseline_ttft > 0
                else None
            ),
            "paired_median_improvement_rate": (
                1.0 - statistics.median(ttft_ratios)
                if ttft_ratios
                else None
            ),
        },
        "throughput_tokens_per_sec": {
            "baseline_mean": baseline_throughput,
            "optimized_mean": optimized_throughput,
            "improvement_rate": (
                optimized_throughput / baseline_throughput - 1.0
                if baseline_throughput > 0
                else None
            ),
            "paired_median_improvement_rate": (
                statistics.median(throughput_ratios) - 1.0
                if throughput_ratios
                else None
            ),
        },
        "validation": {
            "baseline_passed": bool(
                baseline.get("public_validation", {}).get("passed")
            ),
            "optimized_passed": bool(
                optimized.get("public_validation", {}).get("passed")
            ),
        },
    }


def main() -> None:
    args = parse_args()
    comparison = compare_payloads(_load(args.baseline), _load(args.optimized))
    rendered = json.dumps(comparison, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
