#!/usr/bin/env python3
"""Summarize repeated optimization-profile benchmark results."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a profile matrix")
    parser.add_argument("--glob", required=True, dest="result_glob")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _profile(payload: dict[str, Any]) -> str:
    answers = payload.get("answers", [])
    if not answers:
        raise ValueError("Result has no answers")
    profile = answers[0].get("meta", {}).get("optimization_profile")
    if not profile:
        raise ValueError("Result is missing optimization_profile metadata")
    return str(profile)


def _sample_signature(payload: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    return {
        str(answer["question_id"]): (
            answer.get("parsed_answer"),
            answer.get("token_count"),
        )
        for answer in payload.get("answers", [])
    }


def summarize_payloads(
    named_payloads: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int], list[tuple[str, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for name, payload in named_payloads:
        key = (
            Path(payload["dataset_path"]).name,
            _profile(payload),
            int(payload["sample_count"]),
        )
        grouped[key].append((name, payload))

    groups: list[dict[str, Any]] = []
    group_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for key in sorted(grouped):
        dataset, profile, sample_count = key
        runs = sorted(grouped[key])
        canonical = _sample_signature(runs[0][1])
        mismatched_runs = [
            name
            for name, payload in runs[1:]
            if _sample_signature(payload) != canonical
        ]
        accuracies = [
            float(payload["accuracy"]["score"])
            for _, payload in runs
        ]
        ttfts = [
            float(payload["performance"]["avg_ttft_ms"])
            for _, payload in runs
        ]
        throughputs = [
            float(
                payload["performance"]["avg_throughput_tokens_per_sec"]
            )
            for _, payload in runs
        ]
        group = {
            "dataset": dataset,
            "profile": profile,
            "sample_count": sample_count,
            "run_count": len(runs),
            "runs": [name for name, _ in runs],
            "all_validation_passed": all(
                bool(payload["public_validation"]["passed"])
                for _, payload in runs
            ),
            "stable_sample_answers_and_tokens": not mismatched_runs,
            "mismatched_runs": mismatched_runs,
            "accuracy": {
                "median": statistics.median(accuracies),
                "min": min(accuracies),
                "max": max(accuracies),
            },
            "ttft_ms": {
                "median": statistics.median(ttfts),
                "min": min(ttfts),
                "max": max(ttfts),
            },
            "throughput_tokens_per_sec": {
                "median": statistics.median(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            },
        }
        groups.append(group)
        group_index[key] = group

    comparisons: list[dict[str, Any]] = []
    dataset_shapes = sorted(
        {(dataset, sample_count) for dataset, _, sample_count in grouped}
    )
    for dataset, sample_count in dataset_shapes:
        baseline = group_index.get(
            (dataset, "o0_no_grad", sample_count)
        )
        optimized = group_index.get(
            (dataset, "o1_inference_mode", sample_count)
        )
        if baseline is None or optimized is None:
            continue
        baseline_ttft = float(baseline["ttft_ms"]["median"])
        optimized_ttft = float(optimized["ttft_ms"]["median"])
        baseline_throughput = float(
            baseline["throughput_tokens_per_sec"]["median"]
        )
        optimized_throughput = float(
            optimized["throughput_tokens_per_sec"]["median"]
        )
        comparisons.append({
            "dataset": dataset,
            "sample_count": sample_count,
            "baseline_profile": "o0_no_grad",
            "optimized_profile": "o1_inference_mode",
            "accuracy_absolute_change": (
                float(optimized["accuracy"]["median"])
                - float(baseline["accuracy"]["median"])
            ),
            "ttft_improvement_rate": (
                1.0 - optimized_ttft / baseline_ttft
                if baseline_ttft > 0
                else None
            ),
            "throughput_improvement_rate": (
                optimized_throughput / baseline_throughput - 1.0
                if baseline_throughput > 0
                else None
            ),
        })

    return {
        "groups": groups,
        "comparisons": comparisons,
    }


def main() -> None:
    args = parse_args()
    paths = sorted(Path(path).resolve() for path in glob.glob(args.result_glob))
    if not paths:
        raise ValueError(f"No results matched: {args.result_glob}")
    named_payloads = [
        (
            path.name,
            json.loads(path.read_text(encoding="utf-8")),
        )
        for path in paths
    ]
    summary = summarize_payloads(named_payloads)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
