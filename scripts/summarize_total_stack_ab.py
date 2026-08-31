#!/usr/bin/env python3
"""Summarize repeated eager/current public-benchmark runs.

This script is intentionally separate from incremental paired A/B summaries:
it compares independently loaded original eager and complete optimized stacks,
checks that every run covers the same public samples, and makes the weaker
cross-process consistency boundary explicit (the public result does not store
full generated text hashes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", type=Path, action="append", required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_run(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("backend") != "transformers":
        raise ValueError(f"{path}: expected transformers backend")
    answers = payload.get("answers")
    if not isinstance(answers, list) or len(answers) != payload.get("sample_count"):
        raise ValueError(f"{path}: answer count does not match sample_count")
    if not payload.get("public_validation", {}).get("passed"):
        raise ValueError(f"{path}: public validation did not pass")
    payload["_source"] = str(path)
    payload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def answer_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {str(record["question_id"]): record for record in run["answers"]}
    if len(records) != len(run["answers"]):
        raise ValueError(f"{run['_source']}: duplicate question_id")
    return records


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def summarize(eager_paths: list[Path], candidate_paths: list[Path]) -> dict[str, Any]:
    if len(eager_paths) != len(candidate_paths):
        raise ValueError("eager and candidate run counts must match")
    if len(eager_paths) < 2:
        raise ValueError("at least two runs per arm are required")

    eager_runs = [load_run(path) for path in eager_paths]
    candidate_runs = [load_run(path) for path in candidate_paths]
    all_runs = eager_runs + candidate_runs
    maps = [answer_map(run) for run in all_runs]
    sample_ids = list(maps[0])
    for run, records in zip(all_runs[1:], maps[1:]):
        if list(records) != sample_ids:
            raise ValueError(f"{run['_source']}: sample IDs/order differ")

    def run_metrics(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": run["_source"],
            "sha256": run["_sha256"],
            "avg_ttft_ms": float(run["performance"]["avg_ttft_ms"]),
            "avg_throughput_tokens_per_sec": float(
                run["performance"]["avg_throughput_tokens_per_sec"]
            ),
            "accuracy": float(run["accuracy"]["score"]),
            "correct": int(run["accuracy"]["correct"]),
            "public_validation_passed": True,
        }

    eager_tps = [
        float(run["performance"]["avg_throughput_tokens_per_sec"])
        for run in eager_runs
    ]
    candidate_tps = [
        float(run["performance"]["avg_throughput_tokens_per_sec"])
        for run in candidate_runs
    ]
    eager_ttft = [float(run["performance"]["avg_ttft_ms"]) for run in eager_runs]
    candidate_ttft = [
        float(run["performance"]["avg_ttft_ms"]) for run in candidate_runs
    ]
    eager_median_tps = statistics.median(eager_tps)
    candidate_median_tps = statistics.median(candidate_tps)
    eager_median_ttft = statistics.median(eager_ttft)
    candidate_median_ttft = statistics.median(candidate_ttft)

    per_sample_ratios: list[float] = []
    same_answer = 0
    same_correctness = 0
    same_token_count = 0
    for sample_id in sample_ids:
        eager_records = [answer_map(run)[sample_id] for run in eager_runs]
        candidate_records = [answer_map(run)[sample_id] for run in candidate_runs]
        eager_sample_tps = statistics.median(
            float(record["throughput_tokens_per_sec"]) for record in eager_records
        )
        candidate_sample_tps = statistics.median(
            float(record["throughput_tokens_per_sec"])
            for record in candidate_records
        )
        per_sample_ratios.append(candidate_sample_tps / eager_sample_tps)
        answers = {record.get("parsed_answer") for record in eager_records + candidate_records}
        correctness = {bool(record.get("correct")) for record in eager_records + candidate_records}
        token_counts = {int(record["token_count"]) for record in eager_records + candidate_records}
        same_answer += len(answers) == 1
        same_correctness += len(correctness) == 1
        same_token_count += len(token_counts) == 1

    accuracies = [float(run["accuracy"]["score"]) for run in all_runs]
    module_signatures = []
    for run in candidate_runs:
        signatures = {
            json.dumps(
                {
                    key: value
                    for key, value in record.get("meta", {}).items()
                    if key.startswith("ppu_")
                },
                sort_keys=True,
            )
            for record in run["answers"]
        }
        if len(signatures) != 1:
            raise ValueError(f"{run['_source']}: inconsistent PPU module metadata")
        module_signatures.append(json.loads(signatures.pop()))

    payload = {
        "comparison": "original_eager_vs_complete_optimized_stack",
        "method": {
            "process_order": "eager_A, candidate_A, candidate_B, eager_B",
            "runs_per_arm": len(eager_runs),
            "sample_count": len(sample_ids),
            "metric": "official decode throughput: (tokens - 1) / (elapsed - TTFT)",
            "full_text_hash_available": False,
        },
        "eager": {
            "runs": [run_metrics(run) for run in eager_runs],
            "median_avg_ttft_ms": eager_median_ttft,
            "median_avg_throughput_tokens_per_sec": eager_median_tps,
        },
        "candidate": {
            "runs": [run_metrics(run) for run in candidate_runs],
            "median_avg_ttft_ms": candidate_median_ttft,
            "median_avg_throughput_tokens_per_sec": candidate_median_tps,
            "module_metadata": module_signatures,
        },
        "aggregate": {
            "throughput_speedup": candidate_median_tps / eager_median_tps,
            "throughput_improvement_percent": (
                candidate_median_tps / eager_median_tps - 1.0
            )
            * 100.0,
            "ttft_reduction_percent": (
                1.0 - candidate_median_ttft / eager_median_ttft
            )
            * 100.0,
            "run_paired_speedups": [
                candidate / eager
                for eager, candidate in zip(eager_tps, candidate_tps)
            ],
            "per_sample_median_speedup": statistics.median(per_sample_ratios),
            "per_sample_mean_speedup": mean(per_sample_ratios),
            "per_sample_wins": sum(ratio > 1.0 for ratio in per_sample_ratios),
            "per_sample_ratios": per_sample_ratios,
        },
        "consistency": {
            "accuracy_values": accuracies,
            "same_accuracy_all_runs": len(set(accuracies)) == 1,
            "same_parsed_answer_all_runs": same_answer,
            "same_correctness_all_runs": same_correctness,
            "same_token_count_all_runs": same_token_count,
            "total": len(sample_ids),
            "strict_full_text_comparison": "unavailable_in_benchmark_public_output",
        },
    }
    payload["passed"] = (
        payload["aggregate"]["throughput_speedup"] > 1.0
        and payload["consistency"]["same_accuracy_all_runs"]
        and same_answer == len(sample_ids)
        and same_correctness == len(sample_ids)
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = summarize(args.eager, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
