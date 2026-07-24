#!/usr/bin/env python3
"""Validate and merge restartable benchmark chunk results."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge benchmark chunk results")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results-glob", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _profile(payload: dict[str, Any]) -> str:
    answers = payload.get("answers", [])
    if not answers:
        raise ValueError("Chunk result has no answers")
    profiles = {
        answer.get("meta", {}).get("optimization_profile")
        for answer in answers
    }
    if len(profiles) != 1 or None in profiles:
        raise ValueError(f"Chunk has inconsistent optimization profiles: {profiles}")
    return str(next(iter(profiles)))


def _timing_measurement(payload: dict[str, Any]) -> str:
    measurements = {
        answer.get("meta", {}).get("ttft_measurement")
        for answer in payload.get("answers", [])
    }
    if len(measurements) != 1 or None in measurements:
        raise ValueError(f"Chunk has inconsistent TTFT measurements: {measurements}")
    return str(next(iter(measurements)))


def _result_chunk_index(path: Path) -> int:
    marker = ".chunk-"
    name = Path(
        json.loads(path.read_text(encoding="utf-8"))["dataset_path"]
    ).name
    if marker not in name:
        raise ValueError(f"Result dataset is not a prepared chunk: {name}")
    return int(name.split(marker, 1)[1].split("-of-", 1)[0])


def merge_results(
    manifest: dict[str, Any],
    result_paths: list[Path],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    expected_chunks = {
        int(chunk["index"]): chunk
        for chunk in manifest["chunks"]
    }
    payloads_by_index: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk_index = _result_chunk_index(path)
        if chunk_index not in expected_chunks:
            raise ValueError(f"Unexpected chunk index {chunk_index}: {path}")
        if chunk_index in payloads_by_index:
            raise ValueError(f"Duplicate result for chunk {chunk_index}")
        expected_count = int(expected_chunks[chunk_index]["sample_count"])
        if int(payload["sample_count"]) != expected_count:
            raise ValueError(
                f"Chunk {chunk_index} sample count mismatch: "
                f"{payload['sample_count']} != {expected_count}"
            )
        payloads_by_index[chunk_index] = (path, payload)

    missing = sorted(set(expected_chunks) - set(payloads_by_index))
    if missing and not allow_partial:
        raise ValueError(f"Missing chunk results: {missing}")
    if not payloads_by_index:
        raise ValueError("No chunk results were provided")

    ordered = [payloads_by_index[index] for index in sorted(payloads_by_index)]
    versions = {payload["benchmark_version"] for _, payload in ordered}
    backends = {payload["backend"] for _, payload in ordered}
    seeds = {payload["seed"] for _, payload in ordered}
    profiles = {_profile(payload) for _, payload in ordered}
    measurements = {_timing_measurement(payload) for _, payload in ordered}
    for label, values in (
        ("benchmark versions", versions),
        ("backends", backends),
        ("seeds", seeds),
        ("optimization profiles", profiles),
        ("TTFT measurements", measurements),
    ):
        if len(values) != 1:
            raise ValueError(f"Inconsistent {label}: {values}")

    answers: list[dict[str, Any]] = []
    for _, payload in ordered:
        answers.extend(payload["answers"])
    question_ids = [str(answer["question_id"]) for answer in answers]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Merged results contain duplicate question IDs")

    correct = sum(bool(answer["correct"]) for answer in answers)
    validation_failures = sum(
        bool(answer.get("validation_errors"))
        for answer in answers
    )
    ttfts = [float(answer["ttft_ms"]) for answer in answers]
    throughputs = [
        float(answer["throughput_tokens_per_sec"])
        for answer in answers
    ]
    elapsed = sum(
        float(payload["timing"]["benchmark_elapsed_seconds"])
        for _, payload in ordered
    )
    completed_indices = sorted(payloads_by_index)
    complete = not missing
    return {
        "benchmark_version": next(iter(versions)),
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_path": manifest["source_path"],
        "dataset_sha256": manifest["source_sha256"],
        "sample_count": len(answers),
        "seed": next(iter(seeds)),
        "backend": next(iter(backends)),
        "optimization_profile": next(iter(profiles)),
        "ttft_measurement": next(iter(measurements)),
        "performance": {
            "avg_ttft_ms": round(statistics.fmean(ttfts), 3),
            "avg_throughput_tokens_per_sec": round(
                statistics.fmean(throughputs),
                3,
            ),
        },
        "timing": {
            "benchmark_elapsed_seconds": round(elapsed, 3),
            "benchmark_elapsed_minutes": round(elapsed / 60.0, 3),
            "avg_seconds_per_sample": round(elapsed / len(answers), 6),
        },
        "accuracy": {
            "score": correct / len(answers),
            "correct": correct,
            "total": len(answers),
        },
        "public_validation": {
            "passed": validation_failures == 0,
            "failed_samples": validation_failures,
        },
        "chunk_merge": {
            "complete": complete,
            "expected_chunk_count": len(expected_chunks),
            "completed_chunk_count": len(completed_indices),
            "completed_indices": completed_indices,
            "missing_indices": missing,
            "result_files": [path.name for path, _ in ordered],
        },
        "answers": answers,
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    paths = sorted(
        Path(path).resolve()
        for path in glob.glob(args.results_glob)
    )
    merged = merge_results(
        manifest,
        paths,
        allow_partial=args.allow_partial,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sample_count": merged["sample_count"],
                "accuracy": merged["accuracy"]["score"],
                "correct": merged["accuracy"]["correct"],
                "avg_ttft_ms": merged["performance"]["avg_ttft_ms"],
                "avg_throughput_tokens_per_sec": (
                    merged["performance"]["avg_throughput_tokens_per_sec"]
                ),
                "public_validation_passed": (
                    merged["public_validation"]["passed"]
                ),
                "complete": merged["chunk_merge"]["complete"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
