#!/usr/bin/env python3
"""Create a compact, auditable summary from a paired A/B benchmark JSON."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    target = str(payload["ab_target"])
    baseline_records = payload["baseline"]["records"]
    candidate_records = payload[target]["records"]
    if len(baseline_records) != len(candidate_records):
        raise ValueError("baseline/candidate record count mismatch")
    ratios = [
        float(candidate["throughput_tokens_per_sec"])
        / float(baseline["throughput_tokens_per_sec"])
        for baseline, candidate in zip(baseline_records, candidate_records, strict=True)
    ]
    exact_pairs = sum(
        baseline["text_sha256"] == candidate["text_sha256"]
        for baseline, candidate in zip(baseline_records, candidate_records, strict=True)
    )
    answer_pairs = sum(
        baseline["answer"] == candidate["answer"]
        for baseline, candidate in zip(baseline_records, candidate_records, strict=True)
    )
    token_count_pairs = sum(
        baseline["token_count"] == candidate["token_count"]
        for baseline, candidate in zip(baseline_records, candidate_records, strict=True)
    )

    def compact_mode(name: str, records: list[dict[str, object]]) -> dict[str, object]:
        return {
            "avg_ttft_ms": float(payload[name]["avg_ttft_ms"]),
            "avg_throughput_tokens_per_sec": float(
                payload[name]["avg_throughput_tokens_per_sec"]
            ),
            "accuracy": float(payload[name]["accuracy"]),
            "correct": sum(bool(record["correct"]) for record in records),
        }

    summary = {
        "source": str(args.input),
        "sample_offset": int(payload["sample_offset"]),
        "sample_count": int(payload["sample_count"]),
        "max_new_tokens": int(payload["max_new_tokens"]),
        "ab_target": target,
        "projection_backend": payload["projection_backend"],
        "module_counts": {
            "packed_gdn": int(payload["packed_gdn_modules"]),
            "residual_rmsnorm": int(payload["residual_rmsnorm_modules"]),
            "gdn_gate_prep": int(payload["gdn_gate_prep_modules"]),
            **(
                {
                    "acblas_packed_mlp": int(
                        payload["acblas_packed_mlp_modules"]
                    )
                }
                if "acblas_packed_mlp_modules" in payload
                else {}
            ),
        },
        "baseline": compact_mode("baseline", baseline_records),
        target: compact_mode(target, candidate_records),
        "paired_decode": {
            "mean_speedup": statistics.fmean(ratios),
            "median_speedup": statistics.median(ratios),
            "p05_speedup": percentile(ratios, 0.05),
            "p25_speedup": percentile(ratios, 0.25),
            "p75_speedup": percentile(ratios, 0.75),
            "p95_speedup": percentile(ratios, 0.95),
            "min_speedup": min(ratios),
            "max_speedup": max(ratios),
            "wins": sum(value > 1.0 for value in ratios),
            "ties": sum(value == 1.0 for value in ratios),
            "losses": sum(value < 1.0 for value in ratios),
        },
        "pair_consistency": {
            "exact_text": exact_pairs,
            "same_answer": answer_pairs,
            "same_token_count": token_count_pairs,
            "total": len(ratios),
        },
        "passed": bool(payload["passed"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
