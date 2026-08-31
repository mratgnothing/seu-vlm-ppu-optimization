#!/usr/bin/env python3
"""Paired A/B for the reusable first-token cache on the complete PPU stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    custom_ops = repo_root / "ppu" / "custom_ops"
    for path in (repo_root, custom_ops):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from benchmark_public import (
        build_prompt,
        compute_throughput,
        decode_image,
        extract_answer,
        fixed_generation_config,
        load_mmbench_tsv,
        settle_runtime,
    )
    from evaluation_wrapper import VLMModel

    samples = load_mmbench_tsv(
        args.dataset_path, limit=args.sample_offset + args.num_samples
    )[args.sample_offset :]
    if len(samples) != args.num_samples:
        raise ValueError(f"requested {args.num_samples} samples, got {len(samples)}")

    # The environment enables construction of the pool at load time.  Each arm
    # then toggles only whether that same model uses it for the request.
    import os

    os.environ["SEU_PPU_FIRST_TOKEN_CACHE_ENABLE"] = "1"
    os.environ["SEU_PPU_FIRST_TOKEN_CACHE_CAPACITY"] = str(args.cache_capacity)
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens

    def run_sample(sample, enabled: bool, pair_index: int) -> dict[str, object]:
        model._ppu_first_token_cache_enabled = enabled
        settle_runtime(model)
        result = model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        answer = extract_answer(result.text)
        return {
            "sample_id": sample.sample_id,
            "pair_index": pair_index,
            "pair_order": "AB" if pair_index % 2 == 0 else "BA",
            "mode": "reserved_cache" if enabled else "dynamic_cache",
            "token_count": result.token_count,
            "ttft_ms": result.ttft_seconds * 1000.0,
            "elapsed_ms": result.elapsed_seconds * 1000.0,
            "throughput_tokens_per_sec": compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            ),
            "answer": answer,
            "reference": sample.answer,
            "correct": answer == sample.answer,
            "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        }

    # Warm both paths.  In particular this allocates the candidate pool before
    # any measured pair, matching benchmark_public's model warmup behavior.
    run_sample(samples[0], False, -1)
    run_sample(samples[0], True, -1)

    baseline_records = []
    candidate_records = []
    ttft_speedups = []
    throughput_speedups = []
    exact_pairs = 0
    for index, sample in enumerate(samples):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair = {}
        for enabled in order:
            pair[enabled] = run_sample(sample, enabled, index)
        baseline, candidate = pair[False], pair[True]
        baseline_records.append(baseline)
        candidate_records.append(candidate)
        ttft_speedups.append(
            float(baseline["ttft_ms"]) / float(candidate["ttft_ms"])
        )
        throughput_speedups.append(
            float(candidate["throughput_tokens_per_sec"])
            / float(baseline["throughput_tokens_per_sec"])
        )
        exact_pairs += baseline["text_sha256"] == candidate["text_sha256"]
        if (index + 1) % args.progress_every == 0 or index + 1 == len(samples):
            print(
                json.dumps(
                    {
                        "progress": index + 1,
                        "total": len(samples),
                        "exact_pairs": exact_pairs,
                        "median_ttft_speedup": statistics.median(ttft_speedups),
                        "median_throughput_speedup": statistics.median(
                            throughput_speedups
                        ),
                    }
                ),
                flush=True,
            )

    def mean(records, key: str) -> float:
        return statistics.fmean(float(record[key]) for record in records)

    exact = exact_pairs == len(samples)
    accuracy_equal = sum(bool(x["correct"]) for x in baseline_records) == sum(
        bool(x["correct"]) for x in candidate_records
    )
    ttft_passed = (
        statistics.median(ttft_speedups) > 1.0
        and statistics.fmean(ttft_speedups) > 1.0
    )
    throughput_passed = (
        statistics.median(throughput_speedups) >= 1.0
        and statistics.fmean(throughput_speedups) >= 1.0
    )
    payload = {
        "sample_offset": args.sample_offset,
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "cache_capacity": args.cache_capacity,
        "baseline": {
            "mode": "dynamic_cache",
            "avg_ttft_ms": mean(baseline_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean(
                baseline_records, "throughput_tokens_per_sec"
            ),
            "correct": sum(bool(x["correct"]) for x in baseline_records),
        },
        "candidate": {
            "mode": "reserved_cache",
            "avg_ttft_ms": mean(candidate_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean(
                candidate_records, "throughput_tokens_per_sec"
            ),
            "correct": sum(bool(x["correct"]) for x in candidate_records),
        },
        "paired": {
            "median_ttft_speedup": statistics.median(ttft_speedups),
            "mean_ttft_speedup": statistics.fmean(ttft_speedups),
            "ttft_wins": sum(value > 1.0 for value in ttft_speedups),
            "median_throughput_speedup": statistics.median(throughput_speedups),
            "mean_throughput_speedup": statistics.fmean(throughput_speedups),
            "throughput_wins": sum(value > 1.0 for value in throughput_speedups),
            "exact_text_pairs": exact_pairs,
        },
        "gate": {
            "exact_text": exact,
            "accuracy_equal": accuracy_equal,
            "ttft_improved": ttft_passed,
            "throughput_non_regression": throughput_passed,
            "passed": exact and accuracy_equal and ttft_passed and throughput_passed,
        },
        "records": {
            "baseline": baseline_records,
            "candidate": candidate_records,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
