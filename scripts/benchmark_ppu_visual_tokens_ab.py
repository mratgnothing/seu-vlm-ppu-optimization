#!/usr/bin/env python3
"""Paired PPU A/B for a Qwen3.5 visual pixel/token budget.

The model and all accelerator extensions are shared between arms.  Only the
processor that creates image patches changes, avoiding model-load bias while
making the changed visual-token count explicit in every record.
"""

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
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument(
        "--require-answer-equality",
        action="store_true",
        help="Use a conservative gate that rejects any parsed-answer change",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_pixels <= 0:
        raise ValueError("--max-pixels must be positive")
    repo_root = args.repo_root.resolve()
    custom_ops = repo_root / "ppu" / "custom_ops"
    # Insert custom_ops first so the repository root ends up at sys.path[0].
    # A packaged/custom_ops directory may contain a stale wrapper copied by an
    # earlier deployment; the benchmark must always exercise this checkout's
    # evaluation_wrapper.py.
    for path in (custom_ops, repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from transformers import AutoProcessor

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

    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    baseline_processor = model._processor
    candidate_processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        max_pixels=args.max_pixels,
    )
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens

    def run_sample(sample, enabled: bool, pair_index: int) -> dict[str, object]:
        model._processor = candidate_processor if enabled else baseline_processor
        model._tokenizer = getattr(model._processor, "tokenizer", None)
        model._vision_max_pixels = args.max_pixels if enabled else None
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
            "mode": f"max_pixels_{args.max_pixels}" if enabled else "baseline",
            "prompt_tokens": result.meta.get("prompt_tokens"),
            "visual_tokens": result.meta.get("visual_tokens"),
            "image_grid_thw": result.meta.get("image_grid_thw"),
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

    # Warm both processor shapes before measured pairs.  If this changes the
    # model answer already, the subsequent gate still records and rejects it.
    run_sample(samples[0], False, -1)
    run_sample(samples[0], True, -1)

    baseline_records = []
    candidate_records = []
    ttft_speedups = []
    throughput_speedups = []
    visual_ratios = []
    changed_visual_pairs = 0
    exact_text_pairs = 0
    equal_answer_pairs = 0
    for index, sample in enumerate(samples):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair = {enabled: run_sample(sample, enabled, index) for enabled in order}
        baseline, candidate = pair[False], pair[True]
        baseline_records.append(baseline)
        candidate_records.append(candidate)
        ttft_speedups.append(float(baseline["ttft_ms"]) / float(candidate["ttft_ms"]))
        throughput_speedups.append(
            float(candidate["throughput_tokens_per_sec"])
            / float(baseline["throughput_tokens_per_sec"])
        )
        visual_ratios.append(
            float(candidate["visual_tokens"]) / float(baseline["visual_tokens"])
        )
        changed_visual_pairs += candidate["visual_tokens"] != baseline["visual_tokens"]
        exact_text_pairs += baseline["text_sha256"] == candidate["text_sha256"]
        equal_answer_pairs += baseline["answer"] == candidate["answer"]
        if (index + 1) % args.progress_every == 0 or index + 1 == len(samples):
            print(
                json.dumps(
                    {
                        "progress": index + 1,
                        "total": len(samples),
                        "equal_answers": equal_answer_pairs,
                        "median_visual_ratio": statistics.median(visual_ratios),
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

    baseline_correct = sum(bool(record["correct"]) for record in baseline_records)
    candidate_correct = sum(bool(record["correct"]) for record in candidate_records)
    accuracy_passed = candidate_correct >= baseline_correct
    answer_gate_passed = (
        equal_answer_pairs == len(samples) if args.require_answer_equality else True
    )
    ttft_passed = (
        statistics.median(ttft_speedups) > 1.0
        and statistics.fmean(ttft_speedups) > 1.0
    )
    throughput_passed = (
        statistics.median(throughput_speedups) >= 1.0
        and statistics.fmean(throughput_speedups) >= 1.0
    )
    # A mild cap intentionally changes only oversized images, so its median
    # ratio can remain 1.0 even when it performs a real reduction.  Require at
    # least one changed pair and a lower arithmetic mean instead.
    reduction_passed = (
        changed_visual_pairs > 0 and statistics.fmean(visual_ratios) < 1.0
    )
    payload = {
        "sample_offset": args.sample_offset,
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "candidate_max_pixels": args.max_pixels,
        "require_answer_equality": args.require_answer_equality,
        "baseline": {
            "avg_visual_tokens": mean(baseline_records, "visual_tokens"),
            "avg_prompt_tokens": mean(baseline_records, "prompt_tokens"),
            "avg_ttft_ms": mean(baseline_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean(
                baseline_records, "throughput_tokens_per_sec"
            ),
            "correct": baseline_correct,
        },
        "candidate": {
            "avg_visual_tokens": mean(candidate_records, "visual_tokens"),
            "avg_prompt_tokens": mean(candidate_records, "prompt_tokens"),
            "avg_ttft_ms": mean(candidate_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean(
                candidate_records, "throughput_tokens_per_sec"
            ),
            "correct": candidate_correct,
        },
        "paired": {
            "median_visual_token_ratio": statistics.median(visual_ratios),
            "mean_visual_token_ratio": statistics.fmean(visual_ratios),
            "changed_visual_pairs": changed_visual_pairs,
            "median_ttft_speedup": statistics.median(ttft_speedups),
            "mean_ttft_speedup": statistics.fmean(ttft_speedups),
            "ttft_wins": sum(value > 1.0 for value in ttft_speedups),
            "median_throughput_speedup": statistics.median(throughput_speedups),
            "mean_throughput_speedup": statistics.fmean(throughput_speedups),
            "throughput_wins": sum(value > 1.0 for value in throughput_speedups),
            "equal_answer_pairs": equal_answer_pairs,
            "exact_text_pairs": exact_text_pairs,
        },
        "gate": {
            "visual_tokens_reduced": reduction_passed,
            "accuracy_non_regression": accuracy_passed,
            "answer_equality_if_required": answer_gate_passed,
            "ttft_improved": ttft_passed,
            "throughput_non_regression": throughput_passed,
            "passed": (
                reduction_passed
                and accuracy_passed
                and answer_gate_passed
                and ttft_passed
                and throughput_passed
            ),
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
