#!/usr/bin/env python3
"""Paired PPU A/B for multi-row prefill norm/residual fusion."""

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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    custom_ops = repo_root / "ppu" / "custom_ops"
    for path in (custom_ops, repo_root):
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

    samples = load_mmbench_tsv(args.dataset_path, limit=args.num_samples)
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    toggled_modules = [
        module
        for module in model._model.modules()
        if any(
            hasattr(module, name)
            for name in (
                "_seu_prefill_rmsnorm_enabled",
                "_seu_prefill_gated_rmsnorm_enabled",
                "_seu_prefill_residual_rmsnorm_enabled",
            )
        )
    ]
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens

    def set_enabled(enabled: bool) -> None:
        for module in toggled_modules:
            for name in (
                "_seu_prefill_rmsnorm_enabled",
                "_seu_prefill_gated_rmsnorm_enabled",
                "_seu_prefill_residual_rmsnorm_enabled",
            ):
                if hasattr(module, name):
                    setattr(module, name, enabled)
        model._ppu_prefill_row_fusions_enabled = enabled

    def run(sample, enabled: bool) -> dict[str, object]:
        set_enabled(enabled)
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
            "mode": "prefill_row_fusions" if enabled else "baseline",
            "ttft_ms": result.ttft_seconds * 1000.0,
            "throughput_tokens_per_sec": compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            ),
            "token_count": result.token_count,
            "answer": answer,
            "reference": sample.answer,
            "correct": answer == sample.answer,
            "text_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
        }

    run(samples[0], False)
    run(samples[0], True)
    baseline_records = []
    candidate_records = []
    ttft_speedups = []
    throughput_speedups = []
    for index, sample in enumerate(samples):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair = {enabled: run(sample, enabled) for enabled in order}
        baseline, candidate = pair[False], pair[True]
        baseline_records.append(baseline)
        candidate_records.append(candidate)
        ttft_speedups.append(baseline["ttft_ms"] / candidate["ttft_ms"])
        throughput_speedups.append(
            candidate["throughput_tokens_per_sec"]
            / baseline["throughput_tokens_per_sec"]
        )
        if (index + 1) % args.progress_every == 0:
            print(json.dumps({
                "progress": index + 1,
                "median_ttft_speedup": statistics.median(ttft_speedups),
                "equal_answers": sum(
                    a["answer"] == b["answer"]
                    for a, b in zip(baseline_records, candidate_records)
                ),
            }), flush=True)

    equal_answers = sum(
        a["answer"] == b["answer"]
        for a, b in zip(baseline_records, candidate_records)
    )
    exact_text = sum(
        a["text_sha256"] == b["text_sha256"]
        for a, b in zip(baseline_records, candidate_records)
    )
    baseline_correct = sum(record["correct"] for record in baseline_records)
    candidate_correct = sum(record["correct"] for record in candidate_records)
    payload = {
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "toggled_module_count": len(toggled_modules),
        "baseline_correct": baseline_correct,
        "candidate_correct": candidate_correct,
        "equal_answers": equal_answers,
        "exact_text": exact_text,
        "paired": {
            "median_ttft_speedup": statistics.median(ttft_speedups),
            "mean_ttft_speedup": statistics.fmean(ttft_speedups),
            "ttft_wins": sum(value > 1.0 for value in ttft_speedups),
            "median_throughput_speedup": statistics.median(throughput_speedups),
            "mean_throughput_speedup": statistics.fmean(throughput_speedups),
        },
        "passed": (
            candidate_correct >= baseline_correct
            and equal_answers == len(samples)
            and statistics.median(ttft_speedups) > 1.0
            and statistics.fmean(ttft_speedups) > 1.0
            and statistics.median(throughput_speedups) >= 1.0
            and statistics.fmean(throughput_speedups) >= 1.0
        ),
        "records": {"baseline": baseline_records, "candidate": candidate_records},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
