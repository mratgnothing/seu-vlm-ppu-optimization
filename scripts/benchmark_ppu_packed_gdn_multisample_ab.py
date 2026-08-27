#!/usr/bin/env python3
"""Paired multi-sample A/B for packed Qwen3.5 GDN input projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--gdn-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--group-sizes",
        default="4",
        help="Comma-separated consecutive projection groups, for example 2,1,1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    group_sizes = tuple(int(value) for value in args.group_sizes.split(","))
    repo_root = args.repo_root.resolve()
    custom_op_dir = repo_root / "ppu" / "custom_ops"
    for path in (repo_root, custom_op_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(
        {
            "PPU_SDK": "/usr/local/PPU_SDK",
            "PPU_HOME": "/usr/local/PPU_SDK",
            "SEU_PPU_GDN_LIBRARY": str(args.gdn_library.resolve()),
            "SEU_PPU_GDN_PYTHON_DIR": str(custom_op_dir),
            "SEU_PPU_GDN_TILES": "4",
            "SEU_PPU_CONV_ENABLE": "1",
            "SEU_PPU_CONV_THREADS": "96",
            "SEU_PPU_RMSNORM_ENABLE": "1",
            "SEU_PPU_RMSNORM_THREADS": "512",
            "SEU_PPU_GATED_RMSNORM_ENABLE": "1",
            "SEU_PPU_GATED_RMSNORM_THREADS": "128",
            "SEU_PPU_QK_ROPE_ENABLE": "1",
            "SEU_PPU_PACK_MLP_ENABLE": "1",
        }
    )

    import torch

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
    from ppu_gdn_projection_pack import (
        pack_qwen35_gdn_input_projections,
        set_packed_qwen35_gdn_input_projections,
    )

    all_samples = load_mmbench_tsv(
        args.dataset_path,
        limit=args.sample_offset + args.num_samples,
    )
    samples = all_samples[args.sample_offset :]
    if len(samples) != args.num_samples:
        raise ValueError(f"requested {args.num_samples} samples, got {len(samples)}")

    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens
    packed_modules = []
    for module in model._model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        pack_qwen35_gdn_input_projections(module, group_sizes=group_sizes)
        packed_modules.append(module)
    if len(packed_modules) != 18:
        raise RuntimeError(f"expected 18 packed GDN modules, got {len(packed_modules)}")
    torch.cuda.empty_cache()

    def set_enabled(enabled: bool) -> None:
        for module in packed_modules:
            set_packed_qwen35_gdn_input_projections(module, enabled)

    def run_sample(sample, enabled: bool, pair_index: int) -> dict[str, object]:
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
            "pair_index": pair_index,
            "pair_order": "AB" if pair_index % 2 == 0 else "BA",
            "mode": "packed_gdn" if enabled else "optimized_baseline",
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

    run_sample(samples[0], False, -1)
    run_sample(samples[0], True, -1)
    baseline_records: list[dict[str, object]] = []
    packed_records: list[dict[str, object]] = []
    pair_ratios: list[float] = []
    exact_pairs = 0
    for index, sample in enumerate(samples):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair: dict[bool, dict[str, object]] = {}
        for enabled in order:
            pair[enabled] = run_sample(sample, enabled, index)
        baseline_records.append(pair[False])
        packed_records.append(pair[True])
        pair_ratios.append(
            float(pair[True]["throughput_tokens_per_sec"])
            / float(pair[False]["throughput_tokens_per_sec"])
        )
        exact_pairs += pair[True]["text_sha256"] == pair[False]["text_sha256"]

    def mean_metric(records, key: str) -> float:
        return statistics.fmean(float(record[key]) for record in records)

    baseline_accuracy = statistics.fmean(
        bool(record["correct"]) for record in baseline_records
    )
    packed_accuracy = statistics.fmean(bool(record["correct"]) for record in packed_records)
    payload = {
        "sample_offset": args.sample_offset,
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "packed_gdn_modules": len(packed_modules),
        "packed_weight_shape_per_module": [8224, 2048],
        "projection_group_sizes": group_sizes,
        "baseline": {
            "avg_ttft_ms": mean_metric(baseline_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean_metric(
                baseline_records, "throughput_tokens_per_sec"
            ),
            "accuracy": baseline_accuracy,
            "records": baseline_records,
        },
        "packed_gdn": {
            "avg_ttft_ms": mean_metric(packed_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean_metric(
                packed_records, "throughput_tokens_per_sec"
            ),
            "accuracy": packed_accuracy,
            "records": packed_records,
        },
        "paired_decode": {
            "median_speedup": statistics.median(pair_ratios),
            "mean_speedup": statistics.fmean(pair_ratios),
            "wins": sum(value > 1.0 for value in pair_ratios),
            "ratios": pair_ratios,
        },
        "exact_output_pairs": exact_pairs,
        "passed": exact_pairs == len(samples) and baseline_accuracy == packed_accuracy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
