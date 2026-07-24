#!/usr/bin/env python3
"""Profile one public sample without storing its prompt or generated text."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_public import (
    build_prompt,
    compute_throughput,
    decode_image,
    fixed_generation_config,
    load_mmbench_tsv,
    settle_runtime,
)
from evaluation_wrapper import VLMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile one VLM inference sample")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-output", type=Path)
    return parser.parse_args()


def _metric(event: Any, name: str, fallback: str | None = None) -> float:
    value = getattr(event, name, None)
    if value is None and fallback:
        value = getattr(event, fallback, 0.0)
    return float(value or 0.0)


def main() -> None:
    args = parse_args()
    import torch
    from torch.profiler import ProfilerActivity, profile

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local GPU profiler.")

    needed = args.sample_offset + max(args.warmup_samples, 1) + 1
    samples = load_mmbench_tsv(args.dataset_path.resolve(), limit=needed)
    if args.sample_offset < 0 or args.sample_offset >= len(samples):
        raise ValueError(f"sample-offset {args.sample_offset} is outside loaded data")

    model = VLMModel(
        str(args.model_path.resolve()),
        backend="transformers",
        device="auto",
    )
    config = fixed_generation_config()
    sample = samples[args.sample_offset]

    for warmup_index in range(args.warmup_samples):
        warmup_sample = samples[
            min(args.sample_offset + warmup_index, len(samples) - 1)
        ]
        settle_runtime(model)
        model.generate_with_metrics(
            image=decode_image(warmup_sample.image_b64),
            prompt=build_prompt(warmup_sample),
            choices=warmup_sample.choices,
            generation_config=config,
            sample_id=warmup_sample.sample_id,
        )

    settle_runtime(model)
    torch.cuda.reset_peak_memory_stats()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        result = model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        torch.cuda.synchronize()

    events = list(profiler.key_averages())
    ranked = sorted(
        events,
        key=lambda event: _metric(
            event,
            "self_device_time_total",
            "self_cuda_time_total",
        ),
        reverse=True,
    )
    top_cuda_ops = [
        {
            "name": event.key,
            "calls": int(event.count),
            "self_cuda_time_ms": round(
                _metric(
                    event,
                    "self_device_time_total",
                    "self_cuda_time_total",
                )
                / 1000.0,
                3,
            ),
            "cuda_time_total_ms": round(
                _metric(
                    event,
                    "device_time_total",
                    "cuda_time_total",
                )
                / 1000.0,
                3,
            ),
            "self_cpu_time_ms": round(
                _metric(event, "self_cpu_time_total") / 1000.0,
                3,
            ),
        }
        for event in ranked[:30]
    ]

    output = {
        "timestamp": datetime.now().isoformat(),
        "backend": model.backend_name,
        "model_class": type(model._model).__name__,
        "device": str(model._model.device),
        "dataset_file": args.dataset_path.resolve().name,
        "sample_offset": args.sample_offset,
        "warmup_samples": args.warmup_samples,
        "generation": {
            "max_new_tokens": config.max_new_tokens,
            "token_count": result.token_count,
            "ttft_ms": round(result.ttft_seconds * 1000.0, 3),
            "elapsed_ms": round(result.elapsed_seconds * 1000.0, 3),
            "throughput_tokens_per_sec": round(
                compute_throughput(
                    result.token_count,
                    result.ttft_seconds,
                    result.elapsed_seconds,
                ),
                3,
            ),
            "meta": result.meta,
        },
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        },
        "top_cuda_ops": top_cuda_ops,
    }

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.trace_output:
        trace_path = args.trace_output.resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
