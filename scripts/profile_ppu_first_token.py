#!/usr/bin/env python3
"""Profile one warm Qwen3.5 multimodal prefill that emits the first token."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position", type=int, default=0)
    parser.add_argument("--row-limit", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    custom_ops = repo_root / "ppu" / "custom_ops"
    for path in (custom_ops, repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch

    from benchmark_public import (
        build_prompt,
        decode_image,
        load_mmbench_tsv,
    )
    from evaluation_wrapper import VLMModel

    sample = load_mmbench_tsv(
        args.dataset_path, limit=args.position + 1
    )[args.position]
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": decode_image(sample.image_b64)},
            {"type": "text", "text": build_prompt(sample)},
        ],
    }]
    inputs = model._processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model._model.device)
    generation_kwargs = {
        **inputs,
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": True,
    }

    # Compile and populate shape-dependent runtime caches outside the trace.
    with torch.inference_mode():
        model._model.generate(**generation_kwargs)
        torch.cuda.synchronize()

        started = time.perf_counter()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            output_ids = model._model.generate(**generation_kwargs)
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / "first-token-trace.json"
    profiler.export_chrome_trace(str(trace_path))
    averages = profiler.key_averages(group_by_input_shape=True)
    (args.output_dir / "first-token-top-cuda.txt").write_text(
        averages.table(sort_by="self_cuda_time_total", row_limit=args.row_limit)
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "first-token-top-cpu.txt").write_text(
        averages.table(sort_by="self_cpu_time_total", row_limit=args.row_limit)
        + "\n",
        encoding="utf-8",
    )
    payload = {
        "sample_id": sample.sample_id,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(output_ids.shape[1] - inputs.input_ids.shape[1]),
        "profile_scope": "one warm multimodal prefill producing first token",
        "elapsed_ms_with_profiler": elapsed_ms,
        "device": torch.cuda.get_device_name(0),
        "trace": str(trace_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
