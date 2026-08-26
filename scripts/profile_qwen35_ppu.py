#!/usr/bin/env python3
"""Profile a short, real multimodal Qwen3.5 generation on the PPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.max_new_tokens, args.warmup_new_tokens, args.row_limit) <= 0:
        raise ValueError("token counts and row limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_mmbench_tsv(args.dataset_path, limit=1)[0]
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": decode_image(sample.image_b64)},
            {"type": "text", "text": build_prompt(sample)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=args.warmup_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()

    trace_path = args.output_dir / "trace.json"
    profile.export_chrome_trace(str(trace_path))
    key_averages = profile.key_averages(group_by_input_shape=True)
    cuda_table = key_averages.table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit
    )
    cpu_table = key_averages.table(
        sort_by="self_cpu_time_total", row_limit=args.row_limit
    )
    (args.output_dir / "top-cuda.txt").write_text(cuda_table + "\n", encoding="utf-8")
    (args.output_dir / "top-cpu.txt").write_text(cpu_table + "\n", encoding="utf-8")

    generated_tokens = int(output_ids.shape[1]) - prompt_tokens
    summary = {
        "sample_id": sample.sample_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_new_tokens": args.warmup_new_tokens,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "trace_path": str(trace_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n# Top accelerator operations\n")
    print(cuda_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
