#!/usr/bin/env python3
"""Measure Qwen3.5 visual-token counts for processor pixel budgets."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument(
        "--max-pixels",
        default="1048576,589824,262144,147456,65536",
        help="Comma-separated candidate longest-edge pixel budgets",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from transformers import AutoProcessor

    from benchmark_public import build_prompt, decode_image, load_mmbench_tsv

    samples = load_mmbench_tsv(args.dataset_path, limit=args.num_samples)
    budgets = [None] + [int(value) for value in args.max_pixels.split(",")]
    processors = {}
    for budget in budgets:
        kwargs = {} if budget is None else {"max_pixels": budget}
        processors[budget] = AutoProcessor.from_pretrained(
            args.model_path,
            local_files_only=True,
            trust_remote_code=True,
            **kwargs,
        )

    records = []
    for index, sample in enumerate(samples):
        image = decode_image(sample.image_b64)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": build_prompt(sample)},
            ],
        }]
        modes = {}
        for budget, processor in processors.items():
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            grid = inputs.image_grid_thw[0].tolist()
            merge_size = int(processor.image_processor.merge_size)
            modes["baseline" if budget is None else str(budget)] = {
                "grid_thw": grid,
                "visual_tokens": int(grid[0] * grid[1] * grid[2] // merge_size**2),
                "prompt_tokens": int(inputs.input_ids.shape[1]),
            }
        records.append({"sample_id": sample.sample_id, "modes": modes})
        if (index + 1) % 25 == 0:
            print(json.dumps({"progress": index + 1, "total": len(samples)}), flush=True)

    baseline_tokens = [record["modes"]["baseline"]["visual_tokens"] for record in records]
    summary = {}
    for label in records[0]["modes"]:
        values = [record["modes"][label]["visual_tokens"] for record in records]
        ratios = [value / baseline for value, baseline in zip(values, baseline_tokens)]
        summary[label] = {
            "mean_visual_tokens": statistics.fmean(values),
            "median_visual_tokens": statistics.median(values),
            "min_visual_tokens": min(values),
            "max_visual_tokens": max(values),
            "mean_ratio_to_baseline": statistics.fmean(ratios),
            "changed_samples": sum(value != base for value, base in zip(values, baseline_tokens)),
        }
    payload = {
        "sample_count": len(records),
        "dataset_path": str(args.dataset_path),
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(records), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
