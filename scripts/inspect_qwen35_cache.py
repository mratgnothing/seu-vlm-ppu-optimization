#!/usr/bin/env python3
"""Inspect the real Qwen3.5 cache tensors produced by one PPU prefill.

This is a diagnostic only.  It does not patch Transformers and it does not
change the benchmark timing boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_info(value) -> dict | None:
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "bytes": value.numel() * value.element_size(),
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    custom_ops = repo_root / "ppu" / "custom_ops"
    for path in (repo_root, custom_ops):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    import torch
    from transformers.cache_utils import DynamicCache

    from benchmark_public import build_prompt, decode_image, load_mmbench_tsv
    from evaluation_wrapper import VLMModel

    sample = load_mmbench_tsv(args.dataset_path, limit=1)[0]
    wrapper = VLMModel(str(args.model_path), backend="transformers", device="auto")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": decode_image(sample.image_b64)},
            {"type": "text", "text": build_prompt(sample)},
        ],
    }]
    inputs = wrapper._processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(wrapper._model.device)

    cache = DynamicCache(config=wrapper._model.config)
    with torch.inference_mode():
        output_ids = wrapper._model.generate(
            **inputs,
            past_key_values=cache,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
        torch.cuda.synchronize()

    layers = []
    total_bytes = 0
    for index, layer in enumerate(cache.layers):
        record = {"index": index, "type": type(layer).__name__}
        for name in ("keys", "values"):
            info = tensor_info(getattr(layer, name, None))
            record[name] = info
            total_bytes += 0 if info is None else info["bytes"]
        for name in ("conv_states", "recurrent_states"):
            values = getattr(layer, name, None)
            if values is None:
                record[name] = None
                continue
            record[name] = {
                str(key): tensor_info(value) for key, value in values.items()
            }
            total_bytes += sum(
                0 if info is None else info["bytes"]
                for info in record[name].values()
            )
        layers.append(record)

    report = {
        "sample_id": sample.sample_id,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "generated_tokens": int(output_ids.shape[1] - inputs.input_ids.shape[1]),
        "cache_type": type(cache).__name__,
        "total_cache_bytes": total_bytes,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
