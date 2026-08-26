#!/usr/bin/env python3
"""Locate a native PPU failure in a single real Qwen3.5 forward pass.

This diagnostic intentionally runs outside the benchmark wrapper. It prints and
flushes every stage (and optionally every leaf-module entry) so a SIGABRT still
leaves a useful last-known boundary in the captured log.
"""

from __future__ import annotations

import argparse
import faulthandler
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def tensor_metadata(inputs: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name, value in inputs.items():
        if torch.is_tensor(value):
            metadata[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        else:
            metadata[name] = {"type": type(value).__name__}
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-forward", action="store_true")
    parser.add_argument("--trace-leaves", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    faulthandler.enable(all_threads=True)
    emit(
        "runtime",
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )

    sample = load_mmbench_tsv(args.dataset_path, limit=1)[0]
    image = decode_image(sample.image_b64)
    prompt = build_prompt(sample)
    emit("sample", sample_id=sample.sample_id, image_size=list(image.size))

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    emit("processor_loaded", processor_class=type(processor).__name__)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    emit(
        "model_loaded",
        model_class=type(model).__name__,
        model_device=str(model.device),
        top_level_children={name: type(module).__name__ for name, module in model.named_children()},
    )

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    emit("inputs_cpu", tensors=tensor_metadata(inputs))
    inputs = inputs.to(model.device)
    emit("inputs_device", tensors=tensor_metadata(inputs))

    handles = []
    if args.trace_leaves:
        for name, module in model.named_modules():
            if name and not any(module.children()):
                def pre_hook(_module, _args, _kwargs, module_name=name):
                    emit("module_enter", name=module_name, module=type(_module).__name__)

                handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))

    if not args.run_forward:
        emit("prepared", leaf_module_count=len(handles))
        return 0

    emit("forward_start")
    try:
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, return_dict=True)
            torch.cuda.synchronize()
        emit(
            "forward_complete",
            logits_shape=list(outputs.logits.shape),
            logits_dtype=str(outputs.logits.dtype),
            logits_device=str(outputs.logits.device),
        )
    finally:
        for handle in handles:
            handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
