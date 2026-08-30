#!/usr/bin/env python3
"""Validate registered acBLASLt square Linear, scratch reuse, and fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ppu_acblaslt_square import PPUACBLASLtSquareExtension


def measure(callable_, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        callable_()
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--heuristic-index", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(20260828)
    module = torch.nn.Linear(2048, 2048, bias=False).to(
        device="cuda:0", dtype=torch.bfloat16
    ).eval()
    decode_input = torch.randn(1, 1, 2048, device="cuda:0", dtype=torch.bfloat16)
    prefill_input = torch.randn(1, 4, 2048, device="cuda:0", dtype=torch.bfloat16)
    original_forward = module.forward
    extension = PPUACBLASLtSquareExtension(
        args.build_dir, heuristic_index=args.heuristic_index
    )
    with torch.inference_mode():
        expected_decode = original_forward(decode_input)
        expected_prefill = original_forward(prefill_input)
        eager_ms = measure(
            lambda: original_forward(decode_input), args.warmup, args.iters
        )
        extension.patch_linear(module)
        first_decode = module(decode_input).clone()
        first_pointer = module(decode_input).data_ptr()
        second_pointer = module(decode_input).data_ptr()
        actual_prefill = module(prefill_input)
        patched_ms = measure(lambda: module(decode_input), args.warmup, args.iters)
    payload = {
        "shape": [2048, 2048],
        "heuristic_index": args.heuristic_index,
        "eager_ms": eager_ms,
        "patched_ms": patched_ms,
        "speedup": eager_ms / patched_ms,
        "decode_exact": torch.equal(expected_decode, first_decode),
        "prefill_exact": torch.equal(expected_prefill, actual_prefill),
        "scratch_pointer_reused": first_pointer == second_pointer,
    }
    payload["passed"] = all(
        payload[key]
        for key in ("decode_exact", "prefill_exact", "scratch_pointer_reused")
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
