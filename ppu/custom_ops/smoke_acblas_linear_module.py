#!/usr/bin/env python3
"""Validate module-level registered acBLAS integration and fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ppu_acblas_extension import PPUACBLASLinearExtension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--input-features", type=int, required=True)
    parser.add_argument("--output-features", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=2000)
    return parser.parse_args()


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
    args = parse_args()
    torch.manual_seed(20260827)
    module = torch.nn.Linear(
        args.input_features, args.output_features, bias=False
    ).to(device="cuda:0", dtype=torch.bfloat16).eval()
    decode_input = torch.randn(
        1, 1, args.input_features, device="cuda:0", dtype=torch.bfloat16
    )
    prefill_input = torch.randn(
        1, 4, args.input_features, device="cuda:0", dtype=torch.bfloat16
    )
    original_forward = module.forward
    extension = PPUACBLASLinearExtension(args.build_dir)

    with torch.inference_mode():
        expected_decode = original_forward(decode_input)
        expected_prefill = original_forward(prefill_input)
        eager_ms = measure(
            lambda: original_forward(decode_input), args.warmup, args.iters
        )
        extension.patch_linear(module)
        actual_decode = module(decode_input)
        actual_prefill = module(prefill_input)
        patched_ms = measure(lambda: module(decode_input), args.warmup, args.iters)

    payload = {
        "shape": [args.output_features, args.input_features],
        "eager_ms": eager_ms,
        "patched_ms": patched_ms,
        "speedup": eager_ms / patched_ms,
        "decode_exact": torch.equal(expected_decode, actual_decode),
        "prefill_exact": torch.equal(expected_prefill, actual_prefill),
    }
    payload["passed"] = payload["decode_exact"] and payload["prefill_exact"]
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
