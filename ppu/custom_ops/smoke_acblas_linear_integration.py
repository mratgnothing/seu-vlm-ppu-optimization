#!/usr/bin/env python3
"""Validate direct acBLAS BF16 decode linear integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ppu_linear import PPUACBLASLinearLibrary, patch_linear_decode_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path)
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--algorithm", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
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
    module = torch.nn.Linear(args.features, args.features, bias=False).to(
        device="cuda:0", dtype=torch.bfloat16
    ).eval()
    decode_input = torch.randn(
        1, 1, args.features, device="cuda:0", dtype=torch.bfloat16
    )
    prefill_input = torch.randn(
        1, 4, args.features, device="cuda:0", dtype=torch.bfloat16
    )
    eager_forward = module.forward
    library = PPUACBLASLinearLibrary(args.library, algorithm=args.algorithm)

    with torch.inference_mode():
        expected_decode = eager_forward(decode_input)
        expected_prefill = eager_forward(prefill_input)
        module.forward = patch_linear_decode_module(module, library)
        actual_decode = module(decode_input)
        actual_prefill = module(prefill_input)
        torch.cuda.synchronize()
        eager_ms = measure(
            lambda: eager_forward(decode_input), args.warmup, args.iters
        )
        acblas_ms = measure(
            lambda: module(decode_input), args.warmup, args.iters
        )

    result = {
        "candidate": "acblas_bf16_decode_linear",
        "shape": [args.features, args.features],
        "algorithm": args.algorithm,
        "eager_ms": eager_ms,
        "acblas_ms": acblas_ms,
        "speedup": eager_ms / acblas_ms,
        "decode_exact": torch.equal(expected_decode, actual_decode),
        "prefill_exact": torch.equal(expected_prefill, actual_prefill),
    }
    result["passed"] = result["decode_exact"] and result["prefill_exact"]
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
