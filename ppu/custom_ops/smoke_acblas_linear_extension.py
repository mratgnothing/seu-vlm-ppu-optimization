#!/usr/bin/env python3
"""Validate the registered PyTorch/acBLAS BF16 decode linear extension."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, default=2048)
    parser.add_argument("--input-features", type=int)
    parser.add_argument("--output-features", type=int)
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
    input_features = args.input_features or args.features
    output_features = args.output_features or args.features
    build_dir = Path(__file__).resolve().parent / "build" / "acblas_linear_extension"
    sys.path.insert(0, str(build_dir))
    extension = importlib.import_module("seu_acblas_linear_ext")

    torch.manual_seed(20260827)
    device = torch.device("cuda:0")
    linear = torch.nn.Linear(input_features, output_features, bias=False).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    input_tensor = torch.randn(
        1, 1, input_features, device=device, dtype=torch.bfloat16
    )

    with torch.inference_mode():
        expected = linear(input_tensor)
        actual = extension.linear_bf16(
            input_tensor, linear.weight, args.algorithm
        )
        torch.cuda.synchronize()
        eager_ms = measure(
            lambda: linear(input_tensor), args.warmup, args.iters
        )
        extension_ms = measure(
            lambda: extension.linear_bf16(
                input_tensor, linear.weight, args.algorithm
            ),
            args.warmup,
            args.iters,
        )

    exact = torch.equal(expected, actual)
    result = {
        "candidate": "pytorch_cpp_acblas_bf16_decode_linear",
        "shape": [output_features, input_features],
        "algorithm": args.algorithm,
        "eager_ms": eager_ms,
        "extension_ms": extension_ms,
        "speedup": eager_ms / extension_ms,
        "exact": exact,
        "max_abs_error": float(
            (expected.float() - actual.float()).abs().max().item()
        ),
        "passed": exact,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
