#!/usr/bin/env python3
"""Measure the PPU PyTorch BF16 GEMV path with the HGGC benchmark shapes."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--matrix-copies", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.n, args.k, args.iters, args.matrix_copies) <= 0 or args.warmup < 0:
        raise ValueError("dimensions, iterations, and copies must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PPU CUDA compatibility backend is unavailable")

    columns = np.arange(args.k, dtype=np.int64)
    rows = np.arange(args.n, dtype=np.int64)[:, None]
    host_input = torch.from_numpy(((columns % 17) - 8).astype(np.float32) / 32.0)
    host_weights = torch.from_numpy(
        (((rows * 13 + columns[None, :] * 7) % 29) - 14).astype(np.float32)
        / 64.0
    )
    host_input = host_input.to(torch.bfloat16)
    host_weights = host_weights.to(torch.bfloat16)
    reference = torch.mv(host_weights.float(), host_input.float())

    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    input_device = host_input.to(device)
    weights = [host_weights.to(device) for _ in range(args.matrix_copies)]

    output = None
    with torch.inference_mode():
        for iteration in range(args.warmup):
            output = torch.mv(weights[iteration % args.matrix_copies], input_device)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for iteration in range(args.iters):
            copy = (args.warmup + iteration) % args.matrix_copies
            output = torch.mv(weights[copy], input_device)
        stop.record()
        stop.synchronize()

    assert output is not None
    average_ms = float(start.elapsed_time(stop)) / args.iters
    actual = output.float().cpu()
    absolute_error = (actual - reference).abs()
    max_absolute_error = float(absolute_error.max().item())
    relative_error = absolute_error / reference.abs().clamp_min(1.0e-6)
    max_relative_error = float(relative_error.max().item())
    passed = bool(torch.allclose(actual, reference, atol=0.125, rtol=0.01))

    seconds = average_ms / 1000.0
    flops = 2.0 * args.n * args.k
    byte_count = args.n * args.k * 2 + args.k * 2 + args.n * 2
    result = {
        "kernel": "torch_mv_bf16",
        "n": args.n,
        "k": args.k,
        "threads": None,
        "matrix_copies": args.matrix_copies,
        "warmup": args.warmup,
        "iterations": args.iters,
        "average_ms": average_ms,
        "gflops": flops / seconds / 1.0e9,
        "effective_gbps": byte_count / seconds / 1.0e9,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "passed": passed and math.isfinite(average_ms) and average_ms > 0,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(args.device),
        "output_dtype": str(output.dtype),
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
