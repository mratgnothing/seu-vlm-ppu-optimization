#!/usr/bin/env python3
"""Compare the fused PPU decode RMSNorm with Qwen3.5 eager semantics."""

from __future__ import annotations

import argparse
import json

import torch

from ppu_gdn import HIDDEN_SIZE, PPUGDNLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--threads", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + epsilon)
    return (output * (1.0 + weight.float())).type_as(x)


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
    device = torch.device("cuda:0")
    torch.manual_seed(20260826)
    x = torch.randn(1, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    weight = torch.randn(HIDDEN_SIZE, device=device, dtype=torch.bfloat16) / 16
    epsilon = 1.0e-6
    library = PPUGDNLibrary(args.library, rmsnorm_threads=args.threads)
    with torch.inference_mode():
        expected = eager_rmsnorm(x, weight, epsilon)
        actual = library.rmsnorm_decode(x, weight, epsilon)
        torch.cuda.synchronize()
        max_error = float((actual.float() - expected.float()).abs().max().item())
        eager_ms = measure(lambda: eager_rmsnorm(x, weight, epsilon), args.warmup, args.iters)
        fused_ms = measure(lambda: library.rmsnorm_decode(x, weight, epsilon), args.warmup, args.iters)

    passed = max_error <= 7.8125e-3
    result = {
        "kernel": "seu_ppu_rmsnorm_decode",
        "threads": args.threads,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
        "speedup": eager_ms / fused_ms,
        "max_output_error": max_error,
        "passed": passed,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
