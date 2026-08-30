#!/usr/bin/env python3
"""Validate fused BF16 SiLU(gate) * up for Qwen3.5 MLP decode on PPU."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from ppu_gdn import MLP_INTERMEDIATE_SIZE, PPUGDNLibrary


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
    parser.add_argument("--library", required=True)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    torch.manual_seed(20260827)
    device = torch.device("cuda:0")
    gate = torch.randn(
        1, 1, MLP_INTERMEDIATE_SIZE, device=device, dtype=torch.bfloat16
    )
    up = torch.randn_like(gate)
    library = PPUGDNLibrary(args.library, swiglu_threads=args.threads)

    with torch.inference_mode():
        expected = F.silu(gate) * up
        actual = library.swiglu_decode(gate, up)
        baseline_ms = measure(lambda: F.silu(gate) * up, args.warmup, args.iters)
        candidate_ms = measure(
            lambda: library.swiglu_decode(gate, up), args.warmup, args.iters
        )

    exact = torch.equal(expected, actual)
    payload = {
        "candidate": "fused_bf16_swiglu",
        "shape": list(gate.shape),
        "threads": args.threads,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "max_abs_error": float((expected.float() - actual.float()).abs().max()),
        "exact": exact,
        "passed": exact,
    }
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
