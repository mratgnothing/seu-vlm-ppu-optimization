#!/usr/bin/env python3
"""Validate one-dispatch grouped acBLAS GDN projections on PPU."""

from __future__ import annotations

import argparse
import json

import torch

from ppu_acblas_gdn_projection import PPUACBLASGDNProjectionExtension


class Qwen3_5GatedDeltaNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(2048, 6144, bias=False)
        self.in_proj_z = torch.nn.Linear(2048, 2048, bias=False)
        self.in_proj_b = torch.nn.Linear(2048, 16, bias=False)
        self.in_proj_a = torch.nn.Linear(2048, 16, bias=False)

    def project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (
            self.in_proj_qkv(x),
            self.in_proj_z(x),
            self.in_proj_b(x),
            self.in_proj_a(x),
        )


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
    parser.add_argument("--build-dir", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()

    torch.manual_seed(20260827)
    module = Qwen3_5GatedDeltaNet().to(device="cuda:0", dtype=torch.bfloat16).eval()
    decode_input = torch.randn(1, 1, 2048, device="cuda:0", dtype=torch.bfloat16)
    prefill_input = torch.randn(1, 4, 2048, device="cuda:0", dtype=torch.bfloat16)
    originals = [linear.forward for linear in (
        module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a
    )]

    def eager(x):
        return tuple(forward(x) for forward in originals)

    with torch.inference_mode():
        expected_decode = eager(decode_input)
        expected_prefill = eager(prefill_input)
        eager_ms = measure(lambda: eager(decode_input), args.warmup, args.iters)
        extension = PPUACBLASGDNProjectionExtension(args.build_dir)
        extension.pack_module(module)
        actual_decode = module.project(decode_input)
        actual_prefill = module.project(prefill_input)
        grouped_ms = measure(lambda: module.project(decode_input), args.warmup, args.iters)

    decode_exact = all(torch.equal(a, b) for a, b in zip(expected_decode, actual_decode))
    prefill_exact = all(torch.equal(a, b) for a, b in zip(expected_prefill, actual_prefill))
    payload = {
        "candidate": "one_dispatch_four_acblas_gemv",
        "eager_ms": eager_ms,
        "grouped_ms": grouped_ms,
        "speedup": eager_ms / grouped_ms,
        "decode_exact": decode_exact,
        "prefill_exact": prefill_exact,
    }
    payload["passed"] = decode_exact and prefill_exact
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
