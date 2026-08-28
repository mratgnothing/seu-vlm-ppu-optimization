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
    parser.add_argument("--workspace-mib", type=int, default=0)
    parser.add_argument("--batched-ba", action="store_true")
    parser.add_argument("--output-scratch", action="store_true")
    args = parser.parse_args()
    if args.workspace_mib < 0:
        raise ValueError("--workspace-mib must be non-negative")

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
        if args.workspace_mib:
            print("PHASE construct_workspace_extension", flush=True)
        extension = PPUACBLASGDNProjectionExtension(
            args.build_dir,
            workspace_bytes=args.workspace_mib * 1024 * 1024,
            workspace_enabled=False,
        )
        extension.pack_module(module)
        if args.workspace_mib:
            print("PHASE workspace_disabled_first_call", flush=True)
        actual_decode_without_workspace = tuple(
            output.clone() for output in module.project(decode_input)
        )
        actual_prefill = module.project(prefill_input)
        no_workspace_ms = measure(
            lambda: module.project(decode_input), args.warmup, args.iters
        )
        if args.workspace_mib:
            print("PHASE workspace_enable", flush=True)
            extension.set_workspace_enabled(True)
            print("PHASE workspace_enabled_first_call", flush=True)
        if args.batched_ba:
            print("PHASE batched_ba_enable", flush=True)
            extension.set_batched_ba(True)
        if args.output_scratch:
            print("PHASE output_scratch_enable", flush=True)
            extension.set_output_scratch(module, True)
        actual_decode = module.project(decode_input)
        grouped_ms = measure(lambda: module.project(decode_input), args.warmup, args.iters)

    decode_exact = all(torch.equal(a, b) for a, b in zip(expected_decode, actual_decode))
    prefill_exact = all(torch.equal(a, b) for a, b in zip(expected_prefill, actual_prefill))
    workspace_exact = all(
        torch.equal(a, b)
        for a, b in zip(actual_decode_without_workspace, actual_decode)
    )
    payload = {
        "candidate": "one_dispatch_four_acblas_gemv",
        "eager_ms": eager_ms,
        "grouped_ms": grouped_ms,
        "speedup": eager_ms / grouped_ms,
        "workspace_mib": args.workspace_mib,
        "no_workspace_ms": no_workspace_ms,
        "workspace_speedup": no_workspace_ms / grouped_ms,
        "workspace_exact": workspace_exact,
        "batched_ba": args.batched_ba,
        "output_scratch": args.output_scratch,
        "decode_exact": decode_exact,
        "prefill_exact": prefill_exact,
    }
    payload["passed"] = decode_exact and prefill_exact and workspace_exact
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
