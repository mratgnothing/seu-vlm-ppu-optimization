#!/usr/bin/env python3
"""Validate packed Qwen3.5 GDN input projections on random BF16 tensors."""

from __future__ import annotations

import argparse
import json
import threading

import torch

from ppu_gdn_projection_pack import pack_qwen35_gdn_input_projections


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    module = Qwen3_5GatedDeltaNet().to(device="cuda:0", dtype=torch.bfloat16).eval()
    decode_input = torch.randn(1, 1, 2048, device="cuda:0", dtype=torch.bfloat16)
    prefill_input = torch.randn(1, 4, 2048, device="cuda:0", dtype=torch.bfloat16)
    concurrent_inputs = [
        torch.randn(1, 1, 2048, device="cuda:0", dtype=torch.bfloat16)
        for _ in range(2)
    ]
    original_forwards = {
        name: getattr(module, name).forward
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    }

    def original_project(x):
        return tuple(forward(x) for forward in original_forwards.values())

    with torch.inference_mode():
        expected_decode = original_project(decode_input)
        expected_prefill = original_project(prefill_input)
        expected_concurrent = [original_project(x) for x in concurrent_inputs]
        eager_ms = measure(lambda: original_project(decode_input), args.warmup, args.iters)
        pack_qwen35_gdn_input_projections(module)
        actual_decode = module.project(decode_input)
        actual_prefill = module.project(prefill_input)
        packed_ms = measure(lambda: module.project(decode_input), args.warmup, args.iters)

    barrier = threading.Barrier(2)
    actual_concurrent: list[tuple[torch.Tensor, ...] | None] = [None, None]
    thread_errors: list[BaseException] = []

    def concurrent_project(index: int) -> None:
        try:
            with torch.inference_mode():
                x = concurrent_inputs[index]
                qkv = module.in_proj_qkv(x)
                barrier.wait()
                actual_concurrent[index] = (
                    qkv,
                    module.in_proj_z(x),
                    module.in_proj_b(x),
                    module.in_proj_a(x),
                )
        except BaseException as exc:  # surfaced through the smoke result
            thread_errors.append(exc)

    threads = [threading.Thread(target=concurrent_project, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    torch.cuda.synchronize()

    decode_exact = all(torch.equal(a, b) for a, b in zip(expected_decode, actual_decode))
    prefill_exact = all(torch.equal(a, b) for a, b in zip(expected_prefill, actual_prefill))
    concurrent_exact = not thread_errors and all(
        actual is not None
        and all(torch.equal(a, b) for a, b in zip(expected, actual))
        for expected, actual in zip(expected_concurrent, actual_concurrent)
    )
    storage = module._seu_gdn_input_weight.untyped_storage().data_ptr()
    aliases = all(
        getattr(module, name).weight.untyped_storage().data_ptr() == storage
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")
    )
    payload = {
        "candidate": "packed_qwen35_gdn_input_projections",
        "eager_ms": eager_ms,
        "packed_ms": packed_ms,
        "speedup": eager_ms / packed_ms,
        "decode_exact": decode_exact,
        "prefill_exact": prefill_exact,
        "concurrent_exact": concurrent_exact,
        "weights_alias_packed_storage": aliases,
    }
    payload["passed"] = decode_exact and prefill_exact and concurrent_exact and aliases
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
