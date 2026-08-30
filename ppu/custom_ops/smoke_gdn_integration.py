#!/usr/bin/env python3
"""Compare the PPU ctypes GDN operator with the Transformers eager fallback."""

from __future__ import annotations

import argparse
import json
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    torch_recurrent_gated_delta_rule,
)

from ppu_gdn import HEADS, HEAD_DIM, PPUGDNLibrary


def make_inputs(batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260826)
    query = torch.randn(batch_size, 1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    g = -torch.rand(batch_size, 1, HEADS, dtype=torch.float32, device=device) * 0.1
    beta = torch.rand(batch_size, 1, HEADS, dtype=torch.bfloat16, device=device)
    state = torch.randn(
        batch_size, HEADS, HEAD_DIM, HEAD_DIM,
        dtype=torch.float32, device=device,
    ) / 64
    return query, key, value, g, beta, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--tiles-per-head", type=int, choices=(1, 2, 4), default=4)
    return parser.parse_args()


def measure(callable_, args: tuple[torch.Tensor, ...], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        callable_(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        callable_(*args)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0")
    inputs = make_inputs(args.batch_size, device)
    query, key, value, g, beta, initial_state = inputs
    library = PPUGDNLibrary(args.library, tiles_per_head=args.tiles_per_head)

    with torch.inference_mode():
        eager_output, eager_state = torch_recurrent_gated_delta_rule(
            query, key, value, g, beta, initial_state.clone(), True, True
        )
        fused_state = initial_state.clone()
        fused_output = library.recurrent_decode(query, key, value, g, beta, fused_state)
        torch.cuda.synchronize()
        output_error = float((fused_output.float() - eager_output.float()).abs().max().item())
        state_error = float((fused_state - eager_state).abs().max().item())

        def eager(q, k, v, gate, mix, state):
            return torch_recurrent_gated_delta_rule(q, k, v, gate, mix, state, True, True)

        def fused(q, k, v, gate, mix, state):
            return library.recurrent_decode(q, k, v, gate, mix, state)

        eager_ms = measure(eager, inputs[:-1] + (initial_state.clone(),), args.warmup, args.iters)
        fused_ms = measure(fused, inputs[:-1] + (initial_state.clone(),), args.warmup, args.iters)

    passed = output_error <= 1.0e-4 and state_error <= 2.0e-5
    result = {
        "kernel": "seu_ppu_gdn_ctypes",
        "batch_size": args.batch_size,
        "tiles_per_head": args.tiles_per_head,
        "warmup": args.warmup,
        "iterations": args.iters,
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
        "speedup": eager_ms / fused_ms,
        "max_output_error": output_error,
        "max_state_error": state_error,
        "passed": passed,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
