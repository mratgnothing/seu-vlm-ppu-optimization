#!/usr/bin/env python3
"""Validate packed Qwen3.5 MLP projections and steady-state storage aliasing."""

from __future__ import annotations

import argparse
import gc
import json
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppu_gdn import HIDDEN_SIZE, MLP_INTERMEDIATE_SIZE, pack_qwen35_mlp_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iters", type=int, default=400)
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
    torch.manual_seed(20260826)
    device = torch.device("cuda:0")
    config = SimpleNamespace(hidden_size=HIDDEN_SIZE, hidden_act="silu")
    module = Qwen3_5MLP(config, MLP_INTERMEDIATE_SIZE).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    decode_input = torch.randn(
        1, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    prefill_input = torch.randn(
        1, 4, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    eager_forward = module.forward

    with torch.inference_mode():
        expected_decode = eager_forward(decode_input)
        expected_prefill = eager_forward(prefill_input)
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated(device)
        module.forward = pack_qwen35_mlp_module(module)
        gc.collect()
        torch.cuda.empty_cache()
        actual_decode = module(decode_input)
        actual_prefill = module(prefill_input)
        torch.cuda.synchronize()
        memory_after = torch.cuda.memory_allocated(device)
        eager_ms = measure(
            lambda: eager_forward(decode_input), args.warmup, args.iters
        )
        packed_ms = measure(
            lambda: module(decode_input), args.warmup, args.iters
        )

    packed_storage = module._seu_gate_up_weight.untyped_storage().data_ptr()
    gate_alias = module.gate_proj.weight.untyped_storage().data_ptr() == packed_storage
    up_alias = module.up_proj.weight.untyped_storage().data_ptr() == packed_storage
    passed = (
        torch.equal(expected_decode, actual_decode)
        and torch.equal(expected_prefill, actual_prefill)
        and gate_alias
        and up_alias
        and memory_after - memory_before < 10 * 1024 * 1024
    )
    result = {
        "candidate": "qwen35_packed_mlp_gate_up",
        "eager_ms": eager_ms,
        "packed_ms": packed_ms,
        "speedup": eager_ms / packed_ms,
        "decode_exact": torch.equal(expected_decode, actual_decode),
        "prefill_exact": torch.equal(expected_prefill, actual_prefill),
        "gate_alias": gate_alias,
        "up_alias": up_alias,
        "steady_memory_delta_bytes": memory_after - memory_before,
        "passed": passed,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
