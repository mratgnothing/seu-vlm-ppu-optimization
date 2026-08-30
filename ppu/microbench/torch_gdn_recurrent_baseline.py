#!/usr/bin/env python3
"""Measure the eager PyTorch recurrent GDN fallback used by Qwen3.5 decode."""

from __future__ import annotations

import argparse
import json
import math

import torch


HEADS = 16
KEY_DIM = 128
VALUE_DIM = 128
EPSILON = 1.0e-6


def make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    heads = torch.arange(HEADS, dtype=torch.int64)[:, None]
    dims = torch.arange(KEY_DIM, dtype=torch.int64)[None, :]
    query = ((((heads * 17 + dims * 5) % 31) - 15).float() / 64.0).to(torch.bfloat16)
    key = ((((heads * 11 + dims * 3) % 29) - 14).float() / 64.0).to(torch.bfloat16)
    value = ((((heads * 7 + dims * 13) % 37) - 18).float() / 64.0).to(torch.bfloat16)
    g = -0.05 - (torch.arange(HEADS).float() % 7) * 0.01
    beta = 0.2 + (torch.arange(HEADS).float() % 5) * 0.1
    key_indices = torch.arange(KEY_DIM, dtype=torch.int64)[None, :, None]
    value_indices = torch.arange(VALUE_DIM, dtype=torch.int64)[None, None, :]
    head_indices = torch.arange(HEADS, dtype=torch.int64)[:, None, None]
    state = (
        ((head_indices * 19 + key_indices * 7 + value_indices * 3) % 41) - 20
    ).float() / 1024.0
    return tuple(tensor.to(device) for tensor in (query, key, value, g, beta, state))


def recurrent_step(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_norm = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + EPSILON)
    key_norm = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + EPSILON)
    query_float = query_norm.float() / math.sqrt(KEY_DIM)
    key_float = key_norm.float()
    state = state * g.exp()[:, None, None]
    memory_projection = (state * key_float[:, :, None]).sum(dim=1)
    delta = (value.float() - memory_projection) * beta[:, None]
    state = state + key_float[:, :, None] * delta[:, None, :]
    output = (state * query_float[:, :, None]).sum(dim=1).to(torch.bfloat16)
    return output, state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--hggc-dump-prefix",
        help="compare one-step PPU results with PREFIX.state.f32.bin and PREFIX.output.bf16.bin",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if not torch.cuda.is_available():
        raise RuntimeError("PPU CUDA compatibility backend is unavailable")

    cpu_inputs = make_inputs(torch.device("cpu"))
    device = torch.device(f"cuda:{args.device}")
    device_inputs = make_inputs(device)
    with torch.inference_mode():
        expected_output, expected_state = recurrent_step(*cpu_inputs)
        actual_output, actual_state = recurrent_step(*device_inputs)
        torch.cuda.synchronize(device)
        output_error = float(
            (actual_output.float().cpu() - expected_output.float()).abs().max().item()
        )
        state_error = float(
            (actual_state.cpu() - expected_state).abs().max().item()
        )
        # Parallel reduction order differs between CPU and PPU.  This tolerance
        # is tight enough to catch formula/layout mistakes while allowing the
        # observed FP32 summation-order drift.
        correctness_passed = output_error <= 7.8125e-3 and state_error <= 2.0e-4

        hggc_state_error = None
        hggc_output_error = None
        if args.hggc_dump_prefix:
            state_path = args.hggc_dump_prefix + ".state.f32.bin"
            output_path = args.hggc_dump_prefix + ".output.bf16.bin"
            hggc_state = torch.from_file(
                state_path, shared=False, size=HEADS * KEY_DIM * VALUE_DIM,
                dtype=torch.float32,
            ).reshape(HEADS, KEY_DIM, VALUE_DIM)
            hggc_output = torch.from_file(
                output_path, shared=False, size=HEADS * VALUE_DIM,
                dtype=torch.bfloat16,
            ).reshape(HEADS, VALUE_DIM)
            hggc_state_error = float(
                (hggc_state - actual_state.cpu()).abs().max().item()
            )
            hggc_output_error = float(
                (hggc_output.float() - actual_output.float().cpu()).abs().max().item()
            )
            correctness_passed = correctness_passed and (
                hggc_state_error <= 2.0e-3 and hggc_output_error <= 7.8125e-3
            )

        query, key, value, g, beta, initial_state = device_inputs
        timing_state = initial_state.clone()
        for _ in range(args.warmup):
            _, timing_state = recurrent_step(query, key, value, g, beta, timing_state)
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iters):
            output, timing_state = recurrent_step(
                query, key, value, g, beta, timing_state
            )
        stop.record()
        stop.synchronize()

    average_ms = float(start.elapsed_time(stop)) / args.iters
    state_elements = HEADS * KEY_DIM * VALUE_DIM
    vector_elements = HEADS * KEY_DIM
    byte_count = state_elements * 4 * 3 + vector_elements * 2 * 3 + HEADS * 4 * 2
    effective_gbps = byte_count / (average_ms / 1000.0) / 1.0e9
    passed = correctness_passed and math.isfinite(average_ms) and average_ms > 0
    result = {
        "kernel": "torch_gdn_recurrent_eager",
        "heads": HEADS,
        "key_dim": KEY_DIM,
        "value_dim": VALUE_DIM,
        "threads": None,
        "warmup": args.warmup,
        "iterations": args.iters,
        "average_ms": average_ms,
        "effective_gbps": effective_gbps,
        "max_state_error": state_error,
        "max_output_error": output_error,
        "hggc_vs_torch_max_state_error": hggc_state_error,
        "hggc_vs_torch_max_output_error": hggc_output_error,
        "passed": passed,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(args.device),
        "output_dtype": str(output.dtype),
        "state_dtype": str(timing_state.dtype),
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
