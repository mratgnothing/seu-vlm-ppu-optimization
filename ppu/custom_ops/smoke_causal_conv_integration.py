#!/usr/bin/env python3
"""Compare the fused PPU causal-conv decode update with Transformers eager."""

from __future__ import annotations

import argparse
import json

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_causal_conv1d_update

from ppu_gdn import CONV_CHANNELS, CONV_WIDTH, PPUGDNLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--threads", type=int, default=96)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def make_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(20260826)
    hidden = torch.randn(1, CONV_CHANNELS, 1, device=device, dtype=torch.bfloat16)
    state = torch.randn(1, CONV_CHANNELS, CONV_WIDTH, device=device, dtype=torch.bfloat16)
    weight = torch.randn(CONV_CHANNELS, CONV_WIDTH, device=device, dtype=torch.bfloat16) / 8
    bias = torch.randn(CONV_CHANNELS, device=device, dtype=torch.bfloat16) / 8
    return hidden, state, weight, bias


def measure(callable_, inputs: tuple[torch.Tensor, ...], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        callable_(*inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        callable_(*inputs)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0")
    hidden, initial_state, weight, bias = make_inputs(device)
    library = PPUGDNLibrary(args.library, conv_threads=args.threads)
    eager_state = initial_state.clone()
    fused_state = initial_state.clone()
    with torch.inference_mode():
        eager_output = torch_causal_conv1d_update(
            hidden, eager_state, weight, bias, "silu"
        )
        fused_output = library.causal_conv1d_decode(
            hidden, fused_state, weight, bias, "silu"
        )
        torch.cuda.synchronize()
        output_error = float((fused_output.float() - eager_output.float()).abs().max().item())
        state_error = float((fused_state.float() - eager_state.float()).abs().max().item())

        eager_ms = measure(
            lambda h, s, w, b: torch_causal_conv1d_update(h, s, w, b, "silu"),
            (hidden, initial_state.clone(), weight, bias), args.warmup, args.iters,
        )
        fused_ms = measure(
            lambda h, s, w, b: library.causal_conv1d_decode(h, s, w, b, "silu"),
            (hidden, initial_state.clone(), weight, bias), args.warmup, args.iters,
        )

    passed = output_error <= 7.8125e-3 and state_error == 0.0
    result = {
        "kernel": "seu_ppu_causal_conv1d_decode",
        "threads": args.threads,
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
