#!/usr/bin/env python3
"""Compare fused Qwen3.5 decode q/k RMSNorm+RoPE with eager semantics."""

from __future__ import annotations

import argparse
import json

import torch

from ppu_gdn import (
    ATTENTION_HEAD_DIM,
    ATTENTION_HEADS,
    KEY_VALUE_HEADS,
    ROTARY_DIM,
    PPUGDNLibrary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    return parser.parse_args()


def eager_rmsnorm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + epsilon)
    return (output * (1.0 + weight.float())).type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def eager_pipeline(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = eager_rmsnorm(query, query_weight, epsilon).transpose(1, 2)
    key = eager_rmsnorm(key, key_weight, epsilon).transpose(1, 2)
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)

    def apply(tensor: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = tensor[..., :ROTARY_DIM], tensor[..., ROTARY_DIM:]
        embedded = rotary * cosine + rotate_half(rotary) * sine
        return torch.cat((embedded, passthrough), dim=-1)

    return apply(query), apply(key)


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
    query_and_gate = torch.randn(
        1, 1, ATTENTION_HEADS, ATTENTION_HEAD_DIM * 2,
        device=device, dtype=torch.bfloat16,
    )
    query, _ = torch.chunk(query_and_gate, 2, dim=-1)
    key = torch.randn(
        1, 1, KEY_VALUE_HEADS, ATTENTION_HEAD_DIM,
        device=device, dtype=torch.bfloat16,
    )
    query_weight = torch.randn(
        ATTENTION_HEAD_DIM, device=device, dtype=torch.bfloat16
    ) / 16
    key_weight = torch.randn(
        ATTENTION_HEAD_DIM, device=device, dtype=torch.bfloat16
    ) / 16
    angles = torch.randn(1, 1, ROTARY_DIM, device=device, dtype=torch.float32)
    cosine = angles.cos().to(torch.bfloat16)
    sine = angles.sin().to(torch.bfloat16)
    epsilon = 1.0e-6
    library = PPUGDNLibrary(args.library)

    with torch.inference_mode():
        expected_query, expected_key = eager_pipeline(
            query, key, query_weight, key_weight, cosine, sine, epsilon
        )
        actual_query, actual_key = library.qk_rmsnorm_rope_decode(
            query, key, query_weight, key_weight, cosine, sine, epsilon
        )
        torch.cuda.synchronize()
        query_error = float(
            (actual_query.float() - expected_query.float()).abs().max().item()
        )
        key_error = float(
            (actual_key.float() - expected_key.float()).abs().max().item()
        )
        eager_ms = measure(
            lambda: eager_pipeline(
                query, key, query_weight, key_weight, cosine, sine, epsilon
            ),
            args.warmup,
            args.iters,
        )
        fused_ms = measure(
            lambda: library.qk_rmsnorm_rope_decode(
                query, key, query_weight, key_weight, cosine, sine, epsilon
            ),
            args.warmup,
            args.iters,
        )

    passed = query_error == 0.0 and key_error == 0.0
    result = {
        "kernel": "seu_ppu_qk_rmsnorm_rope_decode",
        "eager_ms": eager_ms,
        "fused_ms": fused_ms,
        "speedup": eager_ms / fused_ms,
        "max_query_error": query_error,
        "max_key_error": key_error,
        "passed": passed,
    }
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
