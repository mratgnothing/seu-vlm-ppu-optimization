#!/usr/bin/env python3
"""Numerical and timing gate for the Qwen3.5 decode GDN gate-prep kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from ppu_gdn import HEADS, PPUGDNLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def reference(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    exp_a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    beta = raw_b.sigmoid()
    g = -exp_a_log * F.softplus(raw_a.float() + dt_bias)
    return g, beta


def elapsed_ms(callable_, iters: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        callable_()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def compare_case(
    library: PPUGDNLibrary,
    name: str,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    exp_a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> dict[str, object]:
    expected_g, expected_beta = reference(raw_a, raw_b, exp_a_log, dt_bias)
    actual_g, actual_beta = library.gate_prep_decode(
        raw_a, raw_b, exp_a_log, dt_bias
    )
    torch.cuda.synchronize()
    g_abs = (actual_g - expected_g).abs()
    return {
        "name": name,
        "g_exact": bool(torch.equal(actual_g, expected_g)),
        "g_max_abs_error": float(g_abs.max().item()),
        "g_mean_abs_error": float(g_abs.mean().item()),
        "beta_exact": bool(torch.equal(actual_beta, expected_beta)),
        "finite": bool(torch.isfinite(actual_g).all().item()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260828)
    device = torch.device("cuda")
    library = PPUGDNLibrary(args.library)

    raw_a = (torch.randn(1, 1, HEADS, device=device) * 3).to(torch.bfloat16)
    raw_b = (torch.randn(1, 1, HEADS, device=device) * 3).to(torch.bfloat16)
    a_log = (torch.rand(HEADS, device=device) * 2.75).to(torch.bfloat16)
    exp_a_log = a_log.float().exp().contiguous()
    dt_bias = (torch.randn(HEADS, device=device) * 2).to(torch.bfloat16)

    boundary_values = torch.tensor(
        [-30, -20, -10, -5, -2, -1, -0.0, 0.0, 0.5, 1, 2, 5, 10, 20, 25, 30],
        dtype=torch.bfloat16,
        device=device,
    ).reshape(1, 1, HEADS)
    cases = [
        compare_case(library, "random", raw_a, raw_b, exp_a_log, dt_bias),
        compare_case(
            library,
            "boundary",
            boundary_values,
            boundary_values.flip(-1).contiguous(),
            exp_a_log,
            torch.zeros_like(dt_bias),
        ),
    ]

    for _ in range(args.warmup):
        reference(raw_a, raw_b, exp_a_log, dt_bias)
        library.gate_prep_decode(raw_a, raw_b, exp_a_log, dt_bias)
    baseline_ms = [
        elapsed_ms(lambda: reference(raw_a, raw_b, exp_a_log, dt_bias), args.iters)
        for _ in range(args.repeats)
    ]
    candidate_ms = [
        elapsed_ms(
            lambda: library.gate_prep_decode(raw_a, raw_b, exp_a_log, dt_bias),
            args.iters,
        )
        for _ in range(args.repeats)
    ]
    baseline_median = statistics.median(baseline_ms)
    candidate_median = statistics.median(candidate_ms)
    result = {
        "kernel": "seu_ppu_gdn_gate_prep_decode_bf16",
        "seed": 20260828,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "cases": cases,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "speedup": baseline_median / candidate_median,
        "passed": all(
            case["g_exact"] and case["beta_exact"] and case["finite"]
            for case in cases
        ),
    }
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
