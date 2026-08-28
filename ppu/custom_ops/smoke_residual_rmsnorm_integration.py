#!/usr/bin/env python3
"""Validate fused BF16 residual add + Qwen3.5 RMSNorm on PPU."""

from __future__ import annotations

import argparse
import json

import torch

from ppu_gdn import PPUGDNLibrary


HIDDEN_SIZE = 2048


def qwen35_rmsnorm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    output = input_tensor.float()
    variance = output.pow(2).mean(-1, keepdim=True)
    output = output * torch.rsqrt(variance + epsilon)
    output = output * (1.0 + weight.float())
    return output.to(input_tensor.dtype)


def measure(callable_, updates: list[torch.Tensor], warmup: int, iters: int) -> float:
    for index in range(warmup):
        callable_(updates[index])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for index in range(warmup, warmup + iters):
        callable_(updates[index])
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, required=True)
    parser.add_argument("--threads", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()

    torch.manual_seed(20260827)
    device = torch.device("cuda:0")
    residual = torch.randn(1, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    update_source = torch.randn_like(residual)
    weight = torch.randn(HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.02
    epsilon = 1.0e-6
    library = PPUGDNLibrary(args.library, rmsnorm_threads=args.threads)
    normalized_scratch = torch.empty_like(residual)

    with torch.inference_mode():
        expected_residual = residual + update_source
        expected_normalized = qwen35_rmsnorm_reference(
            expected_residual, weight, epsilon
        )
        actual_update = update_source.clone()
        actual_residual, actual_normalized = library.residual_rmsnorm_decode(
            residual,
            actual_update,
            weight,
            epsilon,
            normalized_output=normalized_scratch,
        )
        baseline_updates = [
            update_source.clone() for _ in range(args.warmup + args.iters)
        ]
        candidate_updates = [
            update_source.clone() for _ in range(args.warmup + args.iters)
        ]
        scratch_updates = [
            update_source.clone() for _ in range(args.warmup + args.iters)
        ]
        raw_stream_updates = [
            update_source.clone() for _ in range(args.warmup + args.iters)
        ]

        def baseline(update: torch.Tensor) -> torch.Tensor:
            added = residual + update
            return library.rmsnorm_decode(added, weight, epsilon)

        def candidate(update: torch.Tensor) -> torch.Tensor:
            return library.residual_rmsnorm_decode(
                residual, update, weight, epsilon
            )[1]

        def scratch_candidate(update: torch.Tensor) -> torch.Tensor:
            return library.residual_rmsnorm_decode(
                residual,
                update,
                weight,
                epsilon,
                normalized_output=normalized_scratch,
            )[1]

        baseline_ms = measure(baseline, baseline_updates, args.warmup, args.iters)
        candidate_ms = measure(candidate, candidate_updates, args.warmup, args.iters)
        scratch_ms = measure(
            scratch_candidate, scratch_updates, args.warmup, args.iters
        )
        object_stream = torch.cuda.current_stream(device).cuda_stream
        raw_stream = torch._C._cuda_getCurrentRawStream(device.index or 0)
        library.set_raw_stream_query(True)
        raw_stream_update = update_source.clone()
        _, raw_stream_actual = library.residual_rmsnorm_decode(
            residual, raw_stream_update, weight, epsilon
        )
        raw_stream_ms = measure(
            candidate, raw_stream_updates, args.warmup, args.iters
        )
        library.set_raw_stream_query(False)
        first_ptr = scratch_candidate(update_source.clone()).data_ptr()
        second_ptr = scratch_candidate(update_source.clone()).data_ptr()

    residual_exact = torch.equal(expected_residual, actual_residual)
    normalized_exact = torch.equal(expected_normalized, actual_normalized)
    payload = {
        "candidate": "inplace_residual_add_rmsnorm",
        "threads": args.threads,
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "scratch_ms": scratch_ms,
        "scratch_over_candidate_speedup": candidate_ms / scratch_ms,
        "raw_stream_ms": raw_stream_ms,
        "raw_stream_query_speedup": candidate_ms / raw_stream_ms,
        "raw_stream_handle_matches": raw_stream == object_stream,
        "raw_stream_exact": torch.equal(expected_normalized, raw_stream_actual),
        "residual_exact": residual_exact,
        "normalized_exact": normalized_exact,
        "inplace_update": actual_residual.data_ptr() == actual_update.data_ptr(),
        "normalized_scratch_reused": first_ptr == second_ptr,
    }
    payload["passed"] = (
        residual_exact
        and normalized_exact
        and payload["inplace_update"]
        and payload["normalized_scratch_reused"]
        and payload["raw_stream_handle_matches"]
        and payload["raw_stream_exact"]
    )
    print("RESULT " + json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
