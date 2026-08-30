#!/usr/bin/env python3
"""Compare host/runtime events for acblasGemvEx and acblasGemmEx(n=1)."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


TARGET_KEYS = (
    "cudaGetDeviceProperties_v2",
    "cudaFree",
    "cudaLaunchKernel",
    "gemvt_op",
)


def bind(library: ctypes.CDLL, symbol: str):
    launch = getattr(library, symbol)
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    return launch


def profile_launch(
    name: str,
    launch,
    weight: torch.Tensor,
    input_vector: torch.Tensor,
    output: torch.Tensor,
    stream: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    def call() -> None:
        status = launch(
            weight.data_ptr(),
            input_vector.data_ptr(),
            output.data_ptr(),
            weight.shape[0],
            weight.shape[1],
            -1,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"{name} failed: {status}")

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        with torch.profiler.record_function(name):
            for _ in range(iters):
                call()
        torch.cuda.synchronize()

    events: dict[str, dict[str, float | int]] = {}
    for event in profiler.key_averages():
        if event.key in TARGET_KEYS:
            events[event.key] = {
                "count": int(event.count),
                "self_cpu_time_total_us": float(event.self_cpu_time_total),
                "self_device_time_total_us": float(
                    getattr(
                        event,
                        "self_device_time_total",
                        getattr(event, "self_cuda_time_total", 0.0),
                    )
                ),
            }
    return {"name": name, "iterations": iters, "events": events}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemv-library", type=Path, required=True)
    parser.add_argument("--gemm-library", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8224)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    gemv_library = ctypes.CDLL(str(args.gemv_library.resolve()))
    gemm_library = ctypes.CDLL(str(args.gemm_library.resolve()))
    gemv = bind(gemv_library, "seu_acblas_linear_bf16")
    gemm = bind(gemm_library, "seu_acblas_gemm_for_gemv_bf16")

    torch.manual_seed(20260829)
    device = torch.device("cuda:0")
    weight = torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16)
    input_vector = torch.randn(args.k, device=device, dtype=torch.bfloat16)
    gemv_output = torch.empty(args.n, device=device, dtype=torch.bfloat16)
    gemm_output = torch.empty_like(gemv_output)
    stream = torch.cuda.current_stream(device).cuda_stream

    gemv_result = profile_launch(
        "acblasGemvEx",
        gemv,
        weight,
        input_vector,
        gemv_output,
        stream,
        args.warmup,
        args.iters,
    )
    gemm_result = profile_launch(
        "acblasGemmEx-n1",
        gemm,
        weight,
        input_vector,
        gemm_output,
        stream,
        args.warmup,
        args.iters,
    )
    exact = torch.equal(gemv_output, gemm_output)
    print(
        "RESULT "
        + json.dumps(
            {"exact": exact, "gemv": gemv_result, "gemm": gemm_result},
            sort_keys=True,
        )
    )
    return 0 if exact else 2


if __name__ == "__main__":
    raise SystemExit(main())
