#!/usr/bin/env python3
"""Compare acblasGemmEx(m=N,n=1,k=K) with torch.mv on decode shapes."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


DEFAULT_ALGORITHMS = "-3,-2,-1," + ",".join(str(value) for value in range(24)) + ",99"


def measure(callable_, warmup: int, iters: int) -> float:
    for iteration in range(warmup):
        status = callable_(iteration)
        if status != 0:
            raise RuntimeError(f"warmup launch failed: {status}")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iters):
        status = callable_(iteration)
        if status != 0:
            raise RuntimeError(f"timed launch failed: {status}")
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--n", type=int, default=8224)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--matrix-copies", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--algorithms", default=DEFAULT_ALGORITHMS)
    args = parser.parse_args()

    library = ctypes.CDLL(str(args.library.resolve()))
    launch = library.seu_acblas_gemm_for_gemv_bf16
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

    torch.manual_seed(20260829)
    device = torch.device("cuda:0")
    input_vector = torch.randn(args.k, device=device, dtype=torch.bfloat16)
    weights = [
        torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16)
        for _ in range(args.matrix_copies)
    ]
    output = torch.empty(args.n, device=device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(device).cuda_stream
    reference = torch.mv(weights[0], input_vector)

    with torch.inference_mode():
        torch_ms = measure(
            lambda iteration: (
                torch.mv(weights[iteration % args.matrix_copies], input_vector),
                0,
            )[1],
            args.warmup,
            args.iters,
        )
        print(
            "RESULT "
            + json.dumps(
                {
                    "backend": "torch.mv",
                    "n": args.n,
                    "k": args.k,
                    "average_ms": torch_ms,
                    "passed": True,
                },
                sort_keys=True,
            )
        )

        for algorithm in (int(value) for value in args.algorithms.split(",")):

            def run(iteration: int) -> int:
                return launch(
                    weights[iteration % args.matrix_copies].data_ptr(),
                    input_vector.data_ptr(),
                    output.data_ptr(),
                    args.n,
                    args.k,
                    algorithm,
                    stream,
                )

            status = run(0)
            if status != 0:
                print(
                    "RESULT "
                    + json.dumps(
                        {
                            "backend": "acblasGemmEx-n1",
                            "algorithm": algorithm,
                            "status": status,
                            "passed": False,
                        },
                        sort_keys=True,
                    )
                )
                continue
            torch.cuda.synchronize()
            exact = torch.equal(output, reference)
            average_ms = measure(run, args.warmup, args.iters)
            print(
                "RESULT "
                + json.dumps(
                    {
                        "backend": "acblasGemmEx-n1",
                        "algorithm": algorithm,
                        "n": args.n,
                        "k": args.k,
                        "average_ms": average_ms,
                        "torch_speedup": torch_ms / average_ms,
                        "exact": exact,
                        "passed": exact,
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
