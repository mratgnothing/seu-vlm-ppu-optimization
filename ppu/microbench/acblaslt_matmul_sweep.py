#!/usr/bin/env python3
"""Compare acBLASLt BF16 heuristics with torch.mv for decode shapes."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--matrix-copies", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--requested-algorithms", type=int, default=32)
    parser.add_argument("--max-workspace-mib", type=int, default=64)
    return parser.parse_args()


def measure(callable_, warmup: int, iters: int) -> float:
    for iteration in range(warmup):
        callable_(iteration)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for iteration in range(iters):
        callable_(iteration)
    stop.record()
    stop.synchronize()
    return float(start.elapsed_time(stop)) / iters


def main() -> int:
    args = parse_args()
    library = ctypes.CDLL(str(args.library.resolve()))
    prepare = library.seu_acblaslt_prepare_bf16
    prepare.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    prepare.restype = ctypes.c_int
    info = library.seu_acblaslt_heuristic_info
    info.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_float),
    ]
    info.restype = ctypes.c_int
    launch = library.seu_acblaslt_matmul_bf16
    launch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int

    max_workspace_bytes = args.max_workspace_mib * 1024 * 1024
    count = ctypes.c_int()
    status = prepare(
        args.n,
        args.k,
        max_workspace_bytes,
        args.requested_algorithms,
        ctypes.byref(count),
    )
    if status != 0:
        raise RuntimeError(f"acBLASLt prepare failed: {status}")

    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    input_vector = torch.randn(args.k, device=device, dtype=torch.bfloat16)
    weights = [
        torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16)
        for _ in range(args.matrix_copies)
    ]
    output = torch.empty(args.n, device=device, dtype=torch.bfloat16)
    workspace = torch.empty(max_workspace_bytes, device=device, dtype=torch.uint8)
    stream = torch.cuda.current_stream(device).cuda_stream
    reference = torch.mv(weights[0], input_vector)

    with torch.inference_mode():
        torch_ms = measure(
            lambda iteration: torch.mv(
                weights[iteration % args.matrix_copies], input_vector
            ),
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
                    "matrix_copies": args.matrix_copies,
                    "average_ms": torch_ms,
                    "passed": True,
                },
                sort_keys=True,
            )
        )

        for index in range(count.value):
            workspace_bytes = ctypes.c_size_t()
            waves_count = ctypes.c_float()
            info_status = info(
                index, ctypes.byref(workspace_bytes), ctypes.byref(waves_count)
            )

            def run(iteration: int) -> int:
                return launch(
                    weights[iteration % args.matrix_copies].data_ptr(),
                    input_vector.data_ptr(),
                    output.data_ptr(),
                    index,
                    workspace.data_ptr(),
                    max_workspace_bytes,
                    stream,
                )

            run_status = run(0)
            if info_status != 0 or run_status != 0:
                print(
                    "RESULT "
                    + json.dumps(
                        {
                            "backend": "acblasLtMatmul",
                            "heuristic_index": index,
                            "info_status": info_status,
                            "run_status": run_status,
                            "passed": False,
                        },
                        sort_keys=True,
                    )
                )
                continue
            torch.cuda.synchronize()
            exact = torch.equal(output, reference)
            max_abs_error = float(
                (output.float() - reference.float()).abs().max().item()
            )
            average_ms = measure(run, args.warmup, args.iters)
            print(
                "RESULT "
                + json.dumps(
                    {
                        "backend": "acblasLtMatmul",
                        "heuristic_index": index,
                        "n": args.n,
                        "k": args.k,
                        "average_ms": average_ms,
                        "torch_speedup": torch_ms / average_ms,
                        "workspace_bytes": workspace_bytes.value,
                        "waves_count": waves_count.value,
                        "exact": exact,
                        "max_abs_error": max_abs_error,
                        "passed": max_abs_error <= 0.125,
                    },
                    sort_keys=True,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
