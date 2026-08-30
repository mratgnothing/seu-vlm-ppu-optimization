#!/usr/bin/env python3
"""Probe PyTorch/HGGC graph capture without assuming CUDA graph support works."""

from __future__ import annotations

import argparse
import json
import statistics
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=65536)
    parser.add_argument("--chain-depth", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    return parser.parse_args()


def synchronized_ms(torch, fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def main() -> int:
    args = parse_args()
    import torch

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "has_cuda_graph": hasattr(torch.cuda, "CUDAGraph"),
        "has_graph_context": hasattr(torch.cuda, "graph"),
        "raw_stream_apis": [
            name for name in dir(torch._C) if "RawStream" in name
        ],
        "elements": args.elements,
        "chain_depth": args.chain_depth,
    }
    if not torch.cuda.is_available():
        report.update(passed=False, blocker="torch.cuda.is_available() is false")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    if not report["has_cuda_graph"] or not report["has_graph_context"]:
        report.update(passed=False, blocker="PyTorch graph API is absent")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    device = torch.device("cuda")
    static_input = torch.linspace(-1.0, 1.0, args.elements, device=device).to(
        torch.bfloat16
    )
    bias = torch.full_like(static_input, 0.03125)

    def workload():
        value = static_input
        for _ in range(args.chain_depth):
            value = torch.nn.functional.silu(value + bias)
        return value

    for _ in range(args.warmup):
        workload()
    torch.cuda.synchronize()

    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = workload()
    except Exception as exc:  # the probe must preserve the runtime's exact blocker
        report.update(
            passed=False,
            capture_supported=False,
            blocker=f"{type(exc).__name__}: {exc}",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    replacement = torch.linspace(1.0, -1.0, args.elements, device=device).to(
        torch.bfloat16
    )
    static_input.copy_(replacement)
    graph.replay()
    torch.cuda.synchronize()
    captured = static_output.clone()
    reference = workload()
    torch.cuda.synchronize()
    exact = torch.equal(captured, reference)
    max_abs_error = float((captured.float() - reference.float()).abs().max().item())

    eager_samples = synchronized_ms(torch, workload, args.repeats)
    graph_samples = synchronized_ms(torch, graph.replay, args.repeats)
    eager_median = statistics.median(eager_samples)
    graph_median = statistics.median(graph_samples)
    report.update(
        capture_supported=True,
        replay_uses_updated_input=exact,
        max_abs_error=max_abs_error,
        eager_median_ms=eager_median,
        graph_median_ms=graph_median,
        microbench_speedup=eager_median / graph_median,
        passed=exact,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
