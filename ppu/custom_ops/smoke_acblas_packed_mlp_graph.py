#!/usr/bin/env python3
"""Test HGGC graph capture around the verified one-entry packed MLP path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppu_acblas_packed_mlp import PPUACBLASPackedMLPExtension
from ppu_gdn import HIDDEN_SIZE, MLP_INTERMEDIATE_SIZE, pack_qwen35_mlp_module


def median_ms(callable_, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    config = SimpleNamespace(hidden_size=HIDDEN_SIZE, hidden_act="silu")
    module = Qwen3_5MLP(config, MLP_INTERMEDIATE_SIZE).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    static_input = torch.randn(1, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    result: dict[str, object] = {"candidate": "graph_captured_acblas_packed_mlp"}
    with torch.inference_mode():
        module.forward = pack_qwen35_mlp_module(module)
        baseline_forward = module.forward
        expected = baseline_forward(static_input).clone()

        extension = PPUACBLASPackedMLPExtension(args.build_dir)
        packed_forward = extension.patch_module(module)
        try:
            graph_forward = extension.capture_module_graph(
                module, static_input, warmup=args.warmup
            )
        except Exception as exc:
            result.update(
                capture_supported=False,
                blocker=f"{type(exc).__name__}: {exc}",
                passed=False,
            )
            print("RESULT " + json.dumps(result, sort_keys=True))
            return 1

        replacement = torch.randn_like(static_input)
        static_input.copy_(replacement)
        captured = graph_forward(static_input).clone()
        torch.cuda.synchronize()
        reference = baseline_forward(replacement)
        torch.cuda.synchronize()

        dynamic_input = torch.randn_like(static_input)
        dynamic_expected = baseline_forward(dynamic_input).clone()
        dynamic_actual = graph_forward(dynamic_input).clone()
        prefill_input = torch.randn(
            1, 4, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
        )
        prefill_expected = baseline_forward(prefill_input).clone()
        prefill_actual = graph_forward(prefill_input).clone()
        alternate_stream = torch.cuda.Stream(device=device)
        alternate_stream_rejected = False
        with torch.cuda.stream(alternate_stream):
            try:
                graph_forward(static_input)
            except RuntimeError as exc:
                alternate_stream_rejected = "bound to one CUDA stream" in str(exc)

        def eager_candidate():
            packed_forward(static_input)

        def graph_candidate_with_copy():
            static_input.copy_(dynamic_input)
            graph_forward(static_input)

        eager_ms = median_ms(eager_candidate, args.repeats)
        graph_ms = median_ms(lambda: graph_forward(static_input), args.repeats)
        graph_with_copy_ms = median_ms(graph_candidate_with_copy, args.repeats)
        exact = torch.equal(captured, reference)
        result.update(
            capture_supported=True,
            updated_input_exact=exact,
            max_abs_error=float((captured.float() - reference.float()).abs().max()),
            eager_candidate_median_ms=eager_ms,
            graph_replay_median_ms=graph_ms,
            graph_over_eager_speedup=eager_ms / graph_ms,
            graph_with_input_copy_median_ms=graph_with_copy_ms,
            graph_with_input_copy_speedup=eager_ms / graph_with_copy_ms,
            dynamic_pointer_fallback_exact=torch.equal(
                dynamic_expected, dynamic_actual
            ),
            prefill_fallback_exact=torch.equal(prefill_expected, prefill_actual),
            alternate_stream_rejected=alternate_stream_rejected,
            passed=(
                exact
                and torch.equal(dynamic_expected, dynamic_actual)
                and torch.equal(prefill_expected, prefill_actual)
                and alternate_stream_rejected
            ),
        )
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
