#!/usr/bin/env python3
"""Validate the one-entry packed MLP extension on one real Qwen3.5 module."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP

from ppu_acblas_packed_mlp import PPUACBLASPackedMLPExtension
from ppu_gdn import HIDDEN_SIZE, MLP_INTERMEDIATE_SIZE, pack_qwen35_mlp_module


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--gate-up-algorithm", type=int, default=-1)
    parser.add_argument("--down-algorithm", type=int, default=-1)
    parser.add_argument("--swiglu-threads", type=int, default=128)
    parser.add_argument("--workspace-mib", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()
    if args.workspace_mib < 0:
        raise ValueError("--workspace-mib must be non-negative")

    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    config = SimpleNamespace(hidden_size=HIDDEN_SIZE, hidden_act="silu")
    module = Qwen3_5MLP(config, MLP_INTERMEDIATE_SIZE).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    decode_input = torch.randn(1, 1, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)
    prefill_input = torch.randn(1, 4, HIDDEN_SIZE, device=device, dtype=torch.bfloat16)

    with torch.inference_mode():
        module.forward = pack_qwen35_mlp_module(module)
        baseline_forward = module.forward
        expected_decode = baseline_forward(decode_input).clone()
        expected_prefill = baseline_forward(prefill_input).clone()
        if args.workspace_mib:
            print("PHASE construct_workspace_extension", flush=True)
        extension = PPUACBLASPackedMLPExtension(
            args.build_dir,
            gate_up_algorithm=args.gate_up_algorithm,
            down_algorithm=args.down_algorithm,
            swiglu_threads=args.swiglu_threads,
            workspace_bytes=args.workspace_mib * 1024 * 1024,
            workspace_enabled=False,
        )
        extension.patch_module(module)
        if args.workspace_mib:
            print("PHASE workspace_disabled_first_call", flush=True)
        actual_decode_without_workspace = module(decode_input).clone()
        actual_prefill = module(prefill_input).clone()
        alternate_stream = torch.cuda.Stream(device=device)
        alternate_stream_rejected = False
        alternate_stream_error = ""
        with torch.cuda.stream(alternate_stream):
            try:
                module(decode_input)
            except RuntimeError as error:
                alternate_stream_error = str(error)
                alternate_stream_rejected = "bound to one CUDA stream" in str(error)
        baseline_ms = measure(lambda: baseline_forward(decode_input), args.warmup, args.iters)
        no_workspace_ms = measure(lambda: module(decode_input), args.warmup, args.iters)
        if args.workspace_mib:
            print("PHASE workspace_enable", flush=True)
            extension.set_workspace_enabled(True)
            print("PHASE workspace_enabled_first_call", flush=True)
        actual_decode = module(decode_input).clone()
        candidate_ms = measure(lambda: module(decode_input), args.warmup, args.iters)
        first_ptr = module(decode_input).data_ptr()
        second_ptr = module(decode_input).data_ptr()

    result = {
        "candidate": "one_entry_acblas_packed_mlp",
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "workspace_mib": args.workspace_mib,
        "no_workspace_ms": no_workspace_ms,
        "workspace_speedup": no_workspace_ms / candidate_ms,
        "workspace_exact": torch.equal(actual_decode_without_workspace, actual_decode),
        "decode_exact": torch.equal(expected_decode, actual_decode),
        "prefill_exact": torch.equal(expected_prefill, actual_prefill),
        "max_abs_error": float((expected_decode.float() - actual_decode.float()).abs().max()),
        "output_scratch_reused": first_ptr == second_ptr,
        "alternate_stream_rejected": alternate_stream_rejected,
        "alternate_stream_error": alternate_stream_error,
    }
    result["passed"] = bool(
        result["decode_exact"]
        and result["workspace_exact"]
        and result["prefill_exact"]
        and result["output_scratch_reused"]
        and result["alternate_stream_rejected"]
    )
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
