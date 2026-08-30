#!/usr/bin/env python3
"""Validate grouped QKV GEMV + q/k RMSNorm+RoPE on a real attention module."""

from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from ppu_acblas_attention_prep import PPUACBLASAttentionPrepExtension
from ppu_gdn import PPUGDNLibrary


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


def clone_tuple(values):
    return tuple(value.clone() for value in values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--gdn-library", required=True)
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--algorithm", type=int, default=-1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=400)
    args = parser.parse_args()

    torch.manual_seed(20260828)
    device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    text_config = getattr(config, "text_config", config)
    module = Qwen3_5Attention(text_config, layer_idx=0).to(
        device=device, dtype=torch.bfloat16
    ).eval()
    library = PPUGDNLibrary(args.gdn_library)
    hidden_states = torch.randn(
        1, 1, 2048, device=device, dtype=torch.bfloat16
    )
    prefill_states = torch.randn(
        1, 4, 2048, device=device, dtype=torch.bfloat16
    )
    cosine = torch.randn(1, 1, 64, device=device, dtype=torch.bfloat16)
    sine = torch.randn(1, 1, 64, device=device, dtype=torch.bfloat16)

    def baseline():
        query, gate = torch.chunk(
            module.q_proj(hidden_states).view(1, 1, -1, 512), 2, dim=-1
        )
        query = query.view(1, 1, -1, 256)
        key = module.k_proj(hidden_states).view(1, 1, -1, 256)
        value = module.v_proj(hidden_states).view(1, 1, -1, 256).transpose(1, 2)
        query, key = library.qk_rmsnorm_rope_decode(
            query,
            key,
            module.q_norm.weight,
            module.k_norm.weight,
            cosine,
            sine,
            module.q_norm.eps,
        )
        return query, key, value, gate

    with torch.inference_mode():
        expected = clone_tuple(baseline())
        expected_prefill = (
            module.q_proj(prefill_states).clone(),
            module.k_proj(prefill_states).clone(),
            module.v_proj(prefill_states).clone(),
        )
        extension = PPUACBLASAttentionPrepExtension(
            args.build_dir, algorithm=args.algorithm
        )
        candidate = extension.patch_module(module)
        actual = clone_tuple(candidate(hidden_states, cosine, sine))
        scratch_pointers = tuple(value.data_ptr() for value in candidate(hidden_states, cosine, sine))
        scratch_reused = scratch_pointers == tuple(
            value.data_ptr() for value in candidate(hidden_states, cosine, sine)
        )
        prefill_falls_back = candidate(
            prefill_states,
            cosine.expand(1, prefill_states.shape[1], -1),
            sine.expand(1, prefill_states.shape[1], -1),
        ) is None
        actual_prefill = (
            module.q_proj(prefill_states).clone(),
            module.k_proj(prefill_states).clone(),
            module.v_proj(prefill_states).clone(),
        )
        alternate_stream = torch.cuda.Stream(device=device)
        alternate_stream_rejected = False
        alternate_stream_error = ""
        with torch.cuda.stream(alternate_stream):
            try:
                candidate(hidden_states, cosine, sine)
            except RuntimeError as error:
                alternate_stream_error = str(error)
                alternate_stream_rejected = "bound to one CUDA stream" in str(error)
        baseline_ms = measure(baseline, args.warmup, args.iters)
        candidate_ms = measure(
            lambda: candidate(hidden_states, cosine, sine),
            args.warmup,
            args.iters,
        )

    labels = ("query", "key", "value", "gate")
    exact = {
        label: torch.equal(reference, output)
        for label, reference, output in zip(labels, expected, actual)
    }
    prefill_exact = all(
        torch.equal(reference, output)
        for reference, output in zip(expected_prefill, actual_prefill)
    )
    max_abs_error = max(
        float((reference.float() - output.float()).abs().max())
        for reference, output in zip(expected, actual)
    )
    result = {
        "candidate": "grouped_acblas_attention_prep",
        "baseline_ms": baseline_ms,
        "candidate_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "exact": exact,
        "prefill_exact": prefill_exact,
        "prefill_falls_back": prefill_falls_back,
        "scratch_reused": scratch_reused,
        "max_abs_error": max_abs_error,
        "alternate_stream_rejected": alternate_stream_rejected,
        "alternate_stream_error": alternate_stream_error,
    }
    result["passed"] = bool(
        all(exact.values())
        and prefill_exact
        and prefill_falls_back
        and scratch_reused
        and alternate_stream_rejected
    )
    print("RESULT " + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
