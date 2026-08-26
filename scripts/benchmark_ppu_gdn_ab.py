#!/usr/bin/env python3
"""A/B the fused PPU GDN decode operator in the full Qwen3.5 VLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OP_DIR = REPO_ROOT / "ppu" / "custom_ops"
DEFAULT_PPU_SDK = Path("/usr/local/PPU_SDK")
if DEFAULT_PPU_SDK.is_dir():
    os.environ.setdefault("PPU_SDK", str(DEFAULT_PPU_SDK))
    os.environ.setdefault("PPU_HOME", str(DEFAULT_PPU_SDK))
for path in (REPO_ROOT, CUSTOM_OP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_public import (
    build_prompt,
    compute_throughput,
    decode_image,
    extract_answer,
    fixed_generation_config,
    load_mmbench_tsv,
    settle_runtime,
)
from evaluation_wrapper import VLMModel
from ppu_gdn import PPUGDNLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--gdn-tiles", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--fuse-conv", action="store_true")
    parser.add_argument("--conv-threads", type=int, default=96)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    parser.add_argument("--rmsnorm-threads", type=int, default=512)
    parser.add_argument("--fuse-gated-rmsnorm", action="store_true")
    parser.add_argument("--gated-rmsnorm-threads", type=int, default=128)
    parser.add_argument("--fuse-qk-rope", action="store_true")
    return parser.parse_args()


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "repeats": len(records),
        "median_ttft_ms": statistics.median(float(r["ttft_ms"]) for r in records),
        "median_elapsed_ms": statistics.median(float(r["elapsed_ms"]) for r in records),
        "median_throughput_tokens_per_sec": statistics.median(
            float(r["throughput_tokens_per_sec"]) for r in records
        ),
        "records": records,
    }


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.max_new_tokens <= 0 or args.sample_offset < 0:
        raise ValueError("repeats/max-new-tokens must be positive and sample-offset nonnegative")

    samples = load_mmbench_tsv(args.dataset_path, limit=args.sample_offset + 1)
    sample = samples[args.sample_offset]
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens
    image = decode_image(sample.image_b64)
    prompt = build_prompt(sample)

    def run_once(mode: str) -> dict[str, object]:
        settle_runtime(model)
        result = model.generate_with_metrics(
            image=image,
            prompt=prompt,
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        return {
            "mode": mode,
            "token_count": result.token_count,
            "ttft_ms": result.ttft_seconds * 1000.0,
            "elapsed_ms": result.elapsed_seconds * 1000.0,
            "throughput_tokens_per_sec": compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            ),
            "answer": extract_answer(result.text),
            "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        }

    run_once("eager_warmup")
    eager_records = [run_once("eager") for _ in range(args.repeats)]

    library = PPUGDNLibrary(
        args.library,
        tiles_per_head=args.gdn_tiles,
        conv_threads=args.conv_threads,
        rmsnorm_threads=args.rmsnorm_threads,
        gated_rmsnorm_threads=args.gated_rmsnorm_threads,
    )
    fused_callable = library.transformers_callable()
    patched_modules = []
    for module in model._model.modules():
        if type(module).__name__ == "Qwen3_5GatedDeltaNet":
            module.recurrent_gated_delta_rule = fused_callable
            if args.fuse_conv:
                module.causal_conv1d_update = library.causal_conv1d_decode
            patched_modules.append(module)
    if len(patched_modules) != 18:
        raise RuntimeError(f"expected 18 Qwen3.5 GDN modules, patched {len(patched_modules)}")
    patched_qk_rope_modules = 0
    if args.fuse_qk_rope:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling

        for module in model._model.modules():
            if type(module).__name__ != "Qwen3_5Attention":
                continue
            module.forward = library.transformers_attention_callable(
                module,
                qwen35_modeling.ALL_ATTENTION_FUNCTIONS,
                qwen35_modeling.eager_attention_forward,
            )
            patched_qk_rope_modules += 1
        if patched_qk_rope_modules != 6:
            raise RuntimeError(
                "expected 6 Qwen3.5 attention modules, "
                f"patched {patched_qk_rope_modules}"
            )
    patched_rmsnorm_modules = 0
    if args.fuse_rmsnorm:
        for module_name, module in model._model.named_modules():
            if (
                type(module).__name__ != "Qwen3_5RMSNorm"
                or not module_name.startswith("model.language_model")
                or module.weight.numel() != 2048
            ):
                continue
            eager_forward = module.forward

            def decode_rmsnorm(x, *, _module=module, _eager=eager_forward):
                if x.ndim >= 2 and x.shape[-2] == 1 and x.shape[-1] == 2048:
                    return library.rmsnorm_decode(x, _module.weight, _module.eps)
                return _eager(x)

            module.forward = decode_rmsnorm
            patched_rmsnorm_modules += 1
        if patched_rmsnorm_modules != 49:
            raise RuntimeError(
                f"expected 49 Qwen3.5 RMSNorm modules, patched {patched_rmsnorm_modules}"
            )
    patched_gated_rmsnorm_modules = 0
    if args.fuse_gated_rmsnorm:
        for module in model._model.modules():
            if type(module).__name__ != "Qwen3_5RMSNormGated":
                continue
            eager_forward = module.forward

            def decode_gated_rmsnorm(
                hidden_states, gate=None, *, _module=module, _eager=eager_forward
            ):
                if (
                    gate is not None
                    and hidden_states.ndim == 2
                    and hidden_states.shape == (16, 128)
                    and gate.shape == hidden_states.shape
                ):
                    return library.gated_rmsnorm_decode(
                        hidden_states, gate, _module.weight, _module.variance_epsilon
                    )
                return _eager(hidden_states, gate)

            module.forward = decode_gated_rmsnorm
            patched_gated_rmsnorm_modules += 1
        if patched_gated_rmsnorm_modules != 18:
            raise RuntimeError(
                "expected 18 Qwen3.5 gated RMSNorm modules, "
                f"patched {patched_gated_rmsnorm_modules}"
            )

    run_once("fused_warmup")
    fused_records = [run_once("fused") for _ in range(args.repeats)]
    eager = summarize(eager_records)
    fused = summarize(fused_records)
    eager_hashes = {str(record["text_sha256"]) for record in eager_records}
    fused_hashes = {str(record["text_sha256"]) for record in fused_records}
    exact_output_match = len(eager_hashes) == 1 and eager_hashes == fused_hashes
    payload = {
        "sample_id": sample.sample_id,
        "sample_offset": args.sample_offset,
        "max_new_tokens": args.max_new_tokens,
        "patched_gdn_modules": len(patched_modules),
        "gdn_tiles_per_head": args.gdn_tiles,
        "fused_causal_conv_modules": len(patched_modules) if args.fuse_conv else 0,
        "conv_threads": args.conv_threads if args.fuse_conv else None,
        "fused_rmsnorm_modules": patched_rmsnorm_modules,
        "rmsnorm_threads": args.rmsnorm_threads if args.fuse_rmsnorm else None,
        "fused_gated_rmsnorm_modules": patched_gated_rmsnorm_modules,
        "gated_rmsnorm_threads": (
            args.gated_rmsnorm_threads if args.fuse_gated_rmsnorm else None
        ),
        "fused_qk_rmsnorm_rope_modules": patched_qk_rope_modules,
        "eager": eager,
        "fused": fused,
        "speedup": {
            "ttft": float(eager["median_ttft_ms"]) / float(fused["median_ttft_ms"]),
            "elapsed": float(eager["median_elapsed_ms"]) / float(fused["median_elapsed_ms"]),
            "decode_throughput": float(fused["median_throughput_tokens_per_sec"])
            / float(eager["median_throughput_tokens_per_sec"]),
        },
        "exact_output_match": exact_output_match,
        "passed": exact_output_match,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
