#!/usr/bin/env python3
"""Paired multi-sample A/B for GDN projection or residual-RMSNorm fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--gdn-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Fail the gate unless paired median and mean throughput both improve",
    )
    parser.add_argument(
        "--pair-log",
        type=Path,
        help="Optional append-only JSONL checkpoint written after every completed pair",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--group-sizes",
        default="4",
        help="Comma-separated consecutive projection groups, for example 2,1,1",
    )
    parser.add_argument(
        "--projection-backend",
        choices=("torch-packed", "acblas-grouped"),
        default="torch-packed",
    )
    parser.add_argument("--acblas-build-dir", type=Path)
    parser.add_argument("--acblas-packed-mlp-build-dir", type=Path)
    parser.add_argument("--acblas-packed-mlp-swiglu-threads", type=int, default=128)
    parser.add_argument("--acblas-attention-prep-build-dir", type=Path)
    parser.add_argument("--acblas-attention-prep-algorithm", type=int, default=-1)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument(
        "--residual-rmsnorm-ab",
        action="store_true",
        help="Keep the selected GDN projection backend on and A/B residual RMSNorm",
    )
    targets.add_argument(
        "--residual-rmsnorm-scratch-ab",
        action="store_true",
        help="Keep fused residual RMSNorm on and A/B persistent output scratch",
    )
    targets.add_argument(
        "--raw-stream-query-ab",
        action="store_true",
        help="Keep the complete optimized stack on and A/B raw stream lookup",
    )
    targets.add_argument(
        "--gate-prep-ab",
        action="store_true",
        help="Keep projections/residual RMSNorm on and A/B GDN gate preparation",
    )
    targets.add_argument(
        "--acblas-packed-mlp-ab",
        action="store_true",
        help="Keep gate-prep stack on and A/B the one-entry packed MLP path",
    )
    targets.add_argument(
        "--acblas-attention-prep-ab",
        action="store_true",
        help="Keep the packed-MLP stack on and A/B grouped attention prep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    group_sizes = tuple(int(value) for value in args.group_sizes.split(","))
    repo_root = args.repo_root.resolve()
    custom_op_dir = repo_root / "ppu" / "custom_ops"
    if not custom_op_dir.is_dir():
        custom_op_dir = Path(__file__).resolve().parent
    for path in (repo_root, custom_op_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.update(
        {
            "PPU_SDK": "/usr/local/PPU_SDK",
            "PPU_HOME": "/usr/local/PPU_SDK",
            "SEU_PPU_GDN_LIBRARY": str(args.gdn_library.resolve()),
            "SEU_PPU_GDN_PYTHON_DIR": str(custom_op_dir),
            "SEU_PPU_GDN_TILES": "4",
            "SEU_PPU_CONV_ENABLE": "1",
            "SEU_PPU_CONV_THREADS": "96",
            "SEU_PPU_RMSNORM_ENABLE": "1",
            "SEU_PPU_RMSNORM_THREADS": "512",
            "SEU_PPU_GATED_RMSNORM_ENABLE": "1",
            "SEU_PPU_GATED_RMSNORM_THREADS": "128",
            "SEU_PPU_QK_ROPE_ENABLE": "1",
            "SEU_PPU_PACK_MLP_ENABLE": "1",
        }
    )

    import torch

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
    from ppu_gdn_projection_pack import set_packed_qwen35_gdn_input_projections

    all_samples = load_mmbench_tsv(
        args.dataset_path,
        limit=args.sample_offset + args.num_samples,
    )
    samples = all_samples[args.sample_offset :]
    if len(samples) != args.num_samples:
        raise ValueError(f"requested {args.num_samples} samples, got {len(samples)}")

    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens
    packed_modules = []
    if args.projection_backend == "acblas-grouped":
        if args.acblas_build_dir is None:
            raise ValueError("--acblas-build-dir is required for acblas-grouped")
        if group_sizes != (4,):
            raise ValueError("acblas-grouped implements the complete four-way group")
        from ppu_acblas_gdn_projection import PPUACBLASGDNProjectionExtension

        projection_packer = PPUACBLASGDNProjectionExtension(
            args.acblas_build_dir
        ).pack_module
    else:
        from ppu_gdn_projection_pack import pack_qwen35_gdn_input_projections

        def projection_packer(module):
            return pack_qwen35_gdn_input_projections(
                module, group_sizes=group_sizes
            )
    for module in model._model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        projection_packer(module)
        packed_modules.append(module)
    if len(packed_modules) != 18:
        raise RuntimeError(f"expected 18 packed GDN modules, got {len(packed_modules)}")
    residual_modules = []
    if (
        args.residual_rmsnorm_ab
        or args.residual_rmsnorm_scratch_ab
        or args.raw_stream_query_ab
        or args.gate_prep_ab
        or args.acblas_packed_mlp_ab
        or args.acblas_attention_prep_ab
    ):
        from ppu_gdn import (
            pack_qwen35_decoder_residual_rmsnorm,
            set_qwen35_decoder_residual_rmsnorm,
            set_qwen35_decoder_residual_rmsnorm_scratch,
        )

        decoder_modules = [
            module
            for module in model._model.modules()
            if type(module).__name__ == "Qwen3_5DecoderLayer"
        ]
        final_norm = model._model.model.language_model.norm
        for index, module in enumerate(decoder_modules):
            next_norm = (
                decoder_modules[index + 1].input_layernorm
                if index + 1 < len(decoder_modules)
                else final_norm
            )
            pack_qwen35_decoder_residual_rmsnorm(
                module, model._ppu_gdn_library, next_norm=next_norm
            )
            residual_modules.append(module)
        if len(residual_modules) != 24:
            raise RuntimeError(
                f"expected 24 residual RMSNorm modules, got {len(residual_modules)}"
            )
    gate_prep_modules = []
    if (
        args.gate_prep_ab
        or args.residual_rmsnorm_scratch_ab
        or args.raw_stream_query_ab
        or args.acblas_packed_mlp_ab
        or args.acblas_attention_prep_ab
    ):
        from ppu_gdn import pack_qwen35_gdn_gate_prep

        for module in packed_modules:
            pack_qwen35_gdn_gate_prep(module, model._ppu_gdn_library)
            gate_prep_modules.append(module)
        if len(gate_prep_modules) != 18:
            raise RuntimeError(
                f"expected 18 GDN gate-prep modules, got {len(gate_prep_modules)}"
            )
    acblas_packed_mlp_modules = []
    if (
        args.acblas_packed_mlp_ab
        or args.acblas_attention_prep_ab
        or args.residual_rmsnorm_scratch_ab
        or args.raw_stream_query_ab
    ):
        if args.acblas_packed_mlp_build_dir is None:
            raise ValueError(
                "--acblas-packed-mlp-build-dir is required for "
                "the packed-MLP or attention-prep A/B target"
            )
        from ppu_acblas_packed_mlp import PPUACBLASPackedMLPExtension

        mlp_extension = PPUACBLASPackedMLPExtension(
            args.acblas_packed_mlp_build_dir,
            swiglu_threads=args.acblas_packed_mlp_swiglu_threads,
        )
        for module in model._model.modules():
            if type(module).__name__ == "Qwen3_5MLP":
                mlp_extension.patch_module(module)
                acblas_packed_mlp_modules.append(module)
        if len(acblas_packed_mlp_modules) != 24:
            raise RuntimeError(
                "expected 24 acBLAS packed MLP modules, got "
                f"{len(acblas_packed_mlp_modules)}"
            )
    acblas_attention_prep_modules = []
    if args.acblas_attention_prep_ab:
        if args.acblas_attention_prep_build_dir is None:
            raise ValueError(
                "--acblas-attention-prep-build-dir is required for "
                "--acblas-attention-prep-ab"
            )
        from ppu_acblas_attention_prep import PPUACBLASAttentionPrepExtension

        attention_extension = PPUACBLASAttentionPrepExtension(
            args.acblas_attention_prep_build_dir,
            algorithm=args.acblas_attention_prep_algorithm,
        )
        for module in model._model.modules():
            if type(module).__name__ == "Qwen3_5Attention":
                attention_extension.patch_module(module)
                acblas_attention_prep_modules.append(module)
        if len(acblas_attention_prep_modules) != 6:
            raise RuntimeError(
                "expected 6 acBLAS attention-prep modules, got "
                f"{len(acblas_attention_prep_modules)}"
            )
    torch.cuda.empty_cache()

    def set_enabled(enabled: bool) -> None:
        for module in packed_modules:
            set_packed_qwen35_gdn_input_projections(
                module,
                True
                if (
                    args.residual_rmsnorm_ab
                    or args.residual_rmsnorm_scratch_ab
                    or args.raw_stream_query_ab
                    or args.gate_prep_ab
                    or args.acblas_packed_mlp_ab
                    or args.acblas_attention_prep_ab
                )
                else enabled,
            )
        if (
            args.residual_rmsnorm_ab
            or args.residual_rmsnorm_scratch_ab
            or args.raw_stream_query_ab
            or args.gate_prep_ab
            or args.acblas_packed_mlp_ab
            or args.acblas_attention_prep_ab
        ):
            for module in residual_modules:
                set_qwen35_decoder_residual_rmsnorm(
                    module,
                    True
                    if (
                        args.residual_rmsnorm_scratch_ab
                        or args.raw_stream_query_ab
                        or args.gate_prep_ab
                        or args.acblas_packed_mlp_ab
                        or args.acblas_attention_prep_ab
                    )
                    else enabled,
                )
                set_qwen35_decoder_residual_rmsnorm_scratch(
                    module, enabled if args.residual_rmsnorm_scratch_ab else False
                )
        if (
            args.gate_prep_ab
            or args.residual_rmsnorm_scratch_ab
            or args.raw_stream_query_ab
            or args.acblas_packed_mlp_ab
            or args.acblas_attention_prep_ab
        ):
            from ppu_gdn import set_qwen35_gdn_gate_prep

            for module in gate_prep_modules:
                set_qwen35_gdn_gate_prep(
                    module,
                    True
                    if (
                        args.acblas_packed_mlp_ab
                        or args.acblas_attention_prep_ab
                        or args.residual_rmsnorm_scratch_ab
                        or args.raw_stream_query_ab
                    )
                    else enabled,
                )
        if (
            args.acblas_packed_mlp_ab
            or args.residual_rmsnorm_scratch_ab
            or args.raw_stream_query_ab
        ):
            for module in acblas_packed_mlp_modules:
                module.forward = module._seu_acblas_packed_mlp_forward
        if args.acblas_attention_prep_ab:
            for module in acblas_packed_mlp_modules:
                module.forward = module._seu_acblas_packed_mlp_forward
            for module in acblas_attention_prep_modules:
                module._seu_attention_prep_decode = (
                    module._seu_acblas_attention_prep_forward if enabled else None
                )
        model._ppu_gdn_library.set_raw_stream_query(
            enabled if args.raw_stream_query_ab else False
        )

    def run_sample(sample, enabled: bool, pair_index: int) -> dict[str, object]:
        set_enabled(enabled)
        settle_runtime(model)
        result = model.generate_with_metrics(
            image=decode_image(sample.image_b64),
            prompt=build_prompt(sample),
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        answer = extract_answer(result.text)
        return {
            "sample_id": sample.sample_id,
            "pair_index": pair_index,
            "pair_order": "AB" if pair_index % 2 == 0 else "BA",
            "mode": (
                "acblas_attention_prep"
                if enabled and args.acblas_attention_prep_ab
                else "acblas_packed_mlp"
                if enabled and args.acblas_packed_mlp_ab
                else "gdn_gate_prep"
                if enabled and args.gate_prep_ab
                else "residual_rmsnorm_scratch"
                if enabled and args.residual_rmsnorm_scratch_ab
                else "raw_stream_query"
                if enabled and args.raw_stream_query_ab
                else "residual_rmsnorm"
                if enabled and args.residual_rmsnorm_ab
                else "packed_gdn"
                if enabled
                else "optimized_baseline"
            ),
            "token_count": result.token_count,
            "ttft_ms": result.ttft_seconds * 1000.0,
            "elapsed_ms": result.elapsed_seconds * 1000.0,
            "throughput_tokens_per_sec": compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            ),
            "answer": answer,
            "reference": sample.answer,
            "correct": answer == sample.answer,
            "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        }

    run_sample(samples[0], False, -1)
    run_sample(samples[0], True, -1)
    baseline_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    pair_ratios: list[float] = []
    exact_pairs = 0
    if args.pair_log is not None:
        args.pair_log.parent.mkdir(parents=True, exist_ok=True)
        args.pair_log.write_text("", encoding="utf-8")
    for index, sample in enumerate(samples):
        order = (False, True) if index % 2 == 0 else (True, False)
        pair: dict[bool, dict[str, object]] = {}
        for enabled in order:
            pair[enabled] = run_sample(sample, enabled, index)
        baseline_records.append(pair[False])
        candidate_records.append(pair[True])
        pair_ratios.append(
            float(pair[True]["throughput_tokens_per_sec"])
            / float(pair[False]["throughput_tokens_per_sec"])
        )
        exact_pairs += pair[True]["text_sha256"] == pair[False]["text_sha256"]
        if args.pair_log is not None:
            with args.pair_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "index": index,
                            "baseline": pair[False],
                            "candidate": pair[True],
                            "speedup": pair_ratios[-1],
                            "exact": (
                                pair[True]["text_sha256"]
                                == pair[False]["text_sha256"]
                            ),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        completed = index + 1
        if completed % args.progress_every == 0 or completed == len(samples):
            print(
                json.dumps(
                    {
                        "progress": completed,
                        "total": len(samples),
                        "exact_pairs": exact_pairs,
                        "wins": sum(value > 1.0 for value in pair_ratios),
                        "median_speedup": statistics.median(pair_ratios),
                    }
                ),
                flush=True,
            )

    def mean_metric(records, key: str) -> float:
        return statistics.fmean(float(record[key]) for record in records)

    baseline_accuracy = statistics.fmean(
        bool(record["correct"]) for record in baseline_records
    )
    candidate_accuracy = statistics.fmean(
        bool(record["correct"]) for record in candidate_records
    )
    candidate_label = (
        "acblas_attention_prep"
        if args.acblas_attention_prep_ab
        else "acblas_packed_mlp"
        if args.acblas_packed_mlp_ab
        else "gdn_gate_prep"
        if args.gate_prep_ab
        else "residual_rmsnorm_scratch"
        if args.residual_rmsnorm_scratch_ab
        else "raw_stream_query"
        if args.raw_stream_query_ab
        else "residual_rmsnorm"
        if args.residual_rmsnorm_ab
        else "packed_gdn"
    )
    median_speedup = statistics.median(pair_ratios)
    mean_speedup = statistics.fmean(pair_ratios)
    performance_passed = median_speedup > 1.0 and mean_speedup > 1.0
    payload = {
        "sample_offset": args.sample_offset,
        "sample_count": len(samples),
        "max_new_tokens": args.max_new_tokens,
        "packed_gdn_modules": len(packed_modules),
        "packed_weight_shape_per_module": [8224, 2048],
        "projection_group_sizes": group_sizes,
        "projection_backend": args.projection_backend,
        "ab_target": candidate_label,
        "residual_rmsnorm_modules": len(residual_modules),
        "residual_rmsnorm_scratch_modules": (
            len(residual_modules) if args.residual_rmsnorm_scratch_ab else 0
        ),
        "raw_stream_query_ab_enabled": args.raw_stream_query_ab,
        "gdn_gate_prep_modules": len(gate_prep_modules),
        "acblas_packed_mlp_modules": len(acblas_packed_mlp_modules),
        "acblas_packed_mlp_swiglu_threads": (
            args.acblas_packed_mlp_swiglu_threads
            if (
                args.acblas_packed_mlp_ab
                or args.acblas_attention_prep_ab
                or args.residual_rmsnorm_scratch_ab
                or args.raw_stream_query_ab
            )
            else None
        ),
        "acblas_attention_prep_modules": len(acblas_attention_prep_modules),
        "acblas_attention_prep_algorithm": (
            args.acblas_attention_prep_algorithm
            if args.acblas_attention_prep_ab
            else None
        ),
        "baseline": {
            "avg_ttft_ms": mean_metric(baseline_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean_metric(
                baseline_records, "throughput_tokens_per_sec"
            ),
            "accuracy": baseline_accuracy,
            "records": baseline_records,
        },
        candidate_label: {
            "avg_ttft_ms": mean_metric(candidate_records, "ttft_ms"),
            "avg_throughput_tokens_per_sec": mean_metric(
                candidate_records, "throughput_tokens_per_sec"
            ),
            "accuracy": candidate_accuracy,
            "records": candidate_records,
        },
        "paired_decode": {
            "median_speedup": median_speedup,
            "mean_speedup": mean_speedup,
            "wins": sum(value > 1.0 for value in pair_ratios),
            "ratios": pair_ratios,
        },
        "exact_output_pairs": exact_pairs,
        "performance_gate_required": args.require_speedup,
        "performance_passed": performance_passed,
        "passed": (
            exact_pairs == len(samples)
            and baseline_accuracy == candidate_accuracy
            and (performance_passed or not args.require_speedup)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
