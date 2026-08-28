#!/usr/bin/env python3
"""Paired A/B for Qwen3.5 GDN projection or residual-RMSNorm fusion on PPU."""

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
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--force-max-new-tokens", action="store_true")
    parser.add_argument(
        "--require-speedup",
        action="store_true",
        help="Fail the gate unless paired median and mean throughput both improve",
    )
    parser.add_argument(
        "--group-sizes",
        default="4",
        help="Comma-separated consecutive projection groups, for example 2,2",
    )
    parser.add_argument("--profile-output-dir", type=Path)
    parser.add_argument("--profile-new-tokens", type=int, default=16)
    parser.add_argument(
        "--projection-backend",
        choices=("torch-packed", "acblas-grouped"),
        default="torch-packed",
    )
    parser.add_argument("--acblas-build-dir", type=Path)
    parser.add_argument("--acblaslt-build-dir", type=Path)
    parser.add_argument("--acblaslt-heuristic-index", type=int, default=25)
    parser.add_argument("--acblas-packed-mlp-build-dir", type=Path)
    parser.add_argument("--acblas-packed-mlp-swiglu-threads", type=int, default=128)
    parser.add_argument("--acblas-workspace-mib", type=int, default=16)
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
        "--acblas-workspace-ab",
        action="store_true",
        help="Keep the raw-stream stack on and A/B persistent acBLAS workspaces",
    )
    targets.add_argument(
        "--acblas-gdn-ba-batched-ab",
        action="store_true",
        help="Keep the raw-stream stack on and A/B batched GDN b/a projections",
    )
    targets.add_argument(
        "--acblas-gdn-output-scratch-ab",
        action="store_true",
        help="Keep the raw-stream stack on and A/B persistent grouped-GDN output",
    )
    targets.add_argument(
        "--gate-prep-ab",
        action="store_true",
        help="Keep projections/residual RMSNorm on and A/B GDN gate preparation",
    )
    targets.add_argument(
        "--acblaslt-square-ab",
        action="store_true",
        help="Keep gate-prep stack on and A/B acBLASLt 2048-square Linears",
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
    if args.acblas_workspace_mib <= 0:
        raise ValueError("--acblas-workspace-mib must be positive")
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

    sample = load_mmbench_tsv(
        args.dataset_path, limit=args.sample_offset + 1
    )[args.sample_offset]
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    if args.force_max_new_tokens:
        model._model.generation_config.eos_token_id = None
    config = fixed_generation_config()
    config.max_new_tokens = args.max_new_tokens
    image = decode_image(sample.image_b64)
    prompt = build_prompt(sample)

    packed_modules = []
    gdn_projection_extension = None
    if args.projection_backend == "acblas-grouped":
        if args.acblas_build_dir is None:
            raise ValueError("--acblas-build-dir is required for acblas-grouped")
        if group_sizes != (4,):
            raise ValueError("acblas-grouped implements the complete four-way group")
        from ppu_acblas_gdn_projection import PPUACBLASGDNProjectionExtension

        gdn_projection_extension = PPUACBLASGDNProjectionExtension(
            args.acblas_build_dir,
            workspace_bytes=(
                args.acblas_workspace_mib * 1024 * 1024
                if args.acblas_workspace_ab
                else 0
            ),
            workspace_enabled=False,
        )
        projection_packer = gdn_projection_extension.pack_module
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
        or args.acblas_workspace_ab
        or args.acblas_gdn_ba_batched_ab
        or args.acblas_gdn_output_scratch_ab
        or args.gate_prep_ab
        or args.acblaslt_square_ab
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
        or args.acblas_workspace_ab
        or args.acblas_gdn_ba_batched_ab
        or args.acblas_gdn_output_scratch_ab
        or args.acblaslt_square_ab
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
    acblaslt_square_modules = []
    acblaslt_square_shape_counts: dict[str, int] = {}
    if args.acblaslt_square_ab:
        if args.acblaslt_build_dir is None:
            raise ValueError(
                "--acblaslt-build-dir is required for --acblaslt-square-ab"
            )
        from ppu_acblaslt_square import PPUACBLASLtSquareExtension

        square_extension = PPUACBLASLtSquareExtension(
            args.acblaslt_build_dir,
            heuristic_index=args.acblaslt_heuristic_index,
        )
        square_names, acblaslt_square_shape_counts = (
            square_extension.patch_qwen35_language_linears(model._model)
        )
        if acblaslt_square_shape_counts != {"2048x2048": 42}:
            raise RuntimeError(
                "expected 42 acBLASLt 2048-square modules, got "
                f"{acblaslt_square_shape_counts}"
            )
        named_modules = dict(model._model.named_modules())
        acblaslt_square_modules = [named_modules[name] for name in square_names]
    acblas_packed_mlp_modules = []
    if (
        args.acblas_packed_mlp_ab
        or args.acblas_attention_prep_ab
        or args.residual_rmsnorm_scratch_ab
        or args.raw_stream_query_ab
        or args.acblas_workspace_ab
        or args.acblas_gdn_ba_batched_ab
        or args.acblas_gdn_output_scratch_ab
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
            workspace_bytes=(
                args.acblas_workspace_mib * 1024 * 1024
                if args.acblas_workspace_ab
                else 0
            ),
            workspace_enabled=False,
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
                    or args.acblas_workspace_ab
                    or args.acblas_gdn_ba_batched_ab
                    or args.acblas_gdn_output_scratch_ab
                    or args.gate_prep_ab
                    or args.acblaslt_square_ab
                    or args.acblas_packed_mlp_ab
                    or args.acblas_attention_prep_ab
                )
                else enabled,
            )
        if (
            args.residual_rmsnorm_ab
            or args.residual_rmsnorm_scratch_ab
            or args.raw_stream_query_ab
            or args.acblas_workspace_ab
            or args.acblas_gdn_ba_batched_ab
            or args.acblas_gdn_output_scratch_ab
            or args.gate_prep_ab
            or args.acblaslt_square_ab
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
                        or args.acblas_workspace_ab
                        or args.acblas_gdn_ba_batched_ab
                        or args.acblas_gdn_output_scratch_ab
                        or args.gate_prep_ab
                        or args.acblaslt_square_ab
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
            or args.acblas_workspace_ab
            or args.acblas_gdn_ba_batched_ab
            or args.acblas_gdn_output_scratch_ab
            or args.acblaslt_square_ab
            or args.acblas_packed_mlp_ab
            or args.acblas_attention_prep_ab
        ):
            from ppu_gdn import set_qwen35_gdn_gate_prep

            for module in gate_prep_modules:
                set_qwen35_gdn_gate_prep(
                    module,
                    True
                    if (
                        args.acblaslt_square_ab
                        or args.acblas_packed_mlp_ab
                        or args.acblas_attention_prep_ab
                        or args.residual_rmsnorm_scratch_ab
                        or args.raw_stream_query_ab
                        or args.acblas_workspace_ab
                        or args.acblas_gdn_ba_batched_ab
                        or args.acblas_gdn_output_scratch_ab
                    )
                    else enabled,
                )
        if args.acblaslt_square_ab:
            for module in acblaslt_square_modules:
                module.forward = (
                    module._seu_acblaslt_square_forward
                    if enabled
                    else module._seu_acblaslt_square_original_forward
                )
        if (
            args.acblas_packed_mlp_ab
            or args.residual_rmsnorm_scratch_ab
            or args.raw_stream_query_ab
            or args.acblas_workspace_ab
            or args.acblas_gdn_ba_batched_ab
            or args.acblas_gdn_output_scratch_ab
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
        if args.acblas_workspace_ab:
            if gdn_projection_extension is None:
                raise RuntimeError("workspace A/B requires acblas-grouped projections")
            gdn_projection_extension.set_workspace_enabled(enabled)
            mlp_extension.set_workspace_enabled(enabled)
        if args.acblas_gdn_ba_batched_ab:
            if gdn_projection_extension is None:
                raise RuntimeError("batched b/a A/B requires acblas-grouped projections")
            gdn_projection_extension.set_batched_ba(enabled)
        if args.acblas_gdn_output_scratch_ab:
            if gdn_projection_extension is None:
                raise RuntimeError("GDN output scratch A/B requires acblas-grouped")
            for module in packed_modules:
                gdn_projection_extension.set_output_scratch(module, enabled)
        model._ppu_gdn_library.set_raw_stream_query(
            True
            if (
                args.acblas_workspace_ab
                or args.acblas_gdn_ba_batched_ab
                or args.acblas_gdn_output_scratch_ab
            )
            else enabled
            if args.raw_stream_query_ab
            else False
        )

    def run_once(enabled: bool, pair_index: int) -> dict[str, object]:
        set_enabled(enabled)
        settle_runtime(model)
        result = model.generate_with_metrics(
            image=image,
            prompt=prompt,
            choices=sample.choices,
            generation_config=config,
            sample_id=sample.sample_id,
        )
        return {
            "mode": (
                "acblas_attention_prep"
                if enabled and args.acblas_attention_prep_ab
                else "acblas_packed_mlp"
                if enabled and args.acblas_packed_mlp_ab
                else "acblaslt_square"
                if enabled and args.acblaslt_square_ab
                else "gdn_gate_prep"
                if enabled and args.gate_prep_ab
                else "residual_rmsnorm_scratch"
                if enabled and args.residual_rmsnorm_scratch_ab
                else "raw_stream_query"
                if enabled and args.raw_stream_query_ab
                else "acblas_workspace"
                if enabled and args.acblas_workspace_ab
                else "acblas_gdn_ba_batched"
                if enabled and args.acblas_gdn_ba_batched_ab
                else "acblas_gdn_output_scratch"
                if enabled and args.acblas_gdn_output_scratch_ab
                else "residual_rmsnorm"
                if enabled and args.residual_rmsnorm_ab
                else "packed_gdn"
                if enabled
                else "optimized_baseline"
            ),
            "pair_index": pair_index,
            "pair_order": "AB" if pair_index % 2 == 0 else "BA",
            "token_count": result.token_count,
            "ttft_ms": result.ttft_seconds * 1000.0,
            "elapsed_ms": result.elapsed_seconds * 1000.0,
            "throughput_tokens_per_sec": compute_throughput(
                result.token_count, result.ttft_seconds, result.elapsed_seconds
            ),
            "answer": extract_answer(result.text),
            "text_sha256": hashlib.sha256(
                result.text.encode("utf-8")
            ).hexdigest(),
        }

    run_once(False, -1)
    run_once(True, -1)
    baseline_records: list[dict[str, object]] = []
    candidate_records: list[dict[str, object]] = []
    for pair_index in range(args.repeats):
        order = (False, True) if pair_index % 2 == 0 else (True, False)
        for enabled in order:
            record = run_once(enabled, pair_index)
            (candidate_records if enabled else baseline_records).append(record)

    baseline = summarize(baseline_records)
    candidate = summarize(candidate_records)
    pair_ratios = [
        float(candidate_records[index]["throughput_tokens_per_sec"])
        / float(baseline_records[index]["throughput_tokens_per_sec"])
        for index in range(args.repeats)
    ]
    baseline_hashes = {str(r["text_sha256"]) for r in baseline_records}
    candidate_hashes = {str(r["text_sha256"]) for r in candidate_records}
    exact = len(baseline_hashes) == 1 and baseline_hashes == candidate_hashes
    median_speedup = statistics.median(pair_ratios)
    mean_speedup = statistics.fmean(pair_ratios)
    performance_passed = median_speedup > 1.0 and mean_speedup > 1.0
    candidate_label = (
        "acblas_attention_prep"
        if args.acblas_attention_prep_ab
        else "acblas_packed_mlp"
        if args.acblas_packed_mlp_ab
        else "acblaslt_square"
        if args.acblaslt_square_ab
        else "gdn_gate_prep"
        if args.gate_prep_ab
        else "residual_rmsnorm_scratch"
        if args.residual_rmsnorm_scratch_ab
        else "raw_stream_query"
        if args.raw_stream_query_ab
        else "acblas_workspace"
        if args.acblas_workspace_ab
        else "acblas_gdn_ba_batched"
        if args.acblas_gdn_ba_batched_ab
        else "acblas_gdn_output_scratch"
        if args.acblas_gdn_output_scratch_ab
        else "residual_rmsnorm"
        if args.residual_rmsnorm_ab
        else "packed_gdn"
    )
    payload = {
        "sample_id": sample.sample_id,
        "max_new_tokens": args.max_new_tokens,
        "force_max_new_tokens": args.force_max_new_tokens,
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
        "acblas_workspace_ab_enabled": args.acblas_workspace_ab,
        "acblas_gdn_ba_batched_ab_enabled": args.acblas_gdn_ba_batched_ab,
        "acblas_gdn_output_scratch_ab_enabled": (
            args.acblas_gdn_output_scratch_ab
        ),
        "acblas_workspace_bytes_per_handle": (
            args.acblas_workspace_mib * 1024 * 1024
            if args.acblas_workspace_ab
            else 0
        ),
        "gdn_gate_prep_modules": len(gate_prep_modules),
        "acblaslt_square_modules": len(acblaslt_square_modules),
        "acblaslt_square_shape_counts": acblaslt_square_shape_counts,
        "acblaslt_heuristic_index": (
            args.acblaslt_heuristic_index if args.acblaslt_square_ab else None
        ),
        "acblas_packed_mlp_modules": len(acblas_packed_mlp_modules),
        "acblas_packed_mlp_swiglu_threads": (
            args.acblas_packed_mlp_swiglu_threads
            if (
                args.acblas_packed_mlp_ab
                or args.acblas_attention_prep_ab
                or args.residual_rmsnorm_scratch_ab
                or args.raw_stream_query_ab
                or args.acblas_workspace_ab
                or args.acblas_gdn_ba_batched_ab
                or args.acblas_gdn_output_scratch_ab
            )
            else None
        ),
        "acblas_attention_prep_modules": len(acblas_attention_prep_modules),
        "acblas_attention_prep_algorithm": (
            args.acblas_attention_prep_algorithm
            if args.acblas_attention_prep_ab
            else None
        ),
        "baseline": baseline,
        "paired_decode": {
            "median_speedup": median_speedup,
            "mean_speedup": mean_speedup,
            "wins": sum(value > 1.0 for value in pair_ratios),
            "ratios": pair_ratios,
        },
        "exact_output_match": exact,
        "performance_gate_required": args.require_speedup,
        "performance_passed": performance_passed,
        "passed": exact and (performance_passed or not args.require_speedup),
    }
    payload[candidate_label] = candidate
    if args.profile_output_dir:
        profile_dir = args.profile_output_dir.resolve()
        profile_dir.mkdir(parents=True, exist_ok=True)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        profile_inputs = model._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model._model.device)

        def profile_mode(enabled: bool, mode: str):
            set_enabled(enabled)
            with torch.inference_mode():
                model._model.generate(
                    **profile_inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                )
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                ) as profiler:
                    output_ids = model._model.generate(
                        **profile_inputs,
                        max_new_tokens=args.profile_new_tokens,
                        do_sample=False,
                        use_cache=True,
                    )
                    torch.cuda.synchronize()
            trace_path = profile_dir / f"{mode}-trace.json"
            profiler.export_chrome_trace(str(trace_path))
            averages = profiler.key_averages(group_by_input_shape=True)
            (profile_dir / f"{mode}-top-cuda.txt").write_text(
                averages.table(sort_by="self_cuda_time_total", row_limit=80) + "\n",
                encoding="utf-8",
            )
            (profile_dir / f"{mode}-top-cpu.txt").write_text(
                averages.table(sort_by="self_cpu_time_total", row_limit=80) + "\n",
                encoding="utf-8",
            )
            return output_ids, trace_path

        baseline_ids, baseline_trace = profile_mode(False, "baseline")
        candidate_ids, candidate_trace = profile_mode(True, candidate_label)
        profile_exact = torch.equal(baseline_ids, candidate_ids)
        payload["profile"] = {
            "new_tokens": args.profile_new_tokens,
            "baseline_trace": str(baseline_trace),
            "candidate_trace": str(candidate_trace),
            "exact_output_match": profile_exact,
        }
        payload["passed"] = bool(payload["passed"] and profile_exact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
