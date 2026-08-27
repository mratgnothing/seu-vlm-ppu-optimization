#!/usr/bin/env python3
"""A/B registered acBLAS decode linears on the fully optimized Qwen3.5 path."""

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
    parser.add_argument("--acblas-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--force-max-new-tokens", action="store_true")
    parser.add_argument("--profile-output-dir", type=Path)
    parser.add_argument("--profile-new-tokens", type=int, default=16)
    return parser.parse_args()


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "repeats": len(records),
        "median_ttft_ms": statistics.median(float(r["ttft_ms"]) for r in records),
        "median_elapsed_ms": statistics.median(
            float(r["elapsed_ms"]) for r in records
        ),
        "median_throughput_tokens_per_sec": statistics.median(
            float(r["throughput_tokens_per_sec"]) for r in records
        ),
        "records": records,
    }


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    custom_op_dir = repo_root / "ppu" / "custom_ops"
    for path in (repo_root, repo_root / "scripts", custom_op_dir):
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
    from ppu_acblas_extension import PPUACBLASLinearExtension

    samples = load_mmbench_tsv(
        args.dataset_path, limit=args.sample_offset + 1
    )
    sample = samples[args.sample_offset]
    model = VLMModel(str(args.model_path), backend="transformers", device="auto")
    if args.force_max_new_tokens:
        # Performance-only mode: keep both A/B paths decoding to the same fixed
        # length even if the model emits EOS. Production generation is unchanged.
        model._model.generation_config.eos_token_id = None
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
                result.token_count,
                result.ttft_seconds,
                result.elapsed_seconds,
            ),
            "answer": extract_answer(result.text),
            "text_sha256": hashlib.sha256(
                result.text.encode("utf-8")
            ).hexdigest(),
        }

    extension = PPUACBLASLinearExtension(args.acblas_build_dir)
    patched_names, shape_counts = extension.patch_qwen35_language_linears(
        model._model
    )
    expected_shape_counts = {
        "512x2048": 12,
        "2048x2048": 42,
        "2048x6144": 24,
        "4096x2048": 6,
        "6144x2048": 18,
    }
    if shape_counts != expected_shape_counts:
        raise RuntimeError(
            f"unexpected Qwen3.5 acBLAS patch counts: {shape_counts}"
        )
    torch.cuda.empty_cache()

    patched_modules = [dict(model._model.named_modules())[name] for name in patched_names]

    def set_acblas_enabled(enabled: bool) -> None:
        for module in patched_modules:
            module.forward = (
                module._seu_acblas_forward
                if enabled
                else module._seu_acblas_original_forward
            )

    set_acblas_enabled(False)
    run_once("optimized_baseline_warmup")
    set_acblas_enabled(True)
    run_once("acblas_warmup")

    baseline_records: list[dict[str, object]] = []
    acblas_records: list[dict[str, object]] = []
    for pair_index in range(args.repeats):
        # Alternate AB and BA order to cancel slow device/load drift.
        order = (False, True) if pair_index % 2 == 0 else (True, False)
        for enabled in order:
            set_acblas_enabled(enabled)
            record = run_once("acblas" if enabled else "optimized_baseline")
            record["pair_index"] = pair_index
            record["pair_order"] = "AB" if pair_index % 2 == 0 else "BA"
            (acblas_records if enabled else baseline_records).append(record)
    baseline = summarize(baseline_records)
    acblas = summarize(acblas_records)
    pair_speedups = [
        float(acblas_records[index]["throughput_tokens_per_sec"])
        / float(baseline_records[index]["throughput_tokens_per_sec"])
        for index in range(args.repeats)
    ]
    baseline_hashes = {str(r["text_sha256"]) for r in baseline_records}
    acblas_hashes = {str(r["text_sha256"]) for r in acblas_records}
    exact_output_match = (
        len(baseline_hashes) == 1 and baseline_hashes == acblas_hashes
    )
    payload = {
        "sample_id": sample.sample_id,
        "sample_offset": args.sample_offset,
        "max_new_tokens": args.max_new_tokens,
        "force_max_new_tokens": args.force_max_new_tokens,
        "baseline": "gdn+conv+rmsnorm+gated_rmsnorm+qk_rope+packed_mlp",
        "candidate": "baseline+registered_acblas_decode_linears",
        "patched_linear_modules": len(patched_names),
        "patched_shape_counts": shape_counts,
        "baseline_metrics": baseline,
        "acblas_metrics": acblas,
        "speedup": {
            "ttft": float(baseline["median_ttft_ms"])
            / float(acblas["median_ttft_ms"]),
            "elapsed": float(baseline["median_elapsed_ms"])
            / float(acblas["median_elapsed_ms"]),
            "decode_throughput": float(
                acblas["median_throughput_tokens_per_sec"]
            )
            / float(baseline["median_throughput_tokens_per_sec"]),
            "paired_decode_median": statistics.median(pair_speedups),
            "paired_decode_wins": sum(value > 1.0 for value in pair_speedups),
            "paired_decode_ratios": pair_speedups,
        },
        "exact_output_match": exact_output_match,
        "passed": exact_output_match,
    }

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
            set_acblas_enabled(enabled)
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

        baseline_profile_ids, baseline_trace = profile_mode(False, "baseline")
        acblas_profile_ids, acblas_trace = profile_mode(True, "acblas")
        profile_exact = torch.equal(baseline_profile_ids, acblas_profile_ids)
        payload["profile"] = {
            "new_tokens": args.profile_new_tokens,
            "baseline_trace": str(baseline_trace),
            "acblas_trace": str(acblas_trace),
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
