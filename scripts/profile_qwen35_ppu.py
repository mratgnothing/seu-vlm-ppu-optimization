#!/usr/bin/env python3
"""Profile a short, real multimodal Qwen3.5 generation on the PPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_OP_DIR = REPO_ROOT / "ppu" / "custom_ops"
DEFAULT_PPU_SDK = Path("/usr/local/PPU_SDK")
if DEFAULT_PPU_SDK.is_dir():
    os.environ.setdefault("PPU_SDK", str(DEFAULT_PPU_SDK))
    os.environ.setdefault("PPU_HOME", str(DEFAULT_PPU_SDK))
for path in (REPO_ROOT, CUSTOM_OP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from benchmark_public import build_prompt, decode_image, load_mmbench_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-new-tokens", type=int, default=2)
    parser.add_argument("--row-limit", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--gdn-library",
        type=Path,
        help="patch all 18 decode GDN modules with this PPU shared library",
    )
    parser.add_argument("--gdn-tiles", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--fuse-conv", action="store_true")
    parser.add_argument("--conv-threads", type=int, default=96)
    parser.add_argument("--fuse-rmsnorm", action="store_true")
    parser.add_argument("--rmsnorm-threads", type=int, default=512)
    parser.add_argument("--fuse-gated-rmsnorm", action="store_true")
    parser.add_argument("--gated-rmsnorm-threads", type=int, default=128)
    parser.add_argument("--fuse-qk-rope", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.max_new_tokens, args.warmup_new_tokens, args.row_limit) <= 0:
        raise ValueError("token counts and row limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_mmbench_tsv(args.dataset_path, limit=1)[0]
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    ).eval()
    patched_gdn_modules = 0
    if args.gdn_library:
        from ppu_gdn import PPUGDNLibrary

        library = PPUGDNLibrary(
            args.gdn_library,
            tiles_per_head=args.gdn_tiles,
            conv_threads=args.conv_threads,
            rmsnorm_threads=args.rmsnorm_threads,
            gated_rmsnorm_threads=args.gated_rmsnorm_threads,
        )
        fused_callable = library.transformers_callable()
        for module in model.modules():
            if type(module).__name__ == "Qwen3_5GatedDeltaNet":
                module.recurrent_gated_delta_rule = fused_callable
                if args.fuse_conv:
                    module.causal_conv1d_update = library.causal_conv1d_decode
                patched_gdn_modules += 1
        if patched_gdn_modules != 18:
            raise RuntimeError(
                f"expected 18 Qwen3.5 GDN modules, patched {patched_gdn_modules}"
            )
        patched_qk_rope_modules = 0
        if args.fuse_qk_rope:
            from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35_modeling

            for module in model.modules():
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
            for module_name, module in model.named_modules():
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
            for module in model.modules():
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
                            hidden_states,
                            gate,
                            _module.weight,
                            _module.variance_epsilon,
                        )
                    return _eager(hidden_states, gate)

                module.forward = decode_gated_rmsnorm
                patched_gated_rmsnorm_modules += 1
            if patched_gated_rmsnorm_modules != 18:
                raise RuntimeError(
                    "expected 18 Qwen3.5 gated RMSNorm modules, "
                    f"patched {patched_gated_rmsnorm_modules}"
                )
    else:
        patched_qk_rope_modules = 0
        patched_rmsnorm_modules = 0
        patched_gated_rmsnorm_modules = 0
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": decode_image(sample.image_b64)},
            {"type": "text", "text": build_prompt(sample)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=args.warmup_new_tokens,
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
        ) as profile:
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            torch.cuda.synchronize()

    trace_path = args.output_dir / "trace.json"
    profile.export_chrome_trace(str(trace_path))
    key_averages = profile.key_averages(group_by_input_shape=True)
    cuda_table = key_averages.table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit
    )
    cpu_table = key_averages.table(
        sort_by="self_cpu_time_total", row_limit=args.row_limit
    )
    (args.output_dir / "top-cuda.txt").write_text(cuda_table + "\n", encoding="utf-8")
    (args.output_dir / "top-cpu.txt").write_text(cpu_table + "\n", encoding="utf-8")

    generated_tokens = int(output_ids.shape[1]) - prompt_tokens
    summary = {
        "sample_id": sample.sample_id,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "max_new_tokens": args.max_new_tokens,
        "warmup_new_tokens": args.warmup_new_tokens,
        "torch_version": torch.__version__,
        "device_name": torch.cuda.get_device_name(0),
        "patched_gdn_modules": patched_gdn_modules,
        "gdn_tiles_per_head": args.gdn_tiles if args.gdn_library else None,
        "fused_causal_conv_modules": (
            patched_gdn_modules if args.gdn_library and args.fuse_conv else 0
        ),
        "fused_rmsnorm_modules": patched_rmsnorm_modules,
        "fused_gated_rmsnorm_modules": patched_gated_rmsnorm_modules,
        "fused_qk_rmsnorm_rope_modules": patched_qk_rope_modules,
        "trace_path": str(trace_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n# Top accelerator operations\n")
    print(cuda_table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
