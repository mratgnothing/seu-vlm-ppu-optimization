#!/usr/bin/env python3
"""Read-only PPU runtime preflight with no third-party Python dependency."""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


PACKAGE_NAMES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "vllm",
    "triton",
    "flash-attn",
    "flashinfer-python",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a PPU inference runtime")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--vllm-source", type=Path, default=Path("/opt/vllm"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _command_output(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "output": None}
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
        timeout=20,
    )
    combined = "\n".join(
        part.strip()
        for part in (completed.stdout, completed.stderr)
        if part.strip()
    )
    return {
        "available": True,
        "returncode": completed.returncode,
        "output": combined[:4000],
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package_name in PACKAGE_NAMES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _source_contains(root: Path, markers: tuple[str, ...]) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*.py"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(marker in content for marker in markers):
            return True
    return False


def _model_config_smoke(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {"requested": False}
    result: dict[str, Any] = {
        "requested": True,
        "model_path": str(model_path.resolve()),
        "exists": model_path.is_dir(),
    }
    if not model_path.is_dir():
        return result
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        result.update(
            loaded=True,
            model_type=getattr(config, "model_type", None),
            architectures=getattr(config, "architectures", None),
        )
    except Exception as exc:
        result.update(loaded=False, error=f"{type(exc).__name__}: {exc}")
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packages = _package_versions()
    device_nodes = sorted(
        set(glob.glob("/dev/alixpu*") + glob.glob("/dev/*ppu*"))
    )
    qwen35_transformers_module = importlib.util.find_spec(
        "transformers.models.qwen3_5"
    )
    vllm_source = args.vllm_source.resolve()
    vllm_qwen35 = _source_contains(
        vllm_source,
        ("Qwen3_5ForConditionalGeneration", "qwen3_5"),
    )
    vllm_gdn = _source_contains(
        vllm_source,
        ("GatedDeltaNet", "gated_delta", "gated delta"),
    )
    vllm_causal_conv = _source_contains(
        vllm_source,
        ("causal_conv1d_fwd", "causal_conv1d_update"),
    )
    transformers_stack_present = all(
        packages[name] is not None for name in ("torch", "transformers")
    )
    vllm_stack_present = transformers_stack_present and packages["vllm"] is not None
    hardware_visible = bool(device_nodes)
    transformers_runtime_ready = bool(
        hardware_visible
        and transformers_stack_present
        and qwen35_transformers_module is not None
    )
    vllm_runtime_ready = bool(
        hardware_visible
        and vllm_stack_present
        and vllm_qwen35
        and vllm_gdn
    )
    model_smoke = _model_config_smoke(args.model_path)
    model_gate = (
        bool(model_smoke.get("loaded"))
        if model_smoke.get("requested")
        else True
    )

    return {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "environment": {
            "PPU_SDK": os.getenv("PPU_SDK"),
            "CUDA_HOME": os.getenv("CUDA_HOME"),
            "LD_LIBRARY_PATH_set": bool(os.getenv("LD_LIBRARY_PATH")),
        },
        "hardware": {
            "device_nodes": device_nodes,
            "ppu_smi": _command_output(["ppu-smi", "--version"]),
        },
        "sdk": {
            "sdk_present": Path(
                os.getenv("PPU_SDK", "/usr/local/PPU_SDK")
            ).is_dir(),
            "hgcc": _command_output(["hgcc", "--version"]),
            "nvcc": _command_output(["nvcc", "--version"]),
        },
        "packages": packages,
        "compatibility": {
            "transformers_stack_present": transformers_stack_present,
            "vllm_stack_present": vllm_stack_present,
            "transformers_qwen35_module": qwen35_transformers_module
            is not None,
            "vllm_source_path": str(vllm_source),
            "vllm_qwen35_source": vllm_qwen35,
            "vllm_gdn_source": vllm_gdn,
            "vllm_causal_conv_source": vllm_causal_conv,
        },
        "model_config_smoke": model_smoke,
        "readiness": {
            "hardware_visible": hardware_visible,
            "transformers_eager_ready": transformers_runtime_ready
            and model_gate,
            "vllm_ready": vllm_runtime_ready and model_gate,
        },
        "deployment_ready": bool(
            model_gate
            and (transformers_runtime_ready or vllm_runtime_ready)
        ),
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
