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


def _module_available(module_name: str) -> bool:
    """Check a possibly nested module without failing when its parent is absent."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


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


def _assess_route(checks: tuple[tuple[bool, str], ...]) -> dict[str, Any]:
    """Return a machine-readable readiness result and actionable blockers."""
    blockers = [blocker for passed, blocker in checks if not passed]
    return {
        "ready": not blockers,
        "blockers": blockers,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packages = _package_versions()
    device_nodes = sorted(
        set(glob.glob("/dev/alixpu*") + glob.glob("/dev/*ppu*"))
    )
    qwen35_transformers_module = _module_available("transformers.models.qwen3_5")
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
    torch_present = packages["torch"] is not None
    transformers_present = packages["transformers"] is not None
    transformers_stack_present = torch_present and transformers_present
    vllm_stack_present = transformers_stack_present and packages["vllm"] is not None
    hardware_visible = bool(device_nodes)
    model_smoke = _model_config_smoke(args.model_path)
    model_checks: tuple[tuple[bool, str], ...] = ()
    if model_smoke.get("requested"):
        model_checks = (
            (bool(model_smoke.get("exists")), "model_path_missing"),
            (bool(model_smoke.get("loaded")), "model_config_load_failed"),
        )

    transformers_route = _assess_route(
        (
            (hardware_visible, "ppu_device_not_visible"),
            (torch_present, "torch_not_installed"),
            (transformers_present, "transformers_not_installed"),
            (
                qwen35_transformers_module,
                "transformers_qwen35_module_missing",
            ),
        )
        + model_checks
    )
    vllm_route = _assess_route(
        (
            (hardware_visible, "ppu_device_not_visible"),
            (torch_present, "torch_not_installed"),
            (transformers_present, "transformers_not_installed"),
            (packages["vllm"] is not None, "vllm_not_installed"),
            (vllm_source.is_dir(), "vllm_source_missing"),
            (vllm_qwen35, "vllm_qwen35_source_missing"),
            (vllm_gdn, "vllm_gdn_source_missing"),
        )
        + model_checks
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
            "transformers_qwen35_module": qwen35_transformers_module,
            "vllm_source_path": str(vllm_source),
            "vllm_qwen35_source": vllm_qwen35,
            "vllm_gdn_source": vllm_gdn,
            "vllm_causal_conv_source": vllm_causal_conv,
        },
        "model_config_smoke": model_smoke,
        "readiness": {
            "hardware_visible": hardware_visible,
            "transformers_eager_ready": transformers_route["ready"],
            "vllm_ready": vllm_route["ready"],
        },
        "routes": {
            "transformers_eager": transformers_route,
            "vllm": vllm_route,
        },
        "deployment_ready": bool(
            transformers_route["ready"] or vllm_route["ready"]
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
