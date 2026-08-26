#!/usr/bin/env python3
"""PPU/Qwen3.5 preflight that is safe to run before model deployment.

The default path is read-only. A tiny PPU tensor operation is available only
through ``--run-device-smoke`` and is refused when PPU hardware is not visible.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAMES = (
    "torch", "torchvision", "transformers", "accelerate", "vllm",
    "triton", "flash-attn", "flashinfer-python", "causal-conv1d",
)

EXPECTED_MODEL_FIELDS: tuple[tuple[tuple[str, ...], Any], ...] = (
    (("model_type",), "qwen3_5"),
    (("architectures",), ["Qwen3_5ForConditionalGeneration"]),
    (("text_config", "model_type"), "qwen3_5_text"),
    (("text_config", "dtype"), "bfloat16"),
    (("text_config", "hidden_size"), 2048),
    (("text_config", "intermediate_size"), 6144),
    (("text_config", "num_hidden_layers"), 24),
    (("text_config", "full_attention_interval"), 4),
    (("text_config", "num_attention_heads"), 8),
    (("text_config", "num_key_value_heads"), 2),
    (("text_config", "head_dim"), 256),
    (("text_config", "linear_num_key_heads"), 16),
    (("text_config", "linear_num_value_heads"), 16),
    (("text_config", "linear_key_head_dim"), 128),
    (("text_config", "linear_value_head_dim"), 128),
    (("text_config", "linear_conv_kernel_dim"), 4),
    (("text_config", "vocab_size"), 248320),
    (("text_config", "use_cache"), True),
    (("vision_config", "hidden_size"), 1024),
    (("vision_config", "intermediate_size"), 4096),
    (("vision_config", "depth"), 24),
    (("vision_config", "num_heads"), 16),
    (("vision_config", "patch_size"), 16),
    (("vision_config", "out_hidden_size"), 2048),
)

KNOWN_RISKS = (
    {
        "id": "cuda_source_not_binary_compatible",
        "summary": "CUDA source can be recompiled for PPU, but NVIDIA binaries cannot run on PPU.",
        "action": "Build every CUDA/HGGC extension with the PPU compiler inside the target image.",
    },
    {
        "id": "cuda_api_coverage_is_partial",
        "summary": "CUDA-like syntax does not guarantee every Runtime, Driver, library, or inline PTX feature.",
        "action": "Check the SDK-specific unsupported API tables and compile every extension on the target node.",
    },
    {
        "id": "qwen35_hybrid_attention",
        "summary": "Qwen3.5-2B mixes 18 linear-attention/GDN layers with 6 full-attention layers.",
        "action": "Validate GDN state, causal-conv1d, full attention, and both prefill/decode paths.",
    },
    {
        "id": "cpu_fallback_can_fake_success",
        "summary": "Generation can succeed while parameters or unsupported operators silently fall back to CPU.",
        "action": "Record parameter devices, profiler kernels, and PPU utilization; reject CPU/meta/disk offload for baselines.",
    },
    {
        "id": "bf16_and_accumulation",
        "summary": "The model uses BF16, while reductions and recurrent state may need FP32 accumulation.",
        "action": "Compare every custom kernel with a trusted reference before timing it.",
    },
    {
        "id": "metric_boundary",
        "summary": "TTFT ends at the first generated token, not the first decoded text chunk.",
        "action": "Reuse the benchmark contract and keep outputs identical across A/B runs.",
    },
)

OPEN_QUESTIONS = (
    "Which exact PPU SDK, driver, compiler, PyTorch, Transformers, and vLLM versions are used by private and final images?",
    "Does PPU-vLLM register Qwen3_5ForConditionalGeneration and implement GDN prefill/decode plus causal-conv1d fast paths?",
    "May the submission install or carry self-built PPU wheels/shared libraries, and what are installation/network limits?",
    "Which quantization, calibration data, offline weight transforms, and mixed-precision formats are formally allowed?",
    "Has organizer package v1.2 been diffed and locked, including benchmark parsing and metric changes?",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect PPU, Qwen3.5, Transformers, and PPU-vLLM readiness"
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--vllm-source", type=Path, default=Path("/opt/vllm"))
    parser.add_argument(
        "--sdk-root", type=Path,
        default=Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK")),
    )
    parser.add_argument(
        "--model-lock", type=Path,
        default=REPO_ROOT / "configs" / "model-lock.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--verify-model-hash", action="store_true",
        help="Hash locked weight files. This reads about 4.6 GB.",
    )
    parser.add_argument(
        "--run-device-smoke", action="store_true",
        help="Run a tiny BF16 tensor operation only when PPU hardware is visible.",
    )
    return parser.parse_args()


def _command_output(command: list[str], timeout: int = 20) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None and not Path(command[0]).is_file():
        return {"available": False, "command": command, "output": None}
    try:
        completed = subprocess.run(
            command, capture_output=True, check=False, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "available": True, "command": command, "returncode": None,
            "error": f"{type(exc).__name__}: {exc}", "output": None,
        }
    combined = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )
    return {
        "available": True, "command": command,
        "returncode": completed.returncode, "output": combined[:8000],
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
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
    for pattern in ("*.py", "*.cpp", "*.cc", "*.cu", "*.h", "*.hpp", "*.hg"):
        for path in root.rglob(pattern):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(marker in content for marker in markers):
                return True
    return False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_layout_probe(
    model_path: Path | None, model_lock_path: Path, verify_hash: bool,
) -> dict[str, Any]:
    if model_path is None:
        return {
            "requested": False, "valid": False,
            "blockers": ["model_path_not_provided"],
        }
    resolved = model_path.resolve()
    result: dict[str, Any] = {
        "requested": True, "model_path": str(resolved),
        "exists": resolved.is_dir(),
        "hash_verification_requested": verify_hash,
    }
    if not resolved.is_dir():
        result.update(valid=False, blockers=["model_path_missing"])
        return result

    config_path = resolved / "config.json"
    config = _load_json(config_path)
    mismatches: list[dict[str, Any]] = []
    if config is None:
        mismatches.append({
            "field": "config.json", "expected": "valid JSON object", "actual": None,
        })
    else:
        for keys, expected in EXPECTED_MODEL_FIELDS:
            actual = _nested_get(config, keys)
            if actual != expected:
                mismatches.append({
                    "field": ".".join(keys), "expected": expected, "actual": actual,
                })
        layer_types = _nested_get(config, ("text_config", "layer_types"))
        actual_counts = Counter(layer_types) if isinstance(layer_types, list) else Counter()
        expected_counts = {"linear_attention": 18, "full_attention": 6}
        if dict(actual_counts) != expected_counts:
            mismatches.append({
                "field": "text_config.layer_types.counts",
                "expected": expected_counts, "actual": dict(actual_counts),
            })
        positions = [
            index for index, value in enumerate(layer_types or [])
            if value == "full_attention"
        ]
        if positions != [3, 7, 11, 15, 19, 23]:
            mismatches.append({
                "field": "text_config.layer_types.full_attention_positions",
                "expected": [3, 7, 11, 15, 19, 23], "actual": positions,
            })

    artifacts = {
        "config.json": config_path.is_file(),
        "tokenizer_config.json": (resolved / "tokenizer_config.json").is_file(),
        "tokenizer.json": (resolved / "tokenizer.json").is_file(),
        "preprocessor_config.json": (resolved / "preprocessor_config.json").is_file(),
        "generation_config.json": (resolved / "generation_config.json").is_file(),
    }
    weight_files = sorted(resolved.glob("*.safetensors"))
    artifacts["safetensors_weights"] = bool(weight_files)
    required = (
        "config.json", "tokenizer_config.json", "tokenizer.json",
        "preprocessor_config.json", "safetensors_weights",
    )
    missing = [name for name in required if not artifacts[name]]

    lock = _load_json(model_lock_path) or {}
    locked_files = lock.get("files", {}) if isinstance(lock.get("files"), dict) else {}
    hash_results: dict[str, Any] = {}
    for filename, metadata in locked_files.items():
        candidate = resolved / filename
        expected_size = metadata.get("size_bytes") if isinstance(metadata, dict) else None
        expected_hash = metadata.get("sha256") if isinstance(metadata, dict) else None
        actual_size = candidate.stat().st_size if candidate.is_file() else None
        entry: dict[str, Any] = {
            "exists": candidate.is_file(), "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
            "size_matches": candidate.is_file() and actual_size == expected_size,
            "expected_sha256": expected_hash, "sha256": None,
            "sha256_matches": None,
        }
        if verify_hash and candidate.is_file() and expected_hash:
            actual_hash = _sha256(candidate)
            entry.update(
                sha256=actual_hash,
                sha256_matches=actual_hash.lower() == expected_hash.lower(),
            )
        hash_results[filename] = entry

    blockers = [f"missing_model_artifact:{name}" for name in missing]
    blockers.extend(
        f"model_config_mismatch:{item['field']}" for item in mismatches
    )
    for filename, entry in hash_results.items():
        if not entry["exists"]:
            blockers.append(f"locked_weight_missing:{filename}")
        elif not entry["size_matches"]:
            blockers.append(f"locked_weight_size_mismatch:{filename}")
        elif verify_hash and not entry["sha256_matches"]:
            blockers.append(f"locked_weight_hash_mismatch:{filename}")

    result.update(
        config_loaded=config is not None,
        config_fingerprint_matches=not mismatches,
        config_mismatches=mismatches,
        artifacts=artifacts,
        weight_file_count=len(weight_files),
        weight_bytes=sum(path.stat().st_size for path in weight_files),
        locked_revision=lock.get("revision"),
        locked_repo_id=lock.get("repo_id"),
        locked_files=hash_results,
        blockers=blockers,
        valid=not blockers,
    )
    return result


def _model_config_smoke(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {"requested": False}
    result: dict[str, Any] = {
        "requested": True, "model_path": str(model_path.resolve()),
        "exists": model_path.is_dir(),
    }
    if not model_path.is_dir():
        return result
    try:
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        result.update(
            loaded=True, model_type=getattr(config, "model_type", None),
            architectures=getattr(config, "architectures", None),
            config_class=type(config).__name__,
            processor_class=type(processor).__name__,
            tokenizer_class=type(tokenizer).__name__ if tokenizer else None,
        )
    except Exception as exc:
        result.update(loaded=False, error=f"{type(exc).__name__}: {exc}")
    return result


def _sdk_probe(sdk_root: Path) -> dict[str, Any]:
    resolved = sdk_root.resolve()
    compiler = resolved / "bin" / "clang++"
    headers = {
        name: (resolved / "include" / name).is_file()
        for name in ("hggc_runtime.h", "hggc_bf16.h")
    }
    library_names = (
        sorted(path.name for path in (resolved / "lib").glob("lib*hggc*"))
        if (resolved / "lib").is_dir() else []
    )
    return {
        "root": str(resolved), "present": resolved.is_dir(),
        "compiler_path": str(compiler),
        "compiler_executable": compiler.is_file() and os.access(compiler, os.X_OK),
        "compiler_version": (
            _command_output([str(compiler), "--version"])
            if compiler.is_file() else {"available": False}
        ),
        "headers": headers, "hggc_libraries": library_names[:100],
        "samples_present": any(
            (resolved / name).is_dir() for name in ("samples", "sample")
        ),
        "tools": {
            name: _command_output([name, "--version"])
            for name in (
                "hgcc", "hggc-memcheck", "ppu-gdb", "asys", "acu",
                "hgobjdump",
            )
        },
    }


def _torch_probe(run_device_smoke: bool, hardware_visible: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "imported": False,
        "device_smoke": {"requested": run_device_smoke, "ran": False},
    }
    try:
        import torch
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    available = bool(torch.cuda.is_available())
    result.update(
        imported=True, version=getattr(torch, "__version__", None),
        compiled_cuda=getattr(getattr(torch, "version", None), "cuda", None),
        compiled_hip=getattr(getattr(torch, "version", None), "hip", None),
        cuda_available=available,
        cuda_device_count=torch.cuda.device_count() if available else 0,
    )
    devices: list[dict[str, Any]] = []
    if available:
        for index in range(torch.cuda.device_count()):
            try:
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index, "name": torch.cuda.get_device_name(index),
                    "total_memory": getattr(properties, "total_memory", None),
                    "major": getattr(properties, "major", None),
                    "minor": getattr(properties, "minor", None),
                })
            except Exception as exc:
                devices.append({
                    "index": index, "error": f"{type(exc).__name__}: {exc}",
                })
    result["devices"] = devices

    if not run_device_smoke:
        return result
    if not hardware_visible:
        result["device_smoke"]["error"] = (
            "PPU hardware not visible; refusing to run on a non-PPU CUDA device"
        )
        return result
    if not available:
        result["device_smoke"]["error"] = (
            "torch.cuda compatibility backend is unavailable"
        )
        return result
    try:
        lhs = torch.arange(
            1024, device="cuda:0", dtype=torch.float32,
        ).reshape(32, 32).to(torch.bfloat16)
        rhs = torch.eye(32, device="cuda:0", dtype=torch.bfloat16)
        output = lhs @ rhs
        torch.cuda.synchronize()
        max_error = float(
            (output.float().cpu() - lhs.float().cpu()).abs().max().item()
        )
        result["device_smoke"].update(
            ran=True, passed=max_error == 0.0,
            operation="BF16 32x32 matmul with identity",
            device=str(output.device), dtype=str(output.dtype),
            max_absolute_error=max_error,
        )
    except Exception as exc:
        result["device_smoke"].update(
            error=f"{type(exc).__name__}: {exc}", passed=False,
        )
    return result


def _assess_route(checks: tuple[tuple[bool, str], ...]) -> dict[str, Any]:
    blockers = [blocker for passed, blocker in checks if not passed]
    return {"ready": not blockers, "blockers": blockers}


def _issue(
    identifier: str, severity: str, summary: str, evidence: Any, action: str,
) -> dict[str, Any]:
    return {
        "id": identifier, "severity": severity, "summary": summary,
        "evidence": evidence, "action": action,
    }


def _build_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    hardware = report["hardware"]
    sdk = report["sdk"]
    packages = report["packages"]
    compatibility = report["compatibility"]
    model = report["model_layout"]

    if not hardware["visible"]:
        issues.append(_issue(
            "ppu_not_visible", "blocker", "No PPU device is visible.",
            hardware["device_nodes"],
            "Check container device passthrough, driver, permissions, and ppu-smi.",
        ))
    if not sdk["present"]:
        issues.append(_issue(
            "ppu_sdk_missing", "blocker", "PPU SDK root is missing.",
            sdk["root"],
            "Use the organizer image or set --sdk-root/PPU_SDK.",
        ))
    elif not sdk["compiler_executable"]:
        issues.append(_issue(
            "ppu_compiler_missing", "blocker",
            "PPU clang++ is not executable.", sdk["compiler_path"],
            "Confirm the SDK package and executable permissions.",
        ))
    missing_headers = [
        name for name, present in sdk["headers"].items() if not present
    ]
    if missing_headers:
        issues.append(_issue(
            "hggc_headers_missing", "blocker",
            "Required HGGC headers are missing.", missing_headers,
            "Install the SDK development package, not runtime-only package.",
        ))
    if packages["torch"] is None:
        issues.append(_issue(
            "torch_missing", "blocker", "PyTorch is not installed.", None,
            "Install the organizer-provided PPU PyTorch wheel/image.",
        ))
    if packages["transformers"] is None:
        issues.append(_issue(
            "transformers_missing", "blocker",
            "Transformers is not installed.", None,
            "Install an organizer-approved version containing Qwen3.5.",
        ))
    elif not compatibility["transformers_qwen35_module"]:
        issues.append(_issue(
            "transformers_qwen35_missing", "blocker",
            "Installed Transformers has no qwen3_5 module.",
            packages["transformers"],
            "Upgrade to an organizer-approved Qwen3.5-capable build.",
        ))
    if model["requested"] and not model["valid"]:
        issues.append(_issue(
            "model_layout_invalid", "blocker",
            "Locked model files/config do not match the expected fingerprint.",
            model["blockers"], "Restore the locked revision before benchmarking.",
        ))
    if not model["requested"]:
        issues.append(_issue(
            "model_not_checked", "warning",
            "No model path was supplied; identity and structure were not checked.",
            None, "Re-run with --model-path pointing to the locked model.",
        ))
    if packages["vllm"] is None:
        issues.append(_issue(
            "vllm_missing", "warning",
            "vLLM is not installed; only Transformers eager may be possible.",
            None, "Use the organizer PPU-vLLM package if it supports Qwen3.5.",
        ))
    source_checks = (
        ("vllm_qwen35_source", "No Qwen3.5 model registration was found."),
        ("vllm_gdn_source", "No GDN implementation was found."),
        ("vllm_causal_conv_source", "No causal-conv1d path was found."),
        ("vllm_ppu_source", "No PPU-specific path was found."),
    )
    for field, summary in source_checks:
        if not compatibility[field]:
            issues.append(_issue(
                field, "warning", summary,
                compatibility["vllm_source_path"],
                "Confirm the private image or port the missing path.",
            ))
    if report["organizer_lock"].get("package") != "dndx_participant-v1.2":
        issues.append(_issue(
            "organizer_v12_not_locked", "warning",
            "Repository organizer lock is not v1.2.",
            report["organizer_lock"].get("package"),
            "Obtain v1.2, diff it, update hashes, and rerun contract tests.",
        ))
    smoke = report["torch"]["device_smoke"]
    if not smoke["requested"]:
        issues.append(_issue(
            "device_smoke_not_run", "info",
            "Tiny PPU tensor execution was not requested.", None,
            "On the approved isolated node, add --run-device-smoke.",
        ))
    elif not smoke.get("passed", False):
        issues.append(_issue(
            "device_smoke_failed", "blocker",
            "Tiny BF16 tensor execution did not pass.", smoke,
            "Fix PPU PyTorch/runtime before loading the full model.",
        ))
    return issues


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    packages = _package_versions()
    device_nodes = sorted(
        set(glob.glob("/dev/alixpu*") + glob.glob("/dev/*ppu*"))
    )
    ppu_smi = _command_output(["ppu-smi", "--version"])
    hardware_visible = bool(device_nodes) or bool(
        ppu_smi.get("available") and ppu_smi.get("returncode") == 0
    )
    qwen35_module = _module_available("transformers.models.qwen3_5")
    vllm_source = args.vllm_source.resolve()
    model_layout = _model_layout_probe(
        args.model_path, args.model_lock.resolve(), args.verify_model_hash,
    )
    model_config_smoke = _model_config_smoke(args.model_path)
    sdk = _sdk_probe(args.sdk_root)

    compatibility = {
        "transformers_stack_present": (
            packages["torch"] is not None
            and packages["transformers"] is not None
        ),
        "transformers_qwen35_module": qwen35_module,
        "vllm_stack_present": (
            packages["torch"] is not None and packages["vllm"] is not None
        ),
        "vllm_source_path": str(vllm_source),
        "vllm_qwen35_source": _source_contains(
            vllm_source, ("Qwen3_5ForConditionalGeneration", "qwen3_5"),
        ),
        "vllm_gdn_source": _source_contains(
            vllm_source, ("GatedDeltaNet", "gated_delta", "gated delta"),
        ),
        "vllm_causal_conv_source": _source_contains(
            vllm_source, ("causal_conv1d_fwd", "causal_conv1d_update"),
        ),
        "vllm_ppu_source": _source_contains(
            vllm_source, ("PPU", "ppu.mma", "__HGGCCC__", "alixpu"),
        ),
    }
    torch_probe = _torch_probe(args.run_device_smoke, hardware_visible)
    organizer_lock = _load_json(
        REPO_ROOT / "configs" / "organizer-lock.json"
    ) or {}
    model_checks = (
        (model_layout["requested"], "model_path_not_provided"),
        (bool(model_layout.get("valid")), "model_layout_invalid"),
        (
            bool(model_config_smoke.get("loaded")),
            "transformers_model_config_or_processor_load_failed",
        ),
    )
    sdk_checks = (
        (hardware_visible, "ppu_device_not_visible"),
        (sdk["present"], "ppu_sdk_missing"),
        (sdk["compiler_executable"], "ppu_compiler_missing"),
        (all(sdk["headers"].values()), "hggc_headers_missing"),
    )
    transformers_route = _assess_route(
        sdk_checks + (
            (packages["torch"] is not None, "torch_not_installed"),
            (packages["transformers"] is not None, "transformers_not_installed"),
            (qwen35_module, "transformers_qwen35_module_missing"),
        ) + model_checks
    )
    vllm_route = _assess_route(
        sdk_checks + (
            (packages["vllm"] is not None, "vllm_not_installed"),
            (vllm_source.is_dir(), "vllm_source_missing"),
            (compatibility["vllm_ppu_source"], "vllm_ppu_source_missing"),
            (compatibility["vllm_qwen35_source"], "vllm_qwen35_source_missing"),
            (compatibility["vllm_gdn_source"], "vllm_gdn_source_missing"),
            (
                compatibility["vllm_causal_conv_source"],
                "vllm_causal_conv_source_missing",
            ),
        ) + model_checks
    )

    report: dict[str, Any] = {
        "schema_version": 2,
        "scope": "preflight_only_not_a_deployment_or_performance_result",
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "hostname": platform.node(),
        },
        "environment": {
            "PPU_SDK": os.getenv("PPU_SDK"),
            "CUDA_HOME": os.getenv("CUDA_HOME"),
            "LD_LIBRARY_PATH_set": bool(os.getenv("LD_LIBRARY_PATH")),
            "PYTHONPATH_set": bool(os.getenv("PYTHONPATH")),
            "visible_device_variables": {
                name: os.getenv(name)
                for name in (
                    "CUDA_VISIBLE_DEVICES", "PPU_VISIBLE_DEVICES",
                    "HGGC_VISIBLE_DEVICES",
                )
                if os.getenv(name) is not None
            },
        },
        "repository": {
            "root": str(REPO_ROOT),
            "commit": _command_output([
                "git", "-C", str(REPO_ROOT), "rev-parse", "HEAD",
            ]),
            "branch": _command_output([
                "git", "-C", str(REPO_ROOT), "branch", "--show-current",
            ]),
            "status": _command_output([
                "git", "-C", str(REPO_ROOT), "status", "--short",
            ]),
        },
        "hardware": {
            "visible": hardware_visible, "device_nodes": device_nodes,
            "ppu_smi": ppu_smi,
        },
        "sdk": sdk,
        "packages": packages,
        "torch": torch_probe,
        "compatibility": compatibility,
        "model_layout": model_layout,
        "model_config_smoke": model_config_smoke,
        "organizer_lock": organizer_lock,
        "routes": {
            "transformers_eager": transformers_route, "vllm": vllm_route,
        },
        "readiness": {
            "toolchain_prerequisites_ready": _assess_route(sdk_checks)["ready"],
            "transformers_route_prerequisites_ready": transformers_route["ready"],
            "vllm_route_prerequisites_ready": vllm_route["ready"],
            "device_smoke_passed": bool(
                torch_probe["device_smoke"].get("passed", False)
            ),
            "full_model_load_tested": False,
            "single_sample_inference_tested": False,
            "accuracy_ttft_throughput_tested": False,
        },
        "known_risks": KNOWN_RISKS,
        "open_questions": OPEN_QUESTIONS,
    }
    report["issues"] = _build_issues(report)
    report["issue_counts"] = dict(
        Counter(issue["severity"] for issue in report["issues"])
    )
    report["deployment_ready"] = False
    report["deployment_ready_reason"] = (
        "Preflight cannot prove deployment; full model load, real multimodal "
        "generation, and benchmark evidence are still required."
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    smoke = report["torch"]["device_smoke"]
    smoke_status: str | bool = (
        readiness["device_smoke_passed"] if smoke["requested"] else "未运行"
    )
    lines = [
        "# PPU / Qwen3.5 首次验证报告",
        "",
        "> 本报告仅表示预检结果，不等于模型已经部署，也不构成比赛性能数据。",
        "",
        "## 摘要",
        "",
        f"- PPU 可见：`{report['hardware']['visible']}`",
        f"- PPU SDK：`{report['sdk']['present']}`，编译器可执行：`{report['sdk']['compiler_executable']}`",
        f"- Transformers 路线前置条件：`{readiness['transformers_route_prerequisites_ready']}`",
        f"- vLLM 路线前置条件：`{readiness['vllm_route_prerequisites_ready']}`",
        f"- PPU 设备冒烟：`{smoke_status}`",
        f"- Blocker/Warning/Info：`{report['issue_counts'].get('blocker', 0)}` / `{report['issue_counts'].get('warning', 0)}` / `{report['issue_counts'].get('info', 0)}`",
        "",
        "## 自动发现的问题",
        "",
        "| 级别 | ID | 现象 | 下一步 |",
        "|---|---|---|---|",
    ]
    for issue in report["issues"]:
        summary = str(issue["summary"]).replace("|", "\\|")
        action = str(issue["action"]).replace("|", "\\|")
        lines.append(
            f"| {issue['severity']} | `{issue['id']}` | {summary} | {action} |"
        )
    lines.extend(["", "## 尚需主办方/环境确认", ""])
    lines.extend(f"- {question}" for question in report["open_questions"])
    lines.extend([
        "", "## 后续验收阶梯", "",
        "1. PPU/SDK/编译器/Python 静态预检。",
        "2. BF16 小张量冒烟，确认 PyTorch 真正落到 PPU。",
        "3. HGGC vectorAdd/BF16 GEMV 编译、数值校验和 memcheck。",
        "4. 锁定模型完整加载，确认无 CPU/meta/disk offload。",
        "5. 单张真实图片生成，确认 processor、GDN、causal-conv1d、attention 和 cache。",
        "6. 固定 20 条三次复测 Accuracy、首 token TTFT 和 decode throughput。",
        "7. PPU Profile 后再选择 GEMV、GDN、causal-conv1d 或 attention 优化。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = build_report(args)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown_path = args.markdown_output.resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
