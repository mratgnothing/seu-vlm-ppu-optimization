from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in [
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "Pillow",
                "numpy",
                "safetensors",
            ]
        },
    }

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        report["torch_runtime"] = {
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "gpu_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory if cuda_available else None
            ),
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
        }
    except Exception as exc:
        report["torch_runtime_error"] = f"{type(exc).__name__}: {exc}"

    output = repo_root / "artifacts" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    runtime = report.get("torch_runtime")
    return 0 if isinstance(runtime, dict) and runtime.get("cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())

