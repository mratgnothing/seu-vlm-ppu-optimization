from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation_wrapper import VLMModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the real model without generating")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="Fail when any parameter is on CPU/meta/disk or no accelerator is visible.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    payload: dict[str, object] = {
        "model_path": str(args.model_path.resolve()),
        "backend": "transformers",
    }

    try:
        model = VLMModel(str(args.model_path), backend="transformers", device="auto")
        torch = model._torch
        loaded = model._model
        device_map = getattr(loaded, "hf_device_map", None)
        parameter_devices = Counter(str(parameter.device) for parameter in loaded.parameters())
        parameter_numel_by_device: Counter[str] = Counter()
        for parameter in loaded.parameters():
            parameter_numel_by_device[str(parameter.device)] += parameter.numel()
        mapped_devices = (
            {str(device) for device in device_map.values()}
            if isinstance(device_map, dict)
            else set()
        )
        observed_devices = set(parameter_devices) | mapped_devices
        offload_devices = {
            device
            for device in observed_devices
            if device.lower().startswith(("cpu", "meta", "disk"))
        }
        cuda_available = bool(torch.cuda.is_available())
        accelerator_resident = bool(parameter_devices) and not offload_devices
        if args.require_accelerator:
            accelerator_resident = accelerator_resident and cuda_available
        payload.update(
            {
                "passed": accelerator_resident if args.require_accelerator else True,
                "backend_name": model.backend_name,
                "model_class": type(loaded).__name__,
                "primary_device": str(getattr(loaded, "device", "unknown")),
                "device_map": device_map,
                "parameter_devices": dict(parameter_devices),
                "parameter_numel_by_device": dict(parameter_numel_by_device),
                "mapped_devices": sorted(mapped_devices),
                "offload_devices": sorted(offload_devices),
                "accelerator_required": args.require_accelerator,
                "accelerator_resident": accelerator_resident,
                "torch_cuda_available": cuda_available,
                "accelerator_devices": (
                    [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                        }
                        for index in range(torch.cuda.device_count())
                    ]
                    if cuda_available
                    else []
                ),
                "memory_footprint_bytes": (
                    loaded.get_memory_footprint()
                    if hasattr(loaded, "get_memory_footprint")
                    else None
                ),
            }
        )
        if args.require_accelerator and not accelerator_resident:
            payload["failure_reason"] = (
                "Model parameters are not fully accelerator-resident or the "
                "PPU PyTorch CUDA-compatibility backend is unavailable."
            )
    except Exception as exc:
        payload.update(
            {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    payload["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
