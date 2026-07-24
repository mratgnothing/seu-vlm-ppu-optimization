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
        loaded = model._model
        device_map = getattr(loaded, "hf_device_map", None)
        parameter_devices = Counter(str(parameter.device) for parameter in loaded.parameters())
        payload.update(
            {
                "passed": True,
                "backend_name": model.backend_name,
                "model_class": type(loaded).__name__,
                "primary_device": str(loaded.device),
                "device_map": device_map,
                "parameter_devices": dict(parameter_devices),
                "memory_footprint_bytes": (
                    loaded.get_memory_footprint()
                    if hasattr(loaded, "get_memory_footprint")
                    else None
                ),
            }
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
