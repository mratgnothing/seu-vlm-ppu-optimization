from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

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
        payload.update(
            {
                "passed": True,
                "backend_name": model.backend_name,
                "model_class": type(loaded).__name__,
                "device_map": device_map,
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

