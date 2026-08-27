#!/usr/bin/env python3
"""Summarize selected events and matrix shapes from a PyTorch Chrome trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TRACKED_NAMES = {
    "aten::mm",
    "aten::linear",
    "aten::cat",
    "aten::empty_strided",
    "cudaLaunchKernel",
}


def normalize_dims(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload)
    counts: Counter[str] = Counter()
    durations_us: defaultdict[str, float] = defaultdict(float)
    mm_shapes: Counter[str] = Counter()
    mm_shape_durations_us: defaultdict[str, float] = defaultdict(float)

    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        if name in TRACKED_NAMES:
            counts[name] += 1
            durations_us[name] += float(event.get("dur", 0.0))
        if name == "aten::mm":
            shape = normalize_dims(event.get("args", {}).get("Input Dims"))
            mm_shapes[shape] += 1
            mm_shape_durations_us[shape] += float(event.get("dur", 0.0))

    summary = {
        "trace": str(args.trace),
        "tracked_events": {
            name: {
                "count": counts[name],
                "duration_ms": round(durations_us[name] / 1000.0, 6),
            }
            for name in sorted(TRACKED_NAMES)
        },
        "aten_mm_shapes": [
            {
                "input_dims": shape,
                "count": count,
                "duration_ms": round(mm_shape_durations_us[shape] / 1000.0, 6),
            }
            for shape, count in mm_shapes.most_common()
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
