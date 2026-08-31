#!/usr/bin/env python3
"""Summarize selected events and matrix shapes from a PyTorch Chrome trace."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TRACKED_NAMES = {
    "aten::_to_copy",
    "aten::add",
    "aten::mm",
    "aten::linear",
    "aten::cat",
    "aten::clone",
    "aten::copy_",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::matmul",
    "aten::mul",
    "aten::sum",
    "aten::to",
    "cuLaunchKernel",
    "cudaDeviceSynchronize",
    "cudaFree",
    "cudaGetDeviceProperties_v2",
    "cudaLaunchKernel",
    "cudaStreamSynchronize",
}

DEFAULT_SHAPE_OPS = {
    "aten::_to_copy",
    "aten::add",
    "aten::cat",
    "aten::clone",
    "aten::copy_",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::linear",
    "aten::matmul",
    "aten::mm",
    "aten::mul",
    "aten::sum",
    "aten::to",
}


def normalize_dims(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def summarize_trace(
    events: list[dict[str, object]],
    *,
    top: int,
    shape_ops: set[str],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    durations_us: defaultdict[str, float] = defaultdict(float)
    operator_shapes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    operator_shape_durations_us: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    category_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    category_durations_us: defaultdict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )

    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", "uncategorized"))
        duration_us = float(event.get("dur", 0.0))
        category_counts[category][name] += 1
        category_durations_us[category][name] += duration_us
        if name in TRACKED_NAMES:
            counts[name] += 1
            durations_us[name] += duration_us
        if name in shape_ops:
            args = event.get("args", {})
            dims = args.get("Input Dims") if isinstance(args, dict) else None
            shape = normalize_dims(dims)
            operator_shapes[name][shape] += 1
            operator_shape_durations_us[name][shape] += duration_us

    shape_summary = {
        name: [
            {
                "input_dims": shape,
                "count": count,
                "duration_ms": round(
                    operator_shape_durations_us[name][shape] / 1000.0, 6
                ),
            }
            for shape, count in shapes.most_common()
        ]
        for name, shapes in sorted(operator_shapes.items())
    }
    return {
        "tracked_events": {
            name: {
                "count": counts[name],
                "duration_ms": round(durations_us[name] / 1000.0, 6),
            }
            for name in sorted(TRACKED_NAMES)
        },
        # Keep the original key for consumers created before operator_shapes.
        "aten_mm_shapes": shape_summary.get("aten::mm", []),
        "operator_shapes": shape_summary,
        "top_events_by_category": {
            category: [
                {
                    "name": name,
                    "count": category_counts[category][name],
                    "duration_ms": round(duration_us / 1000.0, 6),
                }
                for name, duration_us in sorted(
                    durations.items(), key=lambda item: item[1], reverse=True
                )[:top]
            ]
            for category, durations in sorted(category_durations_us.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--shape-op",
        action="append",
        default=[],
        help="CPU operator whose Input Dims should be grouped; repeat as needed",
    )
    args = parser.parse_args()

    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", payload)
    if not isinstance(events, list):
        raise TypeError("traceEvents must be a list")
    shape_ops = set(args.shape_op) if args.shape_op else DEFAULT_SHAPE_OPS
    summary = summarize_trace(events, top=args.top, shape_ops=shape_ops)
    summary["trace"] = str(args.trace)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
