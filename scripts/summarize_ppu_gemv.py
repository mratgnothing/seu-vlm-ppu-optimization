#!/usr/bin/env python3
"""Summarize repeated PPU GEMV RESULT records with reference improvements."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("RESULT "):
                records.append(json.loads(line.removeprefix("RESULT ")))
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["kernel"],
            record["n"],
            record["k"],
            record.get("threads"),
            record.get("matrix_copies", 1),
        )
        grouped[key].append(record)

    reference_medians: dict[tuple[int, int, int], float] = {}
    for key, values in grouped.items():
        kernel, n, k, _threads, copies = key
        if kernel == "bf16_gemv_reference":
            reference_medians[(n, k, copies)] = statistics.median(
                float(value["average_ms"]) for value in values
            )

    rows: list[dict[str, Any]] = []
    for key, values in grouped.items():
        kernel, n, k, threads, copies = key
        timings = [float(value["average_ms"]) for value in values]
        median_ms = statistics.median(timings)
        reference_ms = reference_medians.get((n, k, copies))
        rows.append(
            {
                "kernel": kernel,
                "n": n,
                "k": k,
                "threads": threads,
                "matrix_copies": copies,
                "runs": len(values),
                "all_passed": all(bool(value.get("passed")) for value in values),
                "median_ms": median_ms,
                "min_ms": min(timings),
                "max_ms": max(timings),
                "reference_ms": reference_ms,
                "latency_reduction_pct": (
                    100.0 * (1.0 - median_ms / reference_ms)
                    if reference_ms and kernel != "bf16_gemv_reference"
                    else None
                ),
                "speedup": (
                    reference_ms / median_ms
                    if reference_ms and kernel != "bf16_gemv_reference"
                    else None
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["matrix_copies"], row["n"], row["k"], row["median_ms"]
        ),
    )


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Kernel | N | K | Threads | Copies | Runs | Median ms | Min-Max ms | Latency reduction | Speedup | Correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        reduction = row["latency_reduction_pct"]
        speedup = row["speedup"]
        lines.append(
            f"| {row['kernel']} | {row['n']} | {row['k']} | "
            f"{row['threads'] if row['threads'] is not None else '-'} | "
            f"{row['matrix_copies']} | {row['runs']} | {row['median_ms']:.6f} | "
            f"{row['min_ms']:.6f}-{row['max_ms']:.6f} | "
            f"{f'{reduction:.2f}%' if reduction is not None else '-'} | "
            f"{f'{speedup:.3f}x' if speedup is not None else '-'} | "
            f"{'yes' if row['all_passed'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = render_markdown(summarize(parse_records(args.logs)))
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
