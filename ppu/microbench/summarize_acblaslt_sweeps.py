#!/usr/bin/env python3
"""Summarize RESULT records from one or more acBLASLt sweep logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_log(path: Path) -> dict[str, object]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("RESULT "):
            records.append(json.loads(line.removeprefix("RESULT ")))
    baselines = [record for record in records if record["backend"] == "torch.mv"]
    candidates = [
        record
        for record in records
        if record["backend"] == "acblasLtMatmul" and record.get("passed")
    ]
    if len(baselines) != 1:
        raise ValueError(f"{path}: expected one torch.mv record, got {len(baselines)}")
    if not candidates:
        raise ValueError(f"{path}: no passing acBLASLt heuristic")
    best = min(candidates, key=lambda record: float(record["average_ms"]))
    return {
        "source": str(path),
        "shape": [int(baselines[0]["n"]), int(baselines[0]["k"])],
        "torch_mv_ms": float(baselines[0]["average_ms"]),
        "heuristics_returned": len(
            [record for record in records if record["backend"] == "acblasLtMatmul"]
        ),
        "passing_heuristics": len(candidates),
        "best": best,
        "best_speedup": float(best["torch_speedup"]),
        "integration_candidate": (
            bool(best["exact"]) and float(best["torch_speedup"]) >= 1.03
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {"sweeps": [parse_log(path) for path in args.logs]}
    payload["any_integration_candidate"] = any(
        sweep["integration_candidate"] for sweep in payload["sweeps"]
    )
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
