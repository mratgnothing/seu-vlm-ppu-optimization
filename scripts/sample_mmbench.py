#!/usr/bin/env python3
"""Create a deterministic proportional stratified subset of a public TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample an MMBench public TSV")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-samples", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--strata",
        nargs="+",
        default=["category", "l2-category"],
    )
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def allocate_proportional_counts(
    group_sizes: dict[tuple[str, ...], int],
    target: int,
) -> dict[tuple[str, ...], int]:
    total = sum(group_sizes.values())
    if target <= 0 or target > total:
        raise ValueError(f"target must be within 1..{total}, got {target}")

    exact = {
        key: target * size / total
        for key, size in group_sizes.items()
    }
    allocated = {
        key: min(size, math.floor(exact[key]))
        for key, size in group_sizes.items()
    }
    remaining = target - sum(allocated.values())
    ranking = sorted(
        group_sizes,
        key=lambda key: (
            -(exact[key] - math.floor(exact[key])),
            key,
        ),
    )
    while remaining > 0:
        progressed = False
        for key in ranking:
            if allocated[key] >= group_sizes[key]:
                continue
            allocated[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise RuntimeError("Unable to finish proportional allocation")
    return allocated


def stratified_sample(
    rows: list[dict[str, str]],
    *,
    target: int,
    seed: int,
    strata: tuple[str, ...],
) -> list[dict[str, str]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple((row.get(field) or "").strip() for field in strata)
        groups[key].append(row)
    counts = allocate_proportional_counts(
        {key: len(group) for key, group in groups.items()},
        target,
    )
    randomizer = random.Random(seed)
    selected: list[dict[str, str]] = []
    for key in sorted(groups):
        candidates = list(groups[key])
        randomizer.shuffle(candidates)
        selected.extend(candidates[: counts[key]])
    randomizer.shuffle(selected)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(
    rows: list[dict[str, str]],
    strata: tuple[str, ...],
) -> dict[str, int]:
    counts = Counter(
        " / ".join((row.get(field) or "").strip() for field in strata)
        for row in rows
    )
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    source_path = args.dataset_path.resolve()
    output_path = args.output.resolve()
    strata = tuple(args.strata)

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Dataset has no header")
    missing_fields = [field for field in strata if field not in fieldnames]
    if missing_fields:
        raise ValueError(f"Missing strata fields: {missing_fields}")

    selected = stratified_sample(
        rows,
        target=args.num_samples,
        seed=args.seed,
        strata=strata,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)

    manifest: dict[str, Any] = {
        "source_file": source_path.name,
        "source_sha256": _sha256(source_path),
        "output_file": output_path.name,
        "output_sha256": _sha256(output_path),
        "seed": args.seed,
        "strata": list(strata),
        "source_count": len(rows),
        "sample_count": len(selected),
        "source_distribution": _distribution(rows, strata),
        "sample_distribution": _distribution(selected, strata),
    }
    rendered = json.dumps(manifest, indent=2, ensure_ascii=False)
    print(rendered)
    if args.manifest:
        manifest_path = args.manifest.resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
