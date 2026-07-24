#!/usr/bin/env python3
"""Split a TSV dataset into deterministic, restartable benchmark chunks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare deterministic TSV chunks")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=200)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_chunks(
    dataset: Path,
    output_dir: Path,
    *,
    chunk_size: int,
) -> dict[str, Any]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    dataset = dataset.resolve()
    output_dir = output_dir.resolve()
    csv.field_size_limit(2**31 - 1)
    with dataset.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError("Dataset has no header")
    if not rows:
        raise ValueError("Dataset has no samples")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_count = math.ceil(len(rows) / chunk_size)
    chunks: list[dict[str, Any]] = []
    for zero_index in range(chunk_count):
        start = zero_index * chunk_size
        selected = rows[start : start + chunk_size]
        one_index = zero_index + 1
        filename = (
            f"{dataset.stem}.chunk-{one_index:04d}-of-{chunk_count:04d}.tsv"
        )
        path = output_dir / filename
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(selected)
        chunks.append(
            {
                "index": one_index,
                "file": filename,
                "sample_count": len(selected),
                "sha256": sha256_file(path),
                "first_question_id": str(selected[0].get("index", "")),
                "last_question_id": str(selected[-1].get("index", "")),
            }
        )

    return {
        "source_file": dataset.name,
        "source_path": str(dataset),
        "source_sha256": sha256_file(dataset),
        "source_sample_count": len(rows),
        "chunk_size": chunk_size,
        "chunk_count": chunk_count,
        "output_dir": str(output_dir),
        "chunks": chunks,
    }


def main() -> int:
    args = parse_args()
    manifest = prepare_chunks(
        args.dataset,
        args.output_dir,
        chunk_size=args.chunk_size,
    )
    rendered = json.dumps(manifest, ensure_ascii=False, indent=2)
    manifest_path = args.manifest.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
