#!/usr/bin/env python3
"""Build the isolated acBLASLt BF16 matmul heuristic probe."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "libseu_acblaslt_matmul_probe.so",
    )
    args = parser.parse_args()
    ppu_sdk = Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(ppu_sdk / "bin" / "clang++"),
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(ROOT / "acblaslt_matmul_wrapper.cpp"),
            f"-I{ppu_sdk / 'include'}",
            f"-L{ppu_sdk / 'lib'}",
            f"-Wl,-rpath,{ppu_sdk / 'lib'}",
            "-lacblasLt",
            "-lacblas",
            "-lhggcrt1",
            "-o",
            str(args.output),
        ],
        check=True,
    )
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
