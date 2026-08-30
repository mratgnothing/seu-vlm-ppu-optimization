#!/usr/bin/env python3
"""Build the isolated experimental PyTorch/acBLASLt square extension."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load


ROOT = Path(__file__).resolve().parent
PPU_SDK = Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK"))
BUILD_DIR = ROOT / "build" / "acblaslt_square_extension"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get(
    "PATH", ""
)

bridge_object = BUILD_DIR / "acblaslt_matmul_wrapper.o"
subprocess.run(
    [
        str(PPU_SDK / "bin" / "clang++"),
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-c",
        str(ROOT.parent / "microbench" / "acblaslt_matmul_wrapper.cpp"),
        f"-I{PPU_SDK / 'include'}",
        "-o",
        str(bridge_object),
    ],
    check=True,
)
(BUILD_DIR / "seu_acblaslt_square_ext.so").unlink(missing_ok=True)

module = load(
    name="seu_acblaslt_square_ext",
    sources=[str(ROOT / "acblaslt_square_extension.cpp")],
    extra_include_paths=[str(Path(CUDA_HOME) / "include")],
    extra_cflags=["-O3"],
    extra_ldflags=[
        str(bridge_object),
        f"-L{PPU_SDK / 'lib'}",
        f"-Wl,-rpath,{PPU_SDK / 'lib'}",
        "-lacblasLt",
        "-lacblas",
        "-lhggcrt1",
        "-lc10_cuda",
        "-ltorch_cuda",
    ],
    build_directory=str(BUILD_DIR),
    with_cuda=False,
    verbose=True,
)
print(module.__file__)
