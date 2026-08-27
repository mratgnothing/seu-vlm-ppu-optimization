#!/usr/bin/env python3
"""Build the experimental registered PyTorch/acBLAS linear extension."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load


ROOT = Path(__file__).resolve().parent
PPU_SDK = Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK"))
BUILD_DIR = ROOT / "build" / "acblas_linear_extension"
BUILD_DIR.mkdir(parents=True, exist_ok=True)
# Non-interactive SSH shells do not necessarily activate the virtualenv, so
# make the Ninja installed beside this Python interpreter discoverable.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

bridge_object = BUILD_DIR / "acblas_linear_wrapper.o"
subprocess.run(
    [
        str(PPU_SDK / "bin" / "clang++"),
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-c",
        str(ROOT / "acblas_linear_wrapper.cpp"),
        f"-I{PPU_SDK / 'include'}",
        "-o",
        str(bridge_object),
    ],
    check=True,
)
# The bridge object is passed as a linker flag, so PyTorch's generated Ninja
# graph does not know it is an input. Force the cheap relink after rebuilding
# the object; otherwise a changed bridge can silently leave a stale extension.
(BUILD_DIR / "seu_acblas_linear_ext.so").unlink(missing_ok=True)

module = load(
    name="seu_acblas_linear_ext",
    sources=[str(ROOT / "acblas_linear_extension.cpp")],
    extra_include_paths=[
        str(Path(CUDA_HOME) / "include"),
    ],
    extra_cflags=["-O3"],
    extra_ldflags=[
        str(bridge_object),
        f"-L{PPU_SDK / 'lib'}",
        f"-L{BUILD_DIR}",
        f"-Wl,-rpath,{PPU_SDK / 'lib'}",
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
