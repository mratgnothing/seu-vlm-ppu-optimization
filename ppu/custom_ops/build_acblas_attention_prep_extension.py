#!/usr/bin/env python3
"""Build grouped attention QKV + q/k RMSNorm+RoPE PPU extension."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from torch.utils.cpp_extension import CUDA_HOME, load


ROOT = Path(__file__).resolve().parent
PPU_SDK = Path(os.getenv("PPU_SDK", "/usr/local/PPU_SDK"))
BUILD_DIR = ROOT / "build" / "acblas_attention_prep_extension"
GDN_LIBRARY = Path(
    os.getenv("SEU_PPU_GDN_LIBRARY", str(ROOT / "build" / "libseu_ppu_gdn.so"))
).resolve()
if not GDN_LIBRARY.is_file():
    raise FileNotFoundError(f"GDN library not found: {GDN_LIBRARY}")

BUILD_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get(
    "PATH", ""
)

bridge_object = BUILD_DIR / "acblas_attention_prep_wrapper.o"
subprocess.run(
    [
        str(PPU_SDK / "bin" / "clang++"),
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-c",
        str(ROOT / "acblas_attention_prep_wrapper.cpp"),
        f"-I{PPU_SDK / 'include'}",
        "-o",
        str(bridge_object),
    ],
    check=True,
)
(BUILD_DIR / "seu_acblas_attention_prep_ext.so").unlink(missing_ok=True)

module = load(
    name="seu_acblas_attention_prep_ext",
    sources=[str(ROOT / "acblas_attention_prep_extension.cpp")],
    extra_include_paths=[str(Path(CUDA_HOME) / "include")],
    extra_cflags=["-O3"],
    extra_ldflags=[
        str(bridge_object),
        f"-L{GDN_LIBRARY.parent}",
        f"-L{PPU_SDK / 'lib'}",
        f"-Wl,-rpath,{PPU_SDK / 'lib'}",
        f"-Wl,-rpath,{GDN_LIBRARY.parent}",
        "-lseu_ppu_gdn",
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
