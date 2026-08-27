#!/usr/bin/env python3
"""Report the local PyTorch C++ extension build contract."""

from __future__ import annotations

import json

import torch
from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths


print(
    json.dumps(
        {
            "torch": torch.__version__,
            "includes": include_paths(),
            "libraries": library_paths(),
            "cxx11abi": torch._C._GLIBCXX_USE_CXX11_ABI,
            "cuda_home": CUDA_HOME,
        },
        indent=2,
    )
)
