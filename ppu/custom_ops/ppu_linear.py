"""Direct acBLAS BF16 GEMV integration for fixed batch=1 decode linears."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Callable

import torch


class PPUACBLASLinearLibrary:
    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        algorithm: int = -1,
    ) -> None:
        if library_path is None:
            library_path = (
                Path(__file__).with_name("build") / "libseu_acblas_linear.so"
            )
        self.path = Path(library_path).resolve()
        self.algorithm = algorithm
        self.library = ctypes.CDLL(str(self.path))
        self.launch = self.library.seu_acblas_linear_bf16
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int

    def linear_decode(
        self, input_tensor: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        _validate_linear_inputs(input_tensor, weight)
        output_features, input_features = weight.shape
        output = torch.empty(
            (*input_tensor.shape[:-1], output_features),
            dtype=torch.bfloat16,
            device=input_tensor.device,
        )
        stream = torch.cuda.current_stream(input_tensor.device).cuda_stream
        status = self.launch(
            weight.data_ptr(),
            input_tensor.data_ptr(),
            output.data_ptr(),
            output_features,
            input_features,
            self.algorithm,
            stream,
        )
        if status != 0:
            raise RuntimeError(
                f"PPU acBLAS linear launch failed with status {status}"
            )
        return output


def patch_linear_decode_module(
    module,
    library: PPUACBLASLinearLibrary,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if type(module).__name__ != "Linear":
        raise TypeError("acBLAS decode patch requires torch.nn.Linear")
    if module.bias is not None:
        raise ValueError("acBLAS decode patch requires a bias-free linear")
    weight = module.weight
    if weight.device.type != "cuda" or weight.dtype != torch.bfloat16:
        raise TypeError("acBLAS decode patch requires BF16 PPU weights")
    if not weight.is_contiguous():
        raise ValueError("acBLAS decode patch requires contiguous weights")
    original_forward = module.forward
    launch = library.launch
    algorithm = library.algorithm
    output_features = module.out_features
    input_features = module.in_features

    def acblas_linear_forward(input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.shape == (1, 1, input_features):
            output = torch.empty(
                (1, 1, output_features),
                dtype=torch.bfloat16,
                device=input_tensor.device,
            )
            stream = torch.cuda.current_stream(input_tensor.device).cuda_stream
            status = launch(
                weight.data_ptr(),
                input_tensor.data_ptr(),
                output.data_ptr(),
                output_features,
                input_features,
                algorithm,
                stream,
            )
            if status != 0:
                raise RuntimeError(
                    f"PPU acBLAS linear launch failed with status {status}"
                )
            return output
        return original_forward(input_tensor)

    return acblas_linear_forward


def _validate_linear_inputs(
    input_tensor: torch.Tensor, weight: torch.Tensor
) -> None:
    if input_tensor.device.type != "cuda" or weight.device != input_tensor.device:
        raise ValueError("acBLAS linear input/weight must share the PPU device")
    if input_tensor.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("acBLAS linear input/weight must be torch.bfloat16")
    if input_tensor.shape != (1, 1, weight.shape[1]):
        raise ValueError("acBLAS linear requires batch=1 single-token decode")
    if not weight.is_contiguous() or input_tensor.stride(-1) != 1:
        raise ValueError("acBLAS linear input/weight must be contiguous")
