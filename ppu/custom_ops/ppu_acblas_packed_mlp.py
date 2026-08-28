"""One-entry decode path for packed Qwen3.5 MLP on PPU."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 6144


class PPUACBLASPackedMLPExtension:
    def __init__(
        self,
        build_dir: str | Path,
        *,
        gate_up_algorithm: int = -1,
        down_algorithm: int = -1,
        swiglu_threads: int = 128,
    ) -> None:
        if swiglu_threads <= 0 or swiglu_threads > 1024 or swiglu_threads % 32:
            raise ValueError("swiglu_threads must be a multiple of 32 through 1024")
        build_dir = Path(build_dir).resolve()
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        self.module = importlib.import_module("seu_acblas_packed_mlp_ext")
        self.gate_up_algorithm = gate_up_algorithm
        self.down_algorithm = down_algorithm
        self.swiglu_threads = swiglu_threads

    def patch_module(self, module) -> Callable[[torch.Tensor], torch.Tensor]:
        if type(module).__name__ != "Qwen3_5MLP":
            raise TypeError("acBLAS packed MLP requires Qwen3_5MLP")
        if hasattr(module, "_seu_acblas_packed_mlp_original_forward"):
            raise RuntimeError("module already has the acBLAS packed MLP patch")
        packed_weight = getattr(module, "_seu_gate_up_weight", None)
        if packed_weight is None or packed_weight.shape != (12288, HIDDEN_SIZE):
            raise RuntimeError("run pack_qwen35_mlp_module before this patch")
        if module.down_proj.weight.shape != (HIDDEN_SIZE, INTERMEDIATE_SIZE):
            raise ValueError("unexpected down projection shape")
        if (
            packed_weight.dtype != torch.bfloat16
            or module.down_proj.weight.dtype != torch.bfloat16
            or packed_weight.device.type != "cuda"
            or module.down_proj.weight.device != packed_weight.device
        ):
            raise TypeError("packed MLP weights must be BF16 on one PPU device")

        options = {"device": packed_weight.device, "dtype": torch.bfloat16}
        module.register_buffer(
            "_seu_acblas_packed_projected",
            torch.empty(1, 1, 12288, **options),
            persistent=False,
        )
        module.register_buffer(
            "_seu_acblas_packed_activated",
            torch.empty(1, 1, INTERMEDIATE_SIZE, **options),
            persistent=False,
        )
        module.register_buffer(
            "_seu_acblas_packed_output",
            torch.empty(1, 1, HIDDEN_SIZE, **options),
            persistent=False,
        )

        original_forward = module.forward
        extension = self.module
        gate_up_algorithm = self.gate_up_algorithm
        down_algorithm = self.down_algorithm
        swiglu_threads = self.swiglu_threads
        expected_stream = torch.cuda.current_stream(
            device=packed_weight.device
        ).cuda_stream

        def acblas_packed_forward(input_tensor: torch.Tensor) -> torch.Tensor:
            if (
                input_tensor.shape == (1, 1, HIDDEN_SIZE)
                and input_tensor.dtype == torch.bfloat16
                and input_tensor.device == packed_weight.device
                and input_tensor.is_contiguous()
                and not torch.is_grad_enabled()
            ):
                return extension.packed_mlp_bf16_into(
                    input_tensor,
                    module._seu_gate_up_weight,
                    module.down_proj.weight,
                    module._seu_acblas_packed_projected,
                    module._seu_acblas_packed_activated,
                    module._seu_acblas_packed_output,
                    expected_stream,
                    gate_up_algorithm,
                    down_algorithm,
                    swiglu_threads,
                )
            return original_forward(input_tensor)

        module._seu_acblas_packed_mlp_original_forward = original_forward
        module._seu_acblas_packed_mlp_forward = acblas_packed_forward
        module.forward = acblas_packed_forward
        return acblas_packed_forward

    @staticmethod
    def set_enabled(module, enabled: bool) -> None:
        original = getattr(module, "_seu_acblas_packed_mlp_original_forward", None)
        candidate = getattr(module, "_seu_acblas_packed_mlp_forward", None)
        if original is None or candidate is None:
            raise RuntimeError("module does not have the acBLAS packed MLP patch")
        module.forward = candidate if enabled else original
