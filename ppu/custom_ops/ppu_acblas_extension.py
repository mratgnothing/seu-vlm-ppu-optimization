"""Registered PyTorch/acBLAS decode-linear integration for PPU."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

import torch


# These shapes are selected from random-BF16 microbenchmarks, not task data.
# The tuple order follows torch.nn.Linear weight layout: (out_features, in_features).
DEFAULT_PROFITABLE_SHAPES = frozenset(
    {
        (512, 2048),
        (2048, 2048),
        (2048, 6144),
        (4096, 2048),
        (6144, 2048),
    }
)


class PPUACBLASLinearExtension:
    """Load and apply the registered acBLAS BF16 decode extension."""

    def __init__(self, build_dir: str | Path, *, algorithm: int = -1) -> None:
        self.build_dir = Path(build_dir).resolve()
        if str(self.build_dir) not in sys.path:
            sys.path.insert(0, str(self.build_dir))
        self.module = importlib.import_module("seu_acblas_linear_ext")
        self.algorithm = algorithm

    def patch_linear(self, module: torch.nn.Linear):
        if type(module).__name__ != "Linear":
            raise TypeError("acBLAS decode patch requires torch.nn.Linear")
        if module.bias is not None:
            raise ValueError("acBLAS decode patch requires a bias-free linear")
        if module.weight.device.type != "cuda" or module.weight.dtype != torch.bfloat16:
            raise TypeError("acBLAS decode patch requires BF16 PPU weights")
        if not module.weight.is_contiguous():
            raise ValueError("acBLAS decode patch requires contiguous weights")
        if hasattr(module, "_seu_acblas_original_forward"):
            raise RuntimeError("linear module is already patched with acBLAS")

        original_forward = module.forward
        extension = self.module
        algorithm = self.algorithm
        input_features = module.in_features

        def acblas_decode_forward(input_tensor: torch.Tensor) -> torch.Tensor:
            if (
                input_tensor.shape == (1, 1, input_features)
                and input_tensor.dtype == torch.bfloat16
                and input_tensor.device == module.weight.device
                and input_tensor.stride(-1) == 1
            ):
                return extension.linear_bf16(
                    input_tensor, module.weight, algorithm
                )
            return original_forward(input_tensor)

        module._seu_acblas_original_forward = original_forward
        module._seu_acblas_forward = acblas_decode_forward
        module.forward = acblas_decode_forward
        return acblas_decode_forward

    def patch_qwen35_language_linears(
        self,
        model: torch.nn.Module,
        *,
        profitable_shapes=DEFAULT_PROFITABLE_SHAPES,
    ) -> tuple[list[str], dict[str, int]]:
        patched_names: list[str] = []
        shape_counts: Counter[str] = Counter()
        for name, module in model.named_modules():
            if not name.startswith("model.language_model"):
                continue
            if type(module).__name__ != "Linear" or module.bias is not None:
                continue
            # Packed MLP calls one concatenated F.linear directly, so these
            # two module forwards are intentionally inactive after packing.
            if name.endswith((".gate_proj", ".up_proj")):
                continue
            shape = (module.out_features, module.in_features)
            if shape not in profitable_shapes:
                continue
            self.patch_linear(module)
            patched_names.append(name)
            shape_counts[f"{shape[0]}x{shape[1]}"] += 1
        return patched_names, dict(sorted(shape_counts.items()))
