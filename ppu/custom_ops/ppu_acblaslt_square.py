"""Experimental registered acBLASLt 2048-square decode Linear for PPU."""

from __future__ import annotations

import importlib
import sys
from collections import Counter
from pathlib import Path

import torch


SQUARE_FEATURES = 2048


class PPUACBLASLtSquareExtension:
    """Patch bias-free 2048-square language Linears with scratch output."""

    def __init__(self, build_dir: str | Path, *, heuristic_index: int = 25) -> None:
        self.build_dir = Path(build_dir).resolve()
        if str(self.build_dir) not in sys.path:
            sys.path.insert(0, str(self.build_dir))
        self.module = importlib.import_module("seu_acblaslt_square_ext")
        self.heuristic_index = heuristic_index

    def patch_linear(self, module: torch.nn.Linear):
        if type(module).__name__ != "Linear":
            raise TypeError("acBLASLt square patch requires torch.nn.Linear")
        if module.bias is not None or (module.out_features, module.in_features) != (
            SQUARE_FEATURES,
            SQUARE_FEATURES,
        ):
            raise ValueError("acBLASLt square patch requires bias-free 2048x2048")
        if module.weight.device.type != "cuda" or module.weight.dtype != torch.bfloat16:
            raise TypeError("acBLASLt square patch requires BF16 PPU weights")
        if not module.weight.is_contiguous():
            raise ValueError("acBLASLt square patch requires contiguous weights")
        if hasattr(module, "_seu_acblaslt_square_original_forward"):
            raise RuntimeError("linear module is already patched with acBLASLt")

        original_forward = module.forward
        extension = self.module
        heuristic_index = self.heuristic_index
        scratch = torch.empty(
            (1, 1, SQUARE_FEATURES),
            dtype=module.weight.dtype,
            device=module.weight.device,
        )
        module.register_buffer("_seu_acblaslt_square_output", scratch, persistent=False)

        def acblaslt_decode_forward(input_tensor: torch.Tensor) -> torch.Tensor:
            if (
                input_tensor.shape == (1, 1, SQUARE_FEATURES)
                and input_tensor.dtype == torch.bfloat16
                and input_tensor.device == module.weight.device
                and input_tensor.stride(-1) == 1
            ):
                return extension.acblaslt_square_linear_bf16_into(
                    input_tensor,
                    module.weight,
                    module._seu_acblaslt_square_output,
                    heuristic_index,
                )
            return original_forward(input_tensor)

        module._seu_acblaslt_square_original_forward = original_forward
        module._seu_acblaslt_square_forward = acblaslt_decode_forward
        module.forward = acblaslt_decode_forward
        return acblaslt_decode_forward

    def patch_qwen35_language_linears(
        self, model: torch.nn.Module
    ) -> tuple[list[str], dict[str, int]]:
        patched_names: list[str] = []
        shape_counts: Counter[str] = Counter()
        for name, module in model.named_modules():
            if not name.startswith("model.language_model"):
                continue
            if type(module).__name__ != "Linear" or module.bias is not None:
                continue
            if (module.out_features, module.in_features) != (
                SQUARE_FEATURES,
                SQUARE_FEATURES,
            ):
                continue
            self.patch_linear(module)
            patched_names.append(name)
            shape_counts["2048x2048"] += 1
        return patched_names, dict(shape_counts)
