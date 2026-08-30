"""One-dispatch, four-GEMV acBLAS path for Qwen3.5 GDN projections."""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import torch


HIDDEN_SIZE = 2048
_PROJECTIONS = (
    ("in_proj_qkv", 6144),
    ("in_proj_z", 2048),
    ("in_proj_b", 16),
    ("in_proj_a", 16),
)


class PPUACBLASGDNProjectionExtension:
    """Load the experimental grouped acBLAS extension and patch GDN modules."""

    def __init__(
        self,
        build_dir: str | Path,
        *,
        algorithm: int = -1,
        workspace_bytes: int = 0,
        workspace_enabled: bool = True,
        batched_ba: bool = False,
        ba_gemv: bool = False,
        single_gemv: bool = False,
        tail_gemv: bool = False,
    ) -> None:
        if workspace_bytes < 0:
            raise ValueError("workspace_bytes must be non-negative")
        if sum((batched_ba, ba_gemv, single_gemv, tail_gemv)) > 1:
            raise ValueError(
                "batched_ba, ba_gemv, single_gemv and tail_gemv are mutually exclusive"
            )
        if workspace_enabled and workspace_bytes == 0:
            workspace_enabled = False
        build_dir = Path(build_dir).resolve()
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        self.extension = importlib.import_module("seu_acblas_linear_ext")
        self.algorithm = algorithm
        self.workspace_bytes = workspace_bytes
        self.workspace_enabled = workspace_enabled
        self.workspace: torch.Tensor | None = None
        self.batched_ba = batched_ba
        self.extension.set_gdn_batched_ba(batched_ba)
        self.ba_gemv = ba_gemv
        self.extension.set_gdn_ba_gemv(ba_gemv)
        self.single_gemv = single_gemv
        self.extension.set_gdn_single_gemv(single_gemv)
        self.tail_gemv = tail_gemv
        self.extension.set_gdn_tail_gemv(tail_gemv)

    def set_batched_ba(self, enabled: bool) -> None:
        self.extension.set_gdn_batched_ba(enabled)
        self.batched_ba = enabled
        if enabled:
            self.ba_gemv = False
            self.single_gemv = False
            self.tail_gemv = False

    def set_ba_gemv(self, enabled: bool) -> None:
        self.extension.set_gdn_ba_gemv(enabled)
        self.ba_gemv = enabled
        if enabled:
            self.batched_ba = False
            self.single_gemv = False
            self.tail_gemv = False

    def set_single_gemv(self, enabled: bool) -> None:
        self.extension.set_gdn_single_gemv(enabled)
        self.single_gemv = enabled
        if enabled:
            self.batched_ba = False
            self.ba_gemv = False
            self.tail_gemv = False

    def set_tail_gemv(self, enabled: bool) -> None:
        self.extension.set_gdn_tail_gemv(enabled)
        self.tail_gemv = enabled
        if enabled:
            self.batched_ba = False
            self.ba_gemv = False
            self.single_gemv = False

    def _ensure_workspace(self, device: torch.device) -> None:
        if self.workspace_bytes == 0 or self.workspace is not None:
            return
        self.workspace = torch.empty(
            self.workspace_bytes, dtype=torch.uint8, device=device
        )
        self.set_workspace_enabled(self.workspace_enabled)

    def set_workspace_enabled(self, enabled: bool) -> None:
        if self.workspace_bytes == 0:
            if enabled:
                raise RuntimeError("no acBLAS GDN workspace was allocated")
            self.workspace_enabled = False
            return
        if self.workspace is None:
            self.workspace_enabled = enabled
            return
        if enabled:
            self.extension.set_workspace(self.workspace)
        else:
            self.extension.clear_workspace()
        self.workspace_enabled = enabled

    def pack_module(self, module) -> dict[str, Callable]:
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            raise TypeError("grouped acBLAS requires Qwen3_5GatedDeltaNet")
        weights = []
        originals: dict[str, Callable] = {}
        for name, output_features in _PROJECTIONS:
            linear = getattr(module, name)
            if type(linear).__name__ != "Linear" or linear.bias is not None:
                raise ValueError(f"grouped acBLAS requires bias-free Linear {name}")
            if linear.weight.shape != (output_features, HIDDEN_SIZE):
                raise ValueError(f"unexpected {name} shape {tuple(linear.weight.shape)}")
            if linear.weight.dtype != torch.bfloat16 or linear.weight.device.type != "cuda":
                raise TypeError(f"grouped acBLAS requires BF16 PPU weight {name}")
            if not linear.weight.is_contiguous():
                raise ValueError(f"grouped acBLAS requires contiguous weight {name}")
            weights.append(linear.weight)
            originals[name] = linear.forward

        self._ensure_workspace(weights[0].device)
        # The acBLAS handle stores only the device pointer. Keep the owning
        # tensor alive for every patched module even if this helper is released.
        module._seu_acblas_gdn_workspace = self.workspace
        module.register_buffer(
            "_seu_acblas_gdn_output",
            torch.empty(1, 1, 8224, device=weights[0].device, dtype=torch.bfloat16),
            persistent=False,
        )
        module._seu_acblas_gdn_output_scratch_enabled = False
        expected_stream = torch.cuda.current_stream(weights[0].device).cuda_stream

        with torch.no_grad():
            packed_weight = torch.cat(weights, dim=0).contiguous()
        offset = 0
        for name, output_features in _PROJECTIONS:
            getattr(module, name).weight.data = packed_weight[
                offset : offset + output_features
            ]
            offset += output_features
        module.register_buffer("_seu_acblas_gdn_weight", packed_weight, persistent=False)
        packed_storage = packed_weight.untyped_storage().data_ptr()
        if any(
            getattr(module, name).weight.untyped_storage().data_ptr()
            != packed_storage
            for name, _ in _PROJECTIONS
        ):
            raise RuntimeError("grouped acBLAS weights do not alias packed storage")

        extension = self.extension
        algorithm = self.algorithm
        cache = threading.local()

        def qkv_forward(x: torch.Tensor) -> torch.Tensor:
            if x.ndim >= 2 and x.shape[-2:] == (1, HIDDEN_SIZE):
                if module._seu_acblas_gdn_output_scratch_enabled:
                    packed_output = extension.gdn_projections_bf16_into(
                        x,
                        module._seu_acblas_gdn_weight,
                        module._seu_acblas_gdn_output,
                        expected_stream,
                        algorithm,
                    )
                else:
                    packed_output = extension.gdn_projections_bf16(
                        x, module._seu_acblas_gdn_weight, algorithm
                    )
                outputs = packed_output.split((6144, 2048, 16, 16), dim=-1)
                cache.input = x
                cache.outputs = outputs
                return outputs[0]
            cache.__dict__.clear()
            return originals["in_proj_qkv"](x)

        def follower(index: int, name: str, *, clear: bool = False):
            def forward(x: torch.Tensor) -> torch.Tensor:
                outputs = getattr(cache, "outputs", None)
                if getattr(cache, "input", None) is x and isinstance(outputs, tuple):
                    result = outputs[index]
                    if clear:
                        cache.__dict__.clear()
                    return result
                cache.__dict__.clear()
                return originals[name](x)

            return forward

        forwards = {
            "in_proj_qkv": qkv_forward,
            "in_proj_z": follower(1, "in_proj_z"),
            "in_proj_b": follower(2, "in_proj_b"),
            "in_proj_a": follower(3, "in_proj_a", clear=True),
        }
        module._seu_gdn_original_forwards = originals
        module._seu_gdn_packed_forwards = forwards
        for name, forward in forwards.items():
            getattr(module, name).forward = forward
        return forwards

    @staticmethod
    def set_output_scratch(module, enabled: bool) -> None:
        if not hasattr(module, "_seu_acblas_gdn_output_scratch_enabled"):
            raise RuntimeError("module does not have grouped acBLAS GDN scratch")
        module._seu_acblas_gdn_output_scratch_enabled = enabled
