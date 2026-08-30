"""One-entry Qwen3.5 attention decode preparation for PPU."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path

import torch


HIDDEN_SIZE = 2048
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 256
ROTARY_DIM = 64
Q_PROJECTION_SIZE = 4096
KV_PROJECTION_SIZE = 512
PACKED_SIZE = Q_PROJECTION_SIZE + 2 * KV_PROJECTION_SIZE


class PPUACBLASAttentionPrepExtension:
    def __init__(self, build_dir: str | Path, *, algorithm: int = -1) -> None:
        build_dir = Path(build_dir).resolve()
        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))
        self.module = importlib.import_module("seu_acblas_attention_prep_ext")
        self.algorithm = algorithm

    def patch_module(self, module) -> Callable[..., tuple[torch.Tensor, ...] | None]:
        if type(module).__name__ != "Qwen3_5Attention":
            raise TypeError("acBLAS attention prep requires Qwen3_5Attention")
        if hasattr(module, "_seu_acblas_attention_prep_forward"):
            raise RuntimeError("module already has the acBLAS attention-prep patch")
        specifications = (
            ("q_proj", Q_PROJECTION_SIZE),
            ("k_proj", KV_PROJECTION_SIZE),
            ("v_proj", KV_PROJECTION_SIZE),
        )
        weights = []
        for name, output_features in specifications:
            linear = getattr(module, name)
            if linear.bias is not None or linear.weight.shape != (
                output_features,
                HIDDEN_SIZE,
            ):
                raise ValueError(f"unexpected bias or shape for {name}")
            if (
                linear.weight.dtype != torch.bfloat16
                or linear.weight.device.type != "cuda"
                or not linear.weight.is_contiguous()
            ):
                raise TypeError(f"{name} weight must be contiguous BF16 on PPU")
            weights.append(linear.weight)
        if (
            module.q_norm.weight.shape != (HEAD_DIM,)
            or module.k_norm.weight.shape != (HEAD_DIM,)
        ):
            raise ValueError("unexpected q/k RMSNorm shape")

        with torch.no_grad():
            packed_weight = torch.cat(weights, dim=0).contiguous()
        offset = 0
        for name, output_features in specifications:
            getattr(module, name).weight.data = packed_weight[
                offset : offset + output_features
            ]
            offset += output_features
        module.register_buffer(
            "_seu_acblas_attention_weight", packed_weight, persistent=False
        )
        options = {"device": packed_weight.device, "dtype": torch.bfloat16}
        projected = torch.empty(1, 1, PACKED_SIZE, **options)
        module.register_buffer(
            "_seu_acblas_attention_projected", projected, persistent=False
        )
        module.register_buffer(
            "_seu_acblas_attention_query",
            torch.empty(1, HEADS, 1, HEAD_DIM, **options),
            persistent=False,
        )
        module.register_buffer(
            "_seu_acblas_attention_key",
            torch.empty(1, KV_HEADS, 1, HEAD_DIM, **options),
            persistent=False,
        )
        module.register_buffer(
            "_seu_acblas_attention_value",
            projected[..., Q_PROJECTION_SIZE + KV_PROJECTION_SIZE :]
            .view(1, 1, KV_HEADS, HEAD_DIM)
            .transpose(1, 2),
            persistent=False,
        )
        module.register_buffer(
            "_seu_acblas_attention_gate",
            projected.as_strided(
                (1, 1, HEADS, HEAD_DIM),
                (PACKED_SIZE, PACKED_SIZE, 2 * HEAD_DIM, 1),
                storage_offset=HEAD_DIM,
            ),
            persistent=False,
        )

        extension = self.module
        algorithm = self.algorithm
        expected_stream = torch.cuda.current_stream(
            device=packed_weight.device
        ).cuda_stream

        def attention_prep(
            hidden_states: torch.Tensor,
            cosine: torch.Tensor,
            sine: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
            if not (
                hidden_states.shape == (1, 1, HIDDEN_SIZE)
                and hidden_states.dtype == torch.bfloat16
                and hidden_states.device == packed_weight.device
                and hidden_states.is_contiguous()
                and cosine.shape == (1, 1, ROTARY_DIM)
                and sine.shape == cosine.shape
                and cosine.dtype == torch.bfloat16
                and sine.dtype == torch.bfloat16
                and cosine.device == packed_weight.device
                and sine.device == packed_weight.device
                and not torch.is_grad_enabled()
            ):
                return None
            extension.attention_prep_bf16_into(
                hidden_states,
                module._seu_acblas_attention_weight,
                module.q_norm.weight,
                module.k_norm.weight,
                cosine,
                sine,
                module._seu_acblas_attention_projected,
                module._seu_acblas_attention_query,
                module._seu_acblas_attention_key,
                expected_stream,
                module.q_norm.eps,
                algorithm,
            )
            return (
                module._seu_acblas_attention_query,
                module._seu_acblas_attention_key,
                module._seu_acblas_attention_value,
                module._seu_acblas_attention_gate,
            )

        module._seu_acblas_attention_prep_forward = attention_prep
        module._seu_attention_prep_decode = attention_prep
        return attention_prep

    @staticmethod
    def set_enabled(module, enabled: bool) -> None:
        candidate = getattr(module, "_seu_acblas_attention_prep_forward", None)
        if candidate is None:
            raise RuntimeError("module does not have the attention-prep patch")
        module._seu_attention_prep_decode = candidate if enabled else None
