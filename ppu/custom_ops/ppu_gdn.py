"""Thin ctypes integration for the fixed Qwen3.5-2B PPU decode GDN kernel."""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path
from typing import Callable

import torch


HEADS = 16
HEAD_DIM = 128
CONV_CHANNELS = 6144
CONV_WIDTH = 4
HIDDEN_SIZE = 2048
GATED_NORM_WIDTH = 128
ATTENTION_HEADS = 8
KEY_VALUE_HEADS = 2
ATTENTION_HEAD_DIM = 256
ROTARY_DIM = 64
MLP_INTERMEDIATE_SIZE = 6144


class PPUGDNLibrary:
    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        tiles_per_head: int = 4,
        conv_threads: int = 96,
        rmsnorm_threads: int = 512,
        gated_rmsnorm_threads: int = 128,
        swiglu_threads: int = 256,
    ) -> None:
        if library_path is None:
            library_path = Path(__file__).with_name("build") / "libseu_ppu_gdn.so"
        self.path = Path(library_path).resolve()
        if tiles_per_head not in (1, 2, 4):
            raise ValueError("tiles_per_head must be 1, 2, or 4")
        self.tiles_per_head = tiles_per_head
        if conv_threads <= 0 or conv_threads > 1024 or conv_threads % 32:
            raise ValueError("conv_threads must be a multiple of 32 through 1024")
        self.conv_threads = conv_threads
        if rmsnorm_threads <= 0 or rmsnorm_threads > 1024 or rmsnorm_threads % 32:
            raise ValueError("rmsnorm_threads must be a multiple of 32 through 1024")
        self.rmsnorm_threads = rmsnorm_threads
        if (
            gated_rmsnorm_threads <= 0
            or gated_rmsnorm_threads > 1024
            or gated_rmsnorm_threads % 32
        ):
            raise ValueError(
                "gated_rmsnorm_threads must be a multiple of 32 through 1024"
            )
        self.gated_rmsnorm_threads = gated_rmsnorm_threads
        if swiglu_threads <= 0 or swiglu_threads > 1024 or swiglu_threads % 32:
            raise ValueError("swiglu_threads must be a multiple of 32 through 1024")
        self.swiglu_threads = swiglu_threads
        self.library = ctypes.CDLL(str(self.path))
        self.launch = self.library.seu_ppu_gdn_recurrent_decode_bf16
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.launch.restype = ctypes.c_int
        self.gate_prep_launch = getattr(
            self.library, "seu_ppu_gdn_gate_prep_decode_bf16", None
        )
        if self.gate_prep_launch is not None:
            self.gate_prep_launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            self.gate_prep_launch.restype = ctypes.c_int
        self.conv_launch = self.library.seu_ppu_causal_conv1d_decode_bf16
        self.conv_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.conv_launch.restype = ctypes.c_int
        self.rmsnorm_launch = self.library.seu_ppu_rmsnorm_decode_bf16
        self.rmsnorm_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.rmsnorm_launch.restype = ctypes.c_int
        self.residual_rmsnorm_launch = getattr(
            self.library, "seu_ppu_residual_rmsnorm_decode_bf16", None
        )
        if self.residual_rmsnorm_launch is not None:
            self.residual_rmsnorm_launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self.residual_rmsnorm_launch.restype = ctypes.c_int
        self.swiglu_launch = getattr(
            self.library, "seu_ppu_swiglu_decode_bf16", None
        )
        if self.swiglu_launch is not None:
            self.swiglu_launch.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            self.swiglu_launch.restype = ctypes.c_int
        self.gated_rmsnorm_launch = self.library.seu_ppu_gated_rmsnorm_decode_bf16
        self.gated_rmsnorm_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.gated_rmsnorm_launch.restype = ctypes.c_int
        self.qk_rmsnorm_rope_launch = (
            self.library.seu_ppu_qk_rmsnorm_rope_decode_bf16
        )
        self.qk_rmsnorm_rope_launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_void_p,
        ]
        self.qk_rmsnorm_rope_launch.restype = ctypes.c_int

    def recurrent_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        _validate_inputs(query, key, value, g, beta, state)
        batch_size = query.shape[0]
        output = torch.empty(
            (batch_size, 1, HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        )
        stream = torch.cuda.current_stream(query.device).cuda_stream
        status = self.launch(
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            g.data_ptr(),
            beta.data_ptr(),
            state.data_ptr(),
            output.data_ptr(),
            batch_size,
            query.stride(0),
            key.stride(0),
            value.stride(0),
            g.stride(0),
            beta.stride(0),
            self.tiles_per_head,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"PPU GDN launch failed with HGGC status {status}")
        return output

    def gate_prep_decode(
        self,
        raw_a: torch.Tensor,
        raw_b: torch.Tensor,
        exp_a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        *,
        g_out: torch.Tensor | None = None,
        beta_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare the FP32 decay log and BF16 beta for batch-1 GDN decode."""
        if self.gate_prep_launch is None:
            raise RuntimeError("PPU GDN library does not export gate-prep support")
        _validate_gate_prep_inputs(raw_a, raw_b, exp_a_log, dt_bias)
        batch_size = raw_a.shape[0]
        if g_out is None and beta_out is None:
            g = torch.empty_like(raw_a, dtype=torch.float32)
            beta = torch.empty_like(raw_b)
        elif g_out is not None and beta_out is not None:
            _validate_gate_prep_outputs(raw_a, g_out, beta_out)
            g, beta = g_out, beta_out
        else:
            raise ValueError("g_out and beta_out must be provided together")
        stream = torch.cuda.current_stream(raw_a.device).cuda_stream
        status = self.gate_prep_launch(
            raw_a.data_ptr(),
            raw_b.data_ptr(),
            exp_a_log.data_ptr(),
            dt_bias.data_ptr(),
            g.data_ptr(),
            beta.data_ptr(),
            batch_size,
            raw_a.stride(0),
            raw_b.stride(0),
            g.stride(0),
            beta.stride(0),
            stream,
        )
        if status != 0:
            raise RuntimeError(
                f"PPU GDN gate-prep launch failed with HGGC status {status}"
            )
        return g, beta

    def causal_conv1d_decode(
        self,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
    ) -> torch.Tensor:
        _validate_conv_inputs(hidden_states, conv_state, weight, bias, activation)
        output = torch.empty_like(hidden_states)
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        status = self.conv_launch(
            hidden_states.data_ptr(),
            conv_state.data_ptr(),
            weight.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            output.data_ptr(),
            hidden_states.shape[0],
            hidden_states.stride(0),
            conv_state.stride(0),
            self.conv_threads,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"PPU causal-conv launch failed with HGGC status {status}")
        return output

    def rmsnorm_decode(
        self,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        _validate_rmsnorm_inputs(input_tensor, weight)
        output = torch.empty_like(input_tensor)
        rows = input_tensor.numel() // HIDDEN_SIZE
        stream = torch.cuda.current_stream(input_tensor.device).cuda_stream
        status = self.rmsnorm_launch(
            input_tensor.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            rows,
            input_tensor.stride(-2),
            epsilon,
            self.rmsnorm_threads,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"PPU RMSNorm launch failed with HGGC status {status}")
        return output

    def gated_rmsnorm_decode(
        self,
        hidden_states: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        _validate_gated_rmsnorm_inputs(hidden_states, gate, weight)
        output = torch.empty_like(hidden_states)
        rows = hidden_states.numel() // GATED_NORM_WIDTH
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        status = self.gated_rmsnorm_launch(
            hidden_states.data_ptr(),
            gate.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            rows,
            hidden_states.stride(-2),
            gate.stride(-2),
            epsilon,
            self.gated_rmsnorm_threads,
            stream,
        )
        if status != 0:
            raise RuntimeError(
                f"PPU gated RMSNorm launch failed with HGGC status {status}"
            )
        return output

    def residual_rmsnorm_decode(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_residual_rmsnorm_inputs(residual, update, weight)
        if self.residual_rmsnorm_launch is None:
            raise RuntimeError("PPU GDN library lacks residual RMSNorm symbol")
        normalized_output = torch.empty_like(update)
        rows = update.numel() // HIDDEN_SIZE
        stream = torch.cuda.current_stream(update.device).cuda_stream
        status = self.residual_rmsnorm_launch(
            residual.data_ptr(),
            update.data_ptr(),
            weight.data_ptr(),
            normalized_output.data_ptr(),
            rows,
            residual.stride(-2),
            update.stride(-2),
            epsilon,
            self.rmsnorm_threads,
            stream,
        )
        if status != 0:
            raise RuntimeError(
                f"PPU residual RMSNorm launch failed with HGGC status {status}"
            )
        return update, normalized_output

    def swiglu_decode(
        self,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        _validate_swiglu_inputs(gate, up)
        if self.swiglu_launch is None:
            raise RuntimeError("PPU GDN library lacks SwiGLU symbol")
        output = torch.empty_like(gate)
        stream = torch.cuda.current_stream(gate.device).cuda_stream
        status = self.swiglu_launch(
            gate.data_ptr(),
            up.data_ptr(),
            output.data_ptr(),
            gate.numel(),
            self.swiglu_threads,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"PPU SwiGLU launch failed with HGGC status {status}")
        return output

    def qk_rmsnorm_rope_decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_weight: torch.Tensor,
        key_weight: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_qk_rmsnorm_rope_inputs(
            query, key, query_weight, key_weight, cosine, sine
        )
        batch_size = query.shape[0]
        query_output = torch.empty(
            (batch_size, ATTENTION_HEADS, 1, ATTENTION_HEAD_DIM),
            dtype=torch.bfloat16,
            device=query.device,
        )
        key_output = torch.empty(
            (batch_size, KEY_VALUE_HEADS, 1, ATTENTION_HEAD_DIM),
            dtype=torch.bfloat16,
            device=key.device,
        )
        stream = torch.cuda.current_stream(query.device).cuda_stream
        status = self.qk_rmsnorm_rope_launch(
            query.data_ptr(),
            key.data_ptr(),
            query_weight.data_ptr(),
            key_weight.data_ptr(),
            cosine.data_ptr(),
            sine.data_ptr(),
            query_output.data_ptr(),
            key_output.data_ptr(),
            batch_size,
            query.stride(0),
            query.stride(-2),
            key.stride(0),
            key.stride(-2),
            cosine.stride(0),
            sine.stride(0),
            epsilon,
            stream,
        )
        if status != 0:
            raise RuntimeError(
                f"PPU q/k RMSNorm+RoPE launch failed with HGGC status {status}"
            )
        return query_output, key_output

    def transformers_callable(self) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
        def fused_recurrent_gated_delta_rule(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            initial_state: torch.Tensor,
            output_final_state: bool,
            use_qk_l2norm_in_kernel: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            if not use_qk_l2norm_in_kernel:
                raise ValueError("PPU GDN requires use_qk_l2norm_in_kernel=True")
            if initial_state is None:
                initial_state = torch.zeros(
                    query.shape[0], HEADS, HEAD_DIM, HEAD_DIM,
                    dtype=torch.float32, device=query.device,
                )
            output = self.recurrent_decode(query, key, value, g, beta, initial_state)
            return output, initial_state if output_final_state else None

        return fused_recurrent_gated_delta_rule

    def transformers_attention_callable(
        self,
        module,
        attention_registry,
        eager_attention_forward,
    ) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
        original_forward = module.forward

        def fused_attention_forward(
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: torch.Tensor | None,
            past_key_values=None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            cosine, sine = position_embeddings
            supported_decode = (
                hidden_states.ndim == 3
                and hidden_states.shape[1] == 1
                and hidden_states.shape[-1] == HIDDEN_SIZE
                and module.head_dim == ATTENTION_HEAD_DIM
                and module.q_norm.weight.numel() == ATTENTION_HEAD_DIM
                and module.k_norm.weight.numel() == ATTENTION_HEAD_DIM
                and module.q_norm.eps == module.k_norm.eps
                and cosine.shape == (hidden_states.shape[0], 1, ROTARY_DIM)
                and sine.shape == cosine.shape
            )
            if not supported_decode:
                return original_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query_states, gate = torch.chunk(
                module.q_proj(hidden_states).view(
                    *input_shape, -1, module.head_dim * 2
                ),
                2,
                dim=-1,
            )
            gate = gate.reshape(*input_shape, -1)
            query_states = query_states.view(hidden_shape)
            key_states = module.k_proj(hidden_states).view(hidden_shape)
            value_states = (
                module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            query_states, key_states = self.qk_rmsnorm_rope_decode(
                query_states,
                key_states,
                module.q_norm.weight,
                module.k_norm.weight,
                cosine,
                sine,
                module.q_norm.eps,
            )

            if past_key_values is not None:
                key_states, value_states = past_key_values.update(
                    key_states, value_states, module.layer_idx
                )

            attention_interface = attention_registry.get_interface(
                module.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                module,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not module.training else module.attention_dropout,
                scaling=module.scaling,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = attn_output * torch.sigmoid(gate)
            return module.o_proj(attn_output), attn_weights

        return fused_attention_forward


def pack_qwen35_gdn_gate_prep(module, library: PPUGDNLibrary) -> Callable:
    """Replace only cached one-token GDN decode with fused gate preparation."""
    if type(module).__name__ != "Qwen3_5GatedDeltaNet":
        raise TypeError("gate-prep integration requires Qwen3_5GatedDeltaNet")
    if module.num_v_heads != HEADS or module.head_k_dim != HEAD_DIM:
        raise ValueError("gate-prep integration requires 16 heads of width 128")
    if library.gate_prep_launch is None:
        raise RuntimeError("PPU GDN library does not export gate-prep support")
    if hasattr(module, "_seu_gdn_gate_prep_forwards"):
        return module._seu_gdn_gate_prep_forwards[0]
    if module.A_log.dtype != torch.bfloat16 or module.dt_bias.dtype != torch.bfloat16:
        raise TypeError("gate-prep integration requires BF16 A_log/dt_bias")
    exp_a_log = module.A_log.float().exp().contiguous()
    module.register_buffer("_seu_exp_a_log", exp_a_log, persistent=False)
    original_forward = module.forward
    scratch = threading.local()

    def gate_prep(raw_a: torch.Tensor, raw_b: torch.Tensor):
        buffers = getattr(scratch, "buffers", None)
        if (
            buffers is None
            or buffers[0].device != raw_a.device
            or buffers[0].shape != raw_a.shape
        ):
            buffers = (
                torch.empty_like(raw_a, dtype=torch.float32),
                torch.empty_like(raw_b),
            )
            scratch.buffers = buffers
        return library.gate_prep_decode(
            raw_a,
            raw_b,
            module._seu_exp_a_log,
            module.dt_bias,
            g_out=buffers[0],
            beta_out=buffers[1],
        )

    def fused_forward(
        hidden_states: torch.Tensor,
        cache_params=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        supported_decode = (
            not module.training
            and hidden_states.ndim == 3
            and hidden_states.shape[0] == 1
            and hidden_states.shape[1] == 1
            and hidden_states.shape[2] == HIDDEN_SIZE
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.device.type == "cuda"
            and cache_params is not None
            and cache_params.has_previous_state(module.layer_idx)
        )
        if not supported_decode:
            return original_forward(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                **kwargs,
            )

        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            apply_mask_to_padding_states,
        )

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = cache_params.layers[module.layer_idx].conv_states[0]
        recurrent_state = cache_params.layers[module.layer_idx].recurrent_states[0]

        mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
        z = module.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, module.head_v_dim
        )
        raw_b = module.in_proj_b(hidden_states)
        raw_a = module.in_proj_a(hidden_states)
        mixed_qkv = module.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            module.conv1d.weight.squeeze(1),
            module.conv1d.bias,
            module.activation,
        ).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [module.key_dim, module.key_dim, module.value_dim],
            dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, module.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, module.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, module.head_v_dim)
        g, beta = gate_prep(raw_a, raw_b)
        if module.num_v_heads // module.num_k_heads > 1:
            repeats = module.num_v_heads // module.num_k_heads
            query = query.repeat_interleave(repeats, dim=2)
            key = key.repeat_interleave(repeats, dim=2)
        core_attn_out, last_recurrent_state = module.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        cache_params.update_recurrent_state(last_recurrent_state, module.layer_idx)
        core_attn_out = core_attn_out.reshape(-1, module.head_v_dim)
        z = z.reshape(-1, module.head_v_dim)
        core_attn_out = module.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return module.out_proj(core_attn_out)

    module._seu_gdn_gate_prep_forwards = (original_forward, fused_forward)
    module.forward = fused_forward
    return original_forward


def set_qwen35_gdn_gate_prep(module, enabled: bool) -> None:
    forwards = getattr(module, "_seu_gdn_gate_prep_forwards", None)
    if forwards is None:
        raise RuntimeError("GDN gate-prep module has not been packed")
    module.forward = forwards[1 if enabled else 0]


def pack_qwen35_decoder_residual_rmsnorm(
    module,
    library: PPUGDNLibrary,
    *,
    next_norm=None,
) -> Callable:
    """Fuse both decoder residual-add/RMSNorm edges for one-token decode."""
    if type(module).__name__ != "Qwen3_5DecoderLayer":
        raise TypeError("residual RMSNorm integration requires Qwen3_5DecoderLayer")
    if module.hidden_size != HIDDEN_SIZE:
        raise ValueError("residual RMSNorm requires hidden size 2048")
    if module.block_type not in ("linear_attention", "full_attention"):
        raise ValueError(f"unsupported decoder block type {module.block_type}")
    if module.post_attention_layernorm.weight.shape != (HIDDEN_SIZE,):
        raise ValueError("unexpected post-attention RMSNorm weight shape")
    if (
        module.post_attention_layernorm.weight.dtype != torch.bfloat16
        or module.post_attention_layernorm.weight.device.type != "cuda"
    ):
        raise TypeError("residual RMSNorm requires BF16 PPU decoder weights")

    original_forward = module.forward
    next_norm_cache = None
    if next_norm is not None:
        if (
            type(next_norm).__name__ != "Qwen3_5RMSNorm"
            or next_norm.weight.shape != (HIDDEN_SIZE,)
            or next_norm.weight.dtype != torch.bfloat16
            or next_norm.weight.device.type != "cuda"
        ):
            raise TypeError("next RMSNorm must be a BF16 Qwen3.5 norm on PPU")
        original_next_norm_forward = next_norm.forward
        next_norm_cache = threading.local()

        def cached_next_norm_forward(x: torch.Tensor) -> torch.Tensor:
            if getattr(next_norm_cache, "input", None) is x:
                output = next_norm_cache.output
                next_norm_cache.__dict__.clear()
                return output
            next_norm_cache.__dict__.clear()
            return original_next_norm_forward(x)

        next_norm._seu_residual_rmsnorm_original_forward = (
            original_next_norm_forward
        )
        next_norm._seu_residual_rmsnorm_cached_forward = cached_next_norm_forward
        next_norm.forward = cached_next_norm_forward

    def fused_decoder_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        **kwargs,
    ) -> torch.Tensor:
        # Decoder-layer construction and the PPU library validate hidden size,
        # dtype and device once. Keep the per-token guard to a single shape read.
        if hidden_states.shape[1] != 1:
            return original_forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        residual = hidden_states
        hidden_states = module.input_layernorm(hidden_states)
        if module.block_type == "linear_attention":
            hidden_states = module.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        elif module.block_type == "full_attention":
            hidden_states, _ = module.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        residual, hidden_states = library.residual_rmsnorm_decode(
            residual,
            hidden_states,
            module.post_attention_layernorm.weight,
            module.post_attention_layernorm.eps,
        )
        hidden_states = module.mlp(hidden_states)
        if next_norm is not None:
            next_residual, next_normalized = library.residual_rmsnorm_decode(
                residual,
                hidden_states,
                next_norm.weight,
                next_norm.eps,
            )
            next_norm_cache.input = next_residual
            next_norm_cache.output = next_normalized
            return next_residual
        return residual + hidden_states

    module._seu_residual_rmsnorm_original_forward = original_forward
    module._seu_residual_rmsnorm_fused_forward = fused_decoder_forward
    module._seu_residual_rmsnorm_next_norm = next_norm
    module._seu_residual_rmsnorm_next_cache = next_norm_cache
    module.forward = fused_decoder_forward
    return fused_decoder_forward


def set_qwen35_decoder_residual_rmsnorm(module, enabled: bool) -> None:
    original = getattr(module, "_seu_residual_rmsnorm_original_forward", None)
    fused = getattr(module, "_seu_residual_rmsnorm_fused_forward", None)
    if original is None or fused is None:
        raise RuntimeError("decoder residual RMSNorm has not been packed")
    module.forward = fused if enabled else original
    next_norm = getattr(module, "_seu_residual_rmsnorm_next_norm", None)
    if next_norm is not None:
        next_norm.forward = (
            next_norm._seu_residual_rmsnorm_cached_forward
            if enabled
            else next_norm._seu_residual_rmsnorm_original_forward
        )
        cache = module._seu_residual_rmsnorm_next_cache
        cache.__dict__.clear()


def pack_qwen35_mlp_module(module) -> Callable[[torch.Tensor], torch.Tensor]:
    if type(module).__name__ != "Qwen3_5MLP":
        raise TypeError("packed MLP integration requires Qwen3_5MLP")
    if module.hidden_size != HIDDEN_SIZE or module.intermediate_size != (
        MLP_INTERMEDIATE_SIZE
    ):
        raise ValueError("packed MLP requires hidden/intermediate size 2048/6144")
    if module.gate_proj.bias is not None or module.up_proj.bias is not None:
        raise ValueError("packed MLP requires bias-free gate/up projections")
    gate_weight = module.gate_proj.weight
    up_weight = module.up_proj.weight
    if (
        gate_weight.shape != (MLP_INTERMEDIATE_SIZE, HIDDEN_SIZE)
        or up_weight.shape != gate_weight.shape
    ):
        raise ValueError("packed MLP gate/up weights must both be [6144, 2048]")
    if gate_weight.dtype != torch.bfloat16 or up_weight.dtype != torch.bfloat16:
        raise TypeError("packed MLP gate/up weights must be torch.bfloat16")
    if gate_weight.device.type != "cuda" or up_weight.device != gate_weight.device:
        raise ValueError("packed MLP gate/up weights must share the PPU device")

    original_forward = module.forward
    with torch.no_grad():
        packed_weight = torch.cat((gate_weight, up_weight), dim=0).contiguous()
    module.gate_proj.weight.data = packed_weight[:MLP_INTERMEDIATE_SIZE]
    module.up_proj.weight.data = packed_weight[MLP_INTERMEDIATE_SIZE:]
    module.register_buffer("_seu_gate_up_weight", packed_weight, persistent=False)
    packed_storage = packed_weight.untyped_storage().data_ptr()
    if (
        module.gate_proj.weight.untyped_storage().data_ptr() != packed_storage
        or module.up_proj.weight.untyped_storage().data_ptr() != packed_storage
    ):
        raise RuntimeError("packed MLP gate/up weights do not alias packed storage")

    def packed_mlp_forward(x: torch.Tensor) -> torch.Tensor:
        if x.ndim >= 2 and x.shape[-2] == 1 and x.shape[-1] == HIDDEN_SIZE:
            projected = torch.nn.functional.linear(x, module._seu_gate_up_weight)
            gate, up = projected.split(MLP_INTERMEDIATE_SIZE, dim=-1)
            return module.down_proj(module.act_fn(gate) * up)
        return original_forward(x)

    return packed_mlp_forward


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> None:
    tensors = (query, key, value, g, beta, state)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all PPU GDN tensors must use the CUDA compatibility device")
    expected_vectors = (query.shape[0], 1, HEADS, HEAD_DIM)
    if query.shape != expected_vectors or key.shape != expected_vectors or value.shape != expected_vectors:
        raise ValueError(f"expected q/k/v shape {expected_vectors}")
    if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise TypeError("q/k/v must be torch.bfloat16")
    if g.shape != (query.shape[0], 1, HEADS) or beta.shape != (query.shape[0], 1, HEADS):
        raise ValueError("g/beta must have shape [batch, 1, 16]")
    if g.dtype != torch.float32 or beta.dtype != torch.bfloat16:
        raise TypeError("g must be torch.float32 and beta must be torch.bfloat16")
    if state.shape != (query.shape[0], HEADS, HEAD_DIM, HEAD_DIM):
        raise ValueError("state must have shape [batch, 16, 128, 128]")
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise TypeError("state must be contiguous torch.float32")
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.stride(-1) != 1 or tensor.stride(-2) != HEAD_DIM:
            raise ValueError(f"{name} heads must be contiguous")
    if g.stride(-1) != 1 or beta.stride(-1) != 1:
        raise ValueError("g/beta heads must be contiguous")


def _validate_gate_prep_inputs(
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    exp_a_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> None:
    tensors = (raw_a, raw_b, exp_a_log, dt_bias)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all GDN gate-prep tensors must use the CUDA compatibility device")
    if raw_a.shape != raw_b.shape or raw_a.ndim != 3 or raw_a.shape[1:] != (1, HEADS):
        raise ValueError("raw a/b must share shape [batch, 1, 16]")
    if raw_a.dtype != torch.bfloat16 or raw_b.dtype != torch.bfloat16:
        raise TypeError("raw a/b must be torch.bfloat16")
    if exp_a_log.shape != (HEADS,) or exp_a_log.dtype != torch.float32:
        raise TypeError("exp(A_log) must be contiguous torch.float32 shape [16]")
    if dt_bias.shape != (HEADS,) or dt_bias.dtype != torch.bfloat16:
        raise TypeError("dt_bias must be contiguous torch.bfloat16 shape [16]")
    if any(tensor.device != raw_a.device for tensor in tensors[1:]):
        raise ValueError("all GDN gate-prep tensors must share one PPU device")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("GDN gate-prep head dimensions must be contiguous")


def _validate_gate_prep_outputs(
    raw_a: torch.Tensor,
    g_out: torch.Tensor,
    beta_out: torch.Tensor,
) -> None:
    if g_out.shape != raw_a.shape or beta_out.shape != raw_a.shape:
        raise ValueError("gate-prep outputs must match raw a shape")
    if g_out.dtype != torch.float32 or beta_out.dtype != torch.bfloat16:
        raise TypeError("gate-prep outputs must be FP32 g and BF16 beta")
    if g_out.device != raw_a.device or beta_out.device != raw_a.device:
        raise ValueError("gate-prep outputs must share the input PPU device")
    if not g_out.is_contiguous() or not beta_out.is_contiguous():
        raise ValueError("gate-prep outputs must be contiguous")


def _validate_conv_inputs(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
) -> None:
    tensors = (hidden_states, conv_state, weight) + (() if bias is None else (bias,))
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("all causal-conv tensors must use the CUDA compatibility device")
    if hidden_states.ndim != 3 or hidden_states.shape[1:] != (CONV_CHANNELS, 1):
        raise ValueError("hidden_states must have shape [batch, 6144, 1]")
    if conv_state.shape != (hidden_states.shape[0], CONV_CHANNELS, CONV_WIDTH):
        raise ValueError("conv_state must have shape [batch, 6144, 4]")
    if weight.shape != (CONV_CHANNELS, CONV_WIDTH):
        raise ValueError("weight must have shape [6144, 4]")
    if bias is not None and bias.shape != (CONV_CHANNELS,):
        raise ValueError("bias must have shape [6144]")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("causal-conv tensors must be torch.bfloat16")
    if not conv_state.is_contiguous() or not weight.is_contiguous():
        raise ValueError("conv_state and weight must be contiguous")
    if hidden_states.stride(1) != 1:
        raise ValueError("hidden_states channels must be contiguous for seq_len=1")
    if activation not in (None, "silu", "swish"):
        raise ValueError("only SiLU causal-conv activation is supported")


def _validate_rmsnorm_inputs(input_tensor: torch.Tensor, weight: torch.Tensor) -> None:
    if input_tensor.device.type != "cuda" or weight.device.type != "cuda":
        raise ValueError("RMSNorm input/weight must use the CUDA compatibility device")
    if input_tensor.ndim < 2 or input_tensor.shape[-1] != HIDDEN_SIZE:
        raise ValueError("RMSNorm input last dimension must be 2048")
    if input_tensor.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("RMSNorm input/weight must be torch.bfloat16")
    if weight.shape != (HIDDEN_SIZE,) or not weight.is_contiguous():
        raise ValueError("RMSNorm weight must be contiguous shape [2048]")
    if input_tensor.stride(-1) != 1:
        raise ValueError("RMSNorm input last dimension must be contiguous")


def _validate_residual_rmsnorm_inputs(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    _validate_rmsnorm_inputs(residual, weight)
    _validate_rmsnorm_inputs(update, weight)
    if residual.shape != update.shape or residual.device != update.device:
        raise ValueError("residual/update must share shape and PPU device")
    if residual.data_ptr() == update.data_ptr():
        raise ValueError("residual/update must not alias the same tensor")


def _validate_swiglu_inputs(gate: torch.Tensor, up: torch.Tensor) -> None:
    if gate.device.type != "cuda" or up.device != gate.device:
        raise ValueError("SwiGLU gate/up must share the PPU device")
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise TypeError("SwiGLU gate/up must be torch.bfloat16")
    if gate.shape != up.shape or gate.shape[-1] != MLP_INTERMEDIATE_SIZE:
        raise ValueError("SwiGLU gate/up must share shape [..., 6144]")
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError("SwiGLU gate/up must be contiguous")


def _validate_gated_rmsnorm_inputs(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
) -> None:
    if any(tensor.device.type != "cuda" for tensor in (hidden_states, gate, weight)):
        raise ValueError("gated RMSNorm tensors must use the CUDA compatibility device")
    if hidden_states.shape != gate.shape or hidden_states.shape[-1] != GATED_NORM_WIDTH:
        raise ValueError("hidden_states/gate must share shape [rows, 128]")
    if any(tensor.dtype != torch.bfloat16 for tensor in (hidden_states, gate, weight)):
        raise TypeError("gated RMSNorm tensors must be torch.bfloat16")
    if weight.shape != (GATED_NORM_WIDTH,) or not weight.is_contiguous():
        raise ValueError("gated RMSNorm weight must be contiguous shape [128]")
    if hidden_states.stride(-1) != 1 or gate.stride(-1) != 1:
        raise ValueError("gated RMSNorm rows must be contiguous")


def _validate_qk_rmsnorm_rope_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> None:
    tensors = (query, key, query_weight, key_weight, cosine, sine)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("q/k RMSNorm+RoPE tensors must use the CUDA compatibility device")
    batch_size = query.shape[0]
    if query.shape != (batch_size, 1, ATTENTION_HEADS, ATTENTION_HEAD_DIM):
        raise ValueError("query must have shape [batch, 1, 8, 256]")
    if key.shape != (batch_size, 1, KEY_VALUE_HEADS, ATTENTION_HEAD_DIM):
        raise ValueError("key must have shape [batch, 1, 2, 256]")
    if query_weight.shape != (ATTENTION_HEAD_DIM,) or key_weight.shape != (
        ATTENTION_HEAD_DIM,
    ):
        raise ValueError("q/k RMSNorm weights must have shape [256]")
    if cosine.shape != (batch_size, 1, ROTARY_DIM) or sine.shape != cosine.shape:
        raise ValueError("cosine/sine must have shape [batch, 1, 64]")
    if any(tensor.dtype != torch.bfloat16 for tensor in tensors):
        raise TypeError("q/k RMSNorm+RoPE tensors must be torch.bfloat16")
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("q/k RMSNorm+RoPE last dimensions must be contiguous")
