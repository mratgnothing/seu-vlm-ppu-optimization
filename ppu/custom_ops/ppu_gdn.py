"""Thin ctypes integration for the fixed Qwen3.5-2B PPU decode GDN kernel."""

from __future__ import annotations

import ctypes
import os
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
