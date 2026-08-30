"""Decode-only packing for Qwen3.5 GatedDeltaNet input projections."""

from __future__ import annotations

from collections.abc import Callable
import threading

import torch


HIDDEN_SIZE = 2048
_PROJECTIONS = (
    ("in_proj_qkv", 6144),
    ("in_proj_z", 2048),
    ("in_proj_b", 16),
    ("in_proj_a", 16),
)


def pack_qwen35_gdn_input_projections(
    module,
    group_sizes: tuple[int, ...] = (4,),
) -> dict[str, Callable]:
    """Fuse the four same-input projections during single-token decode.

    Qwen3.5 calls qkv, z, b and a in this order with the same hidden-state
    object. The first closure computes one [8224, 2048] projection; the next
    three return cached views. Prefill and unexpected call orders fall back to
    the original Linear forwards.
    """
    if type(module).__name__ != "Qwen3_5GatedDeltaNet":
        raise TypeError("packed GDN projections require Qwen3_5GatedDeltaNet")
    if not group_sizes or any(size <= 0 for size in group_sizes) or sum(group_sizes) != 4:
        raise ValueError("group_sizes must be a positive partition of four projections")

    weights = []
    original_forwards: dict[str, Callable] = {}
    for name, output_features in _PROJECTIONS:
        linear = getattr(module, name)
        if type(linear).__name__ != "Linear" or linear.bias is not None:
            raise ValueError(f"packed GDN requires bias-free Linear {name}")
        if linear.weight.shape != (output_features, HIDDEN_SIZE):
            raise ValueError(
                f"unexpected {name} weight shape {tuple(linear.weight.shape)}"
            )
        if linear.weight.dtype != torch.bfloat16 or linear.weight.device.type != "cuda":
            raise TypeError(f"packed GDN requires BF16 PPU weight {name}")
        if not linear.weight.is_contiguous():
            raise ValueError(f"packed GDN requires contiguous weight {name}")
        weights.append(linear.weight)
        original_forwards[name] = linear.forward

    with torch.no_grad():
        packed_weight = torch.cat(weights, dim=0).contiguous()
    offset = 0
    for name, output_features in _PROJECTIONS:
        linear = getattr(module, name)
        linear.weight.data = packed_weight[offset : offset + output_features]
        offset += output_features
    module.register_buffer("_seu_gdn_input_weight", packed_weight, persistent=False)

    packed_storage = packed_weight.untyped_storage().data_ptr()
    if any(
        getattr(module, name).weight.untyped_storage().data_ptr() != packed_storage
        for name, _ in _PROJECTIONS
    ):
        raise RuntimeError("packed GDN projection weights do not alias packed storage")

    forwards: dict[str, Callable] = {}
    projection_index = 0
    output_offset = 0
    for group_index, group_size in enumerate(group_sizes):
        group = _PROJECTIONS[projection_index : projection_index + group_size]
        projection_index += group_size
        group_output_features = tuple(features for _, features in group)
        group_weight = module._seu_gdn_input_weight[
            output_offset : output_offset + sum(group_output_features)
        ]
        output_offset += sum(group_output_features)
        if group_size == 1:
            name, _ = group[0]
            forwards[name] = original_forwards[name]
            continue

        cache = threading.local()
        leader_name = group[0][0]

        def leader_forward(
            x: torch.Tensor,
            *,
            _cache=cache,
            _features=group_output_features,
            _leader=leader_name,
            _weight=group_weight,
        ) -> torch.Tensor:
            if x.ndim >= 2 and x.shape[-2:] == (1, HIDDEN_SIZE):
                outputs = torch.nn.functional.linear(x, _weight).split(
                    _features, dim=-1
                )
                _cache.input = x
                _cache.outputs = outputs
                return outputs[0]
            _cache.__dict__.clear()
            return original_forwards[_leader](x)

        forwards[leader_name] = leader_forward
        for follower_index, (name, _) in enumerate(group[1:], start=1):
            clear = follower_index == group_size - 1

            def follower_forward(
                x: torch.Tensor,
                *,
                _cache=cache,
                _index=follower_index,
                _name=name,
                _clear=clear,
            ) -> torch.Tensor:
                outputs = getattr(_cache, "outputs", None)
                if getattr(_cache, "input", None) is x and isinstance(outputs, tuple):
                    result = outputs[_index]
                    if _clear:
                        _cache.__dict__.clear()
                    return result
                _cache.__dict__.clear()
                return original_forwards[_name](x)

            forwards[name] = follower_forward

    module._seu_gdn_original_forwards = original_forwards
    module._seu_gdn_packed_forwards = forwards
    module._seu_gdn_projection_group_sizes = group_sizes
    set_packed_qwen35_gdn_input_projections(module, True)
    return forwards


def set_packed_qwen35_gdn_input_projections(module, enabled: bool) -> None:
    """Toggle an already packed module without reallocating its weights."""
    originals = getattr(module, "_seu_gdn_original_forwards", None)
    packed = getattr(module, "_seu_gdn_packed_forwards", None)
    if not isinstance(originals, dict) or not isinstance(packed, dict):
        raise RuntimeError("GDN input projections have not been packed")
    selected = packed if enabled else originals
    for name, forward in selected.items():
        getattr(module, name).forward = forward
