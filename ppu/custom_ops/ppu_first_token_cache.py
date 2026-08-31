"""Reusable Qwen3.5 cache buffers for lower first-token latency.

The implementation keeps the effective K/V length dynamic.  It only reserves
the backing storage and the fixed-shape Gated DeltaNet states ahead of the
timed generation call, so attention shapes and model arithmetic stay the same.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import (
    CacheLayerMixin,
    DYNAMIC_LAYER_TYPE_MAPPING,
    DynamicCache,
    DynamicLayer,
    LinearAttentionCacheLayerMixin,
    get_layer_types_and_kwargs,
)


class ReservedDynamicLayer(DynamicLayer):
    """A dynamic-length K/V layer backed by a reusable fixed-capacity tensor."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        self.capacity = capacity
        self.current_length = 0

    def reserve(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.dtype = dtype
        self.device = device
        shape = (batch_size, num_heads, self.capacity, head_dim)
        # Inactive positions are never returned to attention, so they do not
        # need to be zero-filled.
        self.keys = torch.empty(shape, dtype=dtype, device=device)
        self.values = torch.empty(shape, dtype=dtype, device=device)
        self.is_initialized = True

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        self.reserve(
            batch_size=key_states.shape[0],
            num_heads=key_states.shape[1],
            head_dim=key_states.shape[-1],
            dtype=key_states.dtype,
            device=key_states.device,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        next_length = self.current_length + key_states.shape[-2]
        if next_length > self.capacity:
            raise RuntimeError(
                f"reserved K/V cache capacity {self.capacity} is smaller than "
                f"the requested length {next_length}"
            )
        self.keys[..., self.current_length : next_length, :].copy_(key_states)
        self.values[..., self.current_length : next_length, :].copy_(value_states)
        self.current_length = next_length
        return (
            self.keys[..., :next_length, :],
            self.values[..., :next_length, :],
        )

    def get_seq_length(self) -> int:
        return self.current_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.current_length + query_length, 0

    def get_max_length(self) -> int:
        # Preserve DynamicLayer semantics for mask construction and generation.
        return -1

    def reset(self) -> None:
        # Every valid position is overwritten before it is exposed again.
        self.current_length = 0


@dataclass(frozen=True)
class CacheAllocation:
    capacity: int
    attention_layers: int
    linear_layers: int
    reserved_bytes: int


class Qwen35CachePool:
    """Own one reusable cache for sequential batch-one benchmark requests."""

    def __init__(self, model, *, capacity: int = 4096, mode: str = "all") -> None:
        if capacity <= 0:
            raise ValueError("cache capacity must be positive")
        if mode not in {"all", "kv", "linear"}:
            raise ValueError("cache mode must be one of: all, kv, linear")
        self.model = model
        self.capacity = capacity
        self.mode = mode
        self.reserve_attention = mode in {"all", "kv"}
        self.reserve_linear = mode in {"all", "linear"}
        self.cache: DynamicCache | None = None
        self.allocation: CacheAllocation | None = None
        self.batch_size: int | None = None

    def acquire(self, *, required_length: int, batch_size: int) -> DynamicCache:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if required_length > self.capacity:
            # Grow outside the timed region instead of rejecting a legitimate
            # long prompt.  A power-of-two capacity limits repeated rebuilds.
            self.capacity = 1 << (required_length - 1).bit_length()
            self.cache = None
            self.allocation = None
        if self.batch_size != batch_size:
            self.cache = None
            self.allocation = None
            self.batch_size = batch_size
        if self.cache is None:
            self.cache, self.allocation = self._allocate(batch_size=batch_size)
        else:
            self._fast_reset(self.cache)
        return self.cache

    def _allocate(self, *, batch_size: int) -> tuple[DynamicCache, CacheAllocation]:
        config = self.model.config
        text_config = config.get_text_config(decoder=True)
        parameter = next(self.model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        cache = DynamicCache(config=config)

        attention_layers = 0
        linear_layers = 0
        reserved_bytes = 0
        num_kv_heads = int(text_config.num_key_value_heads)
        head_dim = int(text_config.head_dim)
        element_size = torch.empty((), dtype=dtype).element_size()

        for index, layer_type in enumerate(text_config.layer_types):
            if layer_type != "full_attention" or not self.reserve_attention:
                continue
            layer = ReservedDynamicLayer(self.capacity)
            layer.reserve(
                batch_size=batch_size,
                num_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
            )
            cache.layers[index] = layer
            attention_layers += 1
            reserved_bytes += (
                2
                * batch_size
                * num_kv_heads
                * self.capacity
                * head_dim
                * element_size
            )

        gdn_modules = {
            module.layer_idx: module
            for module in self.model.modules()
            if type(module).__name__ == "Qwen3_5GatedDeltaNet"
        }
        for index, layer in enumerate(cache.layers):
            if (
                not isinstance(layer, LinearAttentionCacheLayerMixin)
                or not self.reserve_linear
            ):
                continue
            module = gdn_modules.get(index)
            if module is None:
                raise RuntimeError(f"missing Qwen3.5 GDN module for layer {index}")
            conv_template = torch.empty(
                (batch_size, module.conv_dim, module.conv_kernel_size),
                dtype=dtype,
                device=device,
            )
            recurrent_template = torch.empty(
                (
                    batch_size,
                    module.num_v_heads,
                    module.head_k_dim,
                    module.head_v_dim,
                ),
                dtype=torch.float32,
                device=device,
            )
            layer.lazy_initialization(
                conv_states=conv_template,
                recurrent_states=recurrent_template,
            )
            linear_layers += 1
            reserved_bytes += (
                conv_template.numel() * conv_template.element_size()
                + recurrent_template.numel() * recurrent_template.element_size()
            )

        if attention_layers != 6 or linear_layers != 18:
            raise RuntimeError(
                "expected Qwen3.5-2B cache topology 6 attention + 18 linear, "
                f"got {attention_layers} + {linear_layers}"
            )
        return cache, CacheAllocation(
            capacity=self.capacity,
            attention_layers=attention_layers,
            linear_layers=linear_layers,
            reserved_bytes=reserved_bytes,
        )

    def _fresh_layer(self, layer_idx: int):
        text_config = self.model.config.get_text_config(decoder=True)
        layer_types, layer_kwargs = get_layer_types_and_kwargs(text_config)
        return DYNAMIC_LAYER_TYPE_MAPPING[layer_types[layer_idx]](**layer_kwargs)

    def _fast_reset(self, cache: DynamicCache) -> None:
        for index, layer in enumerate(cache.layers):
            if isinstance(layer, ReservedDynamicLayer):
                layer.reset()
            elif isinstance(layer, LinearAttentionCacheLayerMixin):
                if self.reserve_linear:
                    # A fresh prefill overwrites both fixed-shape states in full.
                    # Keeping the backing tensors avoids allocation and memset.
                    for state_idx in range(layer.number_of_states):
                        layer.has_previous_state[state_idx] = False
                else:
                    cache.layers[index] = self._fresh_layer(index)
            elif isinstance(layer, CacheLayerMixin):
                # DynamicLayer.reset() zeroes storage but does not shorten it.
                # Recreate non-reserved attention layers to preserve the exact
                # baseline behavior for the linear-only ablation.
                cache.layers[index] = self._fresh_layer(index)
