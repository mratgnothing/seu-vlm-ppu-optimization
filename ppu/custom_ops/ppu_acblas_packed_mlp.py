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

    def capture_module_graph(
        self, module, static_input: torch.Tensor, *, warmup: int = 3
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Capture the packed decode path for one stable RMSNorm output buffer."""
        packed_forward = getattr(module, "_seu_acblas_packed_mlp_forward", None)
        if packed_forward is None:
            raise RuntimeError("patch_module must run before graph capture")
        if hasattr(module, "_seu_acblas_packed_mlp_graph_forward"):
            raise RuntimeError("module already has a packed MLP graph")
        if (
            static_input.shape != (1, 1, HIDDEN_SIZE)
            or static_input.dtype != torch.bfloat16
            or static_input.device != module._seu_gate_up_weight.device
            or not static_input.is_contiguous()
        ):
            raise ValueError("graph input must be contiguous BF16 [1,1,2048]")
        if warmup < 1:
            raise ValueError("graph warmup must be positive")

        extension = self.module
        capture_stream = torch.cuda.Stream(device=static_input.device)
        capture_stream.wait_stream(torch.cuda.current_stream(static_input.device))

        def raw_submit(expected_stream: int) -> torch.Tensor:
            return extension.packed_mlp_bf16_into(
                static_input,
                module._seu_gate_up_weight,
                module.down_proj.weight,
                module._seu_acblas_packed_projected,
                module._seu_acblas_packed_activated,
                module._seu_acblas_packed_output,
                expected_stream,
                self.gate_up_algorithm,
                self.down_algorithm,
                self.swiglu_threads,
            )

        with torch.inference_mode(), torch.cuda.stream(capture_stream):
            for _ in range(warmup):
                raw_submit(capture_stream.cuda_stream)
        torch.cuda.current_stream(static_input.device).wait_stream(capture_stream)
        torch.cuda.synchronize(static_input.device)

        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, stream=capture_stream):
            graph_output = raw_submit(capture_stream.cuda_stream)
        torch.cuda.synchronize(static_input.device)
        device_index = static_input.device.index or 0
        replay_stream = torch._C._cuda_getCurrentRawStream(device_index)

        def graph_forward(input_tensor: torch.Tensor) -> torch.Tensor:
            if (
                input_tensor.shape == (1, 1, HIDDEN_SIZE)
                and input_tensor.dtype == torch.bfloat16
                and input_tensor.device == static_input.device
                and input_tensor.is_contiguous()
                and input_tensor.data_ptr() == static_input.data_ptr()
                and not torch.is_grad_enabled()
            ):
                current_stream = torch._C._cuda_getCurrentRawStream(device_index)
                if current_stream != replay_stream:
                    raise RuntimeError(
                        "PPU packed MLP graph is bound to one CUDA stream; "
                        "concurrent decode requires per-stream graph state"
                    )
                graph.replay()
                return graph_output
            return packed_forward(input_tensor)

        module._seu_acblas_packed_mlp_graph = graph
        module._seu_acblas_packed_mlp_graph_input = static_input
        module._seu_acblas_packed_mlp_graph_output = graph_output
        module._seu_acblas_packed_mlp_graph_forward = graph_forward
        return graph_forward

    @staticmethod
    def set_enabled(module, enabled: bool) -> None:
        original = getattr(module, "_seu_acblas_packed_mlp_original_forward", None)
        candidate = getattr(module, "_seu_acblas_packed_mlp_forward", None)
        if original is None or candidate is None:
            raise RuntimeError("module does not have the acBLAS packed MLP patch")
        module.forward = candidate if enabled else original

    @staticmethod
    def set_graph_enabled(module, enabled: bool) -> None:
        packed = getattr(module, "_seu_acblas_packed_mlp_forward", None)
        graph = getattr(module, "_seu_acblas_packed_mlp_graph_forward", None)
        if packed is None or graph is None:
            raise RuntimeError("module does not have a captured packed MLP graph")
        module.forward = graph if enabled else packed
