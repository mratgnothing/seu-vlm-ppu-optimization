#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

extern "C" int seu_acblas_packed_mlp_bf16(
    const uint16_t* packed_gate_up_weight,
    const uint16_t* down_weight,
    const uint16_t* input,
    uint16_t* projected,
    uint16_t* activated,
    uint16_t* output,
    int gate_up_algorithm,
    int down_algorithm,
    int swiglu_threads,
    void* stream_handle);

namespace {

torch::Tensor packed_mlp_bf16_into(
    const torch::Tensor& input,
    const torch::Tensor& packed_gate_up_weight,
    const torch::Tensor& down_weight,
    torch::Tensor projected,
    torch::Tensor activated,
    torch::Tensor output,
    int64_t expected_stream,
    int64_t gate_up_algorithm,
    int64_t down_algorithm,
    int64_t swiglu_threads) {
  TORCH_CHECK(
      input.is_cuda() && packed_gate_up_weight.is_cuda() &&
          down_weight.is_cuda() && projected.is_cuda() && activated.is_cuda() &&
          output.is_cuda(),
      "all tensors must be on PPU");
  const auto device = input.device();
  TORCH_CHECK(
      packed_gate_up_weight.device() == device && down_weight.device() == device &&
          projected.device() == device && activated.device() == device &&
          output.device() == device,
      "all tensors must share a device");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          packed_gate_up_weight.scalar_type() == torch::kBFloat16 &&
          down_weight.scalar_type() == torch::kBFloat16 &&
          projected.scalar_type() == torch::kBFloat16 &&
          activated.scalar_type() == torch::kBFloat16 &&
          output.scalar_type() == torch::kBFloat16,
      "all tensors must be BF16");
  TORCH_CHECK(input.sizes() == torch::IntArrayRef({1, 1, 2048}),
              "input must be [1,1,2048]");
  TORCH_CHECK(
      packed_gate_up_weight.sizes() == torch::IntArrayRef({12288, 2048}),
      "packed gate/up weight must be [12288,2048]");
  TORCH_CHECK(down_weight.sizes() == torch::IntArrayRef({2048, 6144}),
              "down weight must be [2048,6144]");
  TORCH_CHECK(projected.sizes() == torch::IntArrayRef({1, 1, 12288}),
              "projected scratch must be [1,1,12288]");
  TORCH_CHECK(activated.sizes() == torch::IntArrayRef({1, 1, 6144}),
              "activated scratch must be [1,1,6144]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({1, 1, 2048}),
              "output scratch must be [1,1,2048]");
  TORCH_CHECK(
      input.is_contiguous() && packed_gate_up_weight.is_contiguous() &&
          down_weight.is_contiguous() && projected.is_contiguous() &&
          activated.is_contiguous() && output.is_contiguous(),
      "all tensors must be contiguous");

  void* stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(stream) ==
          static_cast<uintptr_t>(expected_stream),
      "acBLAS packed MLP persistent scratch is bound to one CUDA stream; "
      "use the stream active when the module was patched");
  const int status = seu_acblas_packed_mlp_bf16(
      static_cast<const uint16_t*>(packed_gate_up_weight.data_ptr()),
      static_cast<const uint16_t*>(down_weight.data_ptr()),
      static_cast<const uint16_t*>(input.data_ptr()),
      static_cast<uint16_t*>(projected.data_ptr()),
      static_cast<uint16_t*>(activated.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      static_cast<int>(gate_up_algorithm),
      static_cast<int>(down_algorithm),
      static_cast<int>(swiglu_threads),
      stream);
  TORCH_CHECK(status == 0, "acBLAS packed MLP failed: ", status);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "packed_mlp_bf16_into",
      &packed_mlp_bf16_into,
      "PPU acBLAS packed Qwen3.5 MLP into persistent scratch");
}
