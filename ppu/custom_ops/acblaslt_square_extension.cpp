#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <cstdint>
#include <mutex>

extern "C" int seu_acblaslt_prepare_bf16(
    int output_features,
    int input_features,
    size_t max_workspace_bytes,
    int requested_algorithms,
    int* returned_algorithms);
extern "C" int seu_acblaslt_matmul_bf16(
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int heuristic_index,
    void* workspace,
    size_t workspace_bytes,
    void* stream_handle);

namespace {

torch::Tensor acblaslt_square_linear_bf16_into(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor output,
    int64_t heuristic_index) {
  TORCH_CHECK(
      input.is_cuda() && weight.is_cuda() && output.is_cuda(),
      "input/weight/output must be on PPU");
  TORCH_CHECK(
      input.device() == weight.device() && input.device() == output.device(),
      "input/weight/output device mismatch");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          weight.scalar_type() == torch::kBFloat16 &&
          output.scalar_type() == torch::kBFloat16,
      "input/weight/output must be BF16");
  TORCH_CHECK(
      input.sizes() == torch::IntArrayRef({1, 1, 2048}),
      "input must be [1,1,2048]");
  TORCH_CHECK(
      weight.sizes() == torch::IntArrayRef({2048, 2048}),
      "weight must be [2048,2048]");
  TORCH_CHECK(
      output.sizes() == torch::IntArrayRef({1, 1, 2048}),
      "output must be [1,1,2048]");
  TORCH_CHECK(
      input.stride(2) == 1 && weight.is_contiguous() && output.is_contiguous(),
      "input last dimension and weight/output must be contiguous");

  static std::once_flag prepare_once;
  static int prepare_status = -1;
  static int heuristic_count = 0;
  std::call_once(prepare_once, []() {
    prepare_status = seu_acblaslt_prepare_bf16(
        2048, 2048, 64ULL * 1024ULL * 1024ULL, 32, &heuristic_count);
  });
  TORCH_CHECK(prepare_status == 0, "acBLASLt prepare failed: ", prepare_status);
  TORCH_CHECK(
      heuristic_index >= 0 && heuristic_index < heuristic_count,
      "acBLASLt heuristic index out of range: ",
      heuristic_index,
      " / ",
      heuristic_count);

  void* stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int status = seu_acblaslt_matmul_bf16(
      static_cast<const uint16_t*>(weight.data_ptr()),
      static_cast<const uint16_t*>(input.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      static_cast<int>(heuristic_index),
      nullptr,
      0,
      stream);
  TORCH_CHECK(status == 0, "acBLASLt BF16 square linear failed: ", status);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "acblaslt_square_linear_bf16_into",
      &acblaslt_square_linear_bf16_into,
      "PPU acBLASLt BF16 2048-square decode linear into scratch output");
}
