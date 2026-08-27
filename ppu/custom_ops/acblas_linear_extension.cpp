#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

// Keep PPU SDK headers out of this translation unit: CUDA compatibility
// headers pulled in by PyTorch and HGGC headers both define half/bfloat16
// types. The acBLAS-only shared library provides this narrow C ABI.
extern "C" int seu_acblas_linear_bf16(
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int output_features,
    int input_features,
    int algorithm,
    void* stream_handle);
extern "C" int seu_acblas_gdn_projections_bf16(
    const uint16_t* packed_weight,
    const uint16_t* input,
    uint16_t* packed_output,
    int algorithm,
    void* stream_handle);

namespace {

torch::Tensor acblas_linear_bf16(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int64_t algorithm) {
  TORCH_CHECK(input.is_cuda() && weight.is_cuda(), "input/weight must be on PPU");
  TORCH_CHECK(input.device() == weight.device(), "input/weight device mismatch");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          weight.scalar_type() == torch::kBFloat16,
      "input/weight must be BF16");
  TORCH_CHECK(input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1,
              "input must be [1,1,K]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(1) == input.size(2),
              "weight must be [N,K]");
  TORCH_CHECK(input.stride(2) == 1 && weight.is_contiguous(),
              "input last dimension and weight must be contiguous");

  const int output_features = static_cast<int>(weight.size(0));
  const int input_features = static_cast<int>(weight.size(1));
  torch::Tensor output = torch::empty(
      {1, 1, output_features}, input.options());
  void* stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int status = seu_acblas_linear_bf16(
      static_cast<const uint16_t*>(weight.data_ptr()),
      static_cast<const uint16_t*>(input.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      output_features,
      input_features,
      static_cast<int>(algorithm),
      stream);
  TORCH_CHECK(status == 0, "acBLAS BF16 linear failed: ", status);
  return output;
}

torch::Tensor acblas_gdn_projections_bf16(
    const torch::Tensor& input,
    const torch::Tensor& packed_weight,
    int64_t algorithm) {
  TORCH_CHECK(
      input.is_cuda() && packed_weight.is_cuda(), "input/weight must be on PPU");
  TORCH_CHECK(input.device() == packed_weight.device(), "device mismatch");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          packed_weight.scalar_type() == torch::kBFloat16,
      "input/weight must be BF16");
  TORCH_CHECK(
      input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1 &&
          input.size(2) == 2048,
      "input must be [1,1,2048]");
  TORCH_CHECK(
      packed_weight.dim() == 2 && packed_weight.size(0) == 8224 &&
          packed_weight.size(1) == 2048,
      "packed weight must be [8224,2048]");
  TORCH_CHECK(input.stride(2) == 1 && packed_weight.is_contiguous(),
              "input last dimension and weight must be contiguous");

  torch::Tensor output = torch::empty({1, 1, 8224}, input.options());
  void* stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  const int status = seu_acblas_gdn_projections_bf16(
      static_cast<const uint16_t*>(packed_weight.data_ptr()),
      static_cast<const uint16_t*>(input.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      static_cast<int>(algorithm),
      stream);
  TORCH_CHECK(status == 0, "acBLAS GDN projections failed: ", status);
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("linear_bf16", &acblas_linear_bf16, "PPU acBLAS BF16 decode linear");
  module.def(
      "gdn_projections_bf16",
      &acblas_gdn_projections_bf16,
      "PPU grouped acBLAS Qwen3.5 GDN projections");
}
