#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
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
extern "C" int seu_acblas_linear_set_workspace(
    void* workspace,
    size_t workspace_bytes);
extern "C" void seu_acblas_gdn_set_batched_ba(int enabled);
extern "C" void seu_acblas_gdn_set_ba_gemv(int enabled);
extern "C" void seu_acblas_gdn_set_single_gemv(int enabled);
extern "C" void seu_acblas_gdn_set_tail_gemv(int enabled);

namespace {

void set_workspace(const torch::Tensor& workspace) {
  TORCH_CHECK(workspace.is_cuda(), "workspace must be on PPU");
  TORCH_CHECK(workspace.scalar_type() == torch::kUInt8,
              "workspace must be uint8");
  TORCH_CHECK(workspace.dim() == 1 && workspace.numel() > 0,
              "workspace must be a non-empty 1D tensor");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  const int status = seu_acblas_linear_set_workspace(
      workspace.data_ptr(), static_cast<size_t>(workspace.numel()));
  TORCH_CHECK(status == 0, "acBLAS linear workspace setup failed: ", status);
}

void clear_workspace() {
  const int status = seu_acblas_linear_set_workspace(nullptr, 0);
  TORCH_CHECK(status == 0, "acBLAS linear workspace clear failed: ", status);
}

void set_gdn_batched_ba(bool enabled) {
  seu_acblas_gdn_set_batched_ba(enabled ? 1 : 0);
}

void set_gdn_ba_gemv(bool enabled) {
  seu_acblas_gdn_set_ba_gemv(enabled ? 1 : 0);
}

void set_gdn_single_gemv(bool enabled) {
  seu_acblas_gdn_set_single_gemv(enabled ? 1 : 0);
}

void set_gdn_tail_gemv(bool enabled) {
  seu_acblas_gdn_set_tail_gemv(enabled ? 1 : 0);
}

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

torch::Tensor acblas_gdn_projections_bf16_into(
    const torch::Tensor& input,
    const torch::Tensor& packed_weight,
    torch::Tensor output,
    int64_t expected_stream,
    int64_t algorithm) {
  TORCH_CHECK(
      input.is_cuda() && packed_weight.is_cuda() && output.is_cuda(),
      "input/weight/output must be on PPU");
  TORCH_CHECK(
      input.device() == packed_weight.device() && input.device() == output.device(),
      "device mismatch");
  TORCH_CHECK(
      input.scalar_type() == torch::kBFloat16 &&
          packed_weight.scalar_type() == torch::kBFloat16 &&
          output.scalar_type() == torch::kBFloat16,
      "input/weight/output must be BF16");
  TORCH_CHECK(
      input.dim() == 3 && input.size(0) == 1 && input.size(1) == 1 &&
          input.size(2) == 2048,
      "input must be [1,1,2048]");
  TORCH_CHECK(
      packed_weight.dim() == 2 && packed_weight.size(0) == 8224 &&
          packed_weight.size(1) == 2048,
      "packed weight must be [8224,2048]");
  TORCH_CHECK(output.sizes() == torch::IntArrayRef({1, 1, 8224}),
              "output scratch must be [1,1,8224]");
  TORCH_CHECK(
      input.stride(2) == 1 && packed_weight.is_contiguous() &&
          output.is_contiguous(),
      "input last dimension, weight, and output must be contiguous");

  void* stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();
  TORCH_CHECK(
      expected_stream < 0 ||
          reinterpret_cast<uintptr_t>(stream) ==
              static_cast<uintptr_t>(expected_stream),
      "acBLAS GDN output scratch is bound to one CUDA stream; "
      "concurrent decode requires per-stream scratch");
  const int status = seu_acblas_gdn_projections_bf16(
      static_cast<const uint16_t*>(packed_weight.data_ptr()),
      static_cast<const uint16_t*>(input.data_ptr()),
      static_cast<uint16_t*>(output.data_ptr()),
      static_cast<int>(algorithm),
      stream);
  TORCH_CHECK(status == 0, "acBLAS GDN projections failed: ", status);
  return output;
}

torch::Tensor acblas_gdn_projections_bf16(
    const torch::Tensor& input,
    const torch::Tensor& packed_weight,
    int64_t algorithm) {
  torch::Tensor output = torch::empty({1, 1, 8224}, input.options());
  return acblas_gdn_projections_bf16_into(
      input, packed_weight, output, -1, algorithm);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("set_workspace", &set_workspace, "Set persistent acBLAS workspace");
  module.def("clear_workspace", &clear_workspace, "Clear acBLAS workspace");
  module.def(
      "set_gdn_batched_ba",
      &set_gdn_batched_ba,
      "Batch the two homogeneous 16x2048 GDN b/a projections");
  module.def(
      "set_gdn_ba_gemv",
      &set_gdn_ba_gemv,
      "Run the adjacent GDN b/a weights as one 32x2048 GEMV");
  module.def(
      "set_gdn_single_gemv",
      &set_gdn_single_gemv,
      "Run the packed 8224x2048 GDN projection as one GEMV");
  module.def(
      "set_gdn_tail_gemv",
      &set_gdn_tail_gemv,
      "Run GDN qkv separately and packed z/b/a as one GEMV");
  module.def("linear_bf16", &acblas_linear_bf16, "PPU acBLAS BF16 decode linear");
  module.def(
      "gdn_projections_bf16",
      &acblas_gdn_projections_bf16,
      "PPU grouped acBLAS Qwen3.5 GDN projections");
  module.def(
      "gdn_projections_bf16_into",
      &acblas_gdn_projections_bf16_into,
      "PPU grouped acBLAS Qwen3.5 GDN projections into persistent scratch");
}
