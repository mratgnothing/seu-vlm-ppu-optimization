#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdint>

extern "C" int seu_acblas_attention_prep_bf16(
    const uint16_t* packed_qkv_weight,
    const uint16_t* hidden_states,
    const uint16_t* query_weight,
    const uint16_t* key_weight,
    const uint16_t* cosine,
    const uint16_t* sine,
    uint16_t* projected,
    uint16_t* query_output,
    uint16_t* key_output,
    int64_t cosine_batch_stride,
    int64_t sine_batch_stride,
    float epsilon,
    int algorithm,
    void* stream_handle);

namespace {

torch::Tensor attention_prep_bf16_into(
    const torch::Tensor& hidden_states,
    const torch::Tensor& packed_qkv_weight,
    const torch::Tensor& query_weight,
    const torch::Tensor& key_weight,
    const torch::Tensor& cosine,
    const torch::Tensor& sine,
    torch::Tensor projected,
    torch::Tensor query_output,
    torch::Tensor key_output,
    int64_t expected_stream,
    double epsilon,
    int64_t algorithm) {
  const torch::Tensor tensors[] = {
      hidden_states, packed_qkv_weight, query_weight, key_weight, cosine, sine,
      projected, query_output, key_output};
  const auto device = hidden_states.device();
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.is_cuda(), "all attention-prep tensors must be on PPU");
    TORCH_CHECK(tensor.device() == device, "attention-prep device mismatch");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kBFloat16,
        "all attention-prep tensors must be BF16");
  }
  TORCH_CHECK(hidden_states.sizes() == torch::IntArrayRef({1, 1, 2048}),
              "hidden_states must be [1,1,2048]");
  TORCH_CHECK(
      packed_qkv_weight.sizes() == torch::IntArrayRef({5120, 2048}),
      "packed QKV weight must be [5120,2048]");
  TORCH_CHECK(query_weight.sizes() == torch::IntArrayRef({256}) &&
                  key_weight.sizes() == torch::IntArrayRef({256}),
              "q/k RMSNorm weights must be [256]");
  TORCH_CHECK(cosine.sizes() == torch::IntArrayRef({1, 1, 64}) &&
                  sine.sizes() == torch::IntArrayRef({1, 1, 64}),
              "cosine/sine must be [1,1,64]");
  TORCH_CHECK(projected.sizes() == torch::IntArrayRef({1, 1, 5120}),
              "projected scratch must be [1,1,5120]");
  TORCH_CHECK(query_output.sizes() == torch::IntArrayRef({1, 8, 1, 256}),
              "query scratch must be [1,8,1,256]");
  TORCH_CHECK(key_output.sizes() == torch::IntArrayRef({1, 2, 1, 256}),
              "key scratch must be [1,2,1,256]");
  TORCH_CHECK(
      hidden_states.is_contiguous() && packed_qkv_weight.is_contiguous() &&
          query_weight.is_contiguous() && key_weight.is_contiguous() &&
          cosine.stride(-1) == 1 && sine.stride(-1) == 1 &&
          projected.is_contiguous() && query_output.is_contiguous() &&
          key_output.is_contiguous(),
      "attention-prep tensors have an unsupported layout");

  void* stream =
      at::cuda::getCurrentCUDAStream(hidden_states.get_device()).stream();
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(stream) ==
          static_cast<uintptr_t>(expected_stream),
      "acBLAS attention prep persistent scratch is bound to one CUDA stream; "
      "use the stream active when the module was patched");
  const int status = seu_acblas_attention_prep_bf16(
      static_cast<const uint16_t*>(packed_qkv_weight.data_ptr()),
      static_cast<const uint16_t*>(hidden_states.data_ptr()),
      static_cast<const uint16_t*>(query_weight.data_ptr()),
      static_cast<const uint16_t*>(key_weight.data_ptr()),
      static_cast<const uint16_t*>(cosine.data_ptr()),
      static_cast<const uint16_t*>(sine.data_ptr()),
      static_cast<uint16_t*>(projected.data_ptr()),
      static_cast<uint16_t*>(query_output.data_ptr()),
      static_cast<uint16_t*>(key_output.data_ptr()),
      cosine.stride(0),
      sine.stride(0),
      static_cast<float>(epsilon),
      static_cast<int>(algorithm),
      stream);
  TORCH_CHECK(status == 0, "acBLAS attention prep failed: ", status);
  return query_output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "attention_prep_bf16_into",
      &attention_prep_bf16_into,
      "PPU grouped QKV GEMV plus q/k RMSNorm+RoPE into persistent scratch");
}
