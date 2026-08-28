#include <acblas_v2.h>

#include <cstdint>
#include <mutex>

extern "C" int seu_ppu_qk_rmsnorm_rope_decode_bf16(
    const uint16_t* query,
    const uint16_t* key,
    const uint16_t* query_weight,
    const uint16_t* key_weight,
    const uint16_t* cosine,
    const uint16_t* sine,
    uint16_t* query_output,
    uint16_t* key_output,
    int batch_size,
    int64_t query_batch_stride,
    int64_t query_head_stride,
    int64_t key_batch_stride,
    int64_t key_head_stride,
    int64_t cosine_batch_stride,
    int64_t sine_batch_stride,
    float epsilon,
    void* stream_handle);

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kQueryProjectionSize = 4096;
constexpr int kKeyProjectionSize = 512;
constexpr int kValueProjectionSize = 512;
constexpr int kPackedProjectionSize =
    kQueryProjectionSize + kKeyProjectionSize + kValueProjectionSize;

acblasHandle_t get_handle(acblasStatus_t* status) {
  static acblasHandle_t handle = nullptr;
  if (handle == nullptr) {
    *status = acblasCreate(&handle);
  } else {
    *status = ACBLAS_STATUS_SUCCESS;
  }
  return handle;
}

std::mutex& get_handle_mutex() {
  static std::mutex mutex;
  return mutex;
}

acblasStatus_t run_gemv(
    acblasHandle_t handle,
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int output_features,
    int algorithm) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  return acblasGemvEx(
      handle,
      ACBLAS_OP_T,
      kHiddenSize,
      output_features,
      HGGC_R_16BF,
      &alpha,
      row_major_weight,
      kHiddenSize,
      input,
      1,
      &beta,
      output,
      1,
      ACBLAS_COMPUTE_32F,
      static_cast<acblasGemmAlgo_t>(algorithm));
}

}  // namespace

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
    void* stream_handle) {
  if (
      packed_qkv_weight == nullptr || hidden_states == nullptr ||
      query_weight == nullptr || key_weight == nullptr || cosine == nullptr ||
      sine == nullptr || projected == nullptr || query_output == nullptr ||
      key_output == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }

  std::lock_guard<std::mutex> lock(get_handle_mutex());
  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  acblasHandle_t handle = get_handle(&status);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasSetStream(
      handle, reinterpret_cast<hggcStream_t>(stream_handle));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  constexpr int kOffsets[] = {
      0, kQueryProjectionSize, kQueryProjectionSize + kKeyProjectionSize};
  constexpr int kSizes[] = {
      kQueryProjectionSize, kKeyProjectionSize, kValueProjectionSize};
  for (int index = 0; index < 3; ++index) {
    status = run_gemv(
        handle,
        packed_qkv_weight + static_cast<int64_t>(kOffsets[index]) * kHiddenSize,
        hidden_states,
        projected + kOffsets[index],
        kSizes[index],
        algorithm);
    if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }

  return seu_ppu_qk_rmsnorm_rope_decode_bf16(
      projected,
      projected + kQueryProjectionSize,
      query_weight,
      key_weight,
      cosine,
      sine,
      query_output,
      key_output,
      1,
      kQueryProjectionSize,
      512,
      kKeyProjectionSize,
      256,
      cosine_batch_stride,
      sine_batch_stride,
      epsilon,
      stream_handle);
}
