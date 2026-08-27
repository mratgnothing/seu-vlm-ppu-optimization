#include <acblas_v2.h>

#include <cstdint>
#include <mutex>

namespace {

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

}  // namespace

extern "C" int seu_acblas_linear_bf16(
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int output_features,
    int input_features,
    int algorithm,
    void* stream_handle) {
  if (
      row_major_weight == nullptr || input == nullptr || output == nullptr ||
      output_features <= 0 || input_features <= 0) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  // acBLAS handles carry stream state. Protect the short host-side SetStream
  // and enqueue sequence while allowing already enqueued device work to run
  // asynchronously on its stream after the API returns.
  std::lock_guard<std::mutex> lock(get_handle_mutex());
  acblasHandle_t handle = get_handle(&status);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasSetStream(
      handle, reinterpret_cast<hggcStream_t>(stream_handle));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major [N, K] has the same bytes as column-major [K, N].
  status = acblasGemvEx(
      handle,
      ACBLAS_OP_T,
      input_features,
      output_features,
      HGGC_R_16BF,
      &alpha,
      row_major_weight,
      input_features,
      input,
      1,
      &beta,
      output,
      1,
      ACBLAS_COMPUTE_32F,
      static_cast<acblasGemmAlgo_t>(algorithm));
  return static_cast<int>(status);
}
