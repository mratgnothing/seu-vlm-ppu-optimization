#include <acblas_v2.h>

#include <cstdint>
#include <mutex>

namespace {

acblasHandle_t& handle() {
  static acblasHandle_t value = nullptr;
  return value;
}

std::mutex& handle_mutex() {
  static std::mutex value;
  return value;
}

}  // namespace

extern "C" int seu_acblas_gemm_for_gemv_bf16(
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int output_features,
    int input_features,
    int algorithm,
    void* stream_handle) {
  if (row_major_weight == nullptr || input == nullptr || output == nullptr ||
      output_features <= 0 || input_features <= 0) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }

  std::lock_guard<std::mutex> lock(handle_mutex());
  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  if (handle() == nullptr) {
    status = acblasCreate(&handle());
  }
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasSetStream(
      handle(), reinterpret_cast<hggcStream_t>(stream_handle));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major [N,K] aliases column-major [K,N]. Transpose it and multiply by
  // a column-major [K,1] vector, yielding column-major [N,1].
  return static_cast<int>(acblasGemmEx(
      handle(),
      ACBLAS_OP_T,
      ACBLAS_OP_N,
      output_features,
      1,
      input_features,
      &alpha,
      row_major_weight,
      HGGC_R_16BF,
      input_features,
      input,
      HGGC_R_16BF,
      input_features,
      &beta,
      output,
      HGGC_R_16BF,
      output_features,
      ACBLAS_COMPUTE_32F,
      static_cast<acblasGemmAlgo_t>(algorithm)));
}

extern "C" void seu_acblas_gemm_for_gemv_destroy() {
  std::lock_guard<std::mutex> lock(handle_mutex());
  if (handle() != nullptr) {
    acblasDestroy(handle());
    handle() = nullptr;
  }
}
