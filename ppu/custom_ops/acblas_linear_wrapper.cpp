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

acblasStatus_t run_gemv(
    acblasHandle_t handle,
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int output_features,
    int input_features,
    int algorithm) {
  const float alpha = 1.0F;
  const float beta = 0.0F;
  // Row-major [N, K] has the same bytes as column-major [K, N].
  return acblasGemvEx(
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

  status = run_gemv(
      handle,
      row_major_weight,
      input,
      output,
      output_features,
      input_features,
      algorithm);
  return static_cast<int>(status);
}

extern "C" int seu_acblas_gdn_projections_bf16(
    const uint16_t* packed_weight,
    const uint16_t* input,
    uint16_t* packed_output,
    int algorithm,
    void* stream_handle) {
  if (packed_weight == nullptr || input == nullptr || packed_output == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  constexpr int kInputFeatures = 2048;
  constexpr int kOutputOffsets[] = {0, 6144, 8192, 8208};
  constexpr int kOutputFeatures[] = {6144, 2048, 16, 16};

  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  std::lock_guard<std::mutex> lock(get_handle_mutex());
  acblasHandle_t handle = get_handle(&status);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasSetStream(
      handle, reinterpret_cast<hggcStream_t>(stream_handle));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  for (int index = 0; index < 4; ++index) {
    const int offset = kOutputOffsets[index];
    status = run_gemv(
        handle,
        packed_weight + static_cast<int64_t>(offset) * kInputFeatures,
        input,
        packed_output + offset,
        kOutputFeatures[index],
        kInputFeatures,
        algorithm);
    if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  }
  return static_cast<int>(ACBLAS_STATUS_SUCCESS);
}
