#include <acblas_v2.h>

#include <cstdint>
#include <cstddef>
#include <mutex>

// Exported by libseu_ppu_gdn.so.  Keep the ABI in integer/opaque-pointer
// terms so this acBLAS translation unit does not need the HGGC kernel source.
extern "C" int seu_ppu_swiglu_decode_bf16(
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* output,
    int elements,
    int threads,
    void* stream_handle);

namespace {

constexpr int kHiddenSize = 2048;
constexpr int kIntermediateSize = 6144;
constexpr int kPackedSize = 2 * kIntermediateSize;

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
  // Row-major [N,K] aliases column-major [K,N].
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

extern "C" int seu_acblas_packed_mlp_set_workspace(
    void* workspace,
    size_t workspace_bytes) {
  if ((workspace == nullptr) != (workspace_bytes == 0)) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  std::lock_guard<std::mutex> lock(get_handle_mutex());
  acblasHandle_t handle = get_handle(&status);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  return static_cast<int>(
      acblasSetWorkspace(handle, workspace, workspace_bytes));
}

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
    void* stream_handle) {
  if (
      packed_gate_up_weight == nullptr || down_weight == nullptr ||
      input == nullptr || projected == nullptr || activated == nullptr ||
      output == nullptr || swiglu_threads <= 0 || swiglu_threads > 1024 ||
      swiglu_threads % 32 != 0) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }

  std::lock_guard<std::mutex> lock(get_handle_mutex());
  acblasStatus_t status = ACBLAS_STATUS_SUCCESS;
  acblasHandle_t handle = get_handle(&status);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasSetStream(
      handle, reinterpret_cast<hggcStream_t>(stream_handle));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  status = run_gemv(
      handle,
      packed_gate_up_weight,
      input,
      projected,
      kPackedSize,
      kHiddenSize,
      gate_up_algorithm);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const int swiglu_status = seu_ppu_swiglu_decode_bf16(
      projected,
      projected + kIntermediateSize,
      activated,
      kIntermediateSize,
      swiglu_threads,
      stream_handle);
  if (swiglu_status != 0) return swiglu_status;

  status = run_gemv(
      handle,
      down_weight,
      activated,
      output,
      kHiddenSize,
      kIntermediateSize,
      down_algorithm);
  return static_cast<int>(status);
}
