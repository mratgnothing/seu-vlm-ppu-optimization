#include <acblasLt.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct Int8MatmulState {
  acblasLtHandle_t handle = nullptr;
  acblasLtMatmulDesc_t operation = nullptr;
  acblasLtMatrixLayout_t input_layout = nullptr;
  acblasLtMatrixLayout_t weight_layout = nullptr;
  acblasLtMatrixLayout_t output_layout = nullptr;
  acblasLtMatmulPreference_t preference = nullptr;
  std::vector<acblasLtMatmulHeuristicResult_t> heuristics;
};

Int8MatmulState& state() {
  static Int8MatmulState value;
  return value;
}

std::mutex& state_mutex() {
  static std::mutex value;
  return value;
}

void destroy_state(Int8MatmulState& value) {
  if (value.preference != nullptr) acblasLtMatmulPreferenceDestroy(value.preference);
  if (value.output_layout != nullptr) acblasLtMatrixLayoutDestroy(value.output_layout);
  if (value.weight_layout != nullptr) acblasLtMatrixLayoutDestroy(value.weight_layout);
  if (value.input_layout != nullptr) acblasLtMatrixLayoutDestroy(value.input_layout);
  if (value.operation != nullptr) acblasLtMatmulDescDestroy(value.operation);
  if (value.handle != nullptr) acblasLtDestroy(value.handle);
  value = Int8MatmulState{};
}

}  // namespace

extern "C" int seu_acblaslt_prepare_int8(
    int output_features,
    int input_features,
    size_t max_workspace_bytes,
    int requested_algorithms,
    int* returned_algorithms) {
  if (output_features <= 0 || input_features <= 0 ||
      requested_algorithms <= 0 || returned_algorithms == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  std::lock_guard<std::mutex> lock(state_mutex());
  Int8MatmulState& value = state();
  destroy_state(value);
  acblasStatus_t status = acblasLtCreate(&value.handle);
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulDescCreate(
        &value.operation, ACBLAS_COMPUTE_32I, HGGC_R_32I);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    const acblasOperation_t transpose_weight = ACBLAS_OP_T;
    status = acblasLtMatmulDescSetAttribute(
        value.operation,
        ACBLASLT_MATMUL_DESC_TRANSA,
        &transpose_weight,
        sizeof(transpose_weight));
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &value.weight_layout,
        HGGC_R_8I,
        input_features,
        output_features,
        input_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &value.input_layout, HGGC_R_8I, input_features, 1, input_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &value.output_layout, HGGC_R_32I, output_features, 1, output_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulPreferenceCreate(&value.preference);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulPreferenceSetAttribute(
        value.preference,
        ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &max_workspace_bytes,
        sizeof(max_workspace_bytes));
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    value.heuristics.resize(static_cast<size_t>(requested_algorithms));
    int count = 0;
    status = acblasLtMatmulAlgoGetHeuristic(
        value.handle,
        value.operation,
        value.weight_layout,
        value.input_layout,
        value.output_layout,
        value.output_layout,
        value.preference,
        requested_algorithms,
        value.heuristics.data(),
        &count);
    if (status == ACBLAS_STATUS_SUCCESS) {
      value.heuristics.resize(static_cast<size_t>(count));
      *returned_algorithms = count;
    }
  }
  return static_cast<int>(status);
}

extern "C" int seu_acblaslt_int8_heuristic_info(
    int index, size_t* workspace_bytes, float* waves_count) {
  std::lock_guard<std::mutex> lock(state_mutex());
  const Int8MatmulState& value = state();
  if (index < 0 || static_cast<size_t>(index) >= value.heuristics.size() ||
      workspace_bytes == nullptr || waves_count == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  const auto& result = value.heuristics[static_cast<size_t>(index)];
  *workspace_bytes = result.workspaceSize;
  *waves_count = result.wavesCount;
  return static_cast<int>(result.state);
}

extern "C" int seu_acblaslt_matmul_int8(
    const int8_t* row_major_weight,
    const int8_t* input,
    int32_t* output,
    int heuristic_index,
    void* workspace,
    size_t workspace_bytes,
    void* stream_handle) {
  if (row_major_weight == nullptr || input == nullptr || output == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  std::lock_guard<std::mutex> lock(state_mutex());
  const Int8MatmulState& value = state();
  if (heuristic_index < 0 ||
      static_cast<size_t>(heuristic_index) >= value.heuristics.size() ||
      workspace_bytes < value.heuristics[static_cast<size_t>(heuristic_index)].workspaceSize) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  const int32_t alpha = 1;
  const int32_t beta = 0;
  const auto& algorithm = value.heuristics[static_cast<size_t>(heuristic_index)].algo;
  return static_cast<int>(acblasLtMatmul(
      value.handle,
      value.operation,
      &alpha,
      row_major_weight,
      value.weight_layout,
      input,
      value.input_layout,
      &beta,
      output,
      value.output_layout,
      output,
      value.output_layout,
      &algorithm,
      workspace,
      workspace_bytes,
      reinterpret_cast<hggcStream_t>(stream_handle)));
}

extern "C" void seu_acblaslt_destroy_int8() {
  std::lock_guard<std::mutex> lock(state_mutex());
  destroy_state(state());
}
