#include <acblasLt.h>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct MatmulState {
  acblasLtHandle_t handle = nullptr;
  acblasLtMatmulDesc_t operation = nullptr;
  acblasLtMatrixLayout_t input_layout = nullptr;
  acblasLtMatrixLayout_t weight_layout = nullptr;
  acblasLtMatrixLayout_t output_layout = nullptr;
  acblasLtMatmulPreference_t preference = nullptr;
  std::vector<acblasLtMatmulHeuristicResult_t> heuristics;
  int output_features = 0;
  int input_features = 0;
  size_t max_workspace_bytes = 0;
};

MatmulState& state() {
  static MatmulState value;
  return value;
}

std::mutex& state_mutex() {
  static std::mutex value;
  return value;
}

void destroy_state(MatmulState& value) {
  if (value.preference != nullptr) {
    acblasLtMatmulPreferenceDestroy(value.preference);
  }
  if (value.output_layout != nullptr) {
    acblasLtMatrixLayoutDestroy(value.output_layout);
  }
  if (value.weight_layout != nullptr) {
    acblasLtMatrixLayoutDestroy(value.weight_layout);
  }
  if (value.input_layout != nullptr) {
    acblasLtMatrixLayoutDestroy(value.input_layout);
  }
  if (value.operation != nullptr) {
    acblasLtMatmulDescDestroy(value.operation);
  }
  if (value.handle != nullptr) {
    acblasLtDestroy(value.handle);
  }
  value = MatmulState{};
}

}  // namespace

extern "C" int seu_acblaslt_prepare_bf16(
    int output_features,
    int input_features,
    size_t max_workspace_bytes,
    int requested_algorithms,
    int* returned_algorithms) {
  if (
      output_features <= 0 || input_features <= 0 ||
      requested_algorithms <= 0 || returned_algorithms == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }

  std::lock_guard<std::mutex> lock(state_mutex());
  MatmulState& value = state();
  destroy_state(value);
  value.output_features = output_features;
  value.input_features = input_features;
  value.max_workspace_bytes = max_workspace_bytes;

  acblasStatus_t status = acblasLtCreate(&value.handle);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasLtMatmulDescCreate(
      &value.operation, ACBLAS_COMPUTE_32F, HGGC_R_32F);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  const acblasOperation_t transpose_weight = ACBLAS_OP_T;
  status = acblasLtMatmulDescSetAttribute(
      value.operation,
      ACBLASLT_MATMUL_DESC_TRANSA,
      &transpose_weight,
      sizeof(transpose_weight));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  // Row-major weight [N,K] has identical bytes to a column-major [K,N]
  // matrix. Use native column-major layouts and compute weight^T [N,K]
  // times input [K,1], avoiding both a weight copy and the PPU acBLASLt
  // unsupported all-row-major layout combination.
  status = acblasLtMatrixLayoutCreate(
      &value.weight_layout,
      HGGC_R_16BF,
      input_features,
      output_features,
      input_features);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasLtMatrixLayoutCreate(
      &value.input_layout, HGGC_R_16BF, input_features, 1, input_features);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasLtMatrixLayoutCreate(
      &value.output_layout, HGGC_R_16BF, output_features, 1, output_features);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  status = acblasLtMatmulPreferenceCreate(&value.preference);
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  status = acblasLtMatmulPreferenceSetAttribute(
      value.preference,
      ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_bytes,
      sizeof(max_workspace_bytes));
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);

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
  if (status != ACBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  value.heuristics.resize(static_cast<size_t>(count));
  *returned_algorithms = count;
  return static_cast<int>(ACBLAS_STATUS_SUCCESS);
}

extern "C" int seu_acblaslt_heuristic_info(
    int index, size_t* workspace_bytes, float* waves_count) {
  std::lock_guard<std::mutex> lock(state_mutex());
  const MatmulState& value = state();
  if (
      index < 0 || static_cast<size_t>(index) >= value.heuristics.size() ||
      workspace_bytes == nullptr || waves_count == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  const auto& result = value.heuristics[static_cast<size_t>(index)];
  *workspace_bytes = result.workspaceSize;
  *waves_count = result.wavesCount;
  return static_cast<int>(result.state);
}

extern "C" int seu_acblaslt_matmul_bf16(
    const uint16_t* row_major_weight,
    const uint16_t* input,
    uint16_t* output,
    int heuristic_index,
    void* workspace,
    size_t workspace_bytes,
    void* stream_handle) {
  if (row_major_weight == nullptr || input == nullptr || output == nullptr) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  std::lock_guard<std::mutex> lock(state_mutex());
  const MatmulState& value = state();
  if (
      heuristic_index < 0 ||
      static_cast<size_t>(heuristic_index) >= value.heuristics.size() ||
      workspace_bytes <
          value.heuristics[static_cast<size_t>(heuristic_index)].workspaceSize) {
    return static_cast<int>(ACBLAS_STATUS_INVALID_VALUE);
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  const auto& algorithm =
      value.heuristics[static_cast<size_t>(heuristic_index)].algo;
  const acblasStatus_t status = acblasLtMatmul(
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
      reinterpret_cast<hggcStream_t>(stream_handle));
  return static_cast<int>(status);
}

extern "C" void seu_acblaslt_destroy() {
  std::lock_guard<std::mutex> lock(state_mutex());
  destroy_state(state());
}
