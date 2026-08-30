#include <acblasLt.h>

#include <cstddef>
#include <cstdio>

namespace {

struct TypeCase {
  const char* name;
  hggcDataType_t weight_type;
  hggcDataType_t input_type;
  hggcDataType_t output_type;
  acblasComputeType_t compute_type;
  hggcDataType_t scale_type;
};

int probe(const TypeCase& type_case, int output_features, int input_features) {
  acblasLtHandle_t handle = nullptr;
  acblasLtMatmulDesc_t operation = nullptr;
  acblasLtMatrixLayout_t weight_layout = nullptr;
  acblasLtMatrixLayout_t input_layout = nullptr;
  acblasLtMatrixLayout_t output_layout = nullptr;
  acblasLtMatmulPreference_t preference = nullptr;
  acblasLtMatmulHeuristicResult_t results[16]{};
  int count = 0;
  acblasStatus_t status = acblasLtCreate(&handle);
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulDescCreate(
        &operation, type_case.compute_type, type_case.scale_type);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    const acblasOperation_t transpose_weight = ACBLAS_OP_T;
    status = acblasLtMatmulDescSetAttribute(
        operation,
        ACBLASLT_MATMUL_DESC_TRANSA,
        &transpose_weight,
        sizeof(transpose_weight));
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &weight_layout,
        type_case.weight_type,
        input_features,
        output_features,
        input_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &input_layout,
        type_case.input_type,
        input_features,
        1,
        input_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatrixLayoutCreate(
        &output_layout,
        type_case.output_type,
        output_features,
        1,
        output_features);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulPreferenceCreate(&preference);
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    const size_t workspace_bytes = 64ULL * 1024ULL * 1024ULL;
    status = acblasLtMatmulPreferenceSetAttribute(
        preference,
        ACBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_bytes,
        sizeof(workspace_bytes));
  }
  if (status == ACBLAS_STATUS_SUCCESS) {
    status = acblasLtMatmulAlgoGetHeuristic(
        handle,
        operation,
        weight_layout,
        input_layout,
        output_layout,
        output_layout,
        preference,
        16,
        results,
        &count);
  }

  std::printf(
      "RESULT name=%s n=%d k=%d status=%d algorithms=%d\n",
      type_case.name,
      output_features,
      input_features,
      static_cast<int>(status),
      count);

  if (preference != nullptr) acblasLtMatmulPreferenceDestroy(preference);
  if (output_layout != nullptr) acblasLtMatrixLayoutDestroy(output_layout);
  if (input_layout != nullptr) acblasLtMatrixLayoutDestroy(input_layout);
  if (weight_layout != nullptr) acblasLtMatrixLayoutDestroy(weight_layout);
  if (operation != nullptr) acblasLtMatmulDescDestroy(operation);
  if (handle != nullptr) acblasLtDestroy(handle);
  return status == ACBLAS_STATUS_SUCCESS ? 0 : 1;
}

}  // namespace

int main() {
  const TypeCase cases[] = {
      {"bf16_bf16_bf16", HGGC_R_16BF, HGGC_R_16BF, HGGC_R_16BF,
       ACBLAS_COMPUTE_32F, HGGC_R_32F},
      {"int8_bf16_bf16", HGGC_R_8I, HGGC_R_16BF, HGGC_R_16BF,
       ACBLAS_COMPUTE_32F, HGGC_R_32F},
      {"fp8e4m3_bf16_bf16", HGGC_R_8F_E4M3, HGGC_R_16BF, HGGC_R_16BF,
       ACBLAS_COMPUTE_32F, HGGC_R_32F},
      {"int8_int8_int32", HGGC_R_8I, HGGC_R_8I, HGGC_R_32I,
       ACBLAS_COMPUTE_32I, HGGC_R_32I},
      {"fp8e4m3_fp8e4m3_bf16", HGGC_R_8F_E4M3, HGGC_R_8F_E4M3,
       HGGC_R_16BF, ACBLAS_COMPUTE_32F, HGGC_R_32F},
  };
  int failures = 0;
  for (const auto& type_case : cases) {
    failures += probe(type_case, 12288, 2048);
  }
  return failures == 0 ? 0 : 2;
}
