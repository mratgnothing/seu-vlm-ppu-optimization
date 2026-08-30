#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-/mnt/workspace/seu/acblas-extension-work-20260827}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/seu/envs/seu-vlm-ppu-20260826/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/seu/Qwen3.5-2B}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/seu/datasets/mmbench/mmbench_dev_cn.tsv}"
REFERENCE_CUSTOM_OPS="${REFERENCE_CUSTOM_OPS:-/mnt/workspace/seu/seu-vlm-ppu-optimization-5070ti/ppu/custom_ops}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORK_ROOT}/results/acblas-packed-mlp-cn-full4029-20260828.json}"
PAIR_LOG_PATH="${PAIR_LOG_PATH:-${WORK_ROOT}/results/acblas-packed-mlp-cn-full4029-pairs-20260828.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-4029}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
AB_TARGET="${AB_TARGET:-acblas-packed-mlp}"
ACBLAS_PACKED_MLP_BUILD_DIR="${ACBLAS_PACKED_MLP_BUILD_DIR:-${WORK_ROOT}/ppu/custom_ops/build/acblas_packed_mlp_extension}"
ACBLAS_ATTENTION_PREP_BUILD_DIR="${ACBLAS_ATTENTION_PREP_BUILD_DIR:-${WORK_ROOT}/ppu/custom_ops/build/acblas_attention_prep_extension}"

export PYTHONPATH="${WORK_ROOT}/ppu/custom_ops:${REFERENCE_CUSTOM_OPS}${PYTHONPATH:+:${PYTHONPATH}}"

case "${AB_TARGET}" in
  acblas-packed-mlp)
    TARGET_ARGS=(
      --acblas-packed-mlp-ab
      --acblas-packed-mlp-build-dir "${ACBLAS_PACKED_MLP_BUILD_DIR}"
    )
    ;;
  acblas-attention-prep)
    TARGET_ARGS=(
      --acblas-attention-prep-ab
      --acblas-packed-mlp-build-dir "${ACBLAS_PACKED_MLP_BUILD_DIR}"
      --acblas-attention-prep-build-dir "${ACBLAS_ATTENTION_PREP_BUILD_DIR}"
    )
    ;;
  *)
    echo "unsupported AB_TARGET: ${AB_TARGET}" >&2
    exit 2
    ;;
esac

exec "${PYTHON_BIN}" "${WORK_ROOT}/scripts/benchmark_ppu_packed_gdn_multisample_ab.py" \
  --repo-root "${WORK_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --gdn-library "${WORK_ROOT}/build/gate-prep/libseu_ppu_gdn.so" \
  --output "${OUTPUT_PATH}" \
  --pair-log "${PAIR_LOG_PATH}" \
  --progress-every 100 \
  --require-speedup \
  --projection-backend acblas-grouped \
  --acblas-build-dir "${WORK_ROOT}/ppu/custom_ops/build/acblas_linear_extension" \
  "${TARGET_ARGS[@]}" \
  --num-samples "${NUM_SAMPLES}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"
