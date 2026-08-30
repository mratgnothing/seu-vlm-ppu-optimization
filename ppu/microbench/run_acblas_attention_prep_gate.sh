#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/mnt/workspace/seu/acblas-extension-work-20260827}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/seu/envs/seu-vlm-ppu-20260826/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/seu/Qwen3.5-2B}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/seu/datasets/mmbench/mmbench_dev_cn.tsv}"
REFERENCE_CUSTOM_OPS="${REFERENCE_CUSTOM_OPS:-/mnt/workspace/seu/seu-vlm-ppu-optimization-5070ti/ppu/custom_ops}"
GDN_LIBRARY="${GDN_LIBRARY:-${WORK_ROOT}/build/gate-prep/libseu_ppu_gdn.so}"
GDN_LIBRARY_DIR="${GDN_LIBRARY_DIR:-${GDN_LIBRARY%/*}}"
ACBLAS_BUILD_DIR="${ACBLAS_BUILD_DIR:-${WORK_ROOT}/ppu/custom_ops/build/acblas_linear_extension}"
ACBLAS_PACKED_MLP_BUILD_DIR="${ACBLAS_PACKED_MLP_BUILD_DIR:-${WORK_ROOT}/ppu/custom_ops/build/acblas_packed_mlp_extension}"
ACBLAS_ATTENTION_PREP_BUILD_DIR="${ACBLAS_ATTENTION_PREP_BUILD_DIR:-${WORK_ROOT}/ppu/custom_ops/build/acblas_attention_prep_extension}"
MODE="${MODE:-fixed}"
OUTPUT_PATH="${OUTPUT_PATH:-${WORK_ROOT}/results/acblas-attention-prep-ab128-20260828.json}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

export PYTHONPATH="${WORK_ROOT}/ppu/custom_ops:${REFERENCE_CUSTOM_OPS}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${GDN_LIBRARY_DIR}:/usr/local/PPU_SDK/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

COMMON_ARGS=(
  --repo-root "${WORK_ROOT}"
  --model-path "${MODEL_PATH}"
  --dataset-path "${DATASET_PATH}"
  --gdn-library "${GDN_LIBRARY}"
  --output "${OUTPUT_PATH}"
  --projection-backend acblas-grouped
  --acblas-build-dir "${ACBLAS_BUILD_DIR}"
  --acblas-packed-mlp-build-dir "${ACBLAS_PACKED_MLP_BUILD_DIR}"
  --acblas-attention-prep-build-dir "${ACBLAS_ATTENTION_PREP_BUILD_DIR}"
  --acblas-attention-prep-ab
  --require-speedup
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

case "${MODE}" in
  fixed)
    REPEATS="${REPEATS:-8}"
    exec "${PYTHON_BIN}" "${WORK_ROOT}/scripts/benchmark_ppu_packed_gdn_ab.py" \
      "${COMMON_ARGS[@]}" \
      --repeats "${REPEATS}" \
      --force-max-new-tokens
    ;;
  multisample)
    NUM_SAMPLES="${NUM_SAMPLES:-20}"
    SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
    PAIR_LOG_PATH="${PAIR_LOG_PATH:-${OUTPUT_PATH%.json}-pairs.jsonl}"
    PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
    exec "${PYTHON_BIN}" "${WORK_ROOT}/scripts/benchmark_ppu_packed_gdn_multisample_ab.py" \
      "${COMMON_ARGS[@]}" \
      --num-samples "${NUM_SAMPLES}" \
      --sample-offset "${SAMPLE_OFFSET}" \
      --pair-log "${PAIR_LOG_PATH}" \
      --progress-every "${PROGRESS_EVERY}"
    ;;
  *)
    echo "unsupported MODE: ${MODE}; expected fixed or multisample" >&2
    exit 2
    ;;
esac
