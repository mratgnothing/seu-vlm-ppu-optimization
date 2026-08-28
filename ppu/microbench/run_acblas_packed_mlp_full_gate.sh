#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${1:-/mnt/workspace/seu/acblas-extension-work-20260827}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/workspace/seu/envs/seu-vlm-ppu-20260826/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/seu/Qwen3.5-2B}"
DATASET_PATH="${DATASET_PATH:-/mnt/workspace/seu/datasets/mmbench/mmbench_dev_cn.tsv}"
REFERENCE_CUSTOM_OPS="${REFERENCE_CUSTOM_OPS:-/mnt/workspace/seu/seu-vlm-ppu-optimization-5070ti/ppu/custom_ops}"

export PYTHONPATH="${WORK_ROOT}/ppu/custom_ops:${REFERENCE_CUSTOM_OPS}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON_BIN}" "${WORK_ROOT}/scripts/benchmark_ppu_packed_gdn_multisample_ab.py" \
  --repo-root "${WORK_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --gdn-library "${WORK_ROOT}/build/gate-prep/libseu_ppu_gdn.so" \
  --output "${WORK_ROOT}/results/acblas-packed-mlp-cn-full4029-20260828.json" \
  --pair-log "${WORK_ROOT}/results/acblas-packed-mlp-cn-full4029-pairs-20260828.jsonl" \
  --progress-every 100 \
  --projection-backend acblas-grouped \
  --acblas-build-dir "${WORK_ROOT}/ppu/custom_ops/build/acblas_linear_extension" \
  --acblas-packed-mlp-ab \
  --acblas-packed-mlp-build-dir "${WORK_ROOT}/ppu/custom_ops/build/acblas_packed_mlp_extension" \
  --num-samples 4029 \
  --max-new-tokens 64
