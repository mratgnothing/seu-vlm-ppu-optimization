#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_submission.sh MODEL_PATH DATASET_PATH [OUTPUT_JSON]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="$1"
DATASET_PATH="$2"
OUTPUT_JSON="${3:-result_submission.json}"

[[ -d "${MODEL_PATH}" ]] || { echo "Model directory not found: ${MODEL_PATH}" >&2; exit 1; }
[[ -f "${DATASET_PATH}" ]] || { echo "Dataset not found: ${DATASET_PATH}" >&2; exit 1; }

source "${SCRIPT_DIR}/activate_ppu_env.sh"
source "${SCRIPT_DIR}/activate_ppu_profile.sh" performance

EXTRA_ARGS=()
if [[ -n "${SEU_NUM_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--num-samples "${SEU_NUM_SAMPLES}")
fi
if [[ -n "${SEU_WARMUP_SAMPLES:-}" ]]; then
  EXTRA_ARGS+=(--warmup-samples "${SEU_WARMUP_SAMPLES}")
fi

python "${REPO_ROOT}/benchmark_public.py" \
  --dataset-path "${DATASET_PATH}" \
  --model-path "${MODEL_PATH}" \
  --backend transformers \
  --output "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
