#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VLLM_SOURCE="${VLLM_SOURCE:-/opt/vllm}"
MODEL_PATH=""
DATASET_PATH=""
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/ppu-first-validation}"
RUN_MICROBENCH=0
RUN_DEVICE_SMOKE=0
RUN_MODEL_LOAD=0
RUN_SINGLE_SAMPLE=0
VERIFY_MODEL_HASH=0
DEVICE="${DEVICE:-0}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"

# Some PPU PyTorch GEMM paths use runtime compilation even when model loading
# and small eager operations already work. The RTC layer aborts the process if
# neither variable points to the installed SDK, so discover the standard image
# location without overriding an organizer-provided value.
if [[ -z "${PPU_SDK:-}" && -z "${PPU_HOME:-}" && -d /usr/local/PPU_SDK ]]; then
  export PPU_SDK=/usr/local/PPU_SDK
fi
if [[ -n "${PPU_SDK:-}" && -d "${PPU_SDK}/bin" ]]; then
  export PATH="${PPU_SDK}/bin:${PATH}"
fi
if [[ -n "${PPU_SDK:-}" && -d "${PPU_SDK}/ppu-smi/bin" ]]; then
  export PATH="${PPU_SDK}/ppu-smi/bin:${PATH}"
fi

usage() {
  cat <<'EOF'
Usage: scripts/run_ppu_first_validation.sh [options]

Runs a read-only PPU runtime preflight. The HGGC microbenchmark is opt-in so
this script cannot accidentally execute competition code on a shared node.

Options:
  --model-path PATH       Optional local Qwen3.5-2B directory.
  --vllm-source PATH      PPU-vLLM source tree (default: /opt/vllm).
  --output-dir PATH       Artifact directory.
  --dataset-path PATH     Optional public MMBench TSV for a one-sample smoke.
  --run-device-smoke      Run a tiny BF16 PyTorch operation on visible PPU.
  --verify-model-hash     Hash the locked model weight file (~4.6 GB read).
  --run-microbench        Build, smoke-test, and run the three BF16 GEMV shapes.
  --run-model-load        Load the full model and reject CPU/meta/disk offload.
  --run-single-sample     Run one real public multimodal sample (never dummy).
  --device INDEX          PPU device index for the microbenchmark (default: 0).
  --warmup COUNT          Suite warmup iterations (default: 10).
  --iterations COUNT      Suite measured iterations (default: 100).
  -h, --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --vllm-source)
      VLLM_SOURCE="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-microbench)
      RUN_MICROBENCH=1
      shift
      ;;
    --run-device-smoke)
      RUN_DEVICE_SMOKE=1
      shift
      ;;
    --verify-model-hash)
      VERIFY_MODEL_HASH=1
      shift
      ;;
    --run-model-load)
      RUN_MODEL_LOAD=1
      shift
      ;;
    --run-single-sample)
      RUN_SINGLE_SAMPLE=1
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --warmup)
      WARMUP="$2"
      shift 2
      ;;
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${RUN_MODEL_LOAD}" -eq 1 && -z "${MODEL_PATH}" ]]; then
  echo "--run-model-load requires --model-path" >&2
  exit 2
fi
if [[ "${RUN_SINGLE_SAMPLE}" -eq 1 ]]; then
  if [[ -z "${MODEL_PATH}" || -z "${DATASET_PATH}" ]]; then
    echo "--run-single-sample requires --model-path and --dataset-path" >&2
    exit 2
  fi
fi

mkdir -p "${OUTPUT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=${PYTHON}"
  echo "ppu_sdk=${PPU_SDK:-unset}"
  echo "ppu_home=${PPU_HOME:-unset}"
  echo "vllm_source=${VLLM_SOURCE}"
  echo "model_path=${MODEL_PATH:-not_requested}"
  echo "dataset_path=${DATASET_PATH:-not_requested}"
  echo "run_device_smoke=${RUN_DEVICE_SMOKE}"
  echo "verify_model_hash=${VERIFY_MODEL_HASH}"
  echo "run_microbench=${RUN_MICROBENCH}"
  echo "run_model_load=${RUN_MODEL_LOAD}"
  echo "run_single_sample=${RUN_SINGLE_SAMPLE}"
  echo "device=${DEVICE}"
  echo "warmup=${WARMUP}"
  echo "iterations=${ITERATIONS}"
} > "${OUTPUT_DIR}/manifest.txt"

{
  echo "# /etc/os-release"
  cat /etc/os-release 2>&1 || true
  echo
  echo "# uname -a"
  uname -a 2>&1 || true
  echo
  echo "# lscpu"
  lscpu 2>&1 || true
  echo
  echo "# free -h"
  free -h 2>&1 || true
  echo
  echo "# df -h"
  df -h 2>&1 || true
} > "${OUTPUT_DIR}/system-info.txt"

"${PYTHON}" -m pip list --format=json \
  > "${OUTPUT_DIR}/python-packages.json" 2>&1 || true

preflight_command=(
  "${PYTHON}"
  "${SCRIPT_DIR}/check_ppu_runtime.py"
  --vllm-source "${VLLM_SOURCE}"
  --output "${OUTPUT_DIR}/runtime.json"
  --markdown-output "${OUTPUT_DIR}/runtime-summary.md"
)
if [[ -n "${MODEL_PATH}" ]]; then
  preflight_command+=(--model-path "${MODEL_PATH}")
fi
if [[ "${RUN_DEVICE_SMOKE}" -eq 1 ]]; then
  preflight_command+=(--run-device-smoke)
fi
if [[ "${VERIFY_MODEL_HASH}" -eq 1 ]]; then
  preflight_command+=(--verify-model-hash)
fi

"${preflight_command[@]}" | tee "${OUTPUT_DIR}/runtime.stdout.log"

if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi > "${OUTPUT_DIR}/ppu-smi.txt" 2>&1 || true
fi

if [[ "${RUN_MICROBENCH}" -eq 1 ]]; then
  MICROBENCH_DIR="${REPO_ROOT}/ppu/microbench"
  "${MICROBENCH_DIR}/build.sh" 2>&1 | tee "${OUTPUT_DIR}/microbench-build.log"

  "${MICROBENCH_DIR}/build/qwen35_bf16_gemv" \
    --n 6144 \
    --k 2048 \
    --warmup 0 \
    --iters 1 \
    --device "${DEVICE}" \
    2>&1 | tee "${OUTPUT_DIR}/microbench-smoke.log"

  DEVICE="${DEVICE}" \
  WARMUP="${WARMUP}" \
  ITERATIONS="${ITERATIONS}" \
    "${MICROBENCH_DIR}/run_qwen35_suite.sh" \
    2>&1 | tee "${OUTPUT_DIR}/microbench-suite.log"
else
  echo "Microbenchmark skipped; pass --run-microbench only on an approved isolated PPU node."
fi

if [[ "${RUN_MODEL_LOAD}" -eq 1 ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/smoke_model_load.py" \
    --model-path "${MODEL_PATH}" \
    --output "${OUTPUT_DIR}/model-load.json" \
    --require-accelerator \
    2>&1 | tee "${OUTPUT_DIR}/model-load.stdout.log"
fi

if [[ "${RUN_SINGLE_SAMPLE}" -eq 1 ]]; then
  "${PYTHON}" "${REPO_ROOT}/benchmark_public.py" \
    --dataset-path "${DATASET_PATH}" \
    --model-path "${MODEL_PATH}" \
    --output "${OUTPUT_DIR}/single-sample.json" \
    --num-samples 1 \
    --warmup-samples 0 \
    --backend transformers \
    --device auto \
    2>&1 | tee "${OUTPUT_DIR}/single-sample.stdout.log"
fi

echo "PPU first validation completed. Artifacts: ${OUTPUT_DIR}"
