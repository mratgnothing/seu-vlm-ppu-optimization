#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VLLM_SOURCE="${VLLM_SOURCE:-/opt/vllm}"
MODEL_PATH=""
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts/ppu-first-validation}"
RUN_MICROBENCH=0
DEVICE="${DEVICE:-0}"
WARMUP="${WARMUP:-10}"
ITERATIONS="${ITERATIONS:-100}"

usage() {
  cat <<'EOF'
Usage: scripts/run_ppu_first_validation.sh [options]

Runs a read-only PPU runtime preflight. The HGGC microbenchmark is opt-in so
this script cannot accidentally execute competition code on a shared node.

Options:
  --model-path PATH       Optional local Qwen3.5-2B directory.
  --vllm-source PATH      PPU-vLLM source tree (default: /opt/vllm).
  --output-dir PATH       Artifact directory.
  --run-microbench        Build, smoke-test, and run the three BF16 GEMV shapes.
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-microbench)
      RUN_MICROBENCH=1
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

mkdir -p "${OUTPUT_DIR}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=${PYTHON}"
  echo "vllm_source=${VLLM_SOURCE}"
  echo "model_path=${MODEL_PATH:-not_requested}"
  echo "run_microbench=${RUN_MICROBENCH}"
  echo "device=${DEVICE}"
  echo "warmup=${WARMUP}"
  echo "iterations=${ITERATIONS}"
} > "${OUTPUT_DIR}/manifest.txt"

preflight_command=(
  "${PYTHON}"
  "${SCRIPT_DIR}/check_ppu_runtime.py"
  --vllm-source "${VLLM_SOURCE}"
  --output "${OUTPUT_DIR}/runtime.json"
)
if [[ -n "${MODEL_PATH}" ]]; then
  preflight_command+=(--model-path "${MODEL_PATH}")
fi

"${preflight_command[@]}" | tee "${OUTPUT_DIR}/runtime.stdout.log"

if command -v ppu-smi >/dev/null 2>&1; then
  ppu-smi > "${OUTPUT_DIR}/ppu-smi.txt" 2>&1 || true
fi

if [[ "${RUN_MICROBENCH}" -eq 0 ]]; then
  echo "Preflight complete. Microbenchmark skipped; pass --run-microbench only on an approved isolated PPU node."
  exit 0
fi

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

echo "PPU first validation completed. Artifacts: ${OUTPUT_DIR}"
