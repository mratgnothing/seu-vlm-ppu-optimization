#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/build/qwen35_bf16_gemv"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
DEVICE="${DEVICE:-0}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found. Run ./build.sh first." >&2
  exit 1
fi

run_shape() {
  local n="$1"
  local k="$2"
  echo "Running Qwen3.5 BF16 GEMV shape N=${n}, K=${k}"
  "${BINARY}" \
    --n "${n}" \
    --k "${k}" \
    --warmup "${WARMUP}" \
    --iters "${ITERATIONS}" \
    --device "${DEVICE}"
}

run_shape 6144 2048
run_shape 2048 6144
run_shape 2048 2048
