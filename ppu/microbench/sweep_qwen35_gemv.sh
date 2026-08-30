#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/build/qwen35_bf16_gemv"
ITERATIONS="${ITERATIONS:-100}"
WARMUP="${WARMUP:-10}"
DEVICE="${DEVICE:-0}"
THREAD_COUNTS="${THREAD_COUNTS:-64 128 256 512}"
MATRIX_COPIES="${MATRIX_COPIES:-1}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found. Run ./build.sh first." >&2
  exit 1
fi

run_case() {
  local kernel="$1"
  local threads="$2"
  local n="$3"
  local k="$4"
  "${BINARY}" \
    --kernel "${kernel}" \
    --threads "${threads}" \
    --matrix-copies "${MATRIX_COPIES}" \
    --n "${n}" \
    --k "${k}" \
    --warmup "${WARMUP}" \
    --iters "${ITERATIONS}" \
    --device "${DEVICE}"
}

for shape in "6144 2048" "2048 6144" "2048 2048"; do
  read -r n k <<< "${shape}"
  echo "Reference N=${n} K=${k} threads=256"
  run_case reference 256 "${n}" "${k}"
  for kernel in warp warp_vec2; do
    for threads in ${THREAD_COUNTS}; do
      echo "Candidate kernel=${kernel} N=${n} K=${k} threads=${threads}"
      run_case "${kernel}" "${threads}" "${n}" "${k}"
    done
  done
done
