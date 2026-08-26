#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="${SCRIPT_DIR}/build/qwen35_bf16_gemv"
PYTHON="${PYTHON:-python3}"
REPEATS="${REPEATS:-3}"
ITERATIONS="${ITERATIONS:-200}"
WARMUP="${WARMUP:-32}"
DEVICE="${DEVICE:-0}"
MATRIX_COPIES="${MATRIX_COPIES:-16}"
CANDIDATE_THREADS="${CANDIDATE_THREADS:-64 128 256}"
RUN_TORCH_BASELINE="${RUN_TORCH_BASELINE:-1}"

if [[ ! -x "${BINARY}" ]]; then
  echo "Binary not found. Run ./build.sh first." >&2
  exit 1
fi
if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "REPEATS must be a positive integer." >&2
  exit 2
fi

run_hggc() {
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

for repeat in $(seq 1 "${REPEATS}"); do
  for shape in "6144 2048" "2048 6144" "2048 2048"; do
    read -r n k <<< "${shape}"
    echo "RUN repeat=${repeat} source=hggc kernel=reference n=${n} k=${k} threads=256 copies=${MATRIX_COPIES}"
    run_hggc reference 256 "${n}" "${k}"

    for threads in ${CANDIDATE_THREADS}; do
      echo "RUN repeat=${repeat} source=hggc kernel=warp_vec2 n=${n} k=${k} threads=${threads} copies=${MATRIX_COPIES}"
      run_hggc warp_vec2 "${threads}" "${n}" "${k}"
    done

    if [[ "${RUN_TORCH_BASELINE}" -eq 1 ]]; then
      echo "RUN repeat=${repeat} source=torch kernel=torch_mv_bf16 n=${n} k=${k} copies=${MATRIX_COPIES}"
      "${PYTHON}" "${SCRIPT_DIR}/torch_gemv_baseline.py" \
        --n "${n}" \
        --k "${k}" \
        --matrix-copies "${MATRIX_COPIES}" \
        --warmup "${WARMUP}" \
        --iters "${ITERATIONS}" \
        --device "${DEVICE}"
    fi
  done
done
