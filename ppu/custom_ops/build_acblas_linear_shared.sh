#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/build}"
mkdir -p "${OUTPUT_DIR}"

"${PPU_SDK}/bin/clang++" \
  -O3 -std=c++17 -fPIC -shared \
  "${SCRIPT_DIR}/acblas_linear_wrapper.cpp" \
  -I"${PPU_SDK}/include" \
  -L"${PPU_SDK}/lib" \
  -Wl,-rpath,"${PPU_SDK}/lib" \
  -lacblas -lhggcrt1 \
  -o "${OUTPUT_DIR}/libseu_acblas_linear.so"

echo "Built ${OUTPUT_DIR}/libseu_acblas_linear.so"
