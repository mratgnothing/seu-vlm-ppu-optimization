#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_SDK_ROOT="${PPU_SDK_ROOT:-/usr/local/PPU_SDK}"
COMPILER="${PPU_SDK_ROOT}/bin/clang++"
HGGC_RUNTIME_LIBRARY="${HGGC_RUNTIME_LIBRARY:-hggcrt1}"
OUTPUT_DIR="${SCRIPT_DIR}/build"

if [[ ! -x "${COMPILER}" ]]; then
  echo "PPU compiler not found: ${COMPILER}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
"${COMPILER}" \
  -O3 \
  -std=c++17 \
  -x hggc \
  "${SCRIPT_DIR}/qwen35_gdn_recurrent.hg" \
  -I"${PPU_SDK_ROOT}/include" \
  -L"${PPU_SDK_ROOT}/lib" \
  -Wl,-rpath,"${PPU_SDK_ROOT}/lib" \
  -l"${HGGC_RUNTIME_LIBRARY}" \
  -o "${OUTPUT_DIR}/qwen35_gdn_recurrent"

echo "Built ${OUTPUT_DIR}/qwen35_gdn_recurrent"
