#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
HGGC_RUNTIME_LIBRARY="${HGGC_RUNTIME_LIBRARY:-hggcrt1}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/build}"
OUTPUT="${OUTPUT_DIR}/libseu_ppu_gdn.so"

mkdir -p "${OUTPUT_DIR}"
"${PPU_SDK}/bin/clang++" \
  -O3 \
  -std=c++17 \
  -fPIC \
  -shared \
  -x hggc \
  "${SCRIPT_DIR}/gdn_recurrent_ppu.hg" \
  -I"${PPU_SDK}/include" \
  -L"${PPU_SDK}/lib" \
  -Wl,-rpath,"${PPU_SDK}/lib" \
  -l"${HGGC_RUNTIME_LIBRARY}" \
  -o "${OUTPUT}"

echo "Built ${OUTPUT}"
