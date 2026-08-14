#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHINESE_DIR="${REPO_ROOT}/conprehension/chinese"
CUDA_DIR="${CHINESE_DIR}/cuda-programming-guide-zh"
PPU_HTML="${CHINESE_DIR}/ppu-sdk-quick-start-zh.html"

CUDA_REPO="https://github.com/bearneck/cuda-programming-guide-zh.git"
PPU_URL="https://help.aliyun.com/zh/document_detail/3030340.html"

for command_name in git curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "error: missing required command: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${CHINESE_DIR}"

if [[ -d "${CUDA_DIR}/.git" ]]; then
  echo "Updating CUDA Chinese guide..."
  git -C "${CUDA_DIR}" pull --ff-only
elif [[ -e "${CUDA_DIR}" ]]; then
  if [[ -n "$(find "${CUDA_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "error: ${CUDA_DIR} exists but is not a Git checkout" >&2
    echo "Move it aside, then rerun this script." >&2
    exit 1
  fi
  rmdir "${CUDA_DIR}"
  git clone --depth 1 "${CUDA_REPO}" "${CUDA_DIR}"
else
  git clone --depth 1 "${CUDA_REPO}" "${CUDA_DIR}"
fi

echo "Downloading the official PPU SDK quick-start page..."
PPU_TMP="${PPU_HTML}.tmp"
curl --fail --location --retry 3 --retry-delay 2 \
  --output "${PPU_TMP}" "${PPU_URL}"

if [[ ! -s "${PPU_TMP}" ]]; then
  echo "error: downloaded PPU page is empty" >&2
  exit 1
fi

mv "${PPU_TMP}" "${PPU_HTML}"

echo
echo "Chinese tutorials are ready:"
echo "  CUDA: ${CUDA_DIR}/README.md"
echo "  PPU:  ${PPU_HTML}"
echo
echo "These third-party copies are intentionally ignored by Git."
