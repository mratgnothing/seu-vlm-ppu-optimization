#!/usr/bin/env bash
# Source this file after scripts/bootstrap_ppu_env.sh has completed.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source scripts/activate_ppu_env.sh" >&2
  exit 2
fi

_seu_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_seu_repo_root="$(cd "${_seu_script_dir}/.." && pwd)"
export SEU_PPU_VENV_DIR="${SEU_PPU_VENV_DIR:-${HOME}/.cache/seu-vlm-ppu/venv}"
export PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
export PPU_HOME="${PPU_HOME:-${PPU_SDK}}"

if [[ ! -f "${SEU_PPU_VENV_DIR}/bin/activate" ]]; then
  echo "PPU virtual environment not found: ${SEU_PPU_VENV_DIR}" >&2
  echo "Run: bash scripts/bootstrap_ppu_env.sh" >&2
  unset _seu_script_dir _seu_repo_root
  return 1
fi

# shellcheck disable=SC1090
source "${SEU_PPU_VENV_DIR}/bin/activate"
export PATH="${PPU_SDK}/bin:${PPU_SDK}/ppu-smi/bin:${PATH}"
export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${_seu_repo_root}:${PYTHONPATH:-}"

unset _seu_script_dir _seu_repo_root
