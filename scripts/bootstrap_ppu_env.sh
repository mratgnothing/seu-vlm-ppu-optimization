#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
PPU_SDK="${PPU_SDK:-/usr/local/PPU_SDK}"
VENV_DIR="${SEU_PPU_VENV_DIR:-${HOME}/.cache/seu-vlm-ppu/venv}"
WHEELHOUSE=""
INSTALL_DEPS=1
BUILD_EXTENSIONS=1
RUN_SMOKE=1
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/bootstrap_ppu_env.sh [options]

Prepare a disposable official PPU image without modifying its system Python.
The repository must first be cloned from GitHub. The script creates an isolated
venv that can still see the image-provided PPU-patched torch, installs only the
non-Torch dependencies, rebuilds native extensions, and runs short device smoke
tests. It is safe to run again on the same instance.

Options:
  --venv-dir PATH       Virtualenv location (default: ~/.cache/seu-vlm-ppu/venv)
  --python PATH         Base image Python (default: /usr/local/bin/python3)
  --ppu-sdk PATH        PPU SDK root (default: /usr/local/PPU_SDK)
  --wheelhouse PATH     Install dependencies from an offline wheel directory
  --skip-deps           Do not install requirements-ppu.txt
  --skip-build          Do not rebuild the three native extensions
  --skip-smoke          Do not run device and extension smoke tests
  --check-only          Read-only check of the official image and repository
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir) VENV_DIR="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --ppu-sdk) PPU_SDK="$2"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="$2"; shift 2 ;;
    --skip-deps) INSTALL_DEPS=0; shift ;;
    --skip-build) BUILD_EXTENSIONS=0; shift ;;
    --skip-smoke) RUN_SMOKE=0; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || fail "required file not found: $1"; }
require_executable() { [[ -x "$1" ]] || fail "required executable not found: $1"; }

[[ "$(uname -s)" == "Linux" ]] || fail "this script must run inside the Linux PPU image"
require_executable "${PYTHON_BIN}"
require_executable "${PPU_SDK}/bin/clang++"
require_file "${PPU_SDK}/lib/libacblas.so"
require_file "${PPU_SDK}/lib/libhggcrt1.so"
require_file "${REPO_ROOT}/requirements-ppu.txt"
require_file "${REPO_ROOT}/ppu/custom_ops/gdn_recurrent_ppu.hg"

export PPU_SDK
export PPU_HOME="${PPU_HOME:-${PPU_SDK}}"
export PATH="${PPU_SDK}/bin:${PPU_SDK}/ppu-smi/bin:${PATH}"
export LD_LIBRARY_PATH="${PPU_SDK}/lib:${PPU_SDK}/lib64:${LD_LIBRARY_PATH:-}"

echo "Repository : ${REPO_ROOT}"
echo "Git commit : $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "Base Python: ${PYTHON_BIN}"
echo "PPU SDK    : ${PPU_SDK}"
echo "Venv       : ${VENV_DIR}"

BASE_TORCH_INFO="$(${PYTHON_BIN} - <<'PY'
import json
import torch
print(json.dumps({"version": torch.__version__, "file": torch.__file__}, sort_keys=True))
PY
)" || fail "the official image Python cannot import its PPU-patched torch"
echo "Base torch : ${BASE_TORCH_INFO}"

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  echo "Read-only image/repository check passed."
  exit 0
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  mkdir -p "$(dirname "${VENV_DIR}")"
  "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi
VENV_PYTHON="${VENV_DIR}/bin/python"

if [[ "${INSTALL_DEPS}" -eq 1 ]]; then
  PIP_ARGS=(install --upgrade-strategy only-if-needed)
  if [[ -n "${WHEELHOUSE}" ]]; then
    [[ -d "${WHEELHOUSE}" ]] || fail "wheelhouse not found: ${WHEELHOUSE}"
    PIP_ARGS+=(--no-index --find-links "${WHEELHOUSE}")
  fi
  "${VENV_PYTHON}" -m pip "${PIP_ARGS[@]}" -r "${REPO_ROOT}/requirements-ppu.txt"
  "${VENV_PYTHON}" -m pip check
fi

VENV_TORCH_INFO="$(${VENV_PYTHON} - <<'PY'
import json
import torch
print(json.dumps({"version": torch.__version__, "file": torch.__file__}, sort_keys=True))
PY
)" || fail "venv cannot import the official image torch"
[[ "${VENV_TORCH_INFO}" == "${BASE_TORCH_INFO}" ]] || {
  echo "Base torch: ${BASE_TORCH_INFO}" >&2
  echo "Venv torch: ${VENV_TORCH_INFO}" >&2
  fail "venv shadowed the official PPU torch; remove the venv and retry"
}

CUSTOM_OPS="${REPO_ROOT}/ppu/custom_ops"
if [[ "${BUILD_EXTENSIONS}" -eq 1 ]]; then
  (
    cd "${CUSTOM_OPS}"
    OUTPUT_DIR="${CUSTOM_OPS}/build" bash ./build_gdn_shared.sh
    "${VENV_PYTHON}" ./build_acblas_linear_extension.py
    SEU_PPU_GDN_LIBRARY="${CUSTOM_OPS}/build/libseu_ppu_gdn.so" \
      "${VENV_PYTHON}" ./build_acblas_packed_mlp_extension.py
  )
fi

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  "${VENV_PYTHON}" - <<'PY'
import torch
assert torch.cuda.is_available(), "PPU is not visible through torch.cuda"
x = torch.ones(32, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
assert float(x.sum().cpu()) == 32.0
print("device:", torch.cuda.get_device_name(0))
print("torch :", torch.__version__)
PY

  (
    cd "${CUSTOM_OPS}"
    "${VENV_PYTHON}" smoke_gdn_gate_prep_integration.py \
      --library build/libseu_ppu_gdn.so --warmup 1 --iters 2 --repeats 1
    "${VENV_PYTHON}" smoke_acblas_linear_module.py \
      --build-dir build/acblas_linear_extension \
      --input-features 2048 --output-features 2048 --warmup 1 --iters 2
    "${VENV_PYTHON}" - <<'PY'
import sys
import torch  # Load the official runtime libraries before the C++ extension.

sys.path.insert(0, "build/acblas_linear_extension")
import seu_acblas_linear_ext as extension

assert hasattr(extension, "gdn_projections_bf16"), (
    "stale acBLAS linear extension: gdn_projections_bf16 is missing"
)
print("acBLAS b/a-GEMV symbol: available")
PY
    "${VENV_PYTHON}" smoke_acblas_packed_mlp_module.py \
      --build-dir build/acblas_packed_mlp_extension --warmup 1 --iters 2
  )
fi

MANIFEST="${VENV_DIR}/bootstrap-manifest.txt"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=${REPO_ROOT}"
  echo "git_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python=${VENV_PYTHON}"
  echo "ppu_sdk=${PPU_SDK}"
  echo "torch=${VENV_TORCH_INFO}"
} > "${MANIFEST}"

echo
echo "PPU environment is ready. Activate it with:"
echo "  export SEU_PPU_VENV_DIR='${VENV_DIR}'"
echo "  source scripts/activate_ppu_env.sh"
echo "Manifest: ${MANIFEST}"
