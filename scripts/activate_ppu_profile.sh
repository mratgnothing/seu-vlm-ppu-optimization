#!/usr/bin/env bash
# Source after scripts/activate_ppu_env.sh to select an evidence-backed PPU stack.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source scripts/activate_ppu_profile.sh [performance]" >&2
  exit 2
fi

_seu_profile="${1:-performance}"
_seu_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_seu_repo_root="$(cd "${_seu_script_dir}/.." && pwd)"
_seu_custom_ops="${_seu_repo_root}/ppu/custom_ops"

case "${_seu_profile}" in
  performance) ;;
  *)
    echo "Unknown PPU profile: ${_seu_profile}" >&2
    unset _seu_profile _seu_script_dir _seu_repo_root _seu_custom_ops
    return 2
    ;;
esac

_seu_gdn_library="${_seu_custom_ops}/build/libseu_ppu_gdn.so"
_seu_gdn_extension="${_seu_custom_ops}/build/acblas_linear_extension"
_seu_mlp_extension="${_seu_custom_ops}/build/acblas_packed_mlp_extension"
for _seu_required in "${_seu_gdn_library}" "${_seu_gdn_extension}" "${_seu_mlp_extension}"; do
  if [[ ! -e "${_seu_required}" ]]; then
    echo "Required PPU artifact is missing: ${_seu_required}" >&2
    echo "Run: bash scripts/bootstrap_ppu_env.sh" >&2
    unset _seu_profile _seu_script_dir _seu_repo_root _seu_custom_ops
    unset _seu_gdn_library _seu_gdn_extension _seu_mlp_extension _seu_required
    return 1
  fi
done

export SEU_PPU_GDN_LIBRARY="${_seu_gdn_library}"
export SEU_PPU_GDN_PYTHON_DIR="${_seu_custom_ops}"
export SEU_PPU_GDN_TILES=4
export SEU_PPU_CONV_ENABLE=1
export SEU_PPU_CONV_THREADS=96
export SEU_PPU_RMSNORM_ENABLE=1
export SEU_PPU_RMSNORM_THREADS=512
export SEU_PPU_GATED_RMSNORM_ENABLE=1
export SEU_PPU_GATED_RMSNORM_THREADS=128
export SEU_PPU_QK_ROPE_ENABLE=1
export SEU_PPU_PACK_MLP_ENABLE=1
export SEU_PPU_RESIDUAL_RMSNORM_ENABLE=1
export SEU_PPU_GDN_GATE_PREP_ENABLE=1
export SEU_PPU_RAW_STREAM_QUERY_ENABLE=1
export SEU_PPU_ACBLAS_GDN_BUILD_DIR="${_seu_gdn_extension}"
export SEU_PPU_ACBLAS_PACKED_MLP_BUILD_DIR="${_seu_mlp_extension}"
export SEU_PPU_ACBLAS_PACKED_MLP_SWIGLU_THREADS=128

# Final evidence-backed stack: exact b/a-GEMV is built into the extension and
# multi-row norm/residual fusions are enabled for prefill.
export SEU_PPU_PREFILL_ROW_FUSIONS_ENABLE=1
export SEU_PPU_ACTIVE_PROFILE="${_seu_profile}"

echo "Activated PPU profile: ${SEU_PPU_ACTIVE_PROFILE}"
echo "  GDN projection: acblas-grouped"
echo "  b/a-GEMV: 1 (final extension path)"
echo "  prefill-row-fusions: ${SEU_PPU_PREFILL_ROW_FUSIONS_ENABLE:-0}"

unset _seu_profile _seu_script_dir _seu_repo_root _seu_custom_ops
unset _seu_gdn_library _seu_gdn_extension _seu_mlp_extension _seu_required
