#!/usr/bin/env bash
# Source after scripts/activate_ppu_env.sh to select an evidence-backed PPU stack.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run this script with: source scripts/activate_ppu_profile.sh [precision|performance|experimental-single]" >&2
  exit 2
fi

_seu_profile="${1:-precision}"
_seu_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_seu_repo_root="$(cd "${_seu_script_dir}/.." && pwd)"
_seu_custom_ops="${_seu_repo_root}/ppu/custom_ops"

case "${_seu_profile}" in
  precision|performance|experimental-single) ;;
  *)
    echo "Unknown PPU profile: ${_seu_profile} (expected precision, performance or experimental-single)" >&2
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
export SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE=0
export SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=0
export SEU_PPU_ACBLAS_PACKED_MLP_BUILD_DIR="${_seu_mlp_extension}"
export SEU_PPU_ACBLAS_PACKED_MLP_SWIGLU_THREADS=128

# Keep all rejected/experimental alternatives off unless a dedicated A/B enables them.
unset SEU_PPU_PACK_GDN_PROJECTIONS_ENABLE
unset SEU_PPU_PACK_GDN_PROJECTIONS_GROUPS
unset SEU_PPU_ACBLAS_ATTENTION_PREP_BUILD_DIR
unset SEU_PPU_ACBLAS_WORKSPACE_MIB
# Reusable first-token cache variants failed the bilingual PPU gate on
# 2026-09-01.  Explicitly clear them so a stale experimental shell cannot
# contaminate the evidence-backed profiles.
unset SEU_PPU_FIRST_TOKEN_CACHE_ENABLE
unset SEU_PPU_FIRST_TOKEN_CACHE_CAPACITY
unset SEU_PPU_FIRST_TOKEN_CACHE_MODE

if [[ "${_seu_profile}" == "performance" ]]; then
  # Bilingual MMBench 4029/4029 exact in both languages. This combines only
  # the adjacent b/a projections and leaves qkv/z on their original GEMVs.
  export SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE=1
elif [[ "${_seu_profile}" == "experimental-single" ]]; then
  echo "WARNING: single-GEMV lost one correct answer on English MMBench 4029." >&2
  echo "Use only for reproducing the accuracy-budget experiment." >&2
  export SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=1
fi
export SEU_PPU_ACTIVE_PROFILE="${_seu_profile}"

echo "Activated PPU profile: ${SEU_PPU_ACTIVE_PROFILE}"
echo "  GDN projection: acblas-grouped"
echo "  b/a-GEMV: ${SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE}"
echo "  single-GEMV: ${SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE}"

unset _seu_profile _seu_script_dir _seu_repo_root _seu_custom_ops
unset _seu_gdn_library _seu_gdn_extension _seu_mlp_extension _seu_required
