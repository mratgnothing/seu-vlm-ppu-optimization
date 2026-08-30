#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHINESE_DIR="${REPO_ROOT}/conprehension/chinese"
CUDA_DIR="${CHINESE_DIR}/cuda-programming-guide-zh"
EXTERNAL_DIR="${CHINESE_DIR}/external"
CUDA_EXTRA_DIR="${EXTERNAL_DIR}/cuda"
PPU_DIR="${EXTERNAL_DIR}/ppu-sdk-v2.1"

CUDA_REPO="https://github.com/bearneck/cuda-programming-guide-zh.git"

for command_name in git curl; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "error: missing required command: ${command_name}" >&2
    exit 1
  fi
done

mkdir -p "${CHINESE_DIR}" "${CUDA_EXTRA_DIR}" "${PPU_DIR}"

if [[ -d "${CUDA_DIR}/.git" ]]; then
  echo "Updating CUDA Chinese guide..."
  git -C "${CUDA_DIR}" pull --ff-only
elif [[ -e "${CUDA_DIR}" ]]; then
  if [[ -f "${CUDA_DIR}/README.md" && -d "${CUDA_DIR}/docs" ]]; then
    echo "Using existing CUDA Chinese guide source snapshot."
  elif [[ -n "$(find "${CUDA_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "error: ${CUDA_DIR} exists but is not a recognizable guide snapshot" >&2
    exit 1
  else
    rmdir "${CUDA_DIR}"
    git clone --depth 1 "${CUDA_REPO}" "${CUDA_DIR}"
  fi
else
  git clone --depth 1 "${CUDA_REPO}" "${CUDA_DIR}"
fi

download_page() {
  local target="$1"
  local url="$2"
  local label="$3"
  local temporary="${target}.tmp"

  echo "Downloading ${label}..."
  curl --fail --location --compressed --retry 3 --retry-delay 2 \
    --user-agent "Mozilla/5.0 (offline documentation mirror)" \
    --output "${temporary}" "${url}"

  if [[ ! -s "${temporary}" ]]; then
    echo "error: downloaded page is empty: ${url}" >&2
    exit 1
  fi
  mv "${temporary}" "${target}"
}

# CUDA 中文补充材料：ReadTheDocs 的最佳实践中文章节和 NVIDIA 中文入门。
download_page "${CUDA_EXTRA_DIR}/best-practices-index.html" \
  "https://cuda-doc.readthedocs.io/zh-cn/latest/CUDA-C-Best-Practices-Guide/index.html" \
  "CUDA Best Practices Chinese index"
download_page "${CUDA_EXTRA_DIR}/best-practices-memory-optimization.html" \
  "https://cuda-doc.readthedocs.io/zh-cn/latest/CUDA-C-Best-Practices-Guide/10_memory_optimizations.html" \
  "CUDA memory optimization chapter"
download_page "${CUDA_EXTRA_DIR}/nvidia-cuda-intro-cn.html" \
  "https://developer.nvidia.cn/blog/cuda-intro-cn/" \
  "NVIDIA CUDA Chinese introduction"
download_page "${CUDA_EXTRA_DIR}/nvidia-cuda-model-intro-cn.html" \
  "https://developer.nvidia.cn/blog/cuda-model-intro-cn/" \
  "NVIDIA CUDA programming model introduction"
download_page "${CUDA_EXTRA_DIR}/nvidia-cuda-programming-model-interface-cn.html" \
  "https://developer.nvidia.cn/blog/cuda-programming-model-interface-cn/" \
  "NVIDIA CUDA programming model interface"

# PPU SDK v2.1.x 官方中文核心开发资料。文件名按使用场景排序，便于离线查找。
PPU_PAGES=(
  "00-sdk-index.html|https://help.aliyun.com/zh/document_detail/3029921.html|PPU SDK index"
  "01-developer-reference.html|https://help.aliyun.com/zh/document_detail/2864730.html|PPU developer reference"
  "02-release-notes.html|https://help.aliyun.com/zh/document_detail/3030339.html|PPU SDK release notes"
  "03-quick-start.html|https://help.aliyun.com/zh/document_detail/3030340.html|PPU SDK quick start"
  "04-compatibility-index.html|https://help.aliyun.com/zh/document_detail/3029924.html|PPU compatibility index"
  "05-programming-guide-v1.4.html|https://help.aliyun.com/zh/document_detail/2871803.html|PPU programming guide"
  "06-compiler-guide.html|https://help.aliyun.com/zh/document_detail/3030344.html|PPU compiler guide"
  "07-hgrtc-api.html|https://help.aliyun.com/zh/document_detail/3031241.html|HGRTC API"
  "10-cuda-unsupported-apis.html|https://help.aliyun.com/zh/document_detail/3029927.html|unsupported CUDA APIs"
  "11-cuda13-unsupported-apis.html|https://help.aliyun.com/zh/document_detail/3030348.html|unsupported CUDA 13 APIs"
  "12-nvcc-options.html|https://help.aliyun.com/zh/document_detail/3029928.html|NVCC option compatibility"
  "13-cuda-samples.html|https://help.aliyun.com/zh/document_detail/3029932.html|CUDA Samples compatibility"
  "14-cublas.html|https://help.aliyun.com/zh/document_detail/3029933.html|CUBLAS compatibility"
  "15-cudnn.html|https://help.aliyun.com/zh/document_detail/3029931.html|CUDNN compatibility"
  "16-cusolver.html|https://help.aliyun.com/zh/document_detail/3029929.html|CUSOLVER compatibility"
  "17-cufft.html|https://help.aliyun.com/zh/document_detail/3029926.html|CUFFT compatibility"
  "18-curand.html|https://help.aliyun.com/zh/document_detail/3029930.html|CURAND compatibility"
  "19-cusparse.html|https://help.aliyun.com/zh/document_detail/3029934.html|CUSPARSE compatibility"
  "20-pccl-nccl.html|https://help.aliyun.com/zh/document_detail/3029925.html|PCCL and NCCL compatibility"
  "21-cupti.html|https://help.aliyun.com/zh/document_detail/3029935.html|CUPTI compatibility"
  "22-nvml.html|https://help.aliyun.com/zh/document_detail/3032000.html|NVML compatibility"
  "23-qlean-quantization.html|https://help.aliyun.com/zh/document_detail/3030283.html|QLean quantization release note"
  "24-acext.html|https://help.aliyun.com/zh/document_detail/3030284.html|acext guide"
  "25-opencv.html|https://help.aliyun.com/zh/document_detail/3030279.html|OpenCV guide"
  "26-dali.html|https://help.aliyun.com/zh/document_detail/3030286.html|DALI guide"
  "27-open3d.html|https://help.aliyun.com/zh/document_detail/3030287.html|Open3D guide"
  "28-video-ffmpeg.html|https://help.aliyun.com/zh/document_detail/3030288.html|Video FFMpeg guide"
  "29-video-codec-image.html|https://help.aliyun.com/zh/document_detail/3030285.html|Video Codec and Image guide"
  "30-ppu-environment-check.html|https://help.aliyun.com/zh/document_detail/2871810.html|PPU environment check"
  "31-ppu-smi.html|https://help.aliyun.com/zh/document_detail/3030119.html|PPU-SMI"
  "32-dcgm.html|https://help.aliyun.com/zh/document_detail/3030118.html|PPU DCGM"
  "33-firmware-install.html|https://help.aliyun.com/zh/document_detail/3030393.html|Firmware installation guide"
  "34-kmd-install.html|https://help.aliyun.com/zh/document_detail/3030394.html|KMD installation guide"
  "35-ppu-xid.html|https://help.aliyun.com/zh/document_detail/3031077.html|PPU XID reference"
  "36-ppu-ecc.html|https://help.aliyun.com/zh/document_detail/3031098.html|PPU ECC handling"
  "37-mps.html|https://help.aliyun.com/zh/document_detail/3031156.html|MPS guide"
  "38-mig.html|https://help.aliyun.com/zh/document_detail/3031169.html|MIG guide"
  "39a-passthrough-icn-isolation.html|https://help.aliyun.com/zh/document_detail/3031166.html|passthrough ICN isolation"
  "39b-container-isolation.html|https://help.aliyun.com/zh/document_detail/3031170.html|container isolation"
  "39c-virtualization.html|https://help.aliyun.com/zh/document_detail/3031173.html|virtualization guide"
  "39d-multicard-id-mapping.html|https://help.aliyun.com/zh/document_detail/3031186.html|multi-card ID mapping"
  "39e-criu.html|https://help.aliyun.com/zh/document_detail/3031227.html|CRIU guide"
  "40-asight-systems-index.html|https://help.aliyun.com/zh/document_detail/3029958.html|Asight Systems index"
  "41-asight-systems-tracing.html|https://help.aliyun.com/zh/document_detail/3029964.html|Asight Systems tracing"
  "42-asight-systems-cli.html|https://help.aliyun.com/zh/document_detail/3029965.html|Asight Systems asys CLI"
  "43-asight-systems-gui.html|https://help.aliyun.com/zh/document_detail/3029966.html|Asight Systems GUI"
  "44-asight-compute-index.html|https://help.aliyun.com/zh/document_detail/3030027.html|Asight Compute index"
  "45-asight-compute-acu.html|https://help.aliyun.com/zh/document_detail/2871816.html|Asight Compute acu CLI"
  "46-asight-compute-gui.html|https://help.aliyun.com/zh/document_detail/2879850.html|Asight Compute GUI"
  "47-asight-faq.html|https://help.aliyun.com/zh/document_detail/3029959.html|Asight FAQ"
  "50-gdb.html|https://help.aliyun.com/zh/document_detail/3031249.html|PPU GDB"
  "51-memcheck.html|https://help.aliyun.com/zh/document_detail/3031865.html|PPU Memcheck"
  "60-hgobjdump.html|https://help.aliyun.com/zh/document_detail/3031866.html|hgobjdump"
  "61-hgprune.html|https://help.aliyun.com/zh/document_detail/3031869.html|hgprune"
  "62-hgfatbinary.html|https://help.aliyun.com/zh/document_detail/3031873.html|hgfatbinary"
  "63-hglink-hgjitlink.html|https://help.aliyun.com/zh/document_detail/3031875.html|hglink and hgJitLink"
  "70-pccl-multicard-debug.html|https://help.aliyun.com/zh/document_detail/3031980.html|PCCL multi-card debugging"
  "71-pccl-p2p-benchmark.html|https://help.aliyun.com/zh/document_detail/3031982.html|PCCL P2P benchmark"
  "72-pccl-device-order-search.html|https://help.aliyun.com/zh/document_detail/3031983.html|PCCL device order search"
  "73-pccl-sailbandwidth.html|https://help.aliyun.com/zh/document_detail/3031984.html|PCCL sailbandwidth"
  "74-deepep.html|https://help.aliyun.com/zh/document_detail/3031987.html|DeepEP guide"
  "75-torchcodec.html|https://help.aliyun.com/zh/document_detail/3032001.html|TorchCodec guide"
  "76-hgdeepstream.html|https://help.aliyun.com/zh/document_detail/3032002.html|HgDeepStream guide"
)

for page in "${PPU_PAGES[@]}"; do
  IFS='|' read -r filename url label <<<"${page}"
  download_page "${PPU_DIR}/${filename}" "${url}" "${label}"
done

echo
echo "Chinese tutorials are ready:"
echo "  CUDA: ${CUDA_DIR}/README.md"
echo "  CUDA optimization: ${CUDA_EXTRA_DIR}"
echo "  PPU:  ${PPU_DIR}/03-quick-start.html"
echo
echo "These third-party copies are intentionally ignored by Git."
