# 当前状态

更新时间：2026-07-24

## 已确认

- 正式工作目录：`D:\GitHub\seu-vlm-ppu-optimization`
- 本机 GPU：NVIDIA GeForce RTX 4050 Laptop GPU，6GB 显存
- 官方模型：`Qwen/Qwen3.5-2B`
- 模型 revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`
- 官方模型完整下载约 4.6GB，单个 BF16 权重文件约 4.5GB
- 模型参数量约 2.274B
- 本地公开数据：MMBench 中文/英文 dev 各 4029 条
- 官方 v1.1 评测代码已导入

## 本地运行环境

- Python 3.12.7
- PyTorch 2.13.0+cu130
- Torchvision 0.28.0+cu130
- Transformers 5.14.1
- CUDA 可用，GPU 支持 BF16
- Qwen3.5 配置、`AutoModelForImageTextToText` 和 `AutoModelForMultimodalLM` 映射均已验证

## 尚未完成

- 模型权重下载和校验
- 真实 Transformers baseline
- PPU 工具链与真实模型验证
- 优化实验和正式报告

## 当前风险

6GB 显存仅略高于 BF16 权重体积，视觉编码器、运行时缓存和中间激活可能导致 OOM。首轮真实 baseline 应先用单样本和较小图片验证；如纯 GPU 加载失败，记录原始错误后再评估 CPU offload，不能把 offload 结果与纯 PPU/GPU 性能直接比较。
