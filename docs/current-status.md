# 当前状态

更新时间：2026-07-24

## 已确认

- 正式工作目录：`D:\GitHub\seu-vlm-ppu-optimization`
- 本机 GPU：NVIDIA GeForce RTX 4050 Laptop GPU，6GB 显存
- 官方模型：`Qwen/Qwen3.5-2B`
- 模型 revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`
- 官方模型完整下载约 4.6GB，单个 BF16 权重文件约 4.5GB
- 模型参数量约 2.274B
- 模型 revision 已完整下载到项目忽略目录
- 主权重 SHA-256：`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`
- 模型类：`Qwen3_5ForConditionalGeneration`
- 617 组参数全部位于 `cuda:0`，模型内存 footprint 为 4,426,483,648 bytes
- 本地公开数据：MMBench 中文/英文 dev 各 4029 条
- 官方 v1.1 评测代码已导入

## 本地运行环境

- Python 3.12.7
- PyTorch 2.13.0+cu130
- Torchvision 0.28.0+cu130
- Transformers 5.14.1
- CUDA 可用，GPU 支持 BF16
- Qwen3.5 配置、`AutoModelForImageTextToText` 和 `AutoModelForMultimodalLM` 映射均已验证

## 已完成的工程基线

- Dummy 中英文各 20 条接口冒烟完成，结果明确标记为非真实模型。
- 真实模型中英文各 1 条生成与解析验证完成。
- 中文公开集前 20 条基线：Accuracy 85%，平均 TTFT 327.451 ms，吞吐 21.813 tokens/s，公开校验通过。
- 英文公开集前 20 条基线：Accuracy 80%，平均 TTFT 409.660 ms，吞吐 28.685 tokens/s，公开校验通过。
- O1 `torch.inference_mode()` 三次中文复测均保持 Accuracy 85%、答案和 token 数不变。
- O1 中文三次保守中位数：TTFT 290.946 ms，提升 11.15%；吞吐 23.150 tokens/s，提升 6.13%。
- O1 英文交叉验证保持 Accuracy 80%、答案和 token 数不变；TTFT 提升 14.10%，吞吐提升 16.65%。
- O2 单样本 CUDA profiler 已完成：GEMV/GEMM 占 self CUDA time 86.18%，elementwise/copy 调用数高。
- Profiler 口径 peak allocated/reserved 为 4.19/4.21 GiB。
- 以上只覆盖固定前 20 条公开样本，不代表完整公开集或私有评测成绩。

## PPU 状态

- 共享节点识别到 4 张 PPU-ZW810E，每张显存约 97.9GB。
- PPU SDK `2.1.0-a5f865`、驱动 `1.3.2-d7f5a2` 和 HGGC 13.0 已确认。
- 官方 vectorAdd 已在临时目录编译、运行并通过，随后清理。
- 共享节点未安装 PyTorch、Transformers、vLLM 或 SGLang。
- 共享节点不上传比赛代码、模型和数据，因此这还不构成 PPU 真实模型部署。

## 尚未完成

- 公开集完整基线与更大样本复测。
- PPU 真实 Qwen3.5-2B 模型部署、Profile 和优化闭环。
- 量化/权重变换的正式允许范围确认。
- 初赛技术报告和最终提交包。

## 当前风险

6GB 显存可以承载当前 BF16 单样本路径，但余量有限。更长输入、更高分辨率、编译缓存或并发请求仍可能 OOM。Qwen3.5 的线性注意力和因果卷积当前使用 PyTorch fallback，缺少 fast-path 扩展；本机 Windows 环境不盲装未经验证的 CUDA 扩展。Profiler 表明 decode 侧 BF16 GEMV 是第一热点，后续优先研究 decode、融合和 PPU 目标 kernel。CPU offload 结果不得与纯 GPU 或 PPU 性能直接比较。
