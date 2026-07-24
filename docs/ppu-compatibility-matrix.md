# Qwen3.5-2B 与当前 PPU 软件栈兼容性矩阵

核验时间：2026-07-24

上游对照：截至核验日，vLLM 主线已公开支持 Qwen3.5 混合注意力模型，但共享节点的 PPU 定制分支仍停留在 0.8.5。参考 [vLLM 仓库](https://github.com/vllm-project/vllm) 与 [Qwen3.5 使用指南](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)。

## 当前共享节点

| 层级 | 镜像预期 | 当前运行态 | Qwen3.5 结论 |
|---|---|---|---|
| PPU SDK | 2.1.0、CUDA 12.6 兼容层 | SDK、驱动、4 张 PPU 正常 | 基础 kernel 链可用 |
| PyTorch | PPU 定制 2.6.0 wheel | 当前 Python 未安装 | 无法直接运行模型 |
| Transformers | 4.51.3 | 当前 Python 未安装 | 该版本早于本地已验证的 Qwen3.5 实现 |
| vLLM | PPU 定制 0.8.5+cu126 | 源码快照存在，wheel 未安装 | 无 Qwen3.5 模型注册 |
| Attention | PPU FlashAttention/FlashInfer | 源码存在 PPU 专用分支 | 可复用，但不覆盖 GDN |
| Causal Conv | vLLM 自定义 CUDA/PPU 编译路径 | BF16 kernel 源码存在 | 可作为 Qwen3.5 移植基础 |
| Gated Delta Network | 镜像未声明 | 未发现对应模型或 kernel | 当前主要缺口 |
| 量化 | bitsandbytes、torchao、vLLM PPU 配置 | 镜像构建说明中存在 | 需先解决模型架构支持并确认规则 |

## 已确认的 PPU-vLLM 特征

- PPU-ZW810E 专用设备判断与 FlashAttention 分支。
- AWQ/GPTQ Marlin 中存在 `ppu.mma`、`ppu.ldmatrix` 和 HGGC 指令。
- PPU-ZW810E 的 INT4、INT8、FP8 kernel 配置较完整。
- vLLM 自带 BF16 causal-conv1d forward/update kernel。
- 当前模型注册包括 Qwen3 和 Qwen2.5-VL，不包括 `Qwen3_5ForConditionalGeneration`。
- 未发现 Qwen3.5 所需 Gated Delta Network 模型实现。
- Qwen3.5-2B 的 `N=6144,K=2048` 与 `N=2048,K=6144` 核心矩阵未发现 PPU-ZW810E 预调优量化配置。

具体层数、矩阵尺寸与 kernel 优先级见 [qwen35-kernel-targets.md](qwen35-kernel-targets.md)。

## 可选部署路线

### 路线 A：主办方提供新版 PPU 镜像

优先级最高。目标镜像应同时具备：

- PPU 定制 PyTorch；
- Transformers 5.x 或等价的 Qwen3.5 支持；
- 支持 `Qwen3_5ForConditionalGeneration` 的 PPU-vLLM；
- GDN/线性注意力和 causal-conv1d 的 PPU fast path；
- 可用的 profiler 与量化工具。

这条路线能直接进入真实 baseline 和算子优化。

### 路线 B：PPU PyTorch + Transformers eager

先用主办方定制 PyTorch 和新版 Transformers 验证 Qwen3.5 功能。若 GDN 与 causal conv 回退到通用 PyTorch 算子，性能可能较低，但可以建立 PPU 功能/精度基线。

前提是获得可安装的 PPU PyTorch wheel 和隔离资源，并确认 Transformers 5.x 与该定制 PyTorch 兼容。

### 路线 C：移植新版 vLLM Qwen3.5 到 PPU 0.8.5 分支

工作量最大。需要移植：

- Qwen3.5 多模态模型与处理器适配；
- 混合 full-attention/GDN cache 管理；
- GDN prefill/decode kernel；
- 当前 PPU FlashAttention、causal conv、量化和调度补丁；
- 完整精度、TTFT、吞吐与稳定性回归。

只有在主办方确认不会提供新版镜像时再进入这条路线。

## 当前结论

共享节点足以验证 SDK 和研究 PPU-vLLM 源码，但不足以完成 Qwen3.5-2B 真实部署。当前最小阻塞项是：

1. 可安装的 PPU Python 栈；
2. Qwen3.5 模型注册；
3. GDN/线性注意力 PPU 路径；
4. 可上传模型和代码的隔离资源。
