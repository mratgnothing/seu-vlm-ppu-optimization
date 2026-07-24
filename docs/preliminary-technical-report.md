# 面向 PPU 的 Qwen3.5-2B VLM 高效推理与优化技术报告（初稿）

> 状态：持续更新。本文已填入本地可复现实验结果；所有尚未在 PPU
> 隔离资源验证的内容均明确标注，不作为已完成成果申报。

## 摘要

本项目面向视觉问答类边缘推理场景，研究 Qwen3.5-2B
视觉语言模型在平头哥 PPU 上的高效部署。当前阶段已建立公开 MMBench
中英文数据的统一评测入口、固定样本与比例分层样本、模型与数据锁定信息，以及
Accuracy、首 Token 时间（TTFT）和生成吞吐的可复现测量链路。在本地
RTX 4050 Laptop GPU 上，`torch.inference_mode()` 相对 `torch.no_grad()`
在中英文各 20 条固定样本上保持答案、Token 数和 Accuracy 不变；中文三次复测的
TTFT 中位数降低 8.25%，吞吐中位数提高 8.82%，英文交叉验证分别提高
9.77% 和 10.57%。

CUDA profile 显示 GEMV/GEMM 占 self CUDA time 的 86.18%，其中 decode
阶段 BF16 GEMV 是第一热点。项目已据此解析 Qwen3.5-2B 的 Gated Delta
Network（GDN）、MLP、全注意力和视觉主干尺寸，并为
`N=6144,K=2048`、`N=2048,K=6144`、`N=2048,K=2048`
三组核心矩阵准备 HGGC BF16 参考微基准。当前共享 PPU 节点的 SDK 与基础
kernel 链可用，但 Python 推理栈缺失，预置 PPU-vLLM 分支也尚未包含
Qwen3.5 模型注册与 GDN 路径。因此 PPU 模型级部署和算子优化结果需要在主办方
提供支持 Qwen3.5 的隔离镜像后完成。

## 1. 应用场景与目标

应用场景为边缘侧单请求视觉问答：系统接收一张图像和一个自然语言问题，输出简短
答案。该场景强调低 TTFT 和稳定的单样本生成速度，同时必须维持视觉语义理解能力。

项目目标包括：

1. 在指定环境完成 Qwen3.5-2B 的真实图文推理部署。
2. 在统一数据、提示词、生成参数和计时边界下建立可复现基线。
3. 在 Accuracy 不下降的前提下降低 TTFT、提高生成吞吐。
4. 根据 profile 结果开展运行时、内存、融合、量化与 PPU kernel 优化。
5. 保存每项优化的代码、环境、原始结果、精度对照和回退方式。

## 2. 软硬件协同设计

### 2.1 评测层

赛事公开入口 `benchmark_public.py` 保持原始评分契约。项目在其外部增加
`evaluation_wrapper.py`，负责模型加载、图像解码、提示词构造、生成和严格计时，
避免修改评分逻辑。所有输出仍交回赛事入口解析与校验。

### 2.2 模型层

模型锁定为官方 `Qwen/Qwen3.5-2B`。本地缓存的 revision、配置和主权重
SHA-256 已写入锁定文件。当前路径使用 BF16 权重、单样本、确定性贪心生成，以便
将性能变化归因于实现优化，而非采样波动。

### 2.3 运行时与硬件层

本地 GPU 用于建立功能、精度、计时和热点基线。PPU 路线按以下优先级推进：

1. 主办方提供支持 Qwen3.5/GDN 的新版 PPU-vLLM 镜像时，优先复用其调度、
   KV Cache、FlashAttention、causal-conv1d 和量化路径。
2. 若只有 PPU 定制 PyTorch，则先以 Transformers eager 建立真实功能和精度基线。
3. 仅在缺少可用新版运行时的情况下，评估将新版 vLLM 的 Qwen3.5/GDN
   路径移植到当前 PPU 0.8.5 分支。

三条路线都必须先通过相同评测入口，再开展 kernel、融合和量化优化。

## 3. 数据与评测协议

### 3.1 数据

- MMBench Dev CN：4029 条，图像、问题、选项和答案字段审计通过。
- MMBench Dev EN：4029 条，字段审计通过。
- 快速迭代集：中英文各固定前 20 条。
- 精度扩展集：中文按 category/二级类别比例分层抽取 200 条，固定随机种子
  `20260625`，覆盖 20 个分组。

分层 200 条用于观察跨类别稳定性，固定前 20 条用于性能回归。两者都不等同于
赛事私有评测集。

### 3.2 指标

- Accuracy：赛事公开入口正确匹配数除以总样本数。
- TTFT：从模型接收完整输入并开始生成，到 streamer 收到第一个新生成 Token
  的时间，不包含 prompt Token 入队。
- Throughput：输出 Token 数除以生成阶段耗时，单样本且不启用 batch。

每次性能对比锁定 question ID、答案、输出 Token 数和验证状态。改变计时边界的
历史结果不与正式 M1 口径混用。

## 4. 已完成优化

### 4.1 O0：答案标记规范化

公开模型可能输出 Markdown 加粗选项，例如 `**A**`。公开解析器只接受规范选项，
因此在不改变语义的前提下规范化答案标记。该修改解决输出格式造成的假性精度损失，
并通过单元测试覆盖。

### 4.2 O1：`torch.inference_mode()`

基线使用 `torch.no_grad()`；O1 改为 `torch.inference_mode()`，关闭推理阶段
不需要的 autograd 版本跟踪。两种 profile 通过环境变量显式选择，便于复现和回退。

验证结果：

- 中英文 question ID、最终答案和输出 Token 数逐项相同。
- 中文 Accuracy 均为 85%，英文 Accuracy 均为 80%。
- 所有公开接口校验通过。

## 5. 性能结果

### 5.1 中文固定 20 条，三次复测中位数

| Profile | Accuracy | TTFT 中位数 | TTFT 范围 | Throughput 中位数 | Throughput 范围 |
|---|---:|---:|---:|---:|---:|
| O0 `no_grad` | 85% | 313.562 ms | 308.458–315.596 ms | 21.328 tokens/s | 21.134–22.538 |
| O1 `inference_mode` | 85% | 287.706 ms | 268.749–290.933 ms | 23.209 tokens/s | 23.074–23.963 |

按赛事提升率公式计算：

- TTFT 提升率：8.2459%。
- Throughput 提升率：8.8194%。

### 5.2 英文固定 20 条交叉验证

| Profile | Accuracy | TTFT | Throughput |
|---|---:|---:|---:|
| O0 `no_grad` | 80% | 298.585 ms | 21.883 tokens/s |
| O1 `inference_mode` | 80% | 269.424 ms | 24.195 tokens/s |

- TTFT 提升率：9.766%。
- Throughput 提升率：10.565%。

英文结果当前各运行 1 次，用于确认收益方向；最终报告前需要扩展为至少三次复测。

### 5.3 中文比例分层 200 条

O1 在 200 条分层样本上的 Accuracy 为 84.5%（169/200），公开接口验证全部通过。
较弱子类包括逻辑推理、关系理解和跨实例推理，后续精度验证应优先观察这些类别，
不得针对单个公开样本手工过拟合。

## 6. Profile 与关键算子

单条中文样本的 CUDA profiler 结果：

| 类别 | Self CUDA time | 占比 |
|---|---:|---:|
| GEMV | 970.766 ms | 60.329% |
| GEMM | 416.038 ms | 25.855% |
| Elementwise | 114.295 ms | 7.103% |
| Memory copy | 73.128 ms | 4.545% |
| Reduction | 28.168 ms | 1.751% |

总 self CUDA time 为 1609.119 ms。峰值 allocated/reserved 显存为
4.19/4.21 GiB。模型 24 个文本层中包含 18 个线性注意力/GDN 层和 6 个全注意力
层；高频 `2048→6144` 与 `6144→2048` 投影、GDN recurrent update 和
width-4 causal conv 是 decode 优化重点。

当前 Transformers 明确回退到 PyTorch 线性注意力和 causal-conv1d 路径。
本机 6GB 显存余量有限，不在 Windows 上盲装未经验证的扩展或直接进行高风险全模型
编译。

## 7. PPU 现状与待验证方案

### 7.1 已确认事实

- 共享节点为 4 张 PPU-ZW810E，每张显存约 97.9GB。
- PPU SDK 2.1、驱动 1.3.2、HGGC 13.0 可用。
- 官方 `vectorAdd` 已完成编译和运行，基础 kernel 链通过。
- 当前共享运行态没有 PyTorch、Transformers、vLLM 或 SGLang。
- `/opt/vllm` 的 PPU 定制 0.8.5 源码包含 PPU 矩阵、FlashAttention、
  causal-conv1d 和量化路径，但缺少 `Qwen3_5ForConditionalGeneration`
  注册和 GDN 实现。

上述结果只能证明 SDK 基础链路，不代表 Qwen3.5-2B 已在 PPU 部署。

### 7.2 已准备验证入口

`ppu/microbench/qwen35_bf16_gemv.hg` 为三组关键尺寸提供正确性优先的
BF16/FP32 累加参考实现，并输出平均延迟、GFLOP/s、有效带宽、误差和机器可解析
JSON。获得隔离资源后，将按“编译—单次冒烟—memcheck—稳定计时—asys/acu
profile—运行时算子对照”的顺序验证。

后续生产优化候选包括：

- BF16 向量化加载和 warp 级归约；
- 针对 PPU 的权重布局与矩阵指令；
- bias、RMSNorm、gate 和激活融合；
- GDN recurrent update 与 causal-conv1d update 融合；
- KV Cache/内存池与单请求调度；
- 经规则允许且精度通过的 INT8/INT4/FP8 量化。

这些候选必须由真实 PPU profile 决定，当前不报告未经实测的提升率。

## 8. 可复现性

仓库包含：

- 模型、数据和主办方文件锁定信息；
- 环境检查、数据审计、固定抽样和基准运行脚本；
- O0/O1 profile 选择与矩阵复测脚本；
- 原始结果对比、统计汇总和 CUDA profile 脚本；
- PPU 零依赖环境预检与 BF16 GEMV 微基准；
- 单元测试和 CI 语法/契约检查。

本地原始数据、模型权重、密钥、日志和运行结果默认被 `.gitignore` 排除。正式提交包
只收录赛事允许的代码、配置模板、报告和必要结果摘要。

## 9. 当前结论与下一步

O1 已在本地固定中英文样本上取得稳定、无精度变化的 TTFT 和吞吐收益，且评测计时
边界已修正到第一个生成 Token。Profile 说明后续优化应集中于 decode GEMV/GEMM、
GDN 和 causal conv，而非在缺少证据时广泛改动。

当前最小外部依赖是主办方明确：

1. 支持 Qwen3.5-2B 的 PPU Python/vLLM 镜像及版本；
2. GDN 和 causal-conv1d 的 PPU fast path；
3. 个性化隔离资源与代码/模型上传时间；
4. 允许的量化格式、校准数据和层级混合精度范围；
5. 初赛复现环境、依赖安装和启动命令限制。

收到答复后即可选择 PPU 路线并进入真实模型功能闭环。

## 附录：证据索引

- 当前状态：`docs/current-status.md`
- 实验计划：`docs/experiment-plan.md`
- O1 实验：`docs/experiments/2026-07-24-o1-inference-mode.md`
- TTFT 边界修正：`docs/experiments/2026-07-24-ttft-token-boundary.md`
- CUDA profile：`docs/experiments/2026-07-24-o2-cuda-profile.md`
- 分层 200 条精度：`docs/experiments/2026-07-24-cn-stratified-n200.md`
- PPU 兼容性矩阵：`docs/ppu-compatibility-matrix.md`
- 关键算子尺寸：`docs/qwen35-kernel-targets.md`
- 主办方问题：`docs/questions-for-organizer.md`
