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
TTFT 中位数降低 8.25%，吞吐中位数提高 8.82%；英文三次复测分别提高
10.22% 和 6.37%。

CUDA profile 显示 GEMV/GEMM 占 self CUDA time 的 86.18%，其中 decode
阶段 BF16 GEMV 是第一热点。项目已据此解析 Qwen3.5-2B 的 Gated Delta
Network（GDN）、MLP、全注意力和视觉主干尺寸，并为
`N=6144,K=2048`、`N=2048,K=6144`、`N=2048,K=2048`
三组核心矩阵准备 HGGC BF16 参考微基准。随后已在隔离 PPU-ZW810E 节点完成
Qwen3.5-2B 全模型部署，并接入五类 HGGC decode 融合与 packed MLP。注册式
acBLAS Linear 通用替换已完成但最终固定长解码无稳定收益，作为负实验保留。新增的
GDN 四路输入投影打包在同模型 CN20 paired 验证中达到 98.430 token/s、85%
Accuracy，但只有 19/20 完整文本一致。进一步实现的结构专用 grouped acBLAS
保留四个原形状 GEMV，CN20 两轮成对中位提升 1.87%/3.91%，Accuracy 均为 85%
且 20/20 全文一致。在此基础上，48-edge residual-add + RMSNorm 跨层融合又在
16-token profile 中减少 720 次 kernel launch，CN20 两轮配对中位提升约 2.1%，
均保持 20/20 全文一致；所有新增路径都默认关闭，需完整公开集和私有集门禁。

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
- 精度扩展集：中英文分别按 category/二级类别比例分层抽取 200 条，固定随机种子
  `20260625`，两种语言均覆盖 20 个分组。

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

### 5.2 英文固定 20 条，三次复测中位数

| Profile | Accuracy | TTFT 中位数 | TTFT 范围 | Throughput 中位数 | Throughput 范围 |
|---|---:|---:|---:|---:|---:|
| O0 `no_grad` | 80% | 300.105 ms | 298.585–300.332 ms | 22.678 tokens/s | 21.883–22.909 |
| O1 `inference_mode` | 80% | 269.424 ms | 268.829–272.349 ms | 24.123 tokens/s | 23.570–24.195 |

- TTFT 提升率：10.2234%。
- Throughput 提升率：6.3718%。

英文 6 个结果文件的 question ID、答案和 Token 数完全一致，全部通过公开接口校验。
最初单次吞吐提升 10.57% 偏乐观，因此正式报告改用三次中位数。

### 5.3 中英文比例分层 200 条

O1 在 200 条分层样本上的 Accuracy 为 84.5%（169/200），公开接口验证全部通过。
较弱子类包括逻辑推理、关系理解和跨实例推理，后续精度验证应优先观察这些类别，
不得针对单个公开样本手工过拟合。

英文使用相同种子和分层方法，Accuracy 为 82.5%（165/200），公开接口验证全部
通过。英文 logic reasoning 为 47.37%，relation reasoning 为 76.19%；
中英文均确认这两类是后续低精度与 kernel 优化的重点精度护栏。

### 5.4 中英文完整公开集

O1 在中文 MMBench Dev 全部 4029 条上的 Accuracy 为 83.94%（3382/4029），
公开接口验证 4029/4029 通过，无无效答案。完整集按 200 条分块运行，共 21 块；
来源文件、分块和合并结果均带 SHA-256 校验。独立审计确认合并结果恰有 4029 个
唯一题目 ID，与原始 TSV 的题号集合完全一致，逐样本重新统计的正确数与汇总字段
一致。

一级类别中，Coarse Perception 为 88.96%，实例级 Fine-grained Perception 为
87.64%，Attribute Reasoning 为 85.52%；较弱的 Logic Reasoning 为 71.05%，
跨实例 Fine-grained Perception 为 76.72%。完整集结果与中文分层 200 条的 84.5%
接近，未观察到规模扩大后的异常精度坍塌。

完整集只运行一次，因此本节仅作为 Accuracy 和管线完整性证据。其 TTFT 与
Throughput 不进入正式性能表；性能结论仍采用固定 20 条、三次运行的中位数。

英文完整公开集原始 Accuracy 为 79.72%（3212/4029）。唯一无效输出在 256-token
上限前已经把 C 标记为加粗的 “matches exactly”，但来不及生成最终答案行。为此，
模型 wrapper 增加严格的通用结论规范化：仅当输出恰有一个加粗正向结论且没有标准
答案行时，追加 `Answer: X`；不读取题号和参考答案。

修复后完整重跑异常所在的 200 条分块。逐样本比较确认 199 个已有解析答案和全部
token 数不变，仅该无效输出恢复为 C。严格重新合并后的英文 Accuracy 为 79.75%
（3213/4029），公开接口验证 4029/4029 通过。英文一级类别中 Coarse Perception
为 87.56%，实例级 Fine-grained Perception 为 85.47%；Logic Reasoning 最弱，
为 56.22%。修复前后的完整 artifact 均保留在本地，以便审计。

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

## 7. PPU 实机部署与优化结果

### 7.1 已确认事实

- 隔离节点为 1 张 PPU-ZW810E，显存 98,304 MiB；SDK 2.1.1、HGGC 13.0 可用。
- 独立 venv 复用 PPU PyTorch 2.11 和定制 Triton，并安装 Transformers 5.14.1；
  没有替换系统环境。
- Qwen3.5-2B 的 617 个参数张量全部驻留 `cuda:0`，无 CPU/meta/disk offload；
  视觉编码、18 层 Gated DeltaNet、6 层全注意力和自回归解码均已闭环。
- 当前没有可直接使用的 PPU-vLLM/Qwen3.5 fast path，生产验证以 Transformers
  eager 加仓库显式 opt-in 算子为基线。

### 7.2 已实测优化

已接入 recurrent GDN、causal-conv、2048 维 RMSNorm、128 维 gated RMSNorm、
q/k RMSNorm+partial RoPE 五类 HGGC decode 核；另以权重共享 view 合并 24 层
MLP gate/up projection，并探索 GDN 每层四个同输入投影的 multi-output packing。
Torch extension + C-ABI acBLAS bridge 也完成了 ABI 隔离和 102 个 Linear 的负实验。

固定中文前 20 条的关键结果为：

| 路径 | 平均 token/s | Accuracy | 相对 eager |
|---|---:|---:|---:|
| eager | 49.737 | 85% | - |
| GDN + causal-conv | 63.911 | 85% | +28.50% |
| all-five | 93.918 / 94.889 | 85% | +88.83% / +90.78% |
| all-five + packed-MLP | 96.506 / 96.715 | 85% | +94.04% / +94.46% |
| + packed GDN projections，paired | 98.430 | 85% | +97.90% |
| + grouped-acBLAS GDN r1/r2 | 98.028 / 99.601 | 85% | +97.10% / +100.26% |
| + 48-edge residual-RMSNorm r1/r2 | 101.616 / 101.507 | 85% | +104.31% / +104.09% |
| + GDN gate-prep r1/r2 | 109.275 / 107.083 | 85% | +119.71% / +115.31% |

最终线程隔离版 packed-GDN 的同模型逐样本 AB/BA 中，基线/候选为
94.099/98.430 token/s，成对速度比中位数 1.0355x，20 条赢 15 条；Accuracy 都是
85%，但 1 条文本多生成 1 个 token。固定 128-token 四对则 4/4 获胜且全文一致。
候选由 Qwen3.5 图结构产生，公开
数据只作回归门禁。acBLAS 最终固定 128-token 八对成对中位仅 0.9997x，未接入。

grouped-acBLAS GDN 不是通用替换：它在一次 pybind/C++ 入口中依次执行 qkv、z、b、a
四个原形状 `acblasGemvEx`。CN20 两轮由 96.409→98.028 和 95.634→99.601
token/s，成对中位为 1.0187x/1.0391x，分别 16/20、17/20 获胜，并保持两轮
20/20 全文一致。Profile 中 `aten::linear/mm` 各减少 1080 次，而设备侧
`gemvt_op` 和 `cudaLaunchKernel` 数不变，说明收益来自主机调度与 handle/stream
设置合并，不是减少数学计算。固定长六对只有 3/6 获胜，所以暂不默认启用。

Qwen3.5 每层 attention 和 MLP 后各有一条 `residual add -> RMSNorm` 相邻边，24 层
共 48 条。新增 HGGC 核保持 residual 的 BF16 舍入点，在同一 kernel 内做 FP32
平方和归约与 weight scaling；跨层 thread-local 缓存把 MLP residual 直接交给下一层
input norm，最后一层连接 final norm。只融合层内 24 条边的第一版固定长中位为
0.9821x，作为负实验保留；完整 48-edge 版固定长两轮中位为 1.0159x/1.0233x。
CN20 两轮由 100.156→101.616 和 98.576→101.507 token/s，配对中位
1.0213x/1.0206x，均 14/20 获胜、85% Accuracy、20/20 全文一致。Profile 中目标
`aten::add` 720→0、`cudaLaunchKernel` 16973→16253；正式 wrapper smoke 也通过
真实 PPU 后端和公开校验。

最后一轮进一步针对 18 个 GDN 层的门控准备：eval 加载时缓存 FP32
`exp(A_log)`，一个 HGGC kernel 合并 `sigmoid(b)`、两个 cast、bias add、Softplus、
乘法和取负，并以 thread-local scratch 复用 FP32 `g` 与 BF16 `beta`。最终固定
128-token 六对全部获胜、全文一致、配对中位 1.0839x；CN20 两轮分别为
101.651→109.275 和 100.085→107.083 token/s，配对中位 1.0811x/1.0863x、
19/20 和 17/20 获胜，两轮均 20/20 全文一致且 Accuracy 85%。Profile 中
`cudaLaunchKernel` 16253→14363，Self CPU/PPU 分别下降 11.48%/5.35%，
`hggc-memcheck` 为 0 errors。完整公开集正在作为独立精度门禁运行，未按公开标签
调节实现。

### 7.3 当前边界

全部自定义路径默认关闭，只有显式环境变量才挂载。五类融合的 reduction 顺序使
all-five 相对 eager 有 5/20 生成长度变化，packed-GDN 又相对 packed 基线新增 1/20
文本漂移；虽未改变 CN20 Accuracy，但最终提交前仍需完整公开集和私有集门禁。

## 8. 可复现性

仓库包含：

- 模型、数据和主办方文件锁定信息；
- 环境检查、数据审计、固定抽样和基准运行脚本；
- O0/O1 profile 选择与矩阵复测脚本；
- 原始结果对比、统计汇总和 CUDA profile 脚本；
- PPU 零依赖环境预检与 BF16 GEMV 微基准；
- 单元测试和 CI 语法/契约检查；
- 使用显式白名单、敏感内容扫描、固定 ZIP 时间戳和逐文件 SHA-256 清单的源码
  候选包生成器。

本地原始数据、模型权重、密钥、日志和运行结果默认被 `.gitignore` 排除。正式提交包
只收录赛事允许的代码、配置模板、报告和必要结果摘要。当前候选包已通过两次相同
SHA-256 的可复现构建验证，但仍需根据主办方最终目录和启动命令要求定稿。

## 9. 当前结论与下一步

O1 已在本地固定中英文样本上取得稳定、无精度变化的 TTFT 和吞吐收益，且评测计时
边界已修正到第一个生成 Token。中英文完整公开集 Accuracy 分别为 83.94% 和
79.75%，结果完整性均已独立审计。Profile 说明后续优化应集中于 decode
GEMV/GEMM、GDN 和 causal conv，而非在缺少证据时广泛改动。

当前最高性能候选已在 PPU 完成编译、模型 A/B、profile 和 CN20 精度闭环。剩余外部
依赖是主办方明确：

1. 支持 Qwen3.5-2B 的 PPU Python/vLLM 镜像及版本；
2. GDN 和 causal-conv1d 的 PPU fast path；
3. 个性化隔离资源与代码/模型上传时间；
4. 允许的量化格式、校准数据和层级混合精度范围；
5. 初赛复现环境、依赖安装和启动命令限制。

收到答复后可固定最终镜像 ABI、完成完整集门禁，并决定是否把当前 extension 打包为
wheel 或迁移到官方 PPU-vLLM custom-op 接口。

## 附录：证据索引

- 当前状态：`docs/current-status.md`
- 实验计划：`docs/experiment-plan.md`
- O1 实验：`docs/experiments/2026-07-24-o1-inference-mode.md`
- TTFT 边界修正：`docs/experiments/2026-07-24-ttft-token-boundary.md`
- 英文三次复测：`docs/experiments/2026-07-24-en-m1-three-runs.md`
- CUDA profile：`docs/experiments/2026-07-24-o2-cuda-profile.md`
- 分层 200 条精度：`docs/experiments/2026-07-24-cn-stratified-n200.md`
- 英文分层 200 条精度：`docs/experiments/2026-07-24-en-stratified-n200.md`
- 中文完整公开集：`docs/experiments/2026-07-26-cn-full-n4029.md`
- 英文完整公开集：`docs/experiments/2026-07-26-en-full-n4029.md`
- PPU 兼容性矩阵：`docs/ppu-compatibility-matrix.md`
- 关键算子尺寸：`docs/qwen35-kernel-targets.md`
- 主办方问题：`docs/questions-for-organizer.md`
