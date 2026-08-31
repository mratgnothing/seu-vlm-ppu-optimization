# Qwen3.5-2B PPU 推理优化完整工作说明

更新时间：2026-08-31

开发分支：`5070ti`

目标硬件：PPU-ZW810E

模型：`Qwen/Qwen3.5-2B`，BF16，batch size 1

## 0. 一页结论

本项目已经完成从原始 Transformers eager、真实 PPU profile、模型专用融合算子、
结构化 acBLAS 提交，到中英文完整公开集门禁的闭环。当前最严格、可直接复述的总性能
结论来自同一张 PPU、同一模型和同一 CN20 数据上的两个独立 ABBA block：原始 eager
每臂四次吞吐中位 `49.3415 token/s`，最终 performance 栈为
`132.1895 token/s`，即 `2.67907x`，吞吐提升 `167.91%`；8 次 Accuracy 均为
85%，20/20 解析答案和正确性一致。该轮公开入口未保存全文哈希，因而不宣称总栈相对
eager 全文 bit-exact，也不把 CN20 结果外推为完整集或私有评测成绩。

性能来源不是 PPU 峰值算力突然提高，而是逐步消除 batch size 1 自回归解码中的主机
调度、短 kernel、重复 stream 查询、临时张量和过细的 GEMV 提交边界。最初五类
HGGC decode 融合把 CN20 吞吐从约 `49.7` 提到约 `94.9 token/s`；之后
gate-prep、单入口 packed-MLP 和 raw-stream 查询在各自前一优化栈上，经中英文完整集
获得约 `8%--11%` 的独立增量。后期 profile 仍显示每个 decode token 有 120 次小
BF16 GEMV，因此最终日集中处理 GDN 四投影的提交次数，而不是继续增加几十微秒的独立
elementwise kernel。

一次合并 qkv/z/b/a 四个投影的 single-GEMV 虽使中文 CN20 总栈达到 `2.7689x`，
并在中英文完整集分别获得约 `2.4%/2.5%` 的中位增量，但英文 4029 条中少答对 1 题，
所以已明确降级为 `experimental_only`。更保守的 b/a-GEMV 只合并两个相邻
`[16,2048]` 投影；中英文 4029 条均得到 4029/4029 全文 exact、正确数分别保持
3374/4029 和 3214/4029，成对中位分别为 `1.00696x/1.00697x`。两轮独立 ABBA
block 的最终总栈复测（每臂 4 次）为 `49.3415→132.1895 token/s`，即
`2.67907x`、提升 `167.91%`，所以它已进入显式 `performance` 配置。

仓库同时保留所有失败路线及其负门禁。它们不是“没来得及删掉的代码”，而是说明
PPU 优化边界的实验记录：局部微基准更快不等于整模更快，减少 API 次数不等于减少设备
kernel，数学等价也不等于 BF16 自回归逐 token 完全等价。

## 1. 文档目的与结论边界

本文按实际工程演进记录本项目完成的环境建设、模型部署、瓶颈分析、算子实现、整模
验证和失败路线。所有性能数字都对应仓库中的 JSON、profile、memcheck 或实验说明。
文中严格区分四类证据：

1. **单算子/模块微基准**：只证明局部实现的正确性和耗时，不能代表整模收益；
2. **固定 128-token paired A/B**：隔离生成长度差异并采用 AB/BA 交错顺序，主要判断
   性能是否稳定；
3. **CN20/EN20 两轮**：使用真实多模态样本检查端到端性能、答案和生成稳定性；
4. **中英文 4029 完整公开集**：验证 Accuracy、完整文本、token 数和长程性能分布。

当前可以直接证明的结论是：五类 decode 融合在 CN20 同口径实验中相对 eager 提升
`88.83%/90.78%`；后续 gate-prep、单入口 packed-MLP 和 raw-stream 查询分别在前一版
优化栈上通过中英文完整集 paired 门禁。最终 performance 栈又以两个独立 ABBA block
直接对比原始 eager，每臂四次中位给出 `2.67907x` 总加速。该直接总加速的范围仍是
CN20，不是 4029 条或私有集；双语完整集承担的是每个后期增量的精度与配对性能门禁。

目前的主要收益是**解码吞吐**。完整集各后期增量的 TTFT 基本持平或轻微上升，不能
宣称 TTFT 获得显著改善。所有结果均为公开开发集工程证据，不代表主办方私有评测。

## 2. 工程基础与可复现环境

### 2.1 本地工程

- 使用 `5070ti` 独立分支开发，不合并到 `main`；
- Windows 本地使用仓库外独立 Conda 环境，避免污染 Conda `base`；
- 锁定 Qwen3.5-2B revision、模型权重 SHA-256、Python 与依赖版本；
- 本地 CUDA 环境用于接口、精度、数据管线和无 PPU 单元测试；
- 模型、数据、密钥和大型 trace 不进入 Git。

### 2.2 PPU 环境闭环

在一张 PPU-ZW810E 上完成真实 Qwen3.5-2B 多模态推理：

- 617/617 个参数张量全部位于 `cuda:0` 暴露的 PPU；
- 无 CPU、meta 或 disk offload；
- 真实图片预处理、视觉主干、GDN/全注意力层和自回归生成全部通过；
- 实测 PPU PyTorch 为 `2.11.0+v0.1.0.ppu2.1.1`，SDK 位于
  `/usr/local/PPU_SDK`；
- 修复 RTC 路径未发现 `PPU_SDK/PPU_HOME` 导致的原生中止问题。

环境证据见 [首次 PPU 基线](../experiments/2026-08-26-ppu-baseline-and-gemv.md)。

### 2.3 新实例恢复

工程权威副本只保留在本机 Git 工作区和 GitHub `5070ti`。PPU 服务器是可销毁执行
节点，不把服务器/CPFS 中的代码、venv 或 `.so` 当作工程恢复来源。新实例流程：

```bash
cd /tmp
git clone --branch 5070ti --single-branch \
  https://github.com/mratgnothing/seu-vlm-ppu-optimization.git
cd seu-vlm-ppu-optimization
bash scripts/bootstrap_ppu_env.sh --check-only
bash scripts/bootstrap_ppu_env.sh
source scripts/activate_ppu_env.sh
source scripts/activate_ppu_profile.sh performance
# 如需保守复测四次原形状 GDN GEMV：
# source scripts/activate_ppu_profile.sh precision
# 只为复现被双语门禁拒绝的性能实验：
# source scripts/activate_ppu_profile.sh experimental-single
```

部署脚本创建仓库外 venv、复用官方镜像的 PPU 定制 Torch、安装不含 Torch 的用户态
依赖、重编译 GDN/acBLAS Linear/单入口 packed-MLP 扩展，并运行短 smoke。脚本会比较
部署前后 Torch 版本与加载路径，防止 PyPI/CUDA wheel 覆盖 PPU 运行时。

## 3. 初始基线与真正瓶颈

固定中文前 20 条、seed `20260625`、temperature 0、2 条预热、真实 Transformers
后端的稳态 eager 基线为：

| 指标 | 实测值 |
|---|---:|
| 平均 TTFT | `118.493 ms` |
| 平均解码吞吐 | `49.737 token/s` |
| Accuracy | `17/20 = 85%` |
| 公开输出校验 | 通过 |

可提交的聚合证据见 [eager CN20 基线汇总](../../results/ppu-eager-cn20-baseline-summary-20260826.json)；逐样本原始结果继续保留在 Git ignored `artifacts/` 中，不进入仓库。

同一真实样本 2-token 预热、16-token profile 显示：

| 指标 | eager |
|---|---:|
| Self CPU | `854.810 ms` |
| Self PPU | `173.799 ms` |
| `cudaLaunchKernel` | `37,293` 次 |
| `empty_strided` | `8,832` 次 |
| `gemvt_op` | `1,906` 次 / `29.343 ms` PPU |

主机调度时间远大于设备计算时间，说明主要矛盾不是单个 GEMV 算力，而是
Transformers eager 将 GDN、norm、RoPE、causal-conv 和状态更新拆成大量短 kernel，
同时产生 Python/ATen 调度、stream 查询、临时张量和频繁 launch。因此路线从“重写
通用 GEMV”转为“融合模型特有算子并扩大一次提交的语义边界”。

这个判断也由反例支持：自写 HGGC GEMV 相对仓库 reference 快 `1.88--2.08x`，但
三个 Qwen3.5 形状仍比 PPU `torch.mv` 慢 `16.7%--62.6%`，所以未接入模型。

## 4. 优化演进

### 4.1 五类 HGGC decode 融合

第一阶段实现并接入五类模型专用核：

1. **recurrent GDN**：融合 q/k L2Norm、state 衰减、`state·k`、delta、rank-1
   state update 和 `state·q`；
2. **causal-conv update**：融合 4-tap depthwise conv、状态移位、SiLU 和 BF16 输出；
3. **RMSNorm**：融合 cast、square、mean、epsilon、rsqrt、weight 和输出 cast；
4. **gated RMSNorm**：在 norm 上继续融合 gate 与 SiLU；
5. **Q/K RMSNorm + partial RoPE**：合并 full-attention 层的两次 norm 和多组
   neg/mul/add/cat。

CN20 同口径演进：

| 路径 | TTFT ms | token/s | 相对 eager | Accuracy | 答案差异 | token 数差异 |
|---|---:|---:|---:|---:|---:|---:|
| eager | 118.493 | 49.737 | - | 17/20 | - | - |
| recurrent GDN | 119.460 | 61.350 | +23.35% | 17/20 | 0/20 | 3/20 |
| GDN + causal-conv | 117.262 | 63.911 | +28.50% | 17/20 | 0/20 | 3/20 |
| all-four | 119.677 | 81.307 | +63.47% | 17/20 | 0/20 | 5/20 |
| all-five r1 | 124.930 | 93.918 | +88.83% | 17/20 | 0/20 | 5/20 |
| all-five r2 | 118.227 | 94.889 | +90.78% | 17/20 | 0/20 | 5/20 |

独立随机/真实模块 smoke 中，causal-conv、RMSNorm、gated RMSNorm 和 Q/K RoPE 均
bit-exact；GDN 最大 state/output 误差为 `5.96e-8/0`；五类调用路径均通过
`hggc-memcheck` 0 errors。自回归会放大归约顺序差异，因此 5/20 token 数漂移被
明确保留为限制，没有将 CN20 Accuracy 不变外推成完整集无损。

机制证据：all-five profile 的 Self CPU/PPU 降至 `409.545/121.871 ms`，相对 eager
明显减少 launch、cat 和临时张量；详见
[五类融合实验](../experiments/2026-08-26-ppu-fused-decode-kernels.md)。

### 4.2 MLP gate/up 权重打包

24 个 MLP 层的 gate/up 投影共享相同输入。将两份 `[6144,2048]` 权重组织为
`[12288,2048]` packed storage，decode 时把两次投影合为一次宽投影，prefill 回退
原路径。Parameter 只是 packed buffer 的 view，没有常驻第二份约 1.2 GiB 权重。

| 路径 | CN20 r1 | CN20 r2 | 相对 all-five |
|---|---:|---:|---:|
| all-five | 93.918 | 94.889 | - |
| + packed gate/up | 96.506 | 96.715 | +2.76% / +2.98% |

两轮相对 all-five 的答案、正确性和 token 数均 20/20 一致。Profile 中
`aten::linear/mm` 各减少 360 次，但底层 launch 数不变，说明收益来自减少
Python/ATen 调度和更高效的宽投影，而非“一次 mm 等于一次设备 kernel”。详见
[packed-MLP 图优化](../experiments/2026-08-27-ppu-packed-mlp.md)。

### 4.3 grouped acBLAS GDN 投影

18 个 GDN 层对同一 hidden state 连续执行 qkv/z/b/a 四个 GEMV。一次
PyTorch C++ extension 入口仍按原顺序提交四个原形状 acBLAS GEMV，从而合并
Python/ATen/pybind 调度且保留原 BF16 数值路径。

- 固定 128-token 六对：中位 `1.0121x`，3/6 获胜，全文一致；
- CN20 r1：中位 `1.0187x`，16/20 获胜，20/20 exact；
- CN20 r2：中位 `1.0391x`，17/20 获胜，20/20 exact；
- 两轮 Accuracy 均为 85%。

固定长结果只有 3/6 获胜，因此该候选仍由显式开关控制；不能仅凭 CN20 最快轮次
宣称稳定 4%。证据见 [结果索引](../../results/README.md#ppu-grouped-acblas-gdn-投影)。

### 4.4 48-edge residual-add + RMSNorm

首版只融合每层内部 24 条 attention residual→norm 边，固定 128-token 中位仅
`0.9821x`，属于失败版本。随后补齐 24 条 MLP residual→下一层 input norm/final norm，
形成 48 条跨层边：

- fixed-128 两轮中位 `1.0159x/1.0233x`；
- CN20 两轮中位 `1.0213x/1.0206x`；
- 两轮均 20/20 完整文本一致、Accuracy 85%；
- 16-token profile 中目标 `aten::add` 从 720 次降为 0，launch 减少 720 次。

这一步说明融合边界必须覆盖完整 producer→consumer 链，局部融合不一定产生端到端
收益。详见 [residual-RMSNorm 实验](../experiments/2026-08-27-ppu-residual-rmsnorm.md)。

### 4.5 GDN gate-prep

加载时缓存 18 层 FP32 `exp(A_log)`，一个 HGGC kernel 合并 Sigmoid、两个 cast、
add、Softplus、mul/neg，并用 thread-local scratch 复用 `g/beta`。

| 口径 | 结果 |
|---|---|
| fixed-128 六对 | 中位 `1.0839x`，6/6 获胜，全文一致 |
| CN20 r1/r2 | 中位 `1.0811x/1.0863x`，20/20 exact，Accuracy 85% |
| 中文完整 4029 | 中位 `1.0862x`，3882/4029 获胜 |
| 完整集精度 | 两路均 3374/4029，文本/答案/token 数 4029/4029 一致 |

Profile 的 launch `16253→14363`，Self CPU/PPU 分别下降 `11.48%/5.35%`，memcheck
为 0 errors。源数据：
[gate-prep 完整集汇总](../../results/gate-prep-scratch-cn-full4029-summary.json)；实验解释见
[gate-prep 实验](../experiments/2026-08-28-ppu-gdn-gate-prep.md)。

### 4.6 单入口 acBLAS packed-MLP

将一次完整 decode MLP 作为优化边界：一个 C++ extension 入口依次提交 packed
gate/up GEMV、bit-exact HGGC SwiGLU 和 down GEMV，并复用每层 scratch。

| 口径 | baseline | candidate | 配对中位 | 一致性 |
|---|---:|---:|---:|---|
| fixed-128 八对 | - | - | `1.1336x` | 8/8 获胜、全文一致 |
| CN20 r1 | 108.451 | 122.350 | `1.1212x` | 20/20 exact |
| CN20 r2 | 109.652 | 121.297 | `1.1122x` | 20/20 exact |
| 中文 4029 | 109.993 | 122.445 | `1.1125x` | 4029/4029 exact |
| 英文 4029 | 107.276 | 118.964 | `1.1093x` | 4029/4029 exact |

中文两路 Accuracy 均为 3374/4029，英文均为 3214/4029。Profile 中 GEMV 数量不变，
但 `aten::linear/mm` 各减少 720 次、launch 减少 360 次，证明收益来自扩大提交边界，
而不是替换底层矩阵算法。

证据：
[中文完整集](../../results/acblas-packed-mlp-cn-full4029-summary.json)、
[英文完整集](../../results/acblas-packed-mlp-en-full4029-summary-20260828.json)、
[实验说明](../experiments/2026-08-28-ppu-acblas-packed-mlp.md)。

### 4.7 raw stream 查询

原路径每次 ctypes 提交都会构造/查询
`torch.cuda.current_stream(...).cuda_stream`。在当前 PPU PyTorch 私有能力存在时，
改为直接查询 raw stream handle；能力不存在或用户未显式启用时立即回退/拒绝。

| 口径 | baseline | candidate | 配对中位 | 一致性 |
|---|---:|---:|---:|---|
| fixed-128 r1/r2 | - | - | `1.1055x/1.0961x` | 全文一致 |
| CN20 r1 | 119.715 | 130.264 | `1.1026x` | 20/20 exact |
| CN20 r2 | 121.953 | 133.136 | `1.0855x` | 20/20 exact |
| 中文 4029 | 120.383 | 131.107 | `1.0906x` | 4029/4029 exact |
| 英文 4029 | 118.577 | 129.398 | `1.0901x` | 4029/4029 exact |

中英文完整集 Accuracy 分别保持 3374/4029 和 3214/4029；memcheck、profile 和
正式 wrapper smoke 均通过。完整集 TTFT 分别为 `119.722→119.867 ms` 和
`118.911→118.950 ms`，因此该优化只宣称解码吞吐收益。

证据：
[中文完整集](../../results/raw-stream-query-cn4029-summary-20260828.json)、
[英文完整集](../../results/raw-stream-query-en4029-summary-20260828.json)、
[实验说明](../experiments/2026-08-28-ppu-raw-stream-query.md)。

### 4.8 最终日 GDN GEMV 收口

资源窗口最后继续调查 GDN 四投影：

- 将 qkv/z/b/a 合为一次 8224-row GEMV，每 token 将 GDN GEMV 从 72 次降到
  18 次；中文 4029 的正确数不变，但英文 4029 少答对 1 题，因此降级为
  `experimental_only`；
- 只将相邻 b/a 两个 `[16,2048]` 合为 `[32,2048]` GEMV，每 token 将完整模型
  GEMV 从 120 次降到 102 次。中文 4029 条中 baseline/candidate 正确数均为
  3374，全文、答案和 token 数均 4029/4029 一致，成对中位/均值
  `1.00696x/1.00814x`；英文对应正确数均为 3214，三类一致性均 4029/4029，
  成对中位/均值 `1.00697x/1.01243x`；
- 固定 128-token 最终复测 2/2 获胜、全文一致，中位 `1.01990x`。16-token profile
  中 `cuLaunchKernel` `2561→2291`，恰好减少 `18 层×15 decode step=270` 次
  acBLAS 提交；540 个小 b/a `gemvt` kernel 被 216 个合并后端 kernel 取代，设备
  kernel 净减 324 个；
- 两个独立 ABBA block、每臂四次的总栈复测为 eager
  `49.3415 token/s`、b/a 性能栈 `132.1895 token/s`，即 `2.67907x`、提升
  `167.91%`。8 次 Accuracy 均为 85%，20/20 解析答案和正确性一致。候选首个
  block 中一次只有 `128.970 token/s` 的低值未剔除，而是通过四次中位吸收。

证据：
[single-GEMV CN100](../../results/acblas-gdn-single-gemv-cn100-r1-20260828.json)、
[b/a-GEMV CN100](../../results/acblas-gdn-ba-gemv-cn100-r1-20260829.json)、
[中文 4029](../../results/acblas-gdn-ba-gemv-cn-full4029-summary-20260831.json)、
[英文 4029](../../results/acblas-gdn-ba-gemv-en-full4029-summary-20260831.json)、
[最终总栈](../../results/ppu-ba-gemv-vs-eager-cn20-abba-20260831.json)、
[最终 profile](../../results/ppu-ba-gemv-profile-20260831.json)和
[运行时瓶颈实验](../experiments/2026-08-28-ppu-acblas-runtime-overhead.md)。

## 5. 当前性能应该怎样表述

### 5.1 可以正式引用

- 原始 eager 与最终 b/a-GEMV 性能栈在同一 PPU 实例跑了两个独立 ABBA block：
  CN20 每臂四次吞吐中位 `49.3415→132.1895 token/s`，即 `2.67907x`、提升
  `167.91%`；8 次 Accuracy 均为 85%，20/20 解析答案和正确性一致。公开入口不保存
  全文哈希，因此该轮不宣称全文 bit-exact；证据见
  [最终总加速](../../results/ppu-ba-gemv-vs-eager-cn20-abba-20260831.json)；
- 不启用 b/a-GEMV 的 `precision` 栈也完成过一次独立进程 ABBA：两次吞吐中位
  `49.445→132.4265 token/s`，即 `2.6783x`、提升 `167.83%`。该短样本总栈数字与
  performance 档的差异小于运行间波动；b/a 的独立增量应引用双语完整集 paired
  `1.00696x/1.00697x`，而不是相减两次 CN20 总加速。证据见
  [精度栈总加速](../../results/ppu-total-stack-vs-eager-cn20-abba-20260831.json)；
- single-GEMV 完成中文 4029 条门禁：两路 Accuracy 均为 3374/4029，答案
  解析结果 4029/4029 一致，成对吞吐中位/均值 `1.0238x/1.0253x`；由于全文仅
  3873/4029 一致，中文范围内只算 accuracy-budget 候选。该档相对 eager 的独立进程
  CN20 ABBA 总加速为 `2.7689x`（`+176.89%`），证据见
  [中文全量](../../results/acblas-gdn-single-gemv-cn-full4029-summary-20260831.json)和
  [直接总加速](../../results/ppu-single-gemv-vs-eager-cn20-abba-20260831.json)。英文
  4029 则出现正确数 `3214→3213`、答案 4028/4029 一致，故双语门禁最终将它降级为
  `experimental_only`，不得进入比赛推荐配置；参见
  [英文全量](../../results/acblas-gdn-single-gemv-en-full4029-summary-20260831.json)和
  [双语决策](../../results/acblas-gdn-single-gemv-bilingual-decision-20260831.json)；
- 五类融合相对初始 eager 的 CN20 吞吐提升：`88.83%/90.78%`；
- gate-prep 相对前一优化栈的中文完整集配对中位提升：`8.62%`；
- 单入口 packed-MLP 相对前一优化栈的中英文完整集提升：`11.25%/10.93%`；
- raw-stream 相对前一优化栈的中英文完整集提升：`9.06%/9.01%`；
- b/a-GEMV 相对精度栈的中英文完整集成对中位提升：`0.696%/0.697%`，两种语言
  都是 4029/4029 全文、答案和 token 数一致，Accuracy 不变；
- 上述完整集增量均保持对应 baseline/candidate 的 Accuracy、完整文本、答案和 token
  数一致。

### 5.2 只能作为演进参考

旧口径把不同阶段的 eager `49.737 token/s` 与后期 CN20 `130.264/133.136 token/s`
直接相除，只能说明约 `2.62x/2.68x` 的量级，现已由同环境、每臂四次独立进程 ABBA
的 `2.67907x` 直接实验替代。不同后期增量的中位 speedup 仍不能简单相乘后当作总加速。

### 5.3 当前不能宣称

- 不能把 CN20 的直接 `2.67907x` 外推成 4029 条或私有集总加速；
- 不能宣称 TTFT 显著提升；
- 不能把模块级 `4x` 或微基准 `7x` 写成整模提升；
- 不能把公开开发集结果外推为私有测试集成绩；
- 不能把双语 4029 paired 的 b/a-GEMV `0.696%/0.697%` 误写成总栈相对 eager
  的额外百分点；两类实验的 baseline 和统计口径不同。

## 6. 保留的失败方向与止损依据

失败结果被保留，因为它们解释了 PPU 上“局部快不等于整模快”的边界，并防止重复
走弯路。

| 方向 | 局部结果 | 整模/最终结果 | 决策与原因 |
|---|---:|---:|---|
| 自写 HGGC GEMV | 比 reference `1.88--2.08x` | 比 `torch.mv` 慢 `16.7%--62.6%` | 不替换成熟通用 GEMV |
| 通用 acBLAS Linear | 模块级约 `1.08--1.17x` | fixed-128 中位 `0.9997x` | 默认关闭，调度收益不稳定 |
| 独立 SwiGLU HGGC | bit-exact | 最优 `0.7901x` | launch 成本超过少一次 elementwise |
| MLP in-place SiLU/mul | 局部约 +5.74% | CN20 仅 +0.12%/+0.16% | 低于稳定保留门限，回退 |
| acBLASLt 方阵 | 模块级 `1.2797x` | fixed-128 `0.9898x`、3/8 胜 | 后端热点增多，停止 |
| Attention Prep | 模块级 `4.1006x` | CN20 `1.0158x/0.9852x` | 第二轮门禁失败 |
| Graph Capture | 16 段链 `1.8303x` | packed-MLP 含 copy `0.9316x` | 动态输入 copy 抵消 replay 收益 |
| residual-RMSNorm scratch | 模块级 `1.3373x` | fixed-128 `0.9862x`、2/8 胜 | 未进入 CN20/profile |
| 24-edge residual 融合 | 语义正确 | fixed-128 `0.9821x` | 扩到完整 48-edge 才晋级 |
| 单次 8224-row GDN GEMV | CN/EN4029 中位 `1.0238x/1.0253x` | 中文 Accuracy 不变；英文少 1 个正确答案 | 双语门禁失败，降级为 experimental-only |
| acBLAS persistent workspace | 可复用工作区 | fixed-128 中位 `0.9846x` | 查询/释放/kernel 数未消除，停止 |
| b/a strided-batched GEMM | host API 更少 | fixed-128 约 `0.9852x` | 未减少设备 kernel，停止 |
| GDN output scratch | 避免重复输出分配 | 两轮 `1.0105x/0.9934x` | 方向不稳定，不晋级 |
| GemmEx-for-GemvEx / 算法枚举 | API/算法可切换 | 最佳约 `1.0036x` | 运行时事件未变，收益低于门限 |
| qkv 保留、仅合并 z/b/a | 比 single-GEMV 保守 | 同一样本仍发生确定性 token 漂移 | 未解决数值问题，停止 |
| A16W8/A16W4 | 理论可减权重带宽 | 当前镜像缺官方 acext/PPU-vLLM weight-only 依赖 | 未伪造本地量化结果，留作平台依赖方向 |

主要证据入口：

- [SwiGLU 负实验](../../results/ppu-swiglu-thread-sweep-negative-20260827.json)
- [通用 acBLAS Linear](../../results/ppu-acblas-ab128-final-20260827.json)
- [acBLASLt 整模 A/B](../../results/acblaslt-square-ab128-20260828.json)
- [Attention Prep r1](../../results/acblas-attention-prep-cn20-r1-20260828.json) /
  [r2](../../results/acblas-attention-prep-cn20-r2-20260828.json)
- [Graph 能力](../../results/ppu-graph-capture-probe-20260828.json) /
  [真实 packed-MLP](../../results/acblas-packed-mlp-graph-smoke-20260828.json)
- [residual scratch A/B](../../results/residual-rmsnorm-scratch-ab128-20260828.json)

## 7. 实验门禁与可追溯性

正式候选按以下顺序晋级：

```text
随机/模块精度
  → decode + prefill 回退
  → 不同 stride/stream/输入地址
  → hggc-memcheck 0 errors
  → 正式 wrapper 与模块挂载计数
  → fixed-128 AB/BA
  → CN20/EN20 两轮
  → 中英文 4029 paired
```

每份长测汇总至少记录：样本数、baseline/candidate 吞吐、Accuracy、配对中位与分位数、
wins/losses、exact text、same answer、same token count、模块挂载数和 `passed`。部分完整集
汇总还保存原始文件和 pair-log SHA-256。原始巨型 trace/逐题 JSONL 由于大小和数据规则
只保存在本机 ignored 快照；可公开的小型汇总、哈希、代码和实验说明进入 GitHub。

自动化复核入口：

```bash
python -m unittest tests.test_work_summary_evidence -v
python -m unittest discover -s tests -v
```

## 8. 代码与证据地图

| 内容 | 路径 |
|---|---|
| 正式模型入口与显式开关 | `evaluation_wrapper.py` |
| HGGC/acBLAS 扩展与 smoke | `ppu/custom_ops/` |
| paired A/B、profile、完整集脚本 | `scripts/` |
| 可公开结果与负实验 | `results/` |
| 每轮实验说明 | `docs/experiments/` |
| 环境恢复 | `scripts/bootstrap_ppu_env.sh` |
| 精度/性能档选择 | `scripts/activate_ppu_profile.sh` |
| 部署/释放手册 | `docs/ppu-resource-release-handoff.md` |
| 本文数字自动核验 | `tests/test_work_summary_evidence.py` |

主要实现提交按演进顺序包括：

```text
d60c797  PPU 真实模型闭环与 profile
149ae8f  五类 decode 融合核
3108b64  MLP gate/up 打包
88cf988  grouped acBLAS GDN
a71108f  residual-RMSNorm
f47fa9d  GDN gate-prep
cdd253e  单入口 packed-MLP
bd82f1c  raw-stream 查询
344fe50  GDN single-GEMV 候选
daa31cb  GDN b/a-GEMV 候选
e1df537  paired 长测可恢复
ccf81e2  新 PPU 镜像环境一键恢复
```

## 9. 当前限制与下一步

1. 在新官方 PPU 镜像上实际执行一次 `bootstrap_ppu_env.sh` 全流程；目前只完成本地
   Bash 语法、帮助入口和无 PPU 测试，不能提前宣称新实例恢复已上机通过。
2. 原始 eager 与最终性能栈的 CN20 两个独立 ABBA block 已完成；若资源允许，再扩大
   总栈直接对比到完整集，但不能把当前 CN20 总加速外推。
3. 当前 profile 已复现 120 次小 BF16 GEMV/token 的主矛盾；减少 54 次 GDN
   GEMV/token 的 single-GEMV 虽通过中文门禁，却因英文 4029 少 1 个正确答案而止损。
   b/a-GEMV 只减少 18 次/token、保持更多原始 BF16 路径，现已通过双语全量门禁。
4. b/a-GEMV 已作为显式 `performance` 档；中英文均 4029/4029 exact 且性能为正。
   `precision` 档仍保留原四 GEMV，`experimental-single` 只用于负实验复现。
5. 所有候选继续保持显式 opt-in；主办方镜像、SDK、模型 revision 或评测入口变化时，
   必须重新跑环境、精度和性能门禁。

## 10. 总结

本项目的核心工作不是把 CUDA 代码机械翻译到 PPU，而是先通过真实模型 profile 确认
Qwen3.5 在 PPU Transformers eager 路径上受到大量小 kernel、临时张量和主机调度限制，
再围绕 GDN、causal-conv、norm、RoPE、MLP 和 stream 提交逐步扩大融合边界。早期五类
融合获得接近 `1.9x` 的 CN20 吞吐，后期通过结构化 C++ extension 和运行时路径继续
获得经完整集验证的 `8%--11%` 级增量。同时保留所有局部快、整模慢的负实验，以实验
门禁而不是直觉决定是否接入。

当前最可信的结论是：最终 performance 栈中的 b/a-GEMV 在中英文 4029 条上均保持
对应 A/B 的 Accuracy、完整文本、答案和 token 数一致，成对中位分别为
`1.00696x/1.00697x`；同环境 CN20、每臂四次的直接总加速为 `2.67907x`，Accuracy、
解析答案和正确性不变。该总加速数字的证据边界是 CN20，完整集和私有集不能由此外推。
最终日已围绕 GEMV 数量完成一次核心迭代，同时通过英文反例拒绝更激进但掉精度的路线。
