# 最终日：完整栈总加速复测与 GEMV 主矛盾迭代

日期：2026-08-31

## 1. 本轮目标

本轮先纠正“把不同日期、不同阶段的吞吐相除”这一不够严格的总提升口径，再只围绕
当前 profile 的最大剩余矛盾继续迭代。所有正式数字必须来自同一 PPU 实例、同一模型、
同一数据、相同 seed/预热和明确的 A/B 顺序；模块微基准只用于定位，不代替整模结果。

## 2. 原始 eager 与当前精度优先完整栈

### 2.1 方法

- 设备：PPU-ZW810E，Driver `2.1.0-ra1f23`，HGGC 13.0；
- 模型：Qwen3.5-2B，Transformers 后端，设备自动选择；
- 数据：MMBench Dev CN 前 20 条；
- 每次运行预热 2 条，运行顺序为 `eager A → candidate A → candidate B → eager B`；
- 四次均为独立 Python 进程，避免上一 arm 的模型 patch、allocator 和缓存污染下一 arm；
- 解码吞吐沿用公开入口口径：`(token_count - 1) / (elapsed - TTFT)`。

当前精度优先完整栈实际挂载：18 GDN、18 causal-conv、49 RMSNorm、18 gated
RMSNorm、6 q/k RMSNorm+RoPE、24 packed gate/up、18 grouped-acBLAS GDN、24
residual-RMSNorm、18 GDN gate-prep、24 单入口 acBLAS packed-MLP，并启用 raw-stream
查询；single-GEMV、b/a-GEMV 和 Attention Prep 均关闭。

### 2.2 结果

| arm | A (token/s) | B (token/s) | 两次中位 (token/s) | Accuracy |
|---|---:|---:|---:|---:|
| 原始 eager | 49.308 | 49.582 | 49.445 | 85% / 85% |
| 当前完整栈 | 132.337 | 132.516 | 132.4265 | 85% / 85% |

直接总加速为：

```text
132.4265 / 49.445 = 2.6782587x
(2.6782587 - 1) × 100% = 167.8259%
```

按 A/B 两次对应相除为 `2.6839x/2.6727x`；逐样本中位/均值为
`2.6810x/2.6805x`，20/20 样本均更快。四次 Accuracy、解析答案和正确性逐题一致。
公开 benchmark JSON 不保存全文哈希，因此这里不能声称四次全文 bit-exact；token 数在
15/20 样本上四次一致，反映早期融合栈已经记录过的生成长度漂移。TTFT 两次中位只从
118.9145 ms 变为 118.8100 ms，下降约 0.088%，应表述为基本持平。

可提交的聚合证据是
[`results/ppu-total-stack-vs-eager-cn20-abba-20260831.json`](../../results/ppu-total-stack-vs-eager-cn20-abba-20260831.json)；
四份原始 JSON 及 SHA-256 保存在本机 ignored `artifacts/ppu-total-ab-20260831/`。

## 3. 当前完整栈 profile 与主要矛盾

重新运行 16-token profile 后，当前完整栈仍有：

| 事件 | 数量 |
|---|---:|
| `cudaLaunchKernel` | 14,003 |
| `cuLaunchKernel` | 2,561 |
| `cudaGetDeviceProperties_v2` | 5,705 |
| `cudaFree` | 3,259 |
| acBLAS GEMV kernel | 2,446 |

16-token 生成包含 1 次 prefill 和 15 个 decode step。结合模块结构与事件邻接关系，每个
decode token 仍有 120 次小 BF16 GEMV：18 层 GDN 每层 qkv/z/b/a 四次，共 72 次；
24 层 MLP 每层 gate-up/down 两次，共 48 次。它们既读取大量权重，也重复进入 acBLAS
运行时。当前不应再新增只有几十微秒的独立 elementwise kernel；可完成的核心方向是减少
GEMV 数量和权重流量。

聚合 profile 见
[`results/ppu-current-stack-profile-20260831.json`](../../results/ppu-current-stack-profile-20260831.json)。
原始 trace 的 SHA-256 为
`8bc8169063b7e437c7a09af5ab31fe6619b4e3a64f3814cd361110aecbe75c95`，保存在本机
ignored `artifacts/ppu-current-profile-20260831/`。

## 4. 最终日候选：GDN 四投影单 GEMV

GDN 四个投影共享同一个输入，且权重已按 qkv/z/b/a 连续存为 `[8224, 2048]`。性能档
用一次 `8224×2048` GEMV 代替每层四次独立 GEMV，使每 token 的 GDN GEMV 从 72 次
降到 18 次，即总 GEMV 从 120 次降到 66 次，减少 54 次/token。

该变换数学表达等价，但矩阵行数改变后 acBLAS 会选择不同 tile 和 BF16 归约顺序，故
不承诺全文 bit-exact。此前 CN100 已得到 Accuracy `93%→93%`、答案 100/100 一致、
全文 99/100 一致和成对中位 `1.0261x`。本轮不再凭 CN100 外推，直接运行中文 4029
条逐样本 AB/BA，全程使用 append-only JSONL 检查点并支持断线恢复。

最终结果完成后，以以下门禁决定归属：

1. 若 Accuracy 或解析答案回退，继续保持实验开关，不进入提交配置；
2. 若 Accuracy/答案守住且性能门禁通过，作为显式 **performance/accuracy-budget** 档；
3. 无论结果如何，四-GEMV 精度优先档继续保留，不能把性能档描述为 bit-exact。

## 5. 已止损、不会在最终日重跑的方向

- workspace：不减少设备查询、释放或 kernel 数，整模中位 `0.9846x`；
- b/a strided-batched GEMM：减少 host API 但不减少 kernel，整模约 `0.9852x`；
- GDN output scratch：两轮结果 `1.0105x/0.9934x`，方向不稳定；
- GemmEx-for-GemvEx 与算法枚举：运行时事件不变，最佳仅约 `1.0036x`；
- Attention Prep、Graph Capture、acBLASLt 方阵等：均已有整模负门禁，不因最后一天
  再挑样本重试；
- A16W8/A16W4：官方 acext/PPU-vLLM weight-only 依赖不在当前镜像，不能在截止日前可靠
  补齐，因此只保留为后续平台依赖方向。

