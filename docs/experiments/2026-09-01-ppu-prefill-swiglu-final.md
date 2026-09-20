# 最后一轮：MLP prefill SwiGLU 融合

## 目标与门槛

在已通过的 multi-row norm/residual prefill 路径上继续优化 MLP，Accuracy 和解析答案
不得下降，允许生成吞吐最多下降 5%。所有数据直接来自 PPU-ZW810E，没有在 Windows
本地执行性能实验。

## 版本一：保留两次投影，只融合激活与乘法

24 个语言 MLP 在 prefill 中仍分别执行 `gate_proj` 和 `up_proj`，候选仅用一个 HGGC
kernel 替代 `SiLU(gate)` 与 `* up` 两个 eager kernel。

| 数据 | TTFT baseline→candidate | TTFT 配对中位 | 吞吐配对中位 | Accuracy | 答案/全文一致 |
|---|---:|---:|---:|---:|---:|
| CN20 | 114.238→113.689 ms | 1.00331x | 1.00633x | 17→17 | 20/20、20/20 |
| EN20 | 114.932→119.674 ms | 0.97769x | 0.96744x | 18→18 | 20/20、20/20 |

中文收益只有噪声量级，英文 TTFT 明确回退。一次 elementwise launch 的节省不足以覆盖
24 次 Python/ctypes 提交，因此拒绝。

## 版本二：宽 gate/up GEMM + packed SwiGLU

把两个 `[rows,2048]×[2048,6144]` 投影改为一个
`[rows,2048]×[2048,12288]` 投影，再由按行读取交错 gate/up 的 HGGC kernel 计算
SwiGLU。CN2 冒烟中 Accuracy、答案和全文均一致，但 TTFT 配对中位为 `0.96925x`，
吞吐为 `0.94718x`，0/2 TTFT 获胜。它已经没有扩大到双语 20 题的必要。

## 决策

- 两种 SwiGLU 候选均不进入正式 `performance`；
- 正式配置采用上一轮 multi-row norm/residual prefill 融合，其 CN20/EN20 TTFT 中位
  分别提升 4.86%/4.48%，Accuracy 与解析答案不变；
- 两个 prefill SwiGLU 候选均已从最终源代码撤回，只保留聚合结果和失败原因；
- 原始逐题结果保存在本机 ignored
  `artifacts/ppu-prefill-swiglu-final-20260901/`。
