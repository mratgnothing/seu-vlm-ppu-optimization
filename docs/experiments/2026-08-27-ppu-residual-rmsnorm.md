# 2026-08-27 PPU residual-add + RMSNorm 跨层融合

## 目标与约束

本轮不使用公开集标签选择题目、阈值或分支，只依据 Qwen3.5-2B 的固定解码图继续减少
小张量 launch。候选建立在已通过精度门禁的 all-five + packed-MLP + grouped-acBLAS
GDN 路径上；权重、提示词、采样策略和评测解析均不改变，且默认关闭。

Qwen3.5 的每个 decoder layer 都有两条 residual 边：

```text
x --input_norm--> mixer --------+--> post_attention_norm --> MLP --+
|                               |                                  |
+-------------------------------+                                  |
                                                                   |
+------------------------------------------------------------------+
```

更精确地，原始 eager 路径每层执行：

```text
h1 = bf16(x + mixer(input_norm(x)))
n1 = RMSNorm(h1)
h2 = bf16(h1 + MLP(n1))
n2 = next_layer.input_norm(h2)  # 最后一层连接最终 language-model norm
```

24 层因此有 48 条 `residual add -> RMSNorm` 相邻边。两者之间没有其他消费者时，可在
一个 HGGC kernel 内计算 BF16 舍入后的 residual sum、FP32 平方和归约、`rsqrt`、
weight scaling，并把 residual sum 原位写回 branch buffer。数学量和 BF16 舍入点保持
与原路径一致；prefill、非 BF16、非 PPU 或形状不匹配时回退原始 forward。

## 实现

- `gdn_recurrent_ppu.hg` 增加 `residual_rmsnorm_decode_kernel` 和稳定 C ABI；
- `ppu_gdn.py` 增加输入契约校验、ctypes 包装和 decoder graph patch；
- 每层第一条边直接融合 `attention residual + post_attention_layernorm`；
- 第二条边把归一化结果通过 thread-local、输入对象身份校验的缓存交给下一层
  `input_layernorm`，最后一层交给模型最终 norm，因此覆盖全部 48 条边；
- 开关关闭时恢复原 forward 并清空缓存，不改变默认行为；
- `evaluation_wrapper.py` 只在 `SEU_PPU_RESIDUAL_RMSNORM_ENABLE=1` 时挂载；
- 新增随机 BF16 单算子 smoke、固定长 AB/BA、CN20 交错 AB/BA 和 trace 汇总测试。

## 单算子验收

PPU-ZW810E 随机 BF16 `[1,1,2048]`：

| 项目 | 结果 |
|---|---:|
| eager add + RMSNorm | 0.017997 ms |
| fused kernel | 0.015732 ms |
| 单算子加速 | 1.1440x |
| residual 输出 | bit-exact |
| normalized 输出 | bit-exact |
| branch buffer 原位复用 | 通过 |

## 从 24-edge 负实验到 48-edge 链

第一版只融合每层内部的 attention residual，共 24 条边。固定 128-token 六对虽然全文
一致，但候选中位吞吐 `101.845 -> 100.275 token/s`，配对中位 `0.9821x`、仅 1/6
获胜。旧 profile 只减少 360 次 residual add/launch，Python wrapper 与额外 kernel
开销抵消了收益，因此没有据此宣称优化成功。

补齐 MLP residual 到下一层 input norm/最终 norm 的另外 24 条边后，固定长两轮为：

| 轮次 | baseline token/s | candidate token/s | 配对中位/均值 | 获胜 | 全文一致 |
|---|---:|---:|---:|---:|---:|
| r3 | 97.865 | 99.552 | 1.0159x / 1.0146x | 4/6 | 6/6 |
| r4 | 101.475 | 102.585 | 1.0233x / 1.0147x | 4/6 | 6/6 |

两轮方向一致，但样本数仍小，固定长结果只作为趋势证据。

## CN20 交错配对复测

两轮均使用同一模型、seed、样本顺序和 grouped-acBLAS 对照，pair 顺序交替 AB/BA：

| 轮次 | baseline token/s | candidate token/s | 配对中位/均值 | 获胜 | Accuracy | 全文一致 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 100.156 | 101.616 | 1.0213x / 1.0154x | 14/20 | 85% / 85% | 20/20 |
| r2 | 98.576 | 101.507 | 1.0206x / 1.0311x | 14/20 | 85% / 85% | 20/20 |

这里的 85% 只表示候选没有改变这 20 条的答案；它不是完整公开集或私有集无损证明。

## 结构 Profile

16-token generate 的 profiler 中，第一 token 来自 prefill，实际有 15 个 decode step。
因此 `48 edges × 15 = 720`：

| 事件 | baseline | candidate | 变化 |
|---|---:|---:|---:|
| `[1,1,2048]` `aten::add` | 720 | 0 | -720 |
| 全部 `aten::add` | 2686 | 1966 | -720 |
| `cudaLaunchKernel` | 16973 | 16253 | -720 |
| 原 RMSNorm kernel | 735 | 15（由总结构推得） | -720 |
| fused residual-RMSNorm kernel | 0 | 720 | +720 |

这说明每条边由原来的 add kernel + norm kernel 变成一个 fused kernel，数学工作量近似
不变，但设备 launch 减少 720 次。Profiler 单次带插桩吞吐为负且不作为性能证据；性能
结论只采用前述无 profiler 的交错重复结果。

## 正式入口与当前决策

正式 `benchmark_public.py` 单样本 smoke 已通过真实 Transformers/PPU 后端和公开校验。
GDN/conv/RMSNorm/gated-RMSNorm/qk-RoPE/packed-MLP/grouped-GDN 计数为
`18/18/49/18/6/24/18`，residual-RMSNorm decoder 计数为 24，backend 为
`acblas-grouped`。

当前保留 `SEU_PPU_RESIDUAL_RMSNORM_ENABLE=1` 作为显式 opt-in 候选，不改默认配置。
下一门禁是完整公开集逐样本答案、文本哈希和 Accuracy；之后再做主办方私有集和最终
镜像复测。公开标签只用于回归验证，不参与 kernel 或图结构设计。

## 证据文件

- `results/ppu-residual-rmsnorm-ab128-20260827.json`：24-edge 负实验；
- `results/ppu-residual-rmsnorm-ab128-chain48-r3-20260827.json`；
- `results/ppu-residual-rmsnorm-ab128-chain48-r4-20260827.json`；
- `results/ppu-residual-rmsnorm-cn20-chain48-r1-20260827.json`；
- `results/ppu-residual-rmsnorm-cn20-chain48-r2-20260827.json`；
- `results/ppu-residual-rmsnorm-profile-chain48-ab-20260827.json`；
- `results/ppu-residual-rmsnorm-profile-chain48-baseline-summary-20260827.json`；
- `results/ppu-residual-rmsnorm-profile-chain48-candidate-summary-20260827.json`；
- `results/ppu-residual-rmsnorm-profile-chain48-gdn-shapes-20260827.json`；
- `results/ppu-residual-rmsnorm-formal-wrapper-chain48-smoke-20260827.json`。
