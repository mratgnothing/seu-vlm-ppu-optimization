# 结果目录

- `results/raw/`：逐样本输出和原始日志，默认被 Git 忽略。
- 可提交到仓库的内容：不含敏感数据的小型汇总表、图表数据和复现实验说明。

所有结论应能追溯到原始结果、配置和代码提交。

## 当前可公开精度结果

- 中文固定前 20 条：Accuracy 85%，公开校验通过。
- 英文固定前 20 条：Accuracy 80%，公开校验通过。
- 中文比例分层 200 条：Accuracy 84.5%（169/200），20 个类别全部覆盖，公开校验通过。
- 英文比例分层 200 条：Accuracy 82.5%（165/200），20 个类别全部覆盖，公开校验通过。
- 中文完整公开集 4029 条：Accuracy 83.94%（3382/4029），21/21 分块和 4029 个唯一题目 ID 完整，公开校验通过。
- 英文完整公开集 4029 条：Accuracy 79.75%（3213/4029），21/21 分块和 4029 个唯一题目 ID 完整，公开校验通过。

英文原始全量运行有一个在输出最终答案前达到 256-token 上限的样本，原始 Accuracy
为 79.72%（3212/4029），公开校验失败 1 条。通用结论规范化经整个 200 条异常分块
复测，只将这一条从空值恢复为 C，其余 199 条答案和全部 token 数保持不变。

## 当前正式本地性能

M1 使用首个生成 token 作为 TTFT 终点。

环境：RTX 4050 Laptop GPU 6GB、BF16、batch size 1、公开集固定前 20 条、2 条预热。

| 数据/版本 | Accuracy | Avg TTFT | Throughput | 校验 |
|---|---:|---:|---:|---|
| 中文 O0 三次中位数 | 85% | 313.562 ms | 21.328 tokens/s | 三次均通过 |
| 中文 O1 三次中位数 | 85% | 287.706 ms | 23.209 tokens/s | 三次均通过 |
| 英文 O0 三次中位数 | 80% | 300.105 ms | 22.678 tokens/s | 三次均通过 |
| 英文 O1 三次中位数 | 80% | 269.424 ms | 24.123 tokens/s | 三次均通过 |

O1 相对中文 O0 的正式中位提升为：TTFT 8.25%，吞吐 8.82%。
英文三次中位数的 TTFT/吞吐提升为 10.22%/6.37%。中英文逐样本答案和 token 数均无变化。

旧文本块计时结果只保留在实验历史中，不进入当前正式性能表。

性能表用于本地工程管线验证，不代表完整公开集速度或主办方私有评测成绩。完整集
Accuracy 采用单次分块运行，不能替代三次性能统计。逐样本原始结果保留在本地忽略目录，
不提交到仓库。

## PPU 注册式 acBLAS Linear

PPU-ZW810E 上，最终单 `.so` + 进程级 handle/mutex 版本虽然随机 BF16 模块级达到
`1.08--1.17x`，但固定 128-token 八对 AB/BA 的成对中位仅 `0.9997x`、4/8 获胜；
CN20 成对中位 `1.0164x`、12/20 获胜。两组均保持完整文本一致，但性能不能稳定
复现，因此不接入正式 wrapper：

- [`ppu-acblas-ab128-final-20260827.json`](ppu-acblas-ab128-final-20260827.json)
- [`ppu-acblas-cn20-final-20260827.json`](ppu-acblas-cn20-final-20260827.json)

## PPU Qwen3.5 GDN 输入投影打包

最终线程隔离版的四投影一次完成在固定 128-token 四对中 4/4 获胜、全文一致，成对
中位 `1.0182x`。CN20 平均吞吐 `94.099→98.430 tokens/s`、成对中位
`1.0355x`、Accuracy 均为
85%，但 19/20 全文一致，唯一差异为相同答案下多生成 1 token。因此它保持默认关闭：

- [`ppu-packed-gdn-ab128-20260827.json`](ppu-packed-gdn-ab128-20260827.json)
- [`ppu-packed-gdn-cn20-20260827.json`](ppu-packed-gdn-cn20-20260827.json)
- [`ppu-packed-gdn-profile-ab-20260827.json`](ppu-packed-gdn-profile-ab-20260827.json)
- [`ppu-packed-gdn-profile-baseline-summary-20260827.json`](ppu-packed-gdn-profile-baseline-summary-20260827.json)
- [`ppu-packed-gdn-profile-candidate-summary-20260827.json`](ppu-packed-gdn-profile-candidate-summary-20260827.json)

只融合 qkv+z 的 `(2,1,1)` 分组可恢复 CN20 20/20 全文一致，但成对中位
`0.9884x`，不具备性能价值：
[`ppu-packed-gdn-exact-cn20-20260827.json`](ppu-packed-gdn-exact-cn20-20260827.json)。

## PPU grouped acBLAS GDN 投影

结构专用 grouped acBLAS 在一次 PyTorch C++ extension 入口内仍按原顺序提交
qkv/z/b/a 四个原形状 GEMV，避免一次 8224 行 GEMV 带来的 BF16 数值路径变化。
固定 128-token 六对成对中位 `1.0121x`、3/6 获胜且全文一致。CN20 两轮分别为：

- `96.409→98.028 tokens/s`，成对中位 `1.0187x`，16/20 获胜；
- `95.634→99.601 tokens/s`，成对中位 `1.0391x`，17/20 获胜。

两轮 Accuracy 均为 85%，并且都是 20/20 全文一致。Profile 中
`aten::linear/mm` 各减少 1080 次，但 `gemvt_op` 和 `cudaLaunchKernel` 数不变，
证明收益来自主机调度合并。当前仍是默认关闭的精度优先候选：

- [`ppu-acblas-grouped-gdn-ab128-20260827.json`](ppu-acblas-grouped-gdn-ab128-20260827.json)
- [`ppu-acblas-grouped-gdn-cn20-r1-20260827.json`](ppu-acblas-grouped-gdn-cn20-r1-20260827.json)
- [`ppu-acblas-grouped-gdn-cn20-r2-20260827.json`](ppu-acblas-grouped-gdn-cn20-r2-20260827.json)
- [`ppu-acblas-grouped-gdn-profile-ab-20260827.json`](ppu-acblas-grouped-gdn-profile-ab-20260827.json)
- [`ppu-acblas-grouped-gdn-profile-baseline-summary-20260827.json`](ppu-acblas-grouped-gdn-profile-baseline-summary-20260827.json)
- [`ppu-acblas-grouped-gdn-profile-candidate-summary-20260827.json`](ppu-acblas-grouped-gdn-profile-candidate-summary-20260827.json)
- [`ppu-acblas-grouped-gdn-profile-candidate-shapes-20260827.json`](ppu-acblas-grouped-gdn-profile-candidate-shapes-20260827.json)
- [`ppu-acblas-grouped-gdn-formal-wrapper-smoke-20260827.json`](ppu-acblas-grouped-gdn-formal-wrapper-smoke-20260827.json)

正式 wrapper 冒烟记录真实 Transformers backend、空校验错误，以及
`18/18/49/18/6/24/18` 的完整模块挂载计数；其冷启动时间不用于性能比较。

## PPU residual-add + RMSNorm 跨层融合

只融合每层内部 24 条边的首版是重要负实验：固定 128-token 六对成对中位
`0.9821x`、仅 1/6 获胜，虽全文一致但没有性能价值。补齐 MLP residual 到下一层
input norm/最终 norm 后覆盖 48 条边，固定长两轮成对中位为 `1.0159x/1.0233x`，
均 4/6 获胜且全文一致。CN20 两轮分别为：

- `100.156→101.616 tokens/s`，成对中位 `1.0213x`，14/20 获胜；
- `98.576→101.507 tokens/s`，成对中位 `1.0206x`，14/20 获胜。

两轮 Accuracy 均为 85%，完整文本均 20/20 一致。16-token profile 中目标形状
`aten::add` 720→0，`cudaLaunchKernel` 16973→16253，证明 48 个边/15 个 decode
step 共减少 720 次设备 launch。正式 wrapper smoke 为真实 Transformers/PPU 后端、
公开校验无错误；冷启动和 profiler 插桩时延不进入性能比较。

- [`ppu-residual-rmsnorm-ab128-20260827.json`](ppu-residual-rmsnorm-ab128-20260827.json)（24-edge 负实验）
- [`ppu-residual-rmsnorm-ab128-chain48-r3-20260827.json`](ppu-residual-rmsnorm-ab128-chain48-r3-20260827.json)
- [`ppu-residual-rmsnorm-ab128-chain48-r4-20260827.json`](ppu-residual-rmsnorm-ab128-chain48-r4-20260827.json)
- [`ppu-residual-rmsnorm-cn20-chain48-r1-20260827.json`](ppu-residual-rmsnorm-cn20-chain48-r1-20260827.json)
- [`ppu-residual-rmsnorm-cn20-chain48-r2-20260827.json`](ppu-residual-rmsnorm-cn20-chain48-r2-20260827.json)
- [`ppu-residual-rmsnorm-profile-chain48-ab-20260827.json`](ppu-residual-rmsnorm-profile-chain48-ab-20260827.json)
- [`ppu-residual-rmsnorm-profile-chain48-baseline-summary-20260827.json`](ppu-residual-rmsnorm-profile-chain48-baseline-summary-20260827.json)
- [`ppu-residual-rmsnorm-profile-chain48-candidate-summary-20260827.json`](ppu-residual-rmsnorm-profile-chain48-candidate-summary-20260827.json)
- [`ppu-residual-rmsnorm-profile-chain48-gdn-shapes-20260827.json`](ppu-residual-rmsnorm-profile-chain48-gdn-shapes-20260827.json)（下一轮 GDN 标量路径形状清单）
- [`ppu-residual-rmsnorm-formal-wrapper-chain48-smoke-20260827.json`](ppu-residual-rmsnorm-formal-wrapper-chain48-smoke-20260827.json)

首版 24-edge profile 的 AB/summary JSON 也保留在同目录，用于解释为何必须补齐跨层
链；它们不能与最终 48-edge profile 混用。

## PPU SwiGLU 独立融合负实验

`SiLU(gate) * up` 自定义 HGGC 核在随机 BF16 `[1,1,6144]` 上四组线程配置均
bit-exact，但最优 128-thread 配置为 `0.7901x`，未通过单算子性能门禁。因此没有
运行公开集挑选样本，也没有接入正式 wrapper。正确后续方向是 GEMM epilogue fusion：

- [`ppu-swiglu-thread-sweep-negative-20260827.json`](ppu-swiglu-thread-sweep-negative-20260827.json)
- [实验说明](../docs/experiments/2026-08-27-ppu-swiglu-negative.md)
