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

## PPU GDN gate-prep 融合

在最终优化栈上缓存 18 层 FP32 `exp(A_log)`，将每层每 token 的 Sigmoid、cast、
Add、Softplus、Mul/Neg 合为一个 HGGC kernel，并用 thread-local scratch 复用
`g/beta`。固定 128-token 六对配对中位 `1.0839x`、6/6 获胜且全文一致。CN20
两轮分别为：

- `101.651→109.275 tokens/s`，配对中位 `1.0811x`，19/20 获胜；
- `100.085→107.083 tokens/s`，配对中位 `1.0863x`，17/20 获胜。

两轮均 20/20 全文一致、Accuracy 85%。16-token profile 的 launch
`16253→14363`，Self CPU/PPU 分别下降 11.48%/5.35%，memcheck 为 0 errors。

中文完整公开集 4029 条最终门禁中，两路 Accuracy 均为 `3374/4029`，完整文本、
答案和 token 数均 4029/4029 一致；成对吞吐中位 `1.0862x`，3882/4029 获胜。
该长测只作为完整精度/一致性门禁，不替代固定 128-token 性能结论。

- [`ppu-gdn-gate-prep-smoke-20260828.json`](ppu-gdn-gate-prep-smoke-20260828.json)
- [`ppu-gdn-gate-prep-ab128-20260828.json`](ppu-gdn-gate-prep-ab128-20260828.json)
- [`ppu-gdn-gate-prep-cn20-r1-20260828.json`](ppu-gdn-gate-prep-cn20-r1-20260828.json)
- [`ppu-gdn-gate-prep-cn20-r2-20260828.json`](ppu-gdn-gate-prep-cn20-r2-20260828.json)
- [`ppu-gdn-gate-prep-profile-ab-20260828.json`](ppu-gdn-gate-prep-profile-ab-20260828.json)
- [`ppu-gdn-gate-prep-profile-baseline-summary-20260828.json`](ppu-gdn-gate-prep-profile-baseline-summary-20260828.json)
- [`ppu-gdn-gate-prep-profile-candidate-summary-20260828.json`](ppu-gdn-gate-prep-profile-candidate-summary-20260828.json)
- [`ppu-gdn-gate-prep-memcheck-20260828.txt`](ppu-gdn-gate-prep-memcheck-20260828.txt)
- [`gate-prep-scratch-cn-full4029-summary.json`](gate-prep-scratch-cn-full4029-summary.json)
- [`ppu-final-formal-wrapper-smoke-20260828.json`](ppu-final-formal-wrapper-smoke-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-gdn-gate-prep.md)

## PPU 单入口 acBLAS packed-MLP

在完整 grouped-GDN/residual-RMSNorm/gate-prep 栈上，一次 extension 入口依次提交
packed gate/up GEMV、HGGC SwiGLU 和 down GEMV。固定 128-token 八对 8/8 获胜，
成对中位 `1.1336x`；CN20 两轮均 20/20 全文一致、20/20 获胜、Accuracy 85%，
成对中位为 `1.1212x/1.1122x`。Profile 中 GEMV 数不变，但 `aten::linear/mm`
各减少 720 次、`cudaLaunchKernel` 减少 360 次。全量 4029 paired 门禁中，两路
Accuracy 均为 3374/4029，4029/4029 文本、答案和 token 数一致；平均吞吐
`109.993→122.445 token/s`，成对中位 `1.1125x`、3939/4029 获胜。

英文完整公开集也以相同口径通过：两路 Accuracy 均为 3214/4029，4029/4029
文本、答案和 token 数一致；平均吞吐 `107.276→118.964 token/s`，成对中位
`1.1093x`、3806/4029 获胜。

- [`acblas-packed-mlp-smoke-20260828.json`](acblas-packed-mlp-smoke-20260828.json)
- [`acblas-packed-mlp-memcheck-20260828.txt`](acblas-packed-mlp-memcheck-20260828.txt)
- [`acblas-packed-mlp-ab128-20260828.json`](acblas-packed-mlp-ab128-20260828.json)
- [`acblas-packed-mlp-cn20-r1-20260828.json`](acblas-packed-mlp-cn20-r1-20260828.json)
- [`acblas-packed-mlp-cn20-r2-20260828.json`](acblas-packed-mlp-cn20-r2-20260828.json)
- [`acblas-packed-mlp-profile-ab-20260828.json`](acblas-packed-mlp-profile-ab-20260828.json)
- [`acblas-packed-mlp-profile-baseline-summary-20260828.json`](acblas-packed-mlp-profile-baseline-summary-20260828.json)
- [`acblas-packed-mlp-profile-candidate-summary-20260828.json`](acblas-packed-mlp-profile-candidate-summary-20260828.json)
- [`acblas-packed-mlp-formal-wrapper-smoke-20260828.json`](acblas-packed-mlp-formal-wrapper-smoke-20260828.json)
- [`acblas-packed-mlp-cn-full4029-summary.json`](acblas-packed-mlp-cn-full4029-summary.json)
- [`acblas-packed-mlp-en20-stream-guard-summary-20260828.json`](acblas-packed-mlp-en20-stream-guard-summary-20260828.json)
- [`acblas-packed-mlp-en-full4029-summary-20260828.json`](acblas-packed-mlp-en-full4029-summary-20260828.json)
- [`acblas-packed-mlp-final-formal-wrapper-smoke-20260828.json`](acblas-packed-mlp-final-formal-wrapper-smoke-20260828.json)
- [`acblas-packed-mlp-final-memcheck-20260828.txt`](acblas-packed-mlp-final-memcheck-20260828.txt)
- [实验说明](../docs/experiments/2026-08-28-ppu-acblas-packed-mlp.md)

## PPU Attention Prep 单入口融合

6 个全注意力层的 Q/K/V 三次 GEMV 与既有 Q/K RMSNorm+RoPE 被放入一次 extension
入口，运算和 BF16 舍入顺序不变。强化 smoke 的真实模块边界
`0.080652→0.019668 ms`，即 `4.1006x`；Q/K/V/gate 与 prefill 均 bit-exact，最大
绝对误差 0，prefill 回退、scratch 复用、异流提交前拒绝均通过，`hggc-memcheck`
为 0 errors。固定 128-token 三组共 56 对全部全文一致，合并中位仅 `1.0047x`；
CN20 r1 为 `1.0158x`，r2 为 `0.9852x`，第二轮严格门禁失败。候选因此停止 profile
和完整集，保持默认关闭。

- [`acblas-attention-prep-smoke-20260828.json`](acblas-attention-prep-smoke-20260828.json)
- [`acblas-attention-prep-memcheck-20260828.txt`](acblas-attention-prep-memcheck-20260828.txt)
- [`acblas-attention-prep-ab128-20260828.json`](acblas-attention-prep-ab128-20260828.json)
- [`acblas-attention-prep-ab128-r2-20260828.json`](acblas-attention-prep-ab128-r2-20260828.json)
- [`acblas-attention-prep-ab128-r3-20260828.json`](acblas-attention-prep-ab128-r3-20260828.json)
- [`acblas-attention-prep-cn20-r1-20260828.json`](acblas-attention-prep-cn20-r1-20260828.json)
- [`acblas-attention-prep-cn20-r2-20260828.json`](acblas-attention-prep-cn20-r2-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-acblas-attention-prep.md)

## PPU Graph Capture 能力与 packed-MLP 止损

当前 PyTorch PPU 已支持 HGGC Graph Capture。65,536 元素、16 段 SiLU 固定链从
`0.153274→0.083743 ms`（`1.8303x`）且更新输入后 exact replay；但已聚合的单入口
packed-MLP 只有 `0.044214→0.043336 ms`（`1.0203x`），加入动态输入地址所需 copy
后为 `0.047462 ms`（`0.9316x`）。候选低于 3% 晋级余量，未进入整模评测。

- [`ppu-graph-capture-probe-20260828.json`](ppu-graph-capture-probe-20260828.json)
- [`acblas-packed-mlp-graph-smoke-20260828.json`](acblas-packed-mlp-graph-smoke-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-graph-capture.md)

## PPU residual-RMSNorm 输出 scratch 负实验

持久 BF16 输出 scratch 相对现有 fused residual-RMSNorm 模块边界为
`0.018729→0.014005 ms`（`1.3373x`），bit-exact、地址复用且 memcheck 0 errors。
但在 grouped-GDN + residual-RMSNorm + gate-prep + 单入口 packed-MLP 完整栈上，
固定 128-token 八对仅 `0.9862x`、2/8 获胜，严格性能门禁失败，未继续 CN20/profile。

- [`residual-rmsnorm-scratch-smoke-20260828.txt`](residual-rmsnorm-scratch-smoke-20260828.txt)
- [`residual-rmsnorm-scratch-memcheck-20260828.txt`](residual-rmsnorm-scratch-memcheck-20260828.txt)
- [`residual-rmsnorm-scratch-ab128-20260828.json`](residual-rmsnorm-scratch-ab128-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-residual-rmsnorm-scratch.md)

## PPU raw stream 查询优化

把每次 ctypes 算子提交的 `torch.cuda.current_stream(...).cuda_stream` 改为受能力检查
保护的 raw stream handle 查询。模块路径 `1.2944x`；固定 128-token 两轮成对中位
`1.1055x/1.0961x`，CN20 两轮成对中位 `1.1026x/1.0855x`。四轮均全文一致，
CN20 两路 Accuracy 均为 85%。中文完整集 4029/4029 文本、答案和 token 数一致，
两路 Accuracy 均为 3374/4029；平均吞吐 `120.383→131.107 token/s`，成对中位
`1.0906x`，3817/4029 获胜。memcheck 0 errors、profile exact 和正式 wrapper
smoke 均通过。英文 20 条两轮也 20/20 exact、Accuracy 90% 不变；英文完整集
4029/4029 exact、Accuracy 均为 3214/4029，平均吞吐
`118.577→129.398 token/s`、成对中位 `1.0901x`，3704/4029 获胜。

- [`raw-stream-query-ab128-r1-20260828.json`](raw-stream-query-ab128-r1-20260828.json)
- [`raw-stream-query-ab128-r2-20260828.json`](raw-stream-query-ab128-r2-20260828.json)
- [`raw-stream-query-cn20-r1-20260828.json`](raw-stream-query-cn20-r1-20260828.json)
- [`raw-stream-query-cn20-r2-20260828.json`](raw-stream-query-cn20-r2-20260828.json)
- [`raw-stream-query-en20-r1-20260828.json`](raw-stream-query-en20-r1-20260828.json)
- [`raw-stream-query-en20-r2-20260828.json`](raw-stream-query-en20-r2-20260828.json)
- [`raw-stream-query-cn4029-summary-20260828.json`](raw-stream-query-cn4029-summary-20260828.json)
- [`raw-stream-query-en4029-summary-20260828.json`](raw-stream-query-en4029-summary-20260828.json)
- [`raw-stream-query-profile-ab-20260828.json`](raw-stream-query-profile-ab-20260828.json)
- [`raw-stream-query-profile-baseline-summary-20260828.json`](raw-stream-query-profile-baseline-summary-20260828.json)
- [`raw-stream-query-profile-candidate-summary-20260828.json`](raw-stream-query-profile-candidate-summary-20260828.json)
- [`raw-stream-query-memcheck-20260828.txt`](raw-stream-query-memcheck-20260828.txt)
- [`raw-stream-query-formal-wrapper-smoke-20260828.json`](raw-stream-query-formal-wrapper-smoke-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-raw-stream-query.md)

## PPU acBLASLt Matmul 负实验

SDK 正式 epilogue 没有 SiLU/SwiGLU。四个 decode 主形状各扫描 32 个 bit-exact
heuristic，只有 `2048x2048` 的 `1.0577x` 通过低层 3% 门槛；隔离 extension 加上
预分配输出后模块级为 `1.2797x`。但完整 gate-prep 栈固定 128-token 八对仅
`0.9898x`、3/8
获胜，因此不接入正式 wrapper。Profile 显示 `aten::linear/mm` 各减少 360 次，但
主 `gemvt_op` 增加 270 次且累计设备时间上升，解释了局部快、整模不快。

- [`acblaslt-sweep-summary-20260828.json`](acblaslt-sweep-summary-20260828.json)
- [`acblaslt-square-isolated-smoke-20260828.json`](acblaslt-square-isolated-smoke-20260828.json)
- [`acblaslt-square-ab128-20260828.json`](acblaslt-square-ab128-20260828.json)
- [`acblaslt-square-profile-ab-20260828.json`](acblaslt-square-profile-ab-20260828.json)
- [`acblaslt-square-profile-baseline-summary-20260828.json`](acblaslt-square-profile-baseline-summary-20260828.json)
- [`acblaslt-square-profile-candidate-summary-20260828.json`](acblaslt-square-profile-candidate-summary-20260828.json)
- [实验说明](../docs/experiments/2026-08-28-ppu-acblaslt-matmul.md)

## PPU SwiGLU 独立融合负实验

`SiLU(gate) * up` 自定义 HGGC 核在随机 BF16 `[1,1,6144]` 上四组线程配置均
bit-exact，但最优 128-thread 配置为 `0.7901x`，未通过单算子性能门禁。因此没有
运行公开集挑选样本，也没有接入正式 wrapper。正确后续方向是 GEMM epilogue fusion：

- [`ppu-swiglu-thread-sweep-negative-20260827.json`](ppu-swiglu-thread-sweep-negative-20260827.json)
- [实验说明](../docs/experiments/2026-08-27-ppu-swiglu-negative.md)
