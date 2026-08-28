# 2026-08-28 PPU residual-RMSNorm 输出 scratch 负实验

## 目标

已验证的 decoder residual-add + RMSNorm 融合每层每 token 仍通过 `torch.empty_like`
分配归一化输出。候选为 24 层各准备两个 BF16 `[1,1,2048]` 输出 scratch，分别供 MLP
输入和下一层 input norm 使用，尝试消除每 token 48 次临时分配。

候选只在显式 A/B 开关下按需分配；prefill 继续走原始 forward。scratch 与 patch 时
默认 stream 绑定，异流在 kernel 提交前拒绝，不宣称支持并发请求。

## 模块验证

PPU-ZW810E 上 1000 次计时：

| 路径 | 平均耗时 | 相对现有 fused 路径 |
|---|---:|---:|
| eager add + RMSNorm | 0.027569 ms | - |
| 现有 fused residual-RMSNorm（每次分配输出） | 0.018729 ms | `1.0000x` |
| fused + 持久输出 scratch | 0.014005 ms | `1.3373x` |

residual、normalized 均 bit-exact，输出地址连续调用保持一致；`hggc-memcheck` 报告
`ERROR SUMMARY: 0 errors`。memcheck 插桩时延不用于性能结论。

## 当前完整优化栈门禁

基线固定为 grouped-acBLAS GDN + residual-RMSNorm + GDN gate-prep + 单入口 acBLAS
packed-MLP，仅 A/B 输出 scratch。固定同一输入强制生成 128 token，交错 8 对：

| 指标 | 结果 |
|---|---:|
| 全文一致 | 8/8 |
| baseline 吞吐中位 | 122.073 token/s |
| scratch 吞吐中位 | 121.204 token/s |
| 成对中位/均值 | `0.9862x / 0.9888x` |
| 候选获胜 | 2/8 |
| 严格性能门禁 | **失败** |

模块少量分配节省不足以覆盖 Python 条件、raw-stream 守卫及整模其余开销。按预先规则，
不追加轮次挑选有利噪声，不运行 CN20/profile/完整集，也不接入 `evaluation_wrapper.py`。
代码只保留为显式实验入口，正式配置和提交候选不变。

## 证据

- `results/residual-rmsnorm-scratch-smoke-20260828.txt`
- `results/residual-rmsnorm-scratch-memcheck-20260828.txt`
- `results/residual-rmsnorm-scratch-ab128-20260828.json`
