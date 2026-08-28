# PPU 后续优化路线图

更新时间：2026-08-28

路线只依据 Qwen3.5 图结构、PPU profile 和跨样本性能/精度门禁，不按公开题号、类别或
答案定制。每一项都必须按“随机/真实张量数值 → 固定长交错 AB/BA → CN20 两轮 →
完整公开集 → 私有集”逐级晋级。

## P0：恢复与精度门禁

1. 用保存镜像 + CPFS 在新 PPU 实例复现最新提交；核对设备/SDK/torch、模型与数据
   哈希，并按名称核对 GDN 18、conv 18、RMSNorm 49、gated norm 18、qk-RoPE 6、
   packed MLP 24、单入口 acBLAS packed-MLP 24、grouped-GDN 18、decoder 24、
   gate-prep 18。
2. grouped-acBLAS GDN + 48-edge residual-RMSNorm + gate-prep + 单入口 packed-MLP 的
   中英文完整公开集均已 4029/4029 exact；下一步按主办方要求进入私有集门禁。
3. 获取主办方私有门禁和最终镜像，固定一次“提交候选”而不是继续追逐小样本噪声。

## P1：真正减少内存中间量和 launch

### GEMM epilogue SwiGLU

独立 HGGC SwiGLU 核虽 bit-exact，但最好仅 0.7901x；将它与前后两次 GEMV 放进一次
C++ extension 入口后，固定长和 CN20 两轮已得到 11%--13% 的稳定端到端收益。这一
版本仍落地 12288 维 projected scratch 和 6144 维 activated scratch。SDK 2.1.1
公开 acBLASLt epilogue 只有 Bias/ReLU/GELU，没有 SiLU；若厂商开放自定义 epilogue，
下一步让 packed gate/up GEMM 直接产出 `SiLU(gate)*up`，才能继续消除中间张量。

### GDN raw-gate 与常量折叠（已完成独立 gate-prep）

已在加载阶段缓存静态 FP32 `exp(A_log)`，并用一个 HGGC kernel 合并 raw `a/b`
门控准备；最终 CN20 两轮配对中位约 +8%，均 20/20 全文 exact。当前采用独立
gate-prep + recurrent 两核，以保留已验证的 state kernel。除非后续 profile 证明
GEMM 路线受阻，不再为省一次 launch 把它强行并入 recurrent kernel。

### Decode scratch arena

单入口 packed-MLP 已为 24 层建立 projected/activated/output scratch，总计约
0.94 MiB，并通过 memcheck。最终 profile 仍有 `empty_like=3982`、
`empty_strided=4391`、`cudaFree=3259` 和大量 clone/copy；下一步继续扩展到 qkv、
gate 和 norm，但必须先验证多请求并发下的 stream/alias 生命周期。

首个扩展候选已经覆盖 6 个全注意力层的 Q/K/V 投影与 Q/K RMSNorm+RoPE：模块级
bit-exact、memcheck 0 errors，并以 patch-time stream guard 明确限制为单流串行 decode。
模块边界虽为 `4.1006x`，固定长 56 对合并中位仅 `1.0047x`，CN20 两轮中位
`1.0158x/0.9852x`、方向不一致，已按停止规则作为“模块快、整模不快”的负实验保留。
下一步不再扩展同类 per-layer scratch，除非能把多个注意力层或 cache 更新纳入更大的
运行时/图边界。

## P2：运行时与调度

1. 把多次 ctypes/Python 调用聚合到一个 C++/pybind decode step，减少 dispatcher、
   device-property 查询和 handle/stream 设置；grouped-acBLAS 已证明主机调度可带来收益。
2. 验证 PPU 是否支持安全的 graph capture。动态 KV 长度、cache 更新和输出停止条件是
   主要障碍，可先捕获单层固定形状子图，不能假定 CUDA Graph 语义完全兼容。
3. 获取主办方 PPU-vLLM/FLA fast path，与当前 Transformers eager 在相同计时口径下
   对照；若官方实现成熟，优先迁移自定义核到其 custom-op 接口。

## P3：矩阵与状态核

1. 向厂商确认 grouped/batched GEMV、multi-output GEMV 和 epilogue API；避免将四路
   GDN 投影强行拼成改变数值路径的单一 8224 行 GEMV。
2. 对 recurrent GDN 做 state tile/向量化和 shared-memory 布局搜索，但必须包含真实
   16×128×128 FP32 state 带宽与跨 token 依赖；微基准不能只测空 state。
3. 评估将 causal-conv、raw gate preparation、recurrent update 合成一层 decode
   pipeline；只有当中间 q/k/v 不再落地时，融合才可能抵消更复杂的 launch 成本。

## 下一轮 GEMM/GEMV 主线

gate-prep 与 acBLASLt heuristic 调查均已完成；方阵整模负实验已止损。当前正收益
来自单入口 packed-MLP：不减少 GEMV 数，但消除 720 次 Linear/ATen 入口和 360 次
elementwise launch；4029 门禁已以 4029/4029 exact、成对中位 `1.1125x` 通过。
Attention Prep 边界已完成并因 CN20 两轮方向不一致止损。下一优先级仍是厂商自定义
SwiGLU epilogue，直接消除 projected/activated 中间张量；其次是官方 PPU-vLLM/FLA
custom-op/graph 接口，最后才是有正式 API 支撑的 grouped/batched GEMV 或权重预打包。
独立 SwiGLU、自写通用 GEMV、通用 acBLAS Linear 和 acBLASLt 方阵均已有负结果，
不重复线程或 heuristic 盲搜。

## P4：量化与低精度

在主办方明确允许范围后，从 weight-only per-channel INT8 开始，优先量化 MLP 与输出
projection，保留 norm、GDN state、gate 和视觉关键层 BF16/FP32。校准集必须与公开
评分集隔离；报告完整集 Accuracy、文本漂移、TTFT、吞吐和显存，不能只报困惑度或
小样本答案。

## 统一停止规则

- 单算子低于 baseline：记录负实验，停止整模评测；
- 只有 profiler 次数下降而无两轮无 profiler 收益：不接入；
- 两轮性能方向不一致：扩大固定输入重复，不换公开样本；
- 文本/答案漂移超过预设门限：回退或恢复原舍入顺序；
- 所有候选默认关闭，完整公开集和私有门禁通过后才改变提交配置。
