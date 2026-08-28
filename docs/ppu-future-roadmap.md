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
   中英文完整公开集均已 4029/4029 exact；叠加 raw-stream 后再次通过中英文各
   4029/4029 exact，成对中位分别 `1.0906x/1.0901x`。下一步按主办方要求进入
   私有集门禁。
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

residual-RMSNorm 的两个输出 scratch 已单独验证：模块级相对现有融合 `1.3373x`、
bit-exact、memcheck 0 errors，但当前完整栈固定长只有 `0.9862x`。因此该方向已止损，
不再逐个小张量扩展 scratch；后续必须在更大 C++/图边界内统一管理 arena，才可能抵消
Python 守卫成本。

首个扩展候选已经覆盖 6 个全注意力层的 Q/K/V 投影与 Q/K RMSNorm+RoPE：模块级
bit-exact、memcheck 0 errors，并以 patch-time stream guard 明确限制为单流串行 decode。
模块边界虽为 `4.1006x`，固定长 56 对合并中位仅 `1.0047x`，CN20 两轮中位
`1.0158x/0.9852x`、方向不一致，已按停止规则作为“模块快、整模不快”的负实验保留。
下一步不再扩展同类 per-layer scratch，除非能把多个注意力层或 cache 更新纳入更大的
运行时/图边界。

## P2：运行时与调度

1. raw-stream 查询已先消除高层 Stream 对象开销：中文 4029 exact、成对中位
   `1.0906x`。下一步把多次 ctypes/Python 调用聚合到一个 C++/pybind decode step，
   减少 dispatcher 和当前 profile 每 16 token 仍有的 5705 次
   `cudaGetDeviceProperties_v2`；不能通过缓存错误的设备/流状态换取速度。父链归因显示
   2105 次来自 `aten::mm/bmm/addmm`，其余 3600 次形成 1800 对连续查询并紧邻
   `cudaFree`。这与 15 个 profiled decode step × 120 次 acBLAS GEMV/step 完全吻合：
   18 层 grouped-GDN 各 4 次（72），24 层 packed-MLP 各 2 次（48）。下一候选应是
   厂商可复用 workspace/handle 或跨层 grouped/batched GEMV，避免每次 acBLAS 内部
   设备查询与释放；不再从 Python stream 侧继续挤微小收益。
   SDK 2.1.1 公开头文件及 `libacblas.so` 动态符号均确认有
   `acblasSetWorkspace_v2`，但 GEMV batched 接口只有
   S/D 精度的 `Sgemv/DgemvStridedBatched`，没有 BF16 `GemvExBatched` 或 heterogeneous
   grouped GEMV。下一轮先给两个进程级 handle 配置持久 workspace 做低风险 A/B；
   若内部查询/释放次数不降，再评估 `acblasGemmBatchedEx` 表达同形状 BF16 GEMV，
   不把 qkv/z/b/a 不同 N 的矩阵强行塞入同一批次。
   该调查现已完成：4/16/64 MiB workspace 未改变设备查询、释放或 kernel 数，固定长
   中位 `0.9846x`；把同形状 b/a 合成 strided-batched GEMM 虽将设备查询
   `5705→4895`、释放 `3259→2989`，kernel 数仍不变且固定长中位 `0.9852x`；GDN
   持久输出 scratch 两轮中位 `1.0105x/0.9934x`、方向不一致。三项均作为负实验停止，
   详见 `docs/experiments/2026-08-28-ppu-acblas-runtime-overhead.md`。
   当前主要矛盾是每 token 120 个 memory-bound BF16 小 GEMV；下一主线改为厂商 acext
   A16W8/A16W4 `WeightOnlyBatchedGemv` 或官方 PPU-vLLM，但当前镜像未包含 acext。
   进一步验证发现，GDN 已连续存放的 qkv/z/b/a 可从每层 4 次 GEMV 合为 1 次：两轮
   fixed-128 中位 `1.0184x/1.0112x`，CN100 中位 `1.0261x`，Accuracy `93%→93%`、
   答案 100/100 一致，但完整文本仅 99/100 一致。该路线作为默认关闭的
   `SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE` accuracy-budget 候选保留，不归类为无损融合。
   另有只合并 b/a 的 3-GEMV 精度优先候选：CN100 100/100 完整文本一致、中位
   `1.0068x`，通过 `SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE` 默认关闭地接入；下一门禁为
   中英文完整公开集。
2. PPU Graph Capture 已完成最小验证：固定 16 段 elementwise 子图 `1.8303x`；但
   已聚合 packed-MLP 仅 `1.0203x`，含动态输入 copy 为 `0.9316x`。当前不改正式路径；
   只有固定 KV/page 地址或完整 decoder-layer/多层图边界出现后才重启该方向。
3. 获取主办方 PPU-vLLM/FLA fast path，与当前 Transformers eager 在相同计时口径下
   对照；若官方实现成熟，优先迁移自定义核到其 custom-op 接口。

## P3：矩阵与状态核

1. 向厂商确认真正保持各投影累加顺序的 grouped/multi-output GEMV 和 epilogue API；
   当前单一 8224 行 GEMV虽有约 2.6% CN100 中位收益，但存在 1/100 文本漂移。
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
