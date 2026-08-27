# PPU 后续优化路线图

更新时间：2026-08-28

路线只依据 Qwen3.5 图结构、PPU profile 和跨样本性能/精度门禁，不按公开题号、类别或
答案定制。每一项都必须按“随机/真实张量数值 → 固定长交错 AB/BA → CN20 两轮 →
完整公开集 → 私有集”逐级晋级。

## P0：恢复与精度门禁

1. 用保存镜像 + CPFS 在新 PPU 实例复现最新 gate-prep 提交；核对设备/SDK/torch、
   模型与数据哈希、18/18/49/18/6/24/18/24/18 模块计数。
2. grouped-acBLAS GDN + 48-edge residual-RMSNorm + gate-prep 的中文完整公开集已
   4029/4029 exact；下一步补英文公开集或直接按主办方要求进入私有集门禁。
3. 获取主办方私有门禁和最终镜像，固定一次“提交候选”而不是继续追逐小样本噪声。

## P1：真正减少内存中间量和 launch

### GEMM epilogue SwiGLU

本轮独立 HGGC SwiGLU 核虽 bit-exact，但最好仅 0.7901x。下一步不是继续调线程，而是
SDK 2.1.1 公开 acBLASLt epilogue 已确认只有 Bias/ReLU/GELU，没有 SiLU。后续需向
厂商请求自定义 epilogue：packed gate/up GEMM 直接产出 `SiLU(gate)*up`，避免
12288 维投影中间张量落地、split、独立 SiLU 和 mul。它对所有 SwiGLU Transformer
通用，预期收益也比单独 elementwise 核更可靠。

### GDN raw-gate 与常量折叠（已完成独立 gate-prep）

已在加载阶段缓存静态 FP32 `exp(A_log)`，并用一个 HGGC kernel 合并 raw `a/b`
门控准备；最终 CN20 两轮配对中位约 +8%，均 20/20 全文 exact。当前采用独立
gate-prep + recurrent 两核，以保留已验证的 state kernel。除非后续 profile 证明
GEMM 路线受阻，不再为省一次 launch 把它强行并入 recurrent kernel。

### Decode scratch arena

最终 gate-prep profile 仍有 `empty_like=3982`、`empty_strided=4391`、`cudaFree=3259` 和大量
clone/copy。为固定 batch=1 decode 建立每层 scratch arena，复用 qkv、gate、norm 和
projection buffer；先检查 cache/stream 生命周期和 alias，再逐个替换临时分配。

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

## 下一轮唯一主线：GEMM/GEMV

gate-prep 与 acBLASLt heuristic 调查均已完成。公开 epilogue 没有 SiLU，四个真实
形状扫参只有方阵过 3% 低层门槛，且整模最终为负。下一轮只在厂商开放自定义
epilogue、grouped/batched GEMV 与权重预打包接口后继续；首选目标是 packed gate/up GEMM
直接产生 `SiLU(gate) * up`，其次才是 2048→6144、6144→2048 的 batch=1 decode
GEMV。独立 SwiGLU、自写通用 GEMV、通用 acBLAS Linear 和 acBLASLt 方阵均已有负
结果，不重复做线程或 heuristic 盲搜。

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
