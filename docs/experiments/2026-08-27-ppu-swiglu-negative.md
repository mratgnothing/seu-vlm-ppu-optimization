# 2026-08-27 PPU SwiGLU 融合负实验

## 动机

最终 residual-RMSNorm profile 中，Qwen3.5-2B 的 24 个 MLP 在每个 decode step 都
执行一次 BF16 `SiLU(gate) * up`。15 个 decode step 对应 360 次 `silu` 和 360 次
`mul`，因此尝试以一个 HGGC elementwise kernel 替代两个 eager kernel。方案只依赖
SwiGLU 数学结构，不使用公开集标签或样本特征。

## 数值契约

自定义核先以 FP32 计算 SiLU，再在与 PyTorch BF16 `silu` 相同的位置舍入为 BF16，
随后与 BF16 `up` 相乘并再次舍入。随机 `[1,1,6144]` 输入在 128/256/512/1024
线程四组中均为 bit-exact，最大绝对误差为 0。

## PPU 微基准

PPU-ZW810E，50 次预热、1000 次计时：

| threads | Torch 两核 ms | HGGC 一核 ms | speedup | exact |
|---:|---:|---:|---:|---:|
| 128 | 0.008775 | 0.011106 | 0.7901x | 是 |
| 256 | 0.008662 | 0.013046 | 0.6639x | 是 |
| 512 | 0.008194 | 0.011181 | 0.7329x | 是 |
| 1024 | 0.008440 | 0.011335 | 0.7446x | 是 |

最优配置仍慢 20.99%。原因是 6144 个元素的工作量过小，PyTorch 已使用高效向量化
kernel，而 HGGC block 调度、输出分配和 Python/ctypes 发射成本大于少一次 launch 的
收益。由于单算子性能门禁已失败，没有继续在 CN20 上寻找有利样本，也没有加入正式
wrapper 开关。

## 保留内容与后续方向

仓库保留独立 C ABI、ctypes 调用和随机 smoke，作为“减少 launch 数不等于降低时延”
的可复现负实验；生产路径不调用它。若将来厂商提供 GEMM epilogue fusion，应把
SwiGLU 直接融合进 packed gate/up GEMM 的 epilogue，从而既不落地 gate/up 中间张量，
也不新增独立 HGGC launch；这才是该结构的正确优化层级。

证据：`results/ppu-swiglu-thread-sweep-negative-20260827.json`。
