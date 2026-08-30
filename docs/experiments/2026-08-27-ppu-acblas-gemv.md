# 2026-08-27 PPU acBLAS BF16 decode GEMV 调查

## 目的与边界

packed-MLP 后的 16-token profile 中，最大设备热点仍是厂商 GEMV/GEMM。该调查只由
Qwen3.5-2B 的固定线性层形状、BF16 dtype、batch=1 decode 和真实冷权重工作集驱动，
不读取题目内容、标签或错误样本；公开集只承担整模型回归门禁，不参与算法选择。

Qwen3.5-2B 的 24 个语言层中，线性投影贯穿两类块：18 个 Gated DeltaNet 层的
q/k/v、z、状态门控和输出投影，6 个 full-attention 层的 q/k/v/o 投影，以及全部
24 层 MLP 的 gate/up/down 投影。decode 时输入通常是一个 `[1,1,2048]` 向量，
所以计算退化为内存带宽和启动开销主导的 GEMV，而不是大 batch GEMM。

## SDK 能力

PPU SDK 提供 `libacblas.so` 和 `libacblasLt.so`：

- `acblasGemvEx/GemmEx` 支持 BF16 输入、FP32 累加；
- acBLASLt 支持 row/column/特殊矩阵布局、heuristic、workspace 和 SM target；
- epilogue 当前公开头文件列出 bias、ReLU、GELU 及其组合，但没有 SiLU/SwiGLU；
- 因而 MLP 的 `SiLU(gate)*up` 不能直接套用现有 epilogue，除非厂商提供额外扩展。

## 裸 acBLAS 冷工作集结果

16 份随机 BF16 权重轮换、32 次预热、200 次计时；packed 12288 输出因显存规模使用
8 份权重。所有 acBLAS 结果与当前 `torch.mv` BF16 输出逐位一致。

| 模型形状 `N×K` | 典型位置 | torch.mv ms | acBLAS 最佳 ms | 加速 |
|---:|---|---:|---:|---:|
| 6144×2048 | GDN qkv、未 packed MLP gate/up | 0.017042 | 0.016665 | 1.023x |
| 2048×6144 | MLP down | 0.014731 | 0.014239 | 1.035x |
| 2048×2048 | attention/state/output 方阵投影 | 0.008047 | 0.007264 | 1.108x |
| 12288×2048 | packed MLP gate/up | 0.026503 | 0.026172 | 1.013x |

`6144×2048` 对算法 `-3/-2/-1/0..23/99` 的完整扫描全部落在近似相同的
`0.01667 ms` 性能簇，说明 `acblasGemvEx` 的算法编号基本映射到同一 GEMV 后端；
不应继续在公开集或单一形状上搜索编号。

## 调用边界问题

裸 kernel 的收益不能直接变成模型收益。通过 Python closure + ctypes 调用
2048×2048 acBLAS：

- 初版（每次 Python 校验）为 `0.018743 ms`，eager `0.010183 ms`；
- 把权重校验和属性捕获移到挂载阶段后为 `0.014877 ms`，eager `0.009059 ms`；
- decode/prefill 均逐位一致，但 ctypes 热路径仍慢 64%。

主要额外成本是 Python 调度、输出张量分配、当前 stream 查询和 FFI 边界。结论是：
直接在 Python 中把 `nn.Linear` 换成 ctypes acBLAS 属于错误集成层，不接入整模型。

## 注册式 C++ extension

最终接入采用两层 ABI，而不是把 Torch 与 PPU 头文件强行放在一起：

```text
PyTorch C++ extension（Torch/CUDA 兼容头）
        -> 窄 C ABI + 当前 stream
acBLAS bridge（只包含 PPU SDK 头）
        -> acblasGemvEx
```

原因是 PyTorch 会包含 CUDA 的 `cuda_fp16.h/cuda_bf16.h`，PPU SDK 的
`hggc_fp16.h/hggc_bf16.h` 又定义同名 `__half/__bfloat16`；在同一翻译单元会产生大量
重定义。ABI 隔离后，bridge 先由 HGGC clang 编译为 PIC object，再与 Torch extension
链接成单一 `.so`；最终 `NEEDED` 不包含项目内 bridge 库，也不含服务器工作目录
rpath。bridge 复用进程级 acBLAS handle，并把 PyTorch 当前 stream 显式传给 acBLAS；
一个短 host mutex 只保护 `SetStream+Gemv` 的提交序列，防止并发线程覆盖 handle 状态，
API 返回后设备工作仍在对应 stream 上异步执行。

随机 BF16 模块级测试把 `nn.Linear.forward`、输出分配、Python closure 和 pybind
边界全部计入；decode 输出逐位一致，4-token prefill 全部回退且逐位一致：

| 权重形状 `N×K` | Qwen3.5 位置 | `nn.Linear` ms | extension ms | 加速 |
|---:|---|---:|---:|---:|
| 2048×2048 | GDN z/out、attention o | 0.008933 | 0.007997 | 1.117x |
| 6144×2048 | GDN qkv | 0.009007 | 0.008342 | 1.080x |
| 2048×6144 | MLP down | 0.009001 | 0.008060 | 1.117x |
| 512×2048 | attention k/v | 0.010124 | 0.008621 | 1.174x |
| 4096×2048 | attention q+gate | 0.009097 | 0.008234 | 1.105x |

`12288×2048` packed gate/up 只有 `1.008x`，不替换；`16×2048` 的 GDN a/b 小投影
也不进入候选。形状集合完全由随机 BF16 微基准决定，不读取公开题目、答案或类别。

## Qwen3.5 整模型验证

在既有 `GDN + conv + RMSNorm + gated RMSNorm + qk-RoPE + packed-MLP` 路径上，
extension 替换 102 个会在 decode 热路径实际调用的 bias-free Linear：

| 形状 | 模块数 |
|---:|---:|
| 2048×2048 | 42 |
| 2048×6144 | 24 |
| 4096×2048 | 6 |
| 512×2048 | 12 |
| 6144×2048 | 18 |

packed MLP 已绕过原始 gate/up module forward，因此不重复挂载那 48 个闲置 forward。

早期 bridge 版本曾在一次 CN20 运行中得到约 `1.041x` 的成对中位提升，但 bridge
仍处于动态库拆分/handle 生命周期变化阶段，不能作为最终结果。最终候选固定为：单一
extension `.so`、进程级 acBLAS handle、短 mutex 保护 `SetStream+Gemv`。该版本的
同模型 CN20 逐样本 AB/BA 为：

| 路径 | 平均 TTFT ms | 平均 token/s | Accuracy |
|---|---:|---:|---:|
| all-five + packed-MLP | 118.699 | 95.879 | 85% |
| + registered acBLAS Linear | 119.101 | 98.537 | 85% |

20 个成对速度比的中位数/均值为 `1.0164x/1.0302x`，只有 12/20 获胜；20/20
完整文本 SHA-256、token 数、答案和正确性一致。进一步用固定 128-token 降低不同
生成长度对计时的影响，8 对测试中基线/候选中位吞吐为 `85.523/85.591 token/s`，
成对中位仅 `0.9997x`、4/8 获胜，完整文本仍一致。因此最终结论是：收益不能跨
工作负载稳定复现，不接入正式 `evaluation_wrapper.py`。

## Profile 机制证据

同一 226-token prompt、16-token profile 中，两条路径的 PPU kernel 数和主要
`gemvt_op` 时间基本不变，但候选恰好减少 `1530 = 102 modules × 15 decode steps`
次 `aten::linear` 和 `aten::mm`：

| 事件 | 基线 | acBLAS extension |
|---|---:|---:|
| `aten::linear` | 3090 / 100.461 ms | 1560 / 55.065 ms |
| `aten::mm` | 2992 / 63.611 ms | 1462 / 32.805 ms |
| `cudaLaunchKernel` | 19878 / 65.787 ms | 19878 / 62.669 ms |
| profiler 总区间 | 672.335 ms | 613.727 ms |

对应 decode mm 形状分别减少：2048 方阵 630 次、MLP down 360 次、GDN qkv
270 次、attention k/v 180 次、attention q+gate 90 次。这个 profile 证明 extension
确实绕过了 ATen 通用 Linear/Matmul dispatcher，但底层 kernel 数不变，且最终固定
长解码没有 wall-clock 收益。它是机制证据，不是可提交性能提升。

## 复现入口

- `ppu/microbench/acblas_gemv_sweep.py`：裸 acBLAS 与 `torch.mv` 多算法/多形状比较；
- `ppu/custom_ops/smoke_acblas_linear_integration.py`：Python/ctypes 集成开销对照；
- `ppu/custom_ops/acblas_linear_extension.cpp`：实验性 PyTorch C++ extension；
- `ppu/custom_ops/build_acblas_linear_extension.py`：使用项目虚拟环境构建 extension。
- `ppu/custom_ops/smoke_acblas_linear_extension.py`：比较完整 C++ extension 调用与
  `nn.Linear`，并要求随机 BF16 输出逐位一致。
- `ppu/custom_ops/smoke_acblas_linear_module.py`：把真实 module forward 与 prefill
  回退开销计入的门禁；
- `scripts/benchmark_ppu_acblas_linear_ab.py`：固定长解码、AB/BA 与 profiler；
- `scripts/benchmark_ppu_acblas_multisample_ab.py`：同模型 CN20 逐样本成对验证；
- `results/ppu-acblas-cn20-final-20260827.json`：最终单 `.so` 版本 CN20 原始记录；
- `results/ppu-acblas-ab128-final-20260827.json`：最终固定 128-token 八对记录。

原始扫参日志与后续二进制保存在隔离 PPU 服务器，不提交模型、数据或大型 trace。

## 决策

保留源码、构建方式和负实验数据，便于后续厂商 runtime/dispatcher 变化后复测；当前
版本不提供 wrapper 环境开关，也不计入最终性能栈。公开集仅承担精度回归，未用于
选择 acBLAS 算法或投影形状。
