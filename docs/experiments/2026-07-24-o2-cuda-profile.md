# O2：CUDA 热点与显存剖析

日期：2026-07-24

## 目标

在 O1 代码基线上定位单样本推理的主要 GPU kernel 和显存峰值，为后续算子融合、编译或 PPU 适配选择方向。

## 配置

- 模型：Qwen3.5-2B，BF16，全部参数位于 `cuda:0`
- 设备：RTX 4050 Laptop GPU 6GB
- 数据：公开中文集固定样本，batch size 1
- 预热：1 条
- 生成：`max_new_tokens = 256`，实际 66 tokens
- 工具：`torch.profiler`，CPU + CUDA，记录 shape 和内存

Profiler 会显著增加运行开销，因此本次 TTFT、端到端时延和吞吐只用于确认采样有效，不参与 O1 性能结论。

## 显存

- PyTorch peak allocated：4,498,557,952 bytes，约 4.19 GiB
- PyTorch peak reserved：4,517,265,408 bytes，约 4.21 GiB

这两个值是 PyTorch allocator 口径，不等于 `nvidia-smi` 的设备进程总占用。6GB 本机显存余量有限，全图编译、较大并发和额外权重副本存在 OOM 风险。

## CUDA 自耗时分布

总 self CUDA time 为 1,609.119 ms：

| 类别 | Calls | Self CUDA time | 占比 |
|---|---:|---:|---:|
| GEMV | 12,936 | 970.766 ms | 60.33% |
| GEMM | 927 | 416.038 ms | 25.86% |
| Elementwise | 79,926 | 114.295 ms | 7.10% |
| Memory copy | 40,859 | 73.128 ms | 4.54% |
| Reduction | 11,668 | 28.168 ms | 1.75% |
| Convolution | 1,188 | 5.168 ms | 0.32% |
| Other | 738 | 1.557 ms | 0.10% |

单个最重 kernel 是 BF16 GEMV：8,970 次，951.873 ms，占全部 self CUDA time 的 59.16%。第二个最重 kernel 是 BF16 GEMM：66 次，359.258 ms，占 22.33%。

## 结论

1. 当前 batch size 1 解码主要受小矩阵 GEMV 支配，GEMV + GEMM 合计占 86.18%。
2. Elementwise 和内存复制单次很小，但调用数分别接近 8 万和 4 万，存在 kernel 启动与融合空间。
3. 视觉卷积只占当前 GPU 自耗时的 0.32%，暂不作为第一优化目标。
4. 本机显存余量不适合直接尝试高风险的全模型 `torch.compile`；先研究 decode 路径、线性注意力 fast path、融合和目标 PPU kernel。
5. Transformers 当前明确提示缺少 `flash-linear-attention` 与 `causal-conv1d` fast path。本机 Windows 不盲装未经验证的扩展，PPU 侧需要确认等价算子或自定义 kernel 支持。

原始 profiler JSON 和 Chrome trace 保存在本地 `artifacts/` 忽略目录，不提交仓库。
