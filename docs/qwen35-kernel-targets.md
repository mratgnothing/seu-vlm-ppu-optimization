# Qwen3.5-2B 关键算子与 PPU kernel 目标

核验时间：2026-07-24

## 模型结构

### 文本主干

- Hidden size：2048
- Intermediate size：6144
- 文本层数：24
- 线性注意力层：18
- 全注意力层：6，每 4 层出现 1 层
- 全注意力 heads：8
- KV heads：2
- Head dim：256

### Gated Delta Network

- Key heads/value heads：16/16
- Key head dim/value head dim：128/128
- Key dim/value dim：2048/2048
- Depthwise causal-conv1d 通道数：6144
- Conv kernel width：4
- 单层 recurrent state：16 × 128 × 128

关键投影：

| 算子 | K（输入） | N（输出） | 层数 |
|---|---:|---:|---:|
| GDN `in_proj_qkv` | 2048 | 6144 | 18 |
| GDN `in_proj_z` | 2048 | 2048 | 18 |
| GDN `in_proj_a/b` | 2048 | 16 | 18 |
| GDN `out_proj` | 2048 | 2048 | 18 |
| GDN causal conv | channels 6144 | width 4 | 18 |
| GDN recurrent update | 16 × 128 × 128 state | 单 token | 18 |

### 全注意力

Q projection 同时输出 query 和 gate：

| 算子 | K | N | 层数 |
|---|---:|---:|---:|
| Q + gate projection | 2048 | 4096 | 6 |
| K projection | 2048 | 512 | 6 |
| V projection | 2048 | 512 | 6 |
| O projection | 2048 | 2048 | 6 |

### MLP 与词表头

| 算子 | K | N | 层数 |
|---|---:|---:|---:|
| Gate projection | 2048 | 6144 | 24 |
| Up projection | 2048 | 6144 | 24 |
| Down projection | 6144 | 2048 | 24 |
| LM head | 2048 | 248320 | 1 |

### 视觉主干

- Hidden size：1024
- Intermediate size：4096
- Depth：24
- Heads：16
- Patch size：16

主要视觉 MLP 尺寸包括 `1024 → 4096` 和 `4096 → 1024`。

## 与当前 PPU-vLLM 量化配置的匹配

当前共享节点的 PPU-ZW810E 配置中确认存在：

- `N=2048, K=2048`：INT4
- `N=4096, K=2048`：INT4
- `N=512, K=2048`：FP8/INT4/INT8
- `N=8192, K=2048`：INT4/INT8
- `N=8192, K=1024`：INT4/INT8
- `N=4096, K=1024`：FP8/INT4

未找到以下 Qwen3.5-2B 核心尺寸的 PPU-ZW810E 预调优配置：

- `N=6144, K=2048`：GDN QKV、MLP gate/up
- `N=2048, K=6144`：MLP down
- `N=248320, K=2048`：LM head

这不代表底层 kernel 无法运行，但意味着量化路线可能需要在线调优、补充配置或保留部分层为 BF16。

## 已准备的尺寸级验证入口

[`ppu/microbench/qwen35_bf16_gemv.hg`](../ppu/microbench/qwen35_bf16_gemv.hg)
覆盖 `N=6144,K=2048`、`N=2048,K=6144` 和 `N=2048,K=2048` 三组解码
关键尺寸。它是 BF16 输入、FP32 累加的正确性参考 kernel，用于隔离 PPU 环境中的
编译、误差、延迟和访存基线，不代表已经完成 PPU 实测或生产级优化。

## 优化优先级

1. GDN 单 token recurrent update：18 层 decode 必经路径。
2. `2048 → 6144` BF16 GEMV/GEMM：GDN QKV 与 MLP 高频复用。
3. `6144 → 2048` MLP down projection。
4. Causal-conv1d width 4 的 decode update。
5. GDN RMSNorm + gate + recurrent update 融合。
6. 全注意力 6 层的 FlashAttention/KV Cache。
7. 视觉 MLP 与 patch/merge，只在 TTFT 优化阶段重点处理。

本地 CUDA profiler 显示 GEMV + GEMM 占 self CUDA time 86.18%，与上述 decode 小矩阵路径一致。
