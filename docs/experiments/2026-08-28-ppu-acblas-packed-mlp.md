# 2026-08-28 PPU 单入口 acBLAS packed-MLP

## 结论

Qwen3.5-2B 的 decode MLP 已实现一个结构专用候选：在一次 PyTorch C++ extension
入口内依次提交 packed gate/up GEMV、bit-exact HGGC SwiGLU 和 down GEMV，并复用
每层三个持久 BF16 scratch。它不改变三段运算或 BF16 舍入顺序，prefill、训练模式和
非 `[1,1,2048]` 输入均回退已有 packed-MLP forward。

当前证据全部通过：

- 单模块 decode/prefill bit-exact，模块级 `1.2288x`；
- 固定 128-token 八对 8/8 获胜，成对中位 `1.1336x`，全文一致；
- CN20 两轮均 20/20 获胜、20/20 全文一致、Accuracy 均为 85%；成对中位分别为
  `1.1212x/1.1122x`；
- `hggc-memcheck` 报告 `ERROR SUMMARY: 0 errors`；
- 正式 wrapper public validation 通过，meta 记录 24 个 acBLAS packed-MLP module。
- 中文完整公开集 4029/4029 文本、答案和 token 数一致，两路 Accuracy 均为
  3374/4029；平均吞吐 `109.993→122.445 token/s`，成对中位 `1.1125x`、
  平均 `1.1146x`，3939/4029 获胜，P05 仍为 `1.0314x`。

这些结果只适用于当前公开集、单请求串行 decode 和当前 PPU 镜像，不能外推为主办方
私有集成绩。

## 为什么上一轮方阵 acBLASLt 失败，而这次成功

上轮只替换 2048 方阵 Linear。虽然单模块变快，但它增加额外设备提交，端到端反而
约退化 1.02%。本轮选择模型结构边界，而不是任意单层边界：

```text
input [1,1,2048]
  -> packed gate/up GEMV [12288,2048]
  -> gate/up views [6144] + HGGC SwiGLU [6144]
  -> down GEMV [2048,6144]
  -> output [1,1,2048]
```

桥接层只调用一次 `acblasSetStream`，然后在同一 stream 上按原顺序提交两次
`acblasGemvEx` 和一次 `seu_ppu_swiglu_decode_bf16`。Torch extension 只负责 tensor
契约和当前 stream；SDK/HGGC 头留在单独翻译单元，避免 half/BF16 类型冲突。

每个 MLP 持久保存：

- projected scratch：`[1,1,12288]` BF16；
- activated scratch：`[1,1,6144]` BF16；
- output scratch：`[1,1,2048]` BF16。

24 层总计约 0.94 MiB，不复制模型权重。该 scratch 设计面向赛事 benchmark 的单请求
串行 decode；同一模型实例并发多请求尚未验证，不能宣称线程安全。

## 模块与端到端结果

单个真实 `Qwen3_5MLP`，10 次预热、100 次计时：

| packed Torch MLP | 单入口候选 | 加速 | decode | prefill | scratch |
|---:|---:|---:|---:|---:|---:|
| 0.041666 ms | 0.033907 ms | 1.2288x | exact | exact | 复用 |

同一 sample 固定生成 128 token，完整 gate-prep/grouped-GDN 基线，8 对 AB/BA：

| 路径 | 中位 TTFT | 中位 token/s | 成对中位/均值 | 获胜 | 全文一致 |
|---|---:|---:|---:|---:|---:|
| 正式 gate-prep 基线 | 116.123 ms | 110.150 | - | - | - |
| + 单入口 packed-MLP | 114.706 ms | 126.043 | `1.1336x/1.1319x` | 8/8 | 是 |

CN20 独立两轮：

| 轮次 | baseline token/s | candidate token/s | 成对中位/均值 | 获胜 | Accuracy | 全文一致 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 108.451 | 122.350 | `1.1212x/1.1301x` | 20/20 | 85%/85% | 20/20 |
| r2 | 109.652 | 121.297 | `1.1122x/1.1065x` | 20/20 | 85%/85% | 20/20 |

完整中文公开集 paired A/B（64 max-new-tokens）：

| baseline token/s | candidate token/s | 成对中位/均值 | P05 | 获胜 | Accuracy | exact |
|---:|---:|---:|---:|---:|---:|---:|
| 109.993 | 122.445 | `1.1125x/1.1146x` | `1.0314x` | 3939/4029 | 3374/4029（两路） | 4029/4029 |

## Profile 机制证据

同一 16-token profile；首 token 为 prefill，因此 decode 有 15 步：

| 事件 | 基线 | 候选 | 差值 |
|---|---:|---:|---:|
| `aten::linear` | 1650 | 930 | -720 |
| `aten::mm` | 1552 | 832 | -720 |
| `aten::mul` | 2422 | 2062 | -360 |
| `cudaLaunchKernel` | 14363 | 14003 | -360 |
| 三类 `gemvt_op` 合计 | 2446 | 2446 | 0 |

`720 = 24 MLP × 2 GEMV × 15 decode steps`，说明两次 Linear 的 Python/ATen 入口被
完全绕过；`360 = 24 × 15`，说明 Torch 的 SiLU+mul 两个 elementwise launch 被一个
HGGC SwiGLU launch 替代。GEMV 数和主 GEMV 时间基本不变，因此本轮加速不是凭空提高
矩阵算力，而是把同一 MLP 的 GEMV、激活和内存生命周期放到正确的提交边界。

## 启用与复现

先构建 GDN 共享库和 packed-MLP extension：

```bash
cd ppu/custom_ops
./build_gdn_shared.sh
python build_acblas_packed_mlp_extension.py
python smoke_acblas_packed_mlp_module.py \
  --build-dir build/acblas_packed_mlp_extension
```

正式 wrapper 在已有高性能配置上增加：

```bash
export SEU_PPU_PACK_MLP_ENABLE=1
export SEU_PPU_ACBLAS_PACKED_MLP_BUILD_DIR="$PWD/ppu/custom_ops/build/acblas_packed_mlp_extension"
export SEU_PPU_ACBLAS_PACKED_MLP_SWIGLU_THREADS=128
```

全量门禁入口：

```bash
bash ppu/microbench/run_acblas_packed_mlp_full_gate.sh
```

## 证据文件

- `results/acblas-packed-mlp-smoke-20260828.json`
- `results/acblas-packed-mlp-memcheck-20260828.txt`
- `results/acblas-packed-mlp-ab128-20260828.json`
- `results/acblas-packed-mlp-cn20-r1-20260828.json`
- `results/acblas-packed-mlp-cn20-r2-20260828.json`
- `results/acblas-packed-mlp-profile-ab-20260828.json`
- `results/acblas-packed-mlp-profile-baseline-summary-20260828.json`
- `results/acblas-packed-mlp-profile-candidate-summary-20260828.json`
- `results/acblas-packed-mlp-formal-wrapper-smoke-20260828.json`
- `results/acblas-packed-mlp-cn-full4029-summary.json`
- `results/acblas-packed-mlp-final-formal-wrapper-smoke-20260828.json`
- `results/acblas-packed-mlp-final-memcheck-20260828.txt`
