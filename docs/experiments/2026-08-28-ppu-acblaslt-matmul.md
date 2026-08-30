# 2026-08-28 PPU acBLASLt decode Matmul 调查

## 结论

在完整 GDN gate-prep 栈通过公开集门禁后，本实验继续检查 PPU acBLASLt 是否能优化
Qwen3.5-2B 的 decode GEMM/GEMV。最终结论是：**不接入正式 wrapper**。

- SDK 公开 epilogue 只有 Bias、ReLU、GELU 及其组合，没有 Qwen MLP 所需的
  `SiLU(gate) * up`；不能用正式接口完成 packed GEMM + SwiGLU 融合。
- 对四个真实 BF16 decode 形状扫描 32 个 heuristic，全部与 `torch.mv` bit-exact。
- packed MLP、MLP down 和 GDN qkv 的最佳低层收益只有 `1.21%/2.69%/1.91%`，未过
  3% 集成门槛。
- `2048x2048` 最佳 heuristic 25 为 `1.0577x`，进入模块集成；用每模块 scratch
  消除输出分配后，隔离 extension 模块级达到 `1.2797x`，decode/prefill 仍 bit-exact。
- 但完整模型固定 128-token 八对 AB/BA 的成对中位只有 `0.9898x`，仅 3/8 获胜；
  因此它是负实验，不设置正式环境开关，也不计入最终性能栈。

## SDK 接口边界

`acblasLt.h` 的 `acblasLtEpilogue_t` 列出 Default、Bias、ReLU、GELU、Aux 和反向
变体，没有 SiLU/Swish。PPU 实现还会拒绝四个矩阵均为 row-major 的组合。探针因此
不复制权重，而采用 BLAS 等价解释：行主序 `[N,K]` 的权重字节等同于列主序
`[K,N]`，再计算其转置与列向量 `[K,1]` 的乘积，输出 `[N,1]`。

## 低层 heuristic 扫描

随机 BF16 输入，32 次预热、200 次计时，并轮换多份权重以避免单权重缓存偏差：

| 权重形状 `N x K` | 位置 | `torch.mv` ms | acBLASLt 最佳 ms | 最佳 heuristic | 加速 | 决策 |
|---:|---|---:|---:|---:|---:|---|
| 12288 x 2048 | packed MLP gate/up | 0.026488 | 0.026171 | 6 | 1.0121x | 止损 |
| 2048 x 6144 | MLP down | 0.014612 | 0.014229 | 14 | 1.0269x | 止损 |
| 2048 x 2048 | attention/GDN output | 0.007633 | 0.007217 | 25 | 1.0577x | 进入模块验证 |
| 6144 x 2048 | GDN qkv | 0.016972 | 0.016653 | 9 | 1.0191x | 止损 |

四个形状的 32/32 heuristic 均通过数值门禁，最佳结果均为 bit-exact。这里的 3%
门槛只决定是否值得支付 PyTorch extension 集成成本，不代表 3% 就足以形成端到端收益。

## 方阵模块与整模结果

方阵候选复用现有 Torch/CUDA 与 PPU SDK 头文件隔离的 C ABI extension，并为每个
Linear 注册独立 `[1,1,2048]` BF16 scratch。prefill、非 BF16、非 decode 形状均回退
原始 forward。

模块级 2000 次计时：

| eager Linear | acBLASLt + scratch | 加速 | decode | prefill | scratch |
|---:|---:|---:|---|---|---|
| 0.009051 ms | 0.007073 ms | 1.2797x | exact | exact | 指针复用 |

完整 gate-prep 栈上，42 个结构匹配模块被 patch；由于 grouped GDN 已绕过部分原始
Linear forward，profile 中实际减少 `360 = 24 x 15` 次 `aten::linear/mm`。固定
128-token 八对结果：

| 指标 | 正式 gate-prep 基线 | + acBLASLt 方阵 |
|---|---:|---:|
| 吞吐中位数 | 111.743 token/s | 109.622 token/s |
| 成对中位速度比 | - | 0.989835x |
| 成对均值速度比 | - | 0.983703x |
| 获胜 | - | 3/8 |
| 完整文本 | - | 8/8 exact |

## Profile 解释

16-token profile 中，候选把 `aten::linear` 从 1650 降至 1290、`aten::mm` 从 1552
降至 1192；但主 `gemvt_op` 聚合从 1546 次/27.772 ms 变为 1816 次/28.889 ms。
即 heuristic 25 在整图中把部分方阵调用拆为更多设备工作，抵消了 dispatcher 与
scratch 的局部收益。`cudaLaunchKernel` 总数仍为 14363，未形成图级 launch 减少。

因此不能用模块级 `1.2797x` 宣称模型加速；固定长 A/B 是最终止损依据。该候选最终
使用独立 `seu_acblaslt_square_ext` 构建，正式 grouped-acBLAS extension 已恢复为不
链接 acBLASLt 的原依赖集合。

## 复现入口与证据

- `ppu/microbench/acblaslt_matmul_wrapper.cpp`
- `ppu/microbench/acblaslt_matmul_sweep.py`
- `ppu/microbench/build_acblaslt_matmul_probe.py`
- `ppu/custom_ops/ppu_acblaslt_square.py`
- `ppu/custom_ops/smoke_acblaslt_square_module.py`
- `scripts/benchmark_ppu_packed_gdn_ab.py --acblaslt-square-ab`
- `results/acblaslt-sweep-summary-20260828.json`
- `results/acblaslt-square-isolated-smoke-20260828.json`
- `results/acblaslt-square-ab128-20260828.json`
- `results/acblaslt-square-profile-ab-20260828.json`
- `results/acblaslt-square-profile-baseline-summary-20260828.json`
- `results/acblaslt-square-profile-candidate-summary-20260828.json`

原始四形状 JSONL 和两份约 76 MiB trace 保存在 ignored artifact/CPFS 快照，不进入
源码提交包。
