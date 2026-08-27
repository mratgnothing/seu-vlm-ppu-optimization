# 2026-08-28 PPU GDN gate-prep 融合

## 结论

Qwen3.5-2B 的 18 个 `Qwen3_5GatedDeltaNet` 层在每个 cached decode token 都会执行：

```python
beta = b.sigmoid()
g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
```

本实验在模型加载时缓存 FP32 `exp(A_log)`，并用一个 HGGC kernel 合并 BF16
`b.sigmoid()`、`a` 的 FP32 cast、bias add、Softplus、乘法和取负；每层用
thread-local scratch 复用 FP32 `g` 与 BF16 `beta` 输出。prefill、训练、非 BF16、
batch/seq 非 1 和无 recurrent cache 的路径全部回退 Transformers 原实现。

最终固定 128-token 六对交错 AB/BA 全部获胜且全文 SHA-256 一致，配对中位
`1.0839x`。中文公开集前 20 条两轮分别得到 `1.0811x/1.0863x` 配对中位提升，
`19/20` 与 `17/20` 获胜；两轮均 `20/20` 全文一致、Accuracy 保持 85%。这是相对
已经包含五类 HGGC、packed-MLP、grouped-acBLAS GDN 和 48-edge residual-RMSNorm
的最终基线的增量，不是相对原始 eager 的数字。

最终中文完整公开集 4029 条 paired AB/BA 也已通过：两路 Accuracy 均为
`3374/4029 = 83.7429%`，`4029/4029` 完整文本、答案和 token 数一致；吞吐成对中位
`1.08623x`，3882/4029 获胜。完整集只承担无回退门禁，长时间运行中的平均吞吐不
替代固定长无 profiler 性能结果。

## 数学与舍入契约

候选保持原模型的精度边界：

- `A_log` 先转 FP32 再求 `exp`，只因 eval 推理期间权重不变而缓存；
- `a + dt_bias` 和 Softplus 在 FP32 中计算，阈值仍为 20；
- `beta` 在 Sigmoid 后显式舍入回 BF16，再交给 recurrent state kernel；
- `g` 保持 FP32，现有 recurrent kernel 继续执行 `decay = exp(g)`；
- recurrent state 的 FP32 更新核没有改动。

随机输入和 `[-30, 30]` 边界输入中，`g` FP32 与 `beta` BF16 均 bit-exact。独立
gate-prep 微基准为 `0.028592 -> 0.020680 ms`，即 `1.3826x`；该数字只证明候选
算子本身值得进入整模门禁，不作为最终模型提升。

## 最终无 profiler 结果

### 固定 128-token，六对 AB/BA

| 指标 | 最终优化基线 | + gate-prep |
|---|---:|---:|
| 吞吐中位数 | 103.895 token/s | 112.606 token/s |
| 配对中位速度比 | - | 1.083854x |
| 配对均值速度比 | - | 1.079947x |
| 获胜 | - | 6/6 |
| 完整文本 | - | 6/6 exact |

### 中文公开集固定前 20 条

| 轮次 | 基线 token/s | 候选 token/s | 配对中位 | 获胜 | 全文 exact | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 101.651 | 109.275 | 1.081132x | 19/20 | 20/20 | 85% / 85% |
| r2 | 100.085 | 107.083 | 1.086290x | 17/20 | 20/20 | 85% / 85% |

TTFT 在两轮中方向不一致，因此本实验只声明稳态 decode throughput 收益，不声明
TTFT 改善。公开题目只用于统一回归门禁，没有根据题号、类别、标签或答案调整实现。

### 中文完整公开集 4029 条

| 指标 | 最终优化基线 | + gate-prep |
|---|---:|---:|
| Accuracy | 3374/4029 (83.7429%) | 3374/4029 (83.7429%) |
| 平均吞吐 | 101.080 token/s | 109.797 token/s |
| 成对中位/均值速度比 | - | 1.086229x / 1.087416x |
| p05 / p95 速度比 | - | 1.009529x / 1.166900x |
| 获胜/失败 | - | 3882 / 147 |
| exact 文本/答案/token 数 | - | 4029/4029 |

该完整集运行没有 profiler 插桩，且未按题目、答案或类别选择候选。平均吞吐受约两小时
长测期间系统状态影响，因此最终性能声明仍以固定 128-token paired A/B 和 CN20 两轮
为主；完整集用于证明 gate-prep 没有扩大数值漂移。

## Profile 机制证据

同一 226-token prompt、2-token warmup、16-token profile：

| 指标 | 基线 | 候选 | 变化 |
|---|---:|---:|---:|
| `cudaLaunchKernel` | 16,253 | 14,363 | -1,890 (-11.63%) |
| `aten::_to_copy` | 1,214 | 674 | -540 |
| `aten::add` | 1,966 | 1,696 | -270 |
| `aten::mul` | 2,692 | 2,422 | -270 |
| `aten::empty_like` | 3,982 | 3,982 | 0 |
| `aten::empty_strided` | 4,931 | 4,391 | -540 |
| Self CPU | 366.100 ms | 324.074 ms | -11.48% |
| Self PPU | 119.365 ms | 112.982 ms | -5.35% |

原始 trace 进一步显示 Sigmoid、Softplus、Exp、Neg、Add、Mul 各减少 270 次，两个
cast 合计减少 540 次；新 `gdn_gate_prep_decode_kernel` 恰好执行 270 次，合计约
0.545 ms。`1890 = 7 x 270` 与 18 层乘 15 个被 profile 的 decode token 完全一致。

首版每次调用为 `g/beta` 新建两个 tensor，使 `empty_like` 增加 540 次；最终版用
thread-local scratch 消除了这项回退，并在第二轮固定长测试中仍保持 exact 和 6/6
获胜。

## 安全与门禁

- `hggc-memcheck`：`ERROR SUMMARY: 0 errors`；
- 随机和边界 `g/beta`：bit-exact；
- 固定 128-token：全文 exact；
- CN20 两轮：均 20/20 全文 exact，Accuracy 不变；
- 中文完整公开集：4029/4029 全文 exact，Accuracy 同为 3374/4029；
- 候选只通过 `SEU_PPU_GDN_GATE_PREP_ENABLE=1` 显式启用；
- 形状、dtype、设备、eval、cache 任一契约不满足时回退原 forward；
- 公开完整集门禁已通过；主办方私有集仍是最终外部门禁。正式 wrapper 继续要求显式
  环境开关，但 gate-prep 已是当前推荐提交配置的一部分。

## 复现

```bash
OUTPUT_DIR=build/gate-prep ./build_gdn_shared.sh

python smoke_gdn_gate_prep_integration.py \
  --library build/gate-prep/libseu_ppu_gdn.so \
  --warmup 50 --iters 1000 --repeats 5

python benchmark_ppu_packed_gdn_ab.py \
  --repo-root . \
  --model-path /mnt/workspace/seu/Qwen3.5-2B \
  --dataset-path /mnt/workspace/seu/datasets/mmbench/mmbench_dev_cn.tsv \
  --gdn-library build/gate-prep/libseu_ppu_gdn.so \
  --projection-backend acblas-grouped \
  --acblas-build-dir build/acblas_linear_extension \
  --gate-prep-ab --force-max-new-tokens \
  --max-new-tokens 128 --repeats 6 \
  --output results/gate-prep-ab128.json
```

正式 wrapper 额外设置：

```bash
export SEU_PPU_GDN_GATE_PREP_ENABLE=1
```

## 证据

- `results/ppu-gdn-gate-prep-smoke-20260828.json`
- `results/ppu-gdn-gate-prep-ab128-20260828.json`
- `results/ppu-gdn-gate-prep-cn20-r1-20260828.json`
- `results/ppu-gdn-gate-prep-cn20-r2-20260828.json`
- `results/ppu-gdn-gate-prep-profile-ab-20260828.json`
- `results/ppu-gdn-gate-prep-profile-baseline-summary-20260828.json`
- `results/ppu-gdn-gate-prep-profile-candidate-summary-20260828.json`
- `results/ppu-gdn-gate-prep-memcheck-20260828.txt`
- `results/gate-prep-scratch-cn-full4029-summary.json`
- `results/ppu-final-formal-wrapper-smoke-20260828.json`
