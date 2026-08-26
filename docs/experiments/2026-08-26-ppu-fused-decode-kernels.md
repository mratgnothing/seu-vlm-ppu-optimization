# 2026-08-26 PPU decode 融合算子迭代

## 结论

在 PPU-ZW810E 的 Transformers eager fallback 上实现并接入五类 HGGC decode
融合算子：recurrent GDN、causal-conv state update、2048 维 RMSNorm、128 维 gated
RMSNorm、full-attention q/k RMSNorm+partial RoPE。固定中文前 20 条的同机对照中，
all-five 两次吞吐为 `93.918/94.889 token/s`，相对 eager `49.737` 提高
`88.83%/90.78%`，Accuracy 均为 85%。

这是公开 dev 小样本工程结果，不代表私有评测；全融合有 5/20 生成 token 数变化，
所以 norm 路径仍是 opt-in 候选。

## 迭代数据

| 路径 | TTFT ms | token/s | Accuracy | 答案差异 | token 数差异 |
|---|---:|---:|---:|---:|---:|
| eager current | 118.493 | 49.737 | 17/20 | - | - |
| fused GDN | 119.460 | 61.350 | 17/20 | 0/20 | 3/20 |
| GDN + conv | 117.262 | 63.911 | 17/20 | 0/20 | 3/20 |
| all-four | 119.677 | 81.307 | 17/20 | 0/20 | 5/20 |
| all-five r1 | 124.930 | 93.918 | 17/20 | 0/20 | 5/20 |
| all-five r2 | 118.227 | 94.889 | 17/20 | 0/20 | 5/20 |

单样本 51-token 三次重复中，all-four 中位吞吐为 `83.841 token/s`，同轮 eager
为 `48.348 token/s`；答案相同，但生成长度为 56 vs 51，因此该单样本不能用
exact-text gate 判为完全等价。

stride 修正后的 all-five 单样本重新三次 A/B：eager/all-five 中位吞吐为
`49.060/93.387 token/s`，51-token 输出的答案、token 数和文本 SHA-256 全部一致。

## Profile 变化

同一 226-token prompt、2-token warmup、16-token profile：

| 路径 | Self CPU ms | Self PPU ms | 主要变化 |
|---|---:|---:|---|
| eager | 854.810 | 173.799 | 37,293 launches，大量 reduce/copy/cat |
| GDN + conv | 648.316 | 159.363 | GDN 收敛为 270 个 kernel；conv decode 的 270 组 cat/conv/silu 消失 |
| GDN + conv + RMSNorm | 498.126 | 143.390 | 735 个标准 RMSNorm 收敛为 2.371 ms 单核组 |
| all-four | 514.366 | 131.899 | gated RMSNorm 也已挂载；四类模块数为 18/18/49/18 |
| all-five | 409.545 | 121.871 | q/k norm+RoPE 挂载 6 层；90 次新核合计约 0.216 ms |

all-four 后最大设备热点重新变为运行时 `gemvt_op`：1,906 次、29.359 ms；
`2048→6144` 的 990 次 `aten::mm` 为 15.959 ms。自定义 GDN 在 profiler 下为
270 次、14.328 ms；标准 RMSNorm 为 735 次、2.377 ms。事件微基准不带
profiler 时 GDN 约 0.03136 ms/call。不同 profile 的 Self CPU 有运行噪声，
瓶颈判断以同口径设备事件和无 profiler 端到端 benchmark 为主。

all-five 相对 all-four 的机制证据：`cudaLaunchKernel` 从 19,878 降到 17,088，
`aten::cat` 从 747 降到 387，`empty_strided` 从 5,472 降到 4,932；Self CPU/PPU
分别再下降 20.38%/7.60%。剩余最大热点仍是 GEMV/GEMM。

## 正确性边界

- causal-conv、RMSNorm、gated RMSNorm、q/k RMSNorm+RoPE 的独立随机 BF16 测试
  均 bit-exact；q/k smoke 使用模型真实的非连续 query head stride。
- GDN 相对 Transformers eager 的独立测试最大 state/output 误差为
  `5.96e-8 / 0`。
- 五类共享库调用路径分别通过 `hggc-memcheck` 单次执行，均为 0 errors。
- 自回归会放大极小归约差异；20 条中 GDN 导致 3 条生成长度变化，标准 RMSNorm
  新增 1 条，gated RMSNorm 再新增 1 条，但解析答案和 Accuracy 未变。
- 在完整公开集、更多类别和主办方私有口径验证前，不把 20 条“Accuracy 不变”外推为
  最终无损。

## 复现

详细构建、smoke、环境开关、问题和方案见
[PPU decode 融合算子 README](../../ppu/custom_ops/README.md)。整模型 A/B 使用：

```bash
python scripts/benchmark_ppu_gdn_ab.py \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_cn.tsv \
  --output /tmp/all4-ab.json \
  --repeats 3 --max-new-tokens 64 \
  --gdn-tiles 4 --fuse-conv --conv-threads 96 \
  --fuse-rmsnorm --rmsnorm-threads 512 \
  --fuse-gated-rmsnorm --gated-rmsnorm-threads 128 \
  --fuse-qk-rope
```

原始 JSON 和 trace 保存在隔离服务器 `/mnt/workspace/seu/results/`，不提交模型、
数据、trace 或私有环境信息。
