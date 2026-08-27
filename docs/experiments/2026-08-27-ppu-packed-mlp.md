# 2026-08-27 PPU packed-MLP gate/up 投影实验

## 结论

在已验证的 all-five 融合路径上，将 Qwen3.5-2B 的 24 个 MLP gate/up 权重重排为
共享的 `[12288, 2048]` packed storage，并只在 batch=1 decode 时把两次
`2048→6144` 投影合成一次 `2048→12288`。这是模型图与权重布局优化，不是第六个
HGGC kernel。

固定中文前 20 条两轮吞吐为 `96.506/96.715 token/s`，相对 eager `49.737`
提高 `94.04%/94.46%`；Accuracy 均为 85%。两轮之间以及相对 all-five 的逐样本
答案、正确性、token 数均完全一致，所以 packed-MLP 没有扩大此前已有的 5/20
生成长度漂移。

## 正确性和存储验收

- 独立真实 `Qwen3_5MLP` smoke：decode 与 prefill BF16 输出 bit-exact。
- gate/up Parameter 均为 packed buffer 的 view；没有保留第二份常驻权重。
- allocator 观测到的常驻增量为 20,480 bytes，而非约 1.2 GiB。
- 51-token 单样本三次 A/B：eager 与候选答案、token 数、文本 SHA-256 完全一致。
- 单样本候选中位吞吐 `99.388 token/s`，同轮 eager 为 `49.358 token/s`；相对之前
  all-five 的 `93.387 token/s` 约提高 6.43%。

## 固定 20 条结果

| 路径 | TTFT ms | token/s | Accuracy | 相对 eager 吞吐 | 相对 all-five r1 |
|---|---:|---:|---:|---:|---:|
| eager current | 118.493 | 49.737 | 17/20 | - | - |
| all-five r1 | 124.930 | 93.918 | 17/20 | +88.83% | - |
| all-five r2 | 118.227 | 94.889 | 17/20 | +90.78% | - |
| all-five + packed-MLP r1 | 119.401 | 96.506 | 17/20 | +94.04% | +2.76% |
| all-five + packed-MLP r2 | 115.916 | 96.715 | 17/20 | +94.46% | +2.98% |

不同轮存在设备运行抖动，因此正式结论取两轮都为正且逐样本一致，而不是挑选最快值。

## Profile 机制证据

同一 226-token prompt、4-token warmup、16-token trace：

| 指标 | all-five | + packed-MLP | 变化 |
|---|---:|---:|---:|
| Self PPU ms | 121.871 | 119.956 | -1.57% |
| `aten::linear` | 3,090 | 2,730 | -360 |
| `aten::mm` | 2,992 | 2,632 | -360 |
| decode `2048→6144` mm | 990 | 270 | -720 |
| decode `2048→12288` mm | 0 | 360 | +360 |
| `cudaLaunchKernel` | 17,088 | 17,088 | 0 |

原来 24 层 × 15 个被 profile 的 decode token × 2 次 gate/up 投影，共 720 次
`2048→6144`；现在变为 360 次 `2048→12288`。剩余 270 次 2048→6144 是
18 层 GDN qkv。底层 launch 数没有减少，说明 PPU GEMV 后端会把更宽输出拆分；端到端
收益来自减少 ATen/Python 调度和更高效的宽矩阵访存，而不是“一次 mm 就一定一次核”。

## 遇到的问题和解决方案

1. A/B 脚本新增 `torch.cuda.empty_cache()` 后漏导入 `torch`，在正式推理前触发
   `NameError`。补导入并跑语法/单元测试后重传。
2. 第一次公开 20 条错误使用 `SEU_PPU_GDN_ENABLE`、`SEU_PPU_CAUSAL_CONV_ENABLE`
   等非正式变量名，结果 meta 显示所有挂载数为 0，吞吐仅 48.617 token/s。改用
   `SEU_PPU_GDN_LIBRARY` 和 `SEU_PPU_CONV_ENABLE` 等正式变量，并以 meta
   `18/18/49/18/6/24` 作运行有效性门禁。
3. profile 传 `--device ppu` 被 Transformers 的 `device_map` 校验拒绝。该 SDK
   通过 PyTorch CUDA 兼容层暴露 PPU，改用 `cuda:0` 后正常运行，设备仍识别为
   PPU-ZW810E。
4. packed 后 `cudaLaunchKernel` 没下降。trace 证明 ATen mm 数和矩阵形状按预期变化，
   但后端内部仍拆分宽 GEMV；因此不能宣称减少了设备 launch，只保留有两轮端到端
   数据支持的约 2.8% 增益。

## 复现

```bash
cd ppu/custom_ops
python smoke_packed_mlp_integration.py --warmup 32 --iters 400

export SEU_PPU_PACK_MLP_ENABLE=1
python scripts/benchmark_ppu_gdn_ab.py \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_cn.tsv \
  --output /tmp/all5-packed-mlp-ab.json \
  --repeats 3 --max-new-tokens 64 \
  --gdn-tiles 4 --fuse-conv --conv-threads 96 \
  --fuse-rmsnorm --rmsnorm-threads 512 \
  --fuse-gated-rmsnorm --gated-rmsnorm-threads 128 \
  --fuse-qk-rope --pack-mlp
```

原始 JSON 与 trace 保存在隔离服务器 `/mnt/workspace/seu/results/`；模型、数据和大型
trace 不进入 Git。
