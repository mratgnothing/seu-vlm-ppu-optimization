# 2026-08-28 PPU Attention Prep 单入口融合

## 目标与边界

Qwen3.5-2B 的 6 个全注意力层在每个 decode token 上分别执行 Q、K、V 三次
Linear，然后对 Q/K 做 RMSNorm 与 RoPE。该候选不改变 BF16 GEMV、归一化或旋转位置
编码的计算顺序，只把下面这段固定形状工作放进一次 PyTorch C++ extension 入口：

```text
hidden [1,1,2048]
  -> q_proj [4096] + k_proj [512] + v_proj [512]
  -> q/k RMSNorm + RoPE
  -> query [1,8,1,256], key/value [1,2,1,256], gate [1,1,8,256]
```

Prefill、训练模式、非 batch-1 或非单 token 输入全部回退原始实现。候选默认关闭，只有
显式设置 `SEU_PPU_ACBLAS_ATTENTION_PREP_BUILD_DIR` 才挂载。

## 实现

- `acblas_attention_prep_wrapper.cpp` 在同一 acBLAS handle、mutex 和当前 stream 上依次
  提交三个原形状 `acblasGemvEx`，随后调用已经验证的 Q/K RMSNorm+RoPE HGGC 核；
- q/k/v 权重在加载阶段拼为 `[5120,2048]` 的连续存储，原 `nn.Linear` 参数改为其 view，
  因而 prefill 仍走标准 Linear 且不保留第二份常驻权重；
- 每个注意力层持久保存 projected/query/key scratch，value 与 gate 是 projected 的 view；
- patch 时绑定当前 PPU stream。其他 stream 会在任何 kernel 提交前被拒绝，防止并发请求
  覆写共享 scratch；当前契约明确是赛事单请求串行 decode，不宣称多流安全；
- 当前 Transformers `DynamicLayer.update()` 使用 `torch.cat` 生成新的 KV cache，历史
  key/value 不会别名到下一 token 会覆写的 scratch。

## 已验证证据

真实 Qwen3.5 Attention 模块在 PPU-ZW810E 上得到：

| 指标 | 原路径 | 候选 | 结果 |
|---|---:|---:|---:|
| 模块边界耗时 | 0.080652 ms | 0.019668 ms | `4.1006x` |
| Q/K/V/gate | - | - | 全部 bit-exact |
| prefill Linear | - | - | bit-exact |
| prefill 回退 | - | - | 已确认不进入自定义入口 |
| 最大绝对误差 | - | - | 0 |
| 非 patch stream | - | - | 提交前拒绝 |
| scratch 生命周期 | - | - | 连续 decode 复用同一存储 |
| `hggc-memcheck` | - | - | 0 errors |

`hggc-memcheck` 插桩后的耗时不用于性能结论。`4.1006x` 也只代表注意力准备子边界，
不能外推为整模型吞吐提升。

## 整模型门禁与结论

固定 128-token 共运行三组独立交错 A/B：

| 轮次 | 配对数 | 全文一致 | 候选获胜 | 成对中位 | 成对均值 | 门禁 |
|---|---:|---:|---:|---:|---:|---|
| r1 | 8 | 8/8 | 3/8 | `0.9900x` | `1.0008x` | 失败 |
| r2 | 16 | 16/16 | 9/16 | `1.0114x` | `1.0105x` | 通过 |
| r3 | 32 | 32/32 | 19/32 | `1.0047x` | `1.0019x` | 通过 |
| 合并 | 56 | 56/56 | 31/56 | `1.0047x` | `1.0042x` | 仅约 +0.4% |

扩大重复后，固定长结果接近 1 且方向受系统噪声影响。继续执行两轮 CN20：

| 轮次 | 平均吞吐 baseline→候选 | 全文一致 | Accuracy | 候选获胜 | 成对中位 | 成对均值 | 门禁 |
|---|---:|---:|---:|---:|---:|---:|---|
| r1 | `119.355→119.988` | 20/20 | 85%→85% | 12/20 | `1.0158x` | `1.0061x` | 通过 |
| r2 | `122.806→119.867` | 20/20 | 85%→85% | 8/20 | `0.9852x` | `0.9765x` | **失败** |

第二轮的中位和均值均低于 1，违反预先写死的“两轮性能方向一致”规则。因此停止
4029 完整集和 profile，不为挑选有利样本追加轮次。该实现保留为默认关闭的负实验：
模块边界减少 Python/ATen 调度很快，但全模型只有 6 个全注意力层，局部节省不足以
稳定覆盖 extension、stream 设置和其余 24 层解码开销。

## 晋级规则

1. 固定同一输入生成 128 token，执行 8 对交错 AB/BA；要求输出 hash 完全一致，且
   成对中位与均值均大于 1；
2. 若固定长通过，执行 CN20 至少两轮；要求每轮 20/20 文本、答案和 token 数一致，
   且性能方向一致；
3. 通过后才运行 profile 和完整公开集。实际第二轮 CN20 已失败，因此这两项未运行；
   候选作为负实验保留、不开启正式 wrapper 配置。

## 复现入口

```bash
cd ppu/custom_ops
SEU_PPU_GDN_LIBRARY=/path/to/libseu_ppu_gdn.so \
  python build_acblas_attention_prep_extension.py

python smoke_acblas_attention_prep_module.py \
  --model-path /path/to/Qwen3.5-2B \
  --gdn-library /path/to/libseu_ppu_gdn.so \
  --build-dir build/acblas_attention_prep_extension

# 固定 128-token 八对；MODE=multisample 可复用同一入口跑 CN20/完整集。
bash ppu/microbench/run_acblas_attention_prep_gate.sh
```

证据：

- `results/acblas-attention-prep-smoke-20260828.json`
- `results/acblas-attention-prep-memcheck-20260828.txt`
- `results/acblas-attention-prep-ab128-20260828.json`
- `results/acblas-attention-prep-ab128-r2-20260828.json`
- `results/acblas-attention-prep-ab128-r3-20260828.json`
- `results/acblas-attention-prep-cn20-r1-20260828.json`
- `results/acblas-attention-prep-cn20-r2-20260828.json`
