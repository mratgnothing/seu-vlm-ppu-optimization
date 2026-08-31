# 独立首 Token profile 与 multi-row prefill 融合

## 本轮范围

本轮只做两个相连方向：

1. 把原来的“1 次 prefill + 15 次 decode”profile 改成只生成 1 token，单独观察
   warm multimodal prefill；
2. 根据该 profile 的最大可复用热点，把已有 RMSNorm、gated-RMSNorm 和
   residual+RMSNorm HGGC 核放开到 multi-row prefill。

没有继续派生第二套 kernel 或多轮参数搜索。

## 独立首 Token profile

样本 3000984 的 prompt 为 226 token。先用相同 shape 预热一次，随后在
`max_new_tokens=1` 下 profile 一次完整多模态 prefill 和首 token 选择，因此 trace 不再
混入后续 decode step。

baseline 的 profiler 聚合结果：

- Self PPU：53.154 ms；
- Self CPU：227.788 ms（profiler 插桩口径，不作为 TTFT）；
- 最大两组 elementwise 合计约 15.449 ms，占 Self PPU 约 29.1%；
- `[226,2048]×[2048,6144]` MLP GEMM 主组为 4.820 ms；
- 另有大量 reduce、copy 和逐行 norm 操作。

因此本轮没有先重写 GEMM，而是复用已经存在、天然按 `rows` 启动的三个 norm/residual
kernel。实际覆盖 49 个 2048 维 RMSNorm、18 个 128 维 gated-RMSNorm 和 24 个 decoder
layer 的两条 residual+RMSNorm 边。

candidate profile 的 Self PPU 为 50.531 ms，相对 baseline 为 `1.0519x`；Self CPU
为 237.487 ms，说明 ctypes/dispatcher 提交成本没有同步下降。profile 插桩总时长不用于
端到端结论，最终数字来自无 profiler 的逐样本 AB/BA。

## 双语配对 A/B

每个模型只加载一次，逐题交替 AB/BA；每题最多生成 64 token。

| 数据 | 平均 TTFT baseline→candidate | TTFT 配对中位/均值 | 获胜 | Accuracy | 答案一致 | 全文一致 |
|---|---:|---:|---:|---:|---:|---:|
| CN20 | 117.099→111.362 ms | 1.04859x / 1.05266x | 18/20 | 17→17 | 20/20 | 12/20 |
| EN20 | 119.189→112.943 ms | 1.04477x / 1.06006x | 18/20 | 18→18 | 20/20 | 20/20 |

TTFT 在两个语言上都获得约 4.5%--4.9% 的配对中位提升，且 18/20 样本获胜，方向比
此前 KV 预分配和视觉 Token 更稳定。

吞吐结果并非双语无回退：

| 数据 | 平均 token/s baseline→candidate | 配对中位/均值 |
|---|---:|---:|
| CN20 | 134.607→133.882 | 0.98318x / 0.99581x |
| EN20 | 133.341→137.426 | 1.02573x / 1.03502x |

理论上该开关只改变 prefill，decode 仍使用原来的单 token kernel；但中文输出长度从平均
39.1 变为 40.5 token，只有 12/20 全文一致，使基于生成序列的吞吐比较受到输出漂移
影响。解析答案和 Accuracy 未变，但不满足当前正式 `performance` 档的严格无回退要求。

## 决策

- 正式 `performance` profile 保持不变；
- 新增显式 `experimental-prefill` 档用于复现本轮 TTFT 收益；
- 默认 profile 明确清除 `SEU_PPU_PREFILL_ROW_FUSIONS_ENABLE`；
- PPU 实机验证从 `experimental-prefill` 切回 `performance` 后，该变量确实被清除；
- 本轮到此止损，不继续为 1%--2% 吞吐变化增加第二套实现。

复现入口：

```bash
python scripts/profile_ppu_first_token.py --help
python scripts/benchmark_ppu_prefill_row_fusions_ab.py --help
source scripts/activate_ppu_profile.sh experimental-prefill
```

精简证据见 `results/ppu-first-token-prefill-row-fusions-20260901.json`；逐题 JSON 和 profiler
热点表保存在本机 ignored `artifacts/ppu-prefill-row-fusions-20260901/`。
