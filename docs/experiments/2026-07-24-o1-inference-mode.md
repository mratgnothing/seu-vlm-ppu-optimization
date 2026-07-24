# O1：纯推理模式与贪心参数收紧

日期：2026-07-24

## 目标

在不改变模型权重、提示词、最大生成长度和输出解析规则的前提下，减少本地 Transformers 基线中的 Python 与 PyTorch 推理开销。

## 修改

- 将 `torch.no_grad()` 替换为更严格的 `torch.inference_mode()`。
- `temperature = 0` 的贪心解码不再传入无效的 `temperature` 和 `top_p` 参数。
- 模型加载参数由已弃用的 `torch_dtype` 更新为 `dtype`。
- 结果元数据记录 `optimization_profile = o1_inference_mode`。

这组修改不改变：

- Qwen3.5-2B 权重与 BF16 精度；
- 公开数据集和主办方 `benchmark_public.py`；
- 输入提示词；
- `max_new_tokens = 256`；
- `do_sample = false`、KV cache 和答案解析方式。

## 验证计划

1. 先通过单元测试和模型加载烟雾测试。
2. 在中文公开集固定前 20 个样本上运行，使用 2 个预热样本。
3. 与 O0 已锁定基线比较 Accuracy、平均 TTFT、平均吞吐量和公开接口校验。
4. 性能结论至少结合重复运行观察，避免把单次系统波动当作优化收益。

## 历史结果：文本块 TTFT 口径

公开中文集固定前 20 条，2 条预热，BF16 纯 GPU。对照基线为 O0：

| 版本 | Accuracy | Avg TTFT | TTFT 提升 | Throughput | 吞吐提升 | 答案/token 漂移 |
|---|---:|---:|---:|---:|---:|---:|
| O0 基线 | 85% | 327.451 ms | - | 21.813 tokens/s | - | - |
| O1 R1 | 85% | 290.946 ms | 11.15% | 23.150 tokens/s | 6.13% | 0/0 |
| O1 R2 | 85% | 286.467 ms | 12.52% | 24.476 tokens/s | 12.21% | 0/0 |
| O1 R3 | 85% | 294.064 ms | 10.20% | 23.064 tokens/s | 5.74% | 0/0 |
| O1 三次中位数 | 85% | 290.946 ms | 11.15% | 23.150 tokens/s | 6.13% | 0/0 |

三次运行使用相同 question ID，解析答案和生成 token 数逐样本一致，公开接口均通过。

英文固定前 20 条交叉验证同样保持 Accuracy 80%、答案和 token 数不变：

- TTFT：409.660 ms → 351.892 ms，提升 14.10%；
- Throughput：28.685 → 33.462 tokens/s，提升 16.65%；
- 公开接口校验通过。

## 结论

O1 在当前 RTX 4050/Windows 本地基线上对中英文均有效，可作为下一阶段的代码基线。表中绝对性能使用旧文本块 TTFT，只保留为历史记录。正式首 token 数值见 [M1 计时边界修正](2026-07-24-ttft-token-boundary.md)。数值只来自固定前 20 条公开样本，暂不外推为完整公开集或 PPU 收益。
