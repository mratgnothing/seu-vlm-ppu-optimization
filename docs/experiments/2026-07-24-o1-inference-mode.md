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
