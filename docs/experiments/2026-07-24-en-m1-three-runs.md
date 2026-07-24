# 英文 M1 三次性能复测

日期：2026-07-24

## 目的

将英文固定前 20 条的 O0/O1 结果从单次方向验证扩展为三次复测，降低单次吞吐波动
对结论的影响，并检查跨运行、跨 profile 的答案与 Token 稳定性。

## 固定条件

- 数据：MMBench Dev EN 固定前 20 条
- 模型：官方 Qwen3.5-2B 锁定 revision
- 硬件：RTX 4050 Laptop GPU 6GB
- dtype：BF16
- batch size：1
- warm-up：2 条
- TTFT 终点：streamer 收到第一个新生成 Token
- O0：`torch.no_grad()`
- O1：`torch.inference_mode()`

原始 JSON 保存在本地忽略目录 `results/raw/`。每个 profile 使用 3 次独立模型加载和
完整运行；汇总使用中位数。

## 结果

| Profile | Accuracy | TTFT 中位数 | TTFT 范围 | Throughput 中位数 | Throughput 范围 |
|---|---:|---:|---:|---:|---:|
| O0 `no_grad` | 80% | 300.105 ms | 298.585–300.332 ms | 22.678 tokens/s | 21.883–22.909 |
| O1 `inference_mode` | 80% | 269.424 ms | 268.829–272.349 ms | 24.123 tokens/s | 23.570–24.195 |

按赛事公式：

- TTFT 提升率：10.2234%
- Throughput 提升率：6.3718%
- Accuracy 绝对变化：0

## 一致性检查

6 个原始结果文件全部满足：

- `public_validation.passed = true`
- `ttft_measurement = first_generated_token_put`
- question ID 顺序和集合一致
- parsed answer 逐样本一致
- output Token 数逐样本一致

## 解释

最初单次英文结果给出的吞吐提升为 10.57%，三次中位数为 6.37%。差异来自 O0
单次吞吐波动，说明性能结论必须使用重复实验统计。TTFT 的三次范围较窄，O1
约 10.22% 的收益方向稳定。

当前正式英文数字改用三次中位数。该结果仍只覆盖公开集固定前 20 条，不代表完整
公开集或赛事私有评测成绩。
