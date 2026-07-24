# M1：TTFT 首 token 计时边界修正

日期：2026-07-24

## 问题

旧 wrapper 使用 `TextIteratorStreamer` 返回的首个非空文本块作为 TTFT 终点。该 streamer 会等待完整词、空格、换行或中文字符后才输出文本，因此首个文本块可能晚于模型生成首个 token。

赛题定义要求从完整输入就绪计时到输出第一个 token，文本块边界不满足该定义。

## 修正

- 保留 `TextIteratorStreamer` 负责最终文本拼接。
- 在其 `put()` 收到首个生成 token 时记录高精度时间戳。
- 跳过 `.generate()` 启动时写入 streamer 的完整 prompt。
- 结果元数据写入 `ttft_measurement = first_generated_token_put`。

不改变模型、权重、prompt、token、答案解析和生成配置。

## 影响

- Accuracy 与生成 token 数应完全不变。
- 新 TTFT 预计低于旧“首文本块”口径。
- Throughput 的 decode window 使用 TTFT，因此也必须重新计算。
- M1 之前的 O0/O1 相对对比仍可作为历史工程记录，但绝对性能数值不再作为正式 TTFT。

## 复测要求

1. 同一固定样本确认答案和 token 数不变。
2. 重新运行中文 O1 三次和英文交叉验证。
3. 重新建立与 M1 同口径的 O0 对照后，才发布新的提升率。

为支持同一 commit 下的严格单变量对比，`scripts/run_benchmark.ps1` 增加：

- `-OptimizationProfile o0_no_grad`
- `-OptimizationProfile o1_inference_mode`

两种 profile 只切换 `no_grad`/`inference_mode`，其余模型与生成配置共用同一实现。

批量复测入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_profile_matrix.ps1 `
  -Profiles o0_no_grad,o1_inference_mode `
  -Languages cn,en `
  -NumSamples 20 `
  -Repeats 3 `
 -RunLabel m1
```

## M1 复测结果

中文固定前 20 条，O0/O1 各运行 3 次：

| Profile | Accuracy | Median TTFT | Median Throughput | 校验 |
|---|---:|---:|---:|---|
| O0 `no_grad` | 85% | 313.562 ms | 21.328 tokens/s | 三次通过 |
| O1 `inference_mode` | 85% | 287.706 ms | 23.209 tokens/s | 三次通过 |

O1 相对 O0：

- TTFT 提升：8.25%
- Throughput 提升：8.82%
- Accuracy 变化：0
- 跨 profile 答案变化：0
- 跨 profile token 数变化：0

英文固定前 20 条三次复测：

| Profile | Accuracy | TTFT 中位数 | TTFT 范围 | Throughput 中位数 | Throughput 范围 |
|---|---:|---:|---:|---:|---:|
| O0 `no_grad` | 80% | 300.105 ms | 298.585–300.332 ms | 22.678 tokens/s | 21.883–22.909 |
| O1 `inference_mode` | 80% | 269.424 ms | 268.829–272.349 ms | 24.123 tokens/s | 23.570–24.195 |

英文 TTFT/Throughput 三次中位提升为 10.22%/6.37%，6 次运行的答案和
token 数同样无变化。完整复测记录见
[英文 M1 三次性能复测](2026-07-24-en-m1-three-runs.md)。

## 结论

M1 修正后的中英文三次结果均支持 O1 有效，当前正式本地性能数字统一采用三次中位数。
