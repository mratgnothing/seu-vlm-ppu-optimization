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
