# 第一阶段：模型推理与评测入门

这份讲义的目标不是一次讲完 Transformer，而是让你先建立一张能用于提问、读代码
和上服务器排错的地图。

## 1. 这个项目的一次请求发生了什么

评测样本包含一张图片、一道问题、A/B/C/D 选项和标准答案。一次推理可以先理解成：

```text
图片 + 问题 + 选项
  -> Processor 整理图片和文字
  -> 视觉编码器把图片变成视觉 token
  -> 文本模型读取全部输入（Prefill）
  -> 模型逐个生成答案 token（Decode）
  -> 从输出中解析 A/B/C/D
  -> 统计 Accuracy、TTFT、Throughput 和显存
```

对应代码：

- `benchmark_public.py`：读取公开数据、构造问题、调用模型并汇总指标；
- `evaluation_wrapper.py`：加载模型、预处理输入、生成答案和记录时间；
- `docs/qwen35-kernel-targets.md`：模型结构和需要重点关注的算子尺寸。

先记住：**模型能加载**、**能生成一条答案**、**答案正确**和**运行得快**是四件不同的
事，必须分别验证。

## 2. Processor、Tokenizer 和模型分别负责什么

### Processor

`AutoProcessor` 同时处理图片和文本：

- 图片会被缩放、归一化并切成视觉 patch；
- 问题和选项会按聊天模板拼接；
- 文本会被 tokenizer 转为整数 token ID；
- 最终返回可以交给模型的张量。

在 `evaluation_wrapper.py` 中，`apply_chat_template(...)` 就在完成这一步。输入不是把
Python 字符串直接塞进 GPU，而是先变成模型约定的张量和 token 序列。

### Tokenizer

Tokenizer 在“字符串”和“整数 token”之间转换。一个 token 不一定等于一个汉字或
一个英文单词。因此吞吐应按生成 token 数计算，不能简单按字数计算。

### 模型

模型接收视觉和文本 token，反复计算下一 token 的概率。项目通过
`AutoModelForImageTextToText` 加载 Qwen3.5-2B，并使用 BF16 放到设备上。

## 3. Prefill 和 Decode

这是推理优化最重要的区分。

### Prefill：一次读完已有上下文

模型一次处理图片 token、问题和选项。假设输入长度为 `S`，线性层通常表现为较大的
矩阵乘法，注意力还需要处理多个输入 token 之间的关系。

Prefill 主要影响：

- 首个 token 出现前的等待时间；
- 图片越大、视觉 token 越多时的计算量；
- 临时激活和缓存占用；
- TTFT。

### Decode：一次只生成一个新 token

首个 token 以后，模型循环生成后续 token。每轮输入通常只有当前的一个 token，很多
线性层因此从 GEMM 变成更像 GEMV 的小批量计算。

Decode 主要影响：

- 连续出字速度；
- tokens/s；
- 权重读取带宽；
- KV Cache 或 GDN recurrent state 的更新成本。

直觉上可以记成：

```text
Prefill = 把题目读懂
Decode  = 一个字一个字作答
```

## 4. 为什么需要 Cache 或 State

如果生成每个 token 时都重新计算全部历史内容，成本会不断增长。因此模型保存历史
计算的中间状态。

### KV Cache

标准全注意力层会保存过去 token 的 Key 和 Value。新 token 只需计算自己的 Q/K/V，
再让 Query 读取历史 KV Cache。

它用显存换计算。上下文越长、batch 越大，KV Cache 通常越大。

### GDN recurrent state

Qwen3.5-2B 不是 24 层全部使用标准全注意力：本项目核验到 18 层为线性注意力/GDN，
6 层为全注意力。GDN 使用固定形状的 recurrent state 逐 token 更新，另有 width=4 的
causal-conv1d 状态。

因此移植旧 vLLM 时，不能只补模型名称；还必须正确处理：

- GDN state 的创建、更新和 batch 重排；
- causal-conv1d 的历史状态；
- 6 个全注意力层的 KV Cache；
- Prefill 和 Decode 两种路径。

## 5. 模型里哪些计算最值得先认识

### GEMM 与 GEMV

- GEMM：矩阵乘矩阵；Prefill 和较大 batch 常见。
- GEMV：矩阵乘向量；单请求 Decode 常见。

Qwen3.5-2B 的高频尺寸包括 `2048 -> 6144`、`6144 -> 2048` 和
`2048 -> 2048`。仓库的 PPU 微基准正是围绕这三组尺寸准备的。本地 CUDA Profile
中 GEMV/GEMM 占主要时间，但到了 PPU 仍需重新 Profile，不能直接照搬结论。

### RMSNorm

对隐藏状态做尺度归一化。单次计算不大，但每层都会调用，容易受内存访问和 kernel
启动开销影响，也常成为算子融合对象。

### 门控 MLP

大致可理解为两条上投影分支经过激活/门控，再下投影回隐藏维度。这里反复出现
`2048 -> 6144 -> 2048`，所以占用大量矩阵计算。

### Attention

Query 和历史 Key/Value 计算相关性，再汇总 Value。全注意力层需要管理 KV Cache；
FlashAttention 是优化其访存的一类方法，不是另一种模型。

### GDN 与 causal-conv1d

它们是本模型区别于普通纯 Transformer 的关键路径，也是旧版 vLLM 适配的主要风险。
第一阶段先理解“它们维护并更新状态”，暂时不要求推导全部数学公式。

## 6. 四个必须分开的指标

### Accuracy

```text
Accuracy = 正确题数 / 总题数
```

它是正确性护栏。优化如果让模型变快但答案变了，通常不能直接接受。公开集结果也不
等于主办方私有集成绩。

### TTFT

Time To First Token，从开始生成到第一个**生成 token**出现的时间。项目中特意跳过了
streamer 放入 prompt 的事件，避免把 prompt 误认成首个生成 token。

TTFT 通常包括预处理、Prefill 和首次 Decode，但具体口径必须始终保持一致。

### Throughput

当前公开评测按 Decode 窗口计算：

```text
Throughput = (生成 token 数 - 1) / (总生成时间 - TTFT)
```

减去第一个 token，是因为它已处在 TTFT 边界。只比较 tokens/s 时必须同时固定输入、
输出长度、batch、采样参数、预热和计时口径。

### 显存

至少区分：

- 模型权重；
- KV Cache/GDN state；
- 中间激活和临时 workspace；
- 框架 allocator 已保留但未实际占用的显存。

本地模型加载约 4.12 GiB 只是权重加载冒烟结果，不等于推理峰值显存。

## 7. 当前代码调用链

从 [benchmark_public.py](../../benchmark_public.py) 开始看：

1. `load_mmbench_tsv` 读取公开样本；
2. `decode_image` 还原图片；
3. `build_prompt` 拼接问题和选项；
4. `VLMModel(...)` 加载模型；
5. `generate_with_metrics(...)` 执行推理；
6. `extract_answer` 从输出解析 A/B/C/D；
7. `compute_throughput` 按固定公式计算 Decode 吞吐；
8. 最后汇总 Accuracy、TTFT 和 Throughput。

再看 [evaluation_wrapper.py](../../evaluation_wrapper.py)：

1. `_load_transformers_backend` 加载 Processor 和 BF16 模型；
2. `_generate_with_transformers` 构造多模态消息；
3. `apply_chat_template` 生成模型输入；
4. `model.generate` 执行 Prefill 和 Decode；
5. `TimedTextIteratorStreamer` 记录第一个生成 token；
6. 返回文本、token 数、TTFT、总耗时和元数据。

## 8. 第一阶段暂时不要混淆的概念

- CUDA runtime 可用，不等于 PPU runtime 可用；
- GPU 上正确，不等于 PPU 上正确；
- 微基准快，不等于完整模型快；
- 模型加载成功，不等于视觉问答成功；
- 单样本结果，不等于 Accuracy；
- 平均延迟，不等于 TTFT；
- 权重占用，不等于推理峰值显存；
- CUDA 源码风格兼容，不等于 GPU/PPU 二进制兼容；
- 有 INT4 权重，不等于硬件存在有效的 INT4 fast path。

## 9. 学完这一阶段应该能回答什么

你不需要背公式，但应该能用自己的话回答：

1. 一张图片怎样变成模型可以计算的输入？
2. Prefill 和 Decode 为什么会出现不同的算子形态？
3. KV Cache 保存了什么，为什么占显存？
4. Qwen3.5 的 GDN state 与 KV Cache 有什么不同？
5. TTFT 和 Throughput 各自覆盖哪段时间？
6. 为什么优化后必须重新跑 Accuracy？
7. 为什么本地 CUDA Profile 不能直接当 PPU Profile？
8. 为什么这个项目优先关注 GEMV、GDN 和 causal-conv1d？

## 10. 你可以直接这样问我

下面的问题都适合作为下一轮入口：

- “Processor 到底把图片变成了什么？能举一个张量形状例子吗？”
- “用一个三 token 的例子演示 Prefill 和 Decode。”
- “KV Cache 为什么存 K/V，不存 Q？”
- “GDN 是什么？先不要公式，用直觉讲。”
- “GEMM 和 GEMV 在代码层面有什么不同？”
- “为什么单 token Decode 容易受显存带宽限制？”
- “逐行带我读 `evaluation_wrapper.py`。”
- “TTFT 的计时代码为什么需要 streamer？”
- “模型加载占 4.12 GiB，为什么运行时还会继续涨？”
- “如何证明一个算子真的在 PPU 上运行而不是 CPU fallback？”
- “给我出五道第一阶段检查题，我回答后你纠正。”

建议一次只问一个概念。先把推理链路讲透，再进入 CUDA 线程模型和 kernel 编程，后面
写算子时会轻松很多。
