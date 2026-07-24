# O0：选项 Markdown 标记归一化

日期：2026-07-24

## 问题

中文公开集样本 `241` 的模型原始回答为：

```text
正确答案是：**B**
```

参考答案是 B，模型语义正确，但 v1.1 `extract_answer` 不接受 Markdown 粗体包裹：

- `parsed_answer = null`
- `validation_errors = ["missing_choice_answer"]`
- Accuracy = 0

## 修改

只在 `evaluation_wrapper.py` 返回文本前去掉 A/B/C/D 周围成对的 Markdown 标记：

- `**B**` → `B`
- `_C_` → `C`
- `` `D` `` → `D`

不读取参考答案，不修改模型 token、不修改时间统计，也不改主办方 `benchmark_public.py`。

## 复测

同一中文样本：

- `parsed_answer = B`
- Accuracy = 1
- `public_validation_passed = true`
- `choice_markup_normalized = true`

英文样本 `241`：

- Accuracy = 1
- `public_validation_passed = true`
- `choice_markup_normalized = false`

两次独立进程的性能数值受到模型加载、首次 kernel 和系统抖动影响，不能把 TTFT 或吞吐变化归因于本项文本归一化。本项结论仅证明准确率解析缺陷得到修复。

固定中文前 20 条复测后：

- Accuracy 从 80% 修正为 85%；
- 公开接口校验从失败变为通过；
- 两条格式漏判被恢复，其中一条语义正确、一条模型本身答错；
- 最终剩余 3 条均为模型真实答错，不再存在解析失败。
