# 英文分层 200 条精度验证

日期：2026-07-24

## 目的

使用与中文相同的确定性比例分层方法，从完整英文 MMBench Dev 中抽取 200 条，
验证 O1 的跨语言精度、答案解析和类别覆盖。

## 子集

- 原始英文 dev：4029 条
- 子集：200 条
- Seed：20260625
- 分层字段：`category`、`l2-category`
- 原始 TSV SHA-256：`59c2418bd0c89a88abe5573461cd98ebad155398af803adb36a396bd6ae31710`
- 派生 TSV SHA-256：`de585a1d05230649353af865d87e169208034acae07c5a1128a0d3204243a4d7`
- Profile：O1 `torch.inference_mode()`
- 后端：Transformers，BF16，纯 GPU

派生 TSV、图片、逐样本结果和类别分析 JSON 均保存在 Git 忽略目录。

## 结果

- Accuracy：82.5%（165/200）
- 公开接口校验：通过，0 条失败
- 错误样本：35 条，均有合法 A/B/C/D 解析
- 20 个 category、6 个二级类别全部覆盖
- TTFT 元数据：`first_generated_token_put`

二级类别：

| 二级类别 | 样本数 | 正确数 | Accuracy |
|---|---:|---:|---:|
| Fine-grained instance-level | 52 | 49 | 94.23% |
| Coarse perception | 51 | 45 | 88.24% |
| Fine-grained cross-instance | 25 | 21 | 84.00% |
| Attribute reasoning | 32 | 25 | 78.12% |
| Relation reasoning | 21 | 16 | 76.19% |
| Logic reasoning | 19 | 9 | 47.37% |

## 性能口径

该长测采用修正后的首个生成 Token 计时，单次结果为 TTFT 270.904 ms、吞吐
24.641 tokens/s。它只用于监控长测是否异常，不与三次固定 20 条性能统计混合。

## 结论

英文分层 200 条 Accuracy 82.5%，相对固定前 20 条的 80% 没有出现异常精度坍塌。
中英文均显示逻辑推理、关系理解是相对弱项。后续量化和 PPU kernel 优化必须保存
分层精度护栏，重点检查这些类别，但不得针对公开样本逐题调参。
