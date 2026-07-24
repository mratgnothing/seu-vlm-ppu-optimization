# B4a：中文分层 200 条精度验证

日期：2026-07-24

## 目的

公开集前 20 条类别集中，Accuracy 波动较大。使用确定性比例分层抽样覆盖完整中文 dev 的 20 个 `category / l2-category` 组合，验证当前答案解析和精度是否稳定。

## 子集

- 原始中文 dev：4029 条
- 子集：200 条
- Seed：20260625
- 分层字段：`category`、`l2-category`
- 原始 TSV SHA-256：`facf92736b4be617e6dfc05dd687e668dec0eb4a6605500b01af385c5d8670eb`
- 派生 TSV SHA-256：`acb84e12985e45929e5f1fcbc5094d8d337a07d6eeb3e54b13e8e08f47d861b0`

派生 TSV、图片和逐样本结果均保存在 Git 忽略目录。

## 结果

- Accuracy：84.5%（169/200）
- 公开接口校验：通过，0 条失败
- 失败样本：31 条，均有合法 A/B/C/D 解析
- 20 个类别全部覆盖

子类别：

| 子类别 | 样本数 | Accuracy |
|---|---:|---:|
| Fine-grained instance-level | 53 | 92.45% |
| Coarse perception | 52 | 90.38% |
| Attribute reasoning | 32 | 84.38% |
| Fine-grained cross-instance | 25 | 80.00% |
| Relation reasoning | 20 | 75.00% |
| Logic reasoning | 18 | 61.11% |

低分方向主要集中在 logic reasoning、空间/物理关系和图像质量判断。单个 category 样本数较小，暂不据此修改 prompt 或做专项过拟合。

## 计时口径

这次长测进程在 M1 TTFT 修正前启动，使用“首个非空文本块”旧口径。Accuracy、答案与接口校验有效；TTFT 和 Throughput 只保留为历史记录，不进入正式性能表。

## 结论

200 条分层 Accuracy 84.5% 与固定前 20 条的 85% 基本一致，说明 O1 当前精度和解析管线具有初步稳定性。下一步应扩大到完整公开集，并优先分析低分子类别，避免按单个题目调参。

