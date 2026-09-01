# 比赛要求与本提交方案

## 赛道信息

根据大赛宣传页与评比规则截图，本项目参加赛道二“端侧 AI 推理优化挑战”，目标是在
阿里云 PPU 服务器上优化 Qwen3.5-2B 多模态推理。评审指标为：

| 指标 | 权重 | 本提交策略 |
|---|---:|---|
| 模型推理精度 | 30% | 不更换权重，不缩减视觉 Token，双语公开样本 Accuracy 不下降 |
| 推理延迟优化 | 30% | multi-row prefill RMSNorm、gated-RMSNorm、residual+RMSNorm 融合 |
| 吞吐量提升 | 20% | GDN/MLP/attention-prep 相关 decode 融合、packed 投影和 b/a-GEMV |
| 系统级优化 | 20% | HGGC/acBLAS 自定义算子、持久权重布局、raw stream、可恢复构建脚本 |

最终私有集得分由主办方环境复测决定；本文数字仅来自 PPU 上的公开 MMBench 开发集。

## 最终保留的技术路径

模型仍按原始 Qwen3.5-2B 权重与计算语义运行。`evaluation_wrapper.py` 在加载完成后把
固定形状的热点模块替换为 PPU 路径：

1. HGGC 融合 GDN recurrent update、causal-conv、RMSNorm、gated-RMSNorm、
   q/k RMSNorm+RoPE、residual+RMSNorm 和 gate-prep；
2. gate/up 权重连续打包，decode MLP 使用单入口 acBLAS + HGGC 路径；
3. GDN 输入投影使用 grouped acBLAS，只将相邻 b/a 投影合并为一次 GEMV；
4. prefill 阶段复用按行 norm/residual 融合核，减少逐元素 kernel 和中间张量；
5. 使用 raw stream handle，减少高频 Python/运行时查询成本。

更激进的 single-GEMV、视觉 Token 缩减、KV 预留和 prefill SwiGLU 均因精度或性能
门槛失败而未进入本分支。

## 最终 PPU 实验

同一 PPU-ZW810E、同一模型与 CN20，采用四个独立进程并按
`eager A → candidate A → candidate B → eager B` 运行：

| 指标 | 原始 eager | 当前提交 | 改善 |
|---|---:|---:|---:|
| 吞吐中位 | 49.2195 token/s | 133.623 token/s | 2.71484x / +171.48% |
| TTFT 中位 | 120.059 ms | 114.313 ms | -4.79% |
| Accuracy | 85% | 85% | 不变 |

四次运行的 20/20 解析答案与正确性一致，20/20 样本吞吐更快。独立的 CN20/EN20
prefill 配对 A/B 中，TTFT 中位分别提升 4.86% 和 4.48%，双语 Accuracy 与 40/40
解析答案不变。

## 复测边界

- 公开入口未保存完整生成文本 hash，因此不宣称 ABBA 四次全文 bit-exact；
- 总加速数字只覆盖 CN20，不能外推到完整公开集或私有集；
- PPU SDK、patched torch 或模型 revision 变化后必须重新构建扩展并复测精度；
- 运行方法见根目录 `README.md`。
