# 比赛结项总结

归档日期：2026-10-29。团队已确认比赛项目完成；本页总结已有实验与交付，不表示本轮重新运行了 PPU、手机或脑电实验。

## PPU 主线

固定模型为 Qwen/Qwen3.5-2B，revision 为 `15852e8c16360a2fea060d615a32b45270f8a8fc`。目标 PPU-ZW810E 上 617 个参数张量均驻留设备，无 CPU/meta/disk offload，已完成真实多模态推理。

优化围绕 BF16 运行时展开：GDN recurrent、causal-conv、RMSNorm/gated RMSNorm、Q/K norm+RoPE、residual+norm、packed MLP、GDN gate-prep、grouped acBLAS、raw-stream 查询、保守 b/a-GEMV 与 multi-row prefill。它不构成已完成 PPU 整模 INT8/INT4 量化的证据。

最终公开 CN20 ABBA：吞吐 49.2195 → 133.623 token/s（2.71484x，+171.48%），TTFT 120.059 → 114.313 ms（-4.79%），四轮 Accuracy 均为 17/20。20/20 答案与正确性一致，15/20 token 数一致，全文严格比较不可用。[对应 JSON](../results/ppu-final-best-vs-eager-cn20-abba-20260901.json)。

保守 b/a-GEMV 相对前序性能栈通过中英文各 4029 条 exact 门禁；该结论只针对对应增量实验，不表示最终栈相对初始 eager 全集逐字一致。激进 single-GEMV、部分 scratch/graph/Attention Prep 以及 prefill SwiGLU 的失败记录继续保留，避免重复尝试已止损的方案。详见 [工作说明](work-summary/README.md) 与 [结果目录](../results/README.md)。

## GUI-VLA 与手机原型

最终材料区分团队最新 GUI-VLA 工程与较早的 Qwen3.5-2B-MNN 本地受控原型。前者以 2026-10-28 的交接与来源记录为准；时钟任务的决策 P50、快路径均值、视频进度和总运行时长各有不同分母，不混为系统整体成功率。153 例截图是领域参考评测，当前材料没有本系统同协议成绩；图中的 >95% 等字段属于目标。

手机原型源码合并到 [applications/mobile-vla](../applications/mobile-vla/README.md)，保留安全门及未提交开发修改。HQQ 4-bit MNN、Android 真机受控视觉点击、电话验收与微信失败分别解释，不替代 PPU 量化结论，不当作通用 GUI Agent 能力。视频展示的工程证据不等于 EEG→手机实时整链已验证。[来源边界](applications/gui-vla-evidence-scope.md)。

## 脑电扩展

最终离线报告及 v4 复现脚本已纳入 [脑电归档](../applications/bci/ARCHIVE.md)。原始数据、训练环境与成功/失败 run 仍在 E 盘保存，不纳入公开 Git。报告属于固定数据上的探索性后续；同次采集、个人校准及已查看测试人群等限制均保留。没有据此证明跨天泛化、临床效果或实时脑控手机。

## 最终交付

[一吴是处决赛汇报.pptx](../deliverables/presentation/一吴是处决赛汇报.pptx) 为 29 页最终版本，四页附录均可翻页，保留用户手动修改和内嵌视频。团队来源 PPT、视频和脑电报告另按成果类别保存。完整入口见 [成果目录](../deliverables/README.md)。

## 证据限制

现有本地 PPU 最终汇总引用的 `artifacts/final-best-vs-eager-20260901` 原始目录在本轮盘点前即不存在；本次保留现有汇总、实验说明和所有仍存在的逐样本结果，不能声称恢复或复核了缺失的原始目录。最终成绩以赛事正式记录为准。
