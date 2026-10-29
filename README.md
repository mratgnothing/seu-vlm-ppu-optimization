# 东南大学 AI+创新应用大赛｜Qwen3.5 PPU 优化与 GUI-VLA

项目已于 2026-10-29 按团队结项要求完成成果归档。本仓库统一保存 PPU 优化源码、实验结论与最终路演材料，并收录 GUI-VLA 手机原型和离线脑电实验。比赛名次、奖项和私有集成绩不在现有证据范围内。

## 从这里阅读

| 内容 | 入口 |
|---|---|
| 最终结论、指标与适用范围 | [结项总结](docs/final-summary.md) |
| 最终汇报与交付文件 | [成果目录](deliverables/README.md) |
| PPU 技术方案与提交说明 | [技术报告](docs/final-submission.md)、[启动方式](README_SUBMISSION.md) |
| 全过程、优化方法与负实验 | [完整工作说明](docs/work-summary/README.md) |
| 可追溯实验数据 | [结果索引](results/README.md) |
| GUI-VLA 最新交接与来源边界 | [2026-10-28 架构与进展](docs/applications/gui-vla-20261028.md)、[证据口径](docs/applications/gui-vla-evidence-scope.md) |
| 手机端受控原型 | [applications/mobile-vla](applications/mobile-vla/README.md) |
| 离线脑电最终实验 | [applications/bci](applications/bci/ARCHIVE.md) |
| 本次合并和清理说明 | [归档说明](docs/archive-20261029.md) |

## 核心 PPU 结果

Qwen3.5-2B、BF16、batch 1，目标为 PPU-ZW810E。同一实例上对公开中文前 20 条，以四个独立进程按 ABBA 顺序复测，结果见 [原始汇总 JSON](results/ppu-final-best-vs-eager-cn20-abba-20260901.json)：

| 指标 | eager | 最终性能栈 |
|---|---:|---:|
| 平均吞吐的两轮中位数 | 49.2195 token/s | 133.623 token/s |
| 平均 TTFT 的两轮中位数 | 120.059 ms | 114.313 ms |
| Accuracy（每轮） | 17/20 | 17/20 |

吞吐为 **2.71484 倍（+171.48%）**，TTFT 降低 **4.79%**。四次运行中 20/20 解析答案与正确性一致；只有 15/20 的生成 token 数一致，公开接口未保存全文哈希，不能声明全文 bit-exact，也不能将 CN20 加速外推为全量或私有集成绩。

## 代码和复现

- `evaluation_wrapper.py`、`ppu/`：评测接口与 HGGC/acBLAS 优化源码。
- `scripts/`、`tests/`：构建、启动、分析与无模型验证。
- `docs/`、`results/`：方法、结论、门禁和失败方向。
- `applications/`：手机原型与脑电扩展；与正式 PPU 评测分别解释。
- `deliverables/`：最终汇报、演示视频及报告。

PPU 环境按 [README_SUBMISSION.md](README_SUBMISSION.md) 配置。官方精简提交保留在独立历史 `submission-source`，固定提交为 `5dd8a0a3f5ca2348062360233a8d70812687544e`；需要该版本时可执行 `git archive --format=zip --output=submission.zip 5dd8a0a3f5ca2348062360233a8d70812687544e`，无需常驻重复源码副本。

本地最小验证：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

模型、数据、私人设备证据、运行环境和 `local-only/` 不上传。研究项目仍使用的 `.venv`、`data/derived` 与模型 junction 在本地保留；来源与复现限制见 [本地接手说明](LOCAL_HANDOFF.md)。
