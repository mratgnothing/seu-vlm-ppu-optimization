# 脑电扩展归档

这是比赛应用扩展的离线实验成果。先读 [最终实验报告](results/改进实验报告.md) 和 [计算核验](results/calculation_verification.json)，再看 [可查看 HTML](../../deliverables/bci/改进实验报告.html) 与 [总结文档](../../deliverables/bci/脑机接口应用方案与实验探索汇报总结.docx)。

`README.md`、`config.json`、`train_v4.py`、`run_training.ps1`、`generate_report.py` 是最终实验的原样快照。数据、个人模型、逐试次结果及历史失败 run 的权威位置仍为 `E:\BCI\datasets\physionet-eegmmidb-1.0.0\training\mi_improvement_20261001_v4`。配置还引用同一数据根下的 pilot/v2/v3 环境；本次没有迁移、删除或重训这些资料。报告生成脚本另依赖该目录的 `report_app/src/data.json`，不能仅靠本目录重建全部报告。

复现训练前先准备上述外部依赖并核对配置。本次归档未启动训练、采集或手机动作。

## 2026-10-30 补充历史资料

前三轮源码、结果说明、数据下载工具和可视化内容组件见 [历史资料入口](history/README.md)。完整原文件和页面快照保存在 Git 忽略目录 `local-only/bci/downloads-output-20261030/`，公开来源校验见 [导入清单](../../docs/bci-output-import-manifest-20261030.json)。下载目录中的旧入口是指向 F 盘的 junction；本轮只移除该入口，保留实际目标和 E 盘训练数据。
