# BCI 历史代码与报告补充归档

2026-10-30 从 Downloads/output 的 F 盘历史归档补充导入。这里保存 2026-09-30 的前三轮代码与结果说明，以及仪表盘和第四轮报告的内容组件；没有重新训练或改变实验结论。最终结论仍以 [BCI 归档入口](../ARCHIVE.md) 和 [第四轮报告](../results/改进实验报告.md) 为准。

| 历史资料 | 内容 |
|---|---|
| [第一轮](mi_task_20260930_v1/README.md) | CSP-LDA / EEGNet 实现、固定配置、安装与核验脚本 |
| [第二轮](mi_task_20260930_v2/README.md) | 扩展训练、可视化准备脚本、结果与计算核验 |
| [第三轮](mi_task_20260930_v3/README.md) | 个人校准、休息拒识、结果与计算核验 |
| [数据说明](dataset/EEGMMIDB_训练说明.md) | 数据来源、历史下载校验与预处理边界；同目录保留下载脚本 |
| visualization-content/ | 两个原可视化工程的 src/content 源码快照 |
| [第四轮页面截图](../../../deliverables/bci/screenshots/bci_v4_report_overview.png) | 历史报告概览，仅作页面参考 |

这些是历史源码快照。脚本中的 E 盘路径、任务目录名、解释器和相邻模块引用仍指向原实验布局，不能把当前 history 目录直接当作独立可运行安装包。原始数据和训练 run 的位置见各轮 README；没有将模型、逐试次结果、脑电波形和本机环境配置上传 GitHub。

visualization-content 仅含原工程内容层，保留了原模板辅助文件；它不是独立 Web 应用。完整工程、src/data.json、导出 HTML、截图、浏览器检查脚本及页面快照已逐文件校验并保存在本仓库 Git 忽略目录 local-only/bci/downloads-output-20261030/ 下，布局沿用原目录。npm 缓存、Playwright 会话文件与 Python 字节码不重复归档。已有最终 HTML 保留在 deliverables/bci/改进实验报告.html。

归档清单见 [导入校验表](../../../docs/bci-output-import-manifest-20261030.json)，迁移说明见 [整理记录](../../../docs/bci-output-import-20261030.md)。
