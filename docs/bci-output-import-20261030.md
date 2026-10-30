# 2026-10-30 BCI 下载目录补充归档

用户授权将 Downloads/output 中有用资料放入比赛仓库，清理下载目录入口并同步 GitHub。本次只整理已有成果，不重新运行训练、手机动作或 PPU 实验。

## 保存结果

- 原入口是 Windows junction，真实目标为 `F:\E盘重要资料\Archive\Downloads-output`。
- 有用原文件 591 个，共 265,790,642 bytes，逐文件 SHA-256 验证后保存在 Git 忽略目录 `local-only/bci/downloads-output-20261030/`，包含完整可视化工程、数据快照、导出页面和浏览器检查资料。
- 公开补充早期三轮源码、配置、结果说明、数据下载工具与说明、可视化内容组件及一张报告概览图，入口为 [history/README.md](../applications/bci/history/README.md)。原有第四轮源码与最终 HTML 保持原位。
- 公开文本规范化为 UTF-8/LF，并移除行尾空白；原字节副本仅保存在本地归档。[公开清单](bci-output-import-manifest-20261030.json) 记录新增来源文件及规范化后的哈希。本次新增导航文件另由 Git 跟踪。
- 244 个缓存/会话文件不重复复制；F 盘原归档继续保留。

## 清理与复现边界

完成复制、哈希核对与公开文件审查后，仅移除 `C:\Users\ASUS\Downloads\output` 这个 junction；不递归删除它指向的 F 盘目标，也不删除 E 盘数据或训练记录。清理回执保存在上述本地归档内的 `cleanup-receipt.json`。

新归档是代码和材料的历史快照，不能据此声称恢复了缺失的 run，或已验证跨天泛化、在线脑控手机。旧脚本继续依赖原任务布局与环境；完整依赖说明见 [BCI 归档](../applications/bci/ARCHIVE.md)。

## 完成核验

591 个原文件与本地副本的 SHA-256 一致；59 个公开来源文件的 Git 暂存内容与公开清单一致。新增 Python 源码通过语法解析，JSON、导航链接与 local-only 忽略规则检查通过；公开文本检查未发现已知凭据格式。仓库要求的 79 项无模型单元测试通过，git diff --check 通过。

已用非递归 rmdir 移除下载目录 junction，并确认 F 盘目标继续存在。此操作清理了下载入口，不代表释放了 F 盘归档占用。
