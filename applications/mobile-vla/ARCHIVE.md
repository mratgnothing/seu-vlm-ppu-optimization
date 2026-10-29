# 手机原型合并说明

2026-10-29 从原 `mratgnothing/zhi-ban-mobile-vla` 仓库合并到本目录；保留原工作区的修改版本与安全门。原 HEAD：`3cdca15ea886826c8fab61c83c2376df26c0387d`。原 Git 全历史已校验并存于主仓库本地忽略项 `local-only/mobile-vla-history.bundle`，未提交修改的补丁也在同目录保存；本目录不再含嵌套 `.git`。

原 README 与 evidence 文档中的日期、设备结果和旧 Downloads 路径是历史记录。现在从本目录运行脚本，优先使用 `$PSScriptRoot` 或项目相对路径。最新团队 GUI-VLA 交接见 [2026-10-28 资料](../../docs/applications/gui-vla-20261028.md)，不要把两条工程线的指标拼接。

`models/Qwen3.5-2B-MNN` 和 `tools/android-build` 在本机仍为指向原 E 盘资料的 junction；不上传、不删除外部目标。设备截图、JSON/XML 记录、数据库、APK 和第三方工具按原 `.gitignore` 留在本地，公开源码不包含这些内容。
