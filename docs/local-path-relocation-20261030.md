# 2026-10-30 本地仓库路径整理

按用户要求，将仓库实体从 `D:\GitHub\VLM-Research\projects\seu-vlm-ppu-optimization` 整理回 `D:\GitHub\seu-vlm-ppu-optimization`，并保留旧路径兼容。两者访问同一份仓库。

## 当前路径

| 路径 | 当前作用 |
|---|---|
| `D:\GitHub\seu-vlm-ppu-optimization` | 唯一仓库实体，普通目录 |
| `D:\GitHub\VLM-Research\projects\seu-vlm-ppu-optimization` | 指向仓库实体的 junction |
| `D:\GitHub\VLM-Research\README.md` | 仓库外的本地导航说明 |

整理前的目录联接方向与上表相反。2026-10-29 归档记录中的路径描述保留其历史含义，当前入口以上表和 `LOCAL_HANDOFF.md` 为准。

## 保留范围

- 在 D 盘内移动完整目录，保留 `.git`、分支历史、`.venv`、`data/derived`、`local-only/`、原始结果和设备资料。
- 三个外部 junction 的目标保持不变：`models/Qwen3.5-2B` → `E:\ModelArchive\Qwen3.5-2B`；`applications/mobile-vla/models/Qwen3.5-2B-MNN` → `E:\ModelArchive\Qwen3.5-2B-MNN`；`applications/mobile-vla/tools/android-build` → `E:\CDriveMigration\zhi-ban-mobile-vla\tools\android-build`。
- 两条仓库路径仍可访问相同的环境、数据和模型入口；F 盘研究项目使用的 `D:\GitHub\seu-vlm-ppu-optimization` 路径保持可用。
- Git 忽略规则保持原有范围。模型、运行环境、本地配置、私人设备证据和 `local-only/` 继续只保留在本地，不随文档提交上传。

## 文件完整性核验

迁移前后分别计算 **24,418 个普通文件、3,927,741,967 bytes** 的 SHA-256，覆盖已跟踪内容、忽略文件与 `.git`，未遍历外部联接目标。普通文件相对路径、大小、哈希、子目录清单及三个外部联接目标全部一致。此核验在更新本次路径文档之前完成；完整清单仅存于本地操作工作区。

运行验证：新旧路径均可启动原 Python 环境；实体路径可导入 PyTorch `2.13.0+cu130` 与 Transformers `5.14.1`。两条路径下的 Git、环境、数据、模型与工具入口逐项确认为同一文件或目录。迁移后的原环境通过仓库现有 79 项无模型单元测试；Git 差异格式检查通过。

随后更新仓库首页、本地接手说明、文档入口及状态中的路径说明，并同步到 GitHub `main`。仓库外的导航 README 单独更新，不属于 GitHub 仓库内容。

本轮为路径整理与文档同步，不重跑 PPU、模型推理、手机动作或脑电训练，也不改变已有实验结论。文件逻辑大小不代表物理磁盘释放量；本次未通过删除项目内容来释放空间。
