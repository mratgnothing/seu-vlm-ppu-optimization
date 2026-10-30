# 比赛项目本地接手

结项归档：2026-10-29（Asia/Singapore）；本地路径于 2026-10-30 整理。仓库实体为 `D:\GitHub\seu-vlm-ppu-optimization`。旧路径 `D:\GitHub\VLM-Research\projects\seu-vlm-ppu-optimization` 是指向该实体的兼容 junction，访问的是同一仓库，不占用第二份文件空间。迁移范围与核验见 [路径整理记录](docs/local-path-relocation-20261030.md)。

先读 `README.md` → `docs/final-summary.md` → `deliverables/README.md` → `docs/archive-20261029.md`。最终 PPU 提交说明在 `README_SUBMISSION.md`；所有历史结果与失败方案在 `results/` 和 `docs/work-summary/`。

## 本地专有内容

- `.venv` 与 `data/derived`：仍被独立 VLM 研究项目作为参考资产使用，保留原位。
- `models/Qwen3.5-2B` → `E:\ModelArchive\Qwen3.5-2B`：模型只保留外部 junction，不是项目内部副本。
- `artifacts`、`results/raw`：保留仍存在的原始结果、摘要与核验；仅删除可重建的 profiler trace 和旧打包 ZIP。
- `applications/mobile-vla/models/Qwen3.5-2B-MNN` → `E:\ModelArchive\Qwen3.5-2B-MNN`。
- `applications/mobile-vla/tools/android-build` → `E:\CDriveMigration\zhi-ban-mobile-vla\tools\android-build`；原工具目标保留。
- 手机截图/设备证据/数据库与第三方工具保留在手机子目录的 Git 忽略项；最终 APK、原 Git bundle、修改补丁与完整整理清单在 `local-only/`，不上传。
- 脑电原始数据、环境和成功/失败 run 仍在 `E:\BCI`；`Downloads/output` 是到 F 盘共享历史归档的 junction，本次未删除或修改该归档及其 Playwright 资料。

GitHub 只保证已提交内容。`local-only/`、忽略文件、模型和外部数据不在 GitHub 备份范围内。无新实验需求时不启动模型、训练、手机通信或旧队列。
