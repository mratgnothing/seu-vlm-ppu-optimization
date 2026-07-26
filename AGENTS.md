# Agent 工作约定

本文件适用于在本仓库中工作的所有自动化 Agent。人类成员也可据此了解协作边界。

## 开始工作前

按顺序阅读：

1. `PROJECT_CONTEXT.md`：项目目标、信息优先级和真实状态边界；
2. `docs/README.md`：各模块当前进度、证据入口和下一步；
3. `docs/rules-and-boundaries.md`：赛事规则与禁止越界事项；
4. `docs/current-status.md`：最近一次经过验证的环境与结果。

若文档冲突，以主办方最新书面通知、固定评测接口和可复现实验记录为准。

## 真实性边界

- `dummy` 只用于接口冒烟，不得写成真实模型结果。
- `--backend auto` 可能回退到 dummy，不得用于正式基线。
- 本地 RTX 4050 结果不等于阿里云 PPU 结果。
- PPU SDK、示例或微基准通过不等于 Qwen3.5-2B 已在 PPU 部署。
- 只有目标 PPU 上真实模型推理可复现，并产生有效 Accuracy、TTFT 和 Throughput，
  才能写“PPU 部署完成”。
- 全量公开集单次运行用于 Accuracy 和管线完整性；正式性能比较使用同口径重复实验。

## 数据与安全

- 不提交模型权重、公开数据副本、逐样本原始结果、本地配置、日志、密钥或 SSH 私钥。
- 本地敏感内容必须继续由 `.gitignore` 和源码候选包白名单排除。
- 不改动主办方评测公式或使用参考答案影响推理输出。
- 精度、量化、算子或调度结论必须能追溯到代码提交、配置和原始实验记录。

## 协作方式

- `main` 保存可复现状态；功能开发使用短分支并通过 Pull Request 合入。
- 每项优化尽量保持单变量，实验记录放在 `docs/experiments/`。
- 修改代码后运行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

- 更新结果时同步维护 `docs/current-status.md`、`docs/README.md` 和对应实验记录。
- 推送前检查 `git status`，只提交本任务相关文件。

## 当前交接点

本地真实 Transformers 基线、O1、M1、全量中英文 Accuracy、CUDA profile、PPU
兼容性调查和源码候选包工具均已完成。下一条关键路径是获得主办方支持
Qwen3.5-2B 的隔离 PPU 环境，然后完成真实 PPU 功能、性能和优化闭环。

