# 协作约定

## 分支

- `main`：可复现的稳定版本。
- `baseline/<topic>`：环境、基线和测量改进。
- `opt/<topic>`：模型或推理路径优化。
- `ppu/<topic>`：PPU 适配和系统级优化。
- `docs/<topic>`：报告、论文和文档。

## Pull Request

每个 PR 应说明：

1. 改了什么以及为什么。
2. 使用的模型、数据、环境和命令。
3. Accuracy、TTFT、Throughput 的前后结果。
4. 精度下降、失败样本和已知限制。
5. 是否在 PPU 上真实验证。

禁止只提交“看起来更快”的实现而没有可复现数据。

## 提交信息

推荐使用简短前缀：

- `baseline: ...`
- `opt: ...`
- `ppu: ...`
- `docs: ...`
- `test: ...`
- `chore: ...`

