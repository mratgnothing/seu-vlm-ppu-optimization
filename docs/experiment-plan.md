# 首轮实验计划

## 阶段 B：建立可信基线

| 编号 | 内容 | 通过条件 |
|---|---|---|
| B0 | 中英文 dummy 各 20 条 | 评测入口通过，结果明确标记 `backend=dummy` |
| B1 | 真实模型仅加载 | 模型 class、device map 和内存占用有记录 |
| B2 | 中英文真实模型各 1 条 | 唯一 A/B/C/D、无验证错误、无 OOM |
| B3 | 中英文真实模型各 20 条 | 保存 Accuracy、TTFT、Throughput 和逐样本结果 |
| B4 | 公开集完整基线 | 环境稳定后再运行，结果与配置可追溯 |

本机 6GB 显存可能触发 CPU offload。含 CPU offload 的本地结果仅用于功能和精度调试，不与纯 GPU 或 PPU 性能横向比较。

## 阶段 O：低风险优化候选

真实基线完成后按单变量方式依次评估：

1. `torch.inference_mode()` 替代 `torch.no_grad()`。
2. 减少重复的 Python/processor 初始化和临时对象。
3. KV Cache 的分配、布局和复用。
4. 目标硬件支持的静态图、算子融合和 attention 实现。
5. 主办方书面认可后的运行时低精度方案。

每项优化都必须同时报告 Accuracy、TTFT、Throughput、端到端耗时和内存，不允许只保留成功样本。

## 阶段 P：PPU

1. 共享节点只检查 OS、驱动、SDK、示例和工具链。
2. 不向共享节点上传正式成果、数据或额外密钥。
3. 个性化资源开放后再运行真实模型、Profile 和正式性能对比。

