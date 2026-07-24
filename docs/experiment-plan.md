# 首轮实验计划

## 阶段 B：建立可信基线

| 编号 | 内容 | 通过条件 |
|---|---|---|
| B0 | 中英文 dummy 各 20 条 | 评测入口通过，结果明确标记 `backend=dummy` |
| B1 | 真实模型仅加载 | 模型 class、device map 和内存占用有记录 |
| B2 | 中英文真实模型各 1 条 | 唯一 A/B/C/D、无验证错误、无 OOM |
| B3 | 中英文真实模型各 20 条 | 保存 Accuracy、TTFT、Throughput 和逐样本结果 |
| B4 | 公开集完整基线 | 环境稳定后再运行，结果与配置可追溯 |

B0-B3 已完成，B4 待运行。当前 BF16 单样本路径全部参数位于 GPU，未使用 CPU offload。

## 阶段 O：低风险优化候选

真实基线完成后按单变量方式依次评估：

1. `torch.inference_mode()` 替代 `torch.no_grad()`：已完成，三次中文 20 条复测有效。
2. GPU profiler、显存峰值和 prefill/decode 热点定位：进行中。
3. 减少重复的 Python/processor 初始化和临时对象。
4. KV Cache 的分配、布局和复用。
5. 目标硬件支持的静态图、算子融合和 attention 实现。
6. 主办方书面认可后的运行时低精度方案。

每项优化都必须同时报告 Accuracy、TTFT、Throughput、端到端耗时和内存，不允许只保留成功样本。

## 阶段 P：PPU

1. 共享节点只检查 OS、驱动、SDK、示例和工具链。
2. 不向共享节点上传正式成果、数据或额外密钥。
3. 个性化资源开放后再运行真实模型、Profile 和正式性能对比。
