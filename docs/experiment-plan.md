# 首轮实验计划

## 阶段 B：建立可信基线

| 编号 | 内容 | 通过条件 |
|---|---|---|
| B0 | 中英文 dummy 各 20 条 | 评测入口通过，结果明确标记 `backend=dummy` |
| B1 | 真实模型仅加载 | 模型 class、device map 和内存占用有记录 |
| B2 | 中英文真实模型各 1 条 | 唯一 A/B/C/D、无验证错误、无 OOM |
| B3 | 中英文真实模型各 20 条 | 保存 Accuracy、TTFT、Throughput 和逐样本结果 |
| B3a | 中英文比例分层各 200 条 | 固定 seed、类别覆盖、Accuracy 和接口校验可追溯 |
| B4 | 公开集完整基线 | 环境稳定后再运行，结果与配置可追溯 |

B0-B4 已完成。中英文各 4029 条完整公开集均通过题号集合、分块和逐样本正确数审计；
英文全量暴露的一条截断输出已通过通用结论规范化和完整异常分块复测解决。当前 BF16
单样本路径全部参数位于 GPU，未使用 CPU offload。

## 阶段 O：低风险优化候选

真实基线完成后按单变量方式依次评估：

1. `torch.inference_mode()` 替代 `torch.no_grad()`：已完成，中英文各三次 20 条复测有效。
2. GPU profiler、显存峰值和热点定位：已完成，GEMV/GEMM 占 self CUDA time 86.18%。
3. Decode 小矩阵 GEMV、线性注意力和因果卷积 fast path。
4. Elementwise/copy kernel 融合与启动开销。
5. KV Cache 的分配、布局和复用。
6. 目标硬件支持的静态图、算子融合和 attention 实现。
7. 主办方书面认可后的运行时低精度方案。

每项优化都必须同时报告 Accuracy、TTFT、Throughput、端到端耗时和内存，不允许只保留成功样本。

## 阶段 P：PPU

1. 共享节点只检查 OS、驱动、SDK、示例和工具链。
2. 只读核验预置 PPU-vLLM 的 Qwen3.5、GDN、causal conv、FlashAttention 和量化兼容性。
3. 不向共享节点上传正式成果、数据或额外密钥。
4. 向主办方确认新版 Qwen3.5 PPU 镜像、隔离资源和量化允许范围。
5. 个性化资源开放后先运行 `scripts/check_ppu_runtime.py`，再进入真实模型、Profile 和正式性能对比。
