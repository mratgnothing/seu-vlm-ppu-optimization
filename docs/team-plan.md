# 两人协作与一个月推进计划

更新时间：2026-07-24

## 角色划分

### 成员 A：模型、精度与评测

- 维护公开评测入口、分层抽样和结果比较器。
- 跑本地 GPU baseline、精度回归和失败样本分类。
- 研究提示词稳定性、运行时精度和主办方允许的量化方案。
- 维护实验表、技术报告中的 Accuracy 与方法章节。

### 成员 B：PPU、运行时与算子

- 维护 PPU 预检、环境复现和一键启动脚本。
- 跟进 PPU-vLLM、Qwen3.5 GDN、causal conv 和 FlashAttention 兼容性。
- 使用 asight/transProfiler 定位 prefill/decode 热点。
- 负责算子融合、调度、内存、量化 kernel 和 PPU 性能章节。

### 共同责任

- 每个优化必须在相同 question ID、模型 revision 和生成配置上比较。
- PR 必须附 Accuracy、TTFT、Throughput、验证状态和失败样本数。
- 任何一方都不得提交密钥、模型权重、评测数据和逐样本原始输出。
- 每两天做一次短合并；`main` 始终保持可复现。

## 一个月节奏

### 第 1 周：可信基线与环境

当前已完成：

- 官方资料、数据、模型 revision 和 SHA-256 锁定。
- 本地 BF16 纯 GPU 加载与中英文 20 条真实基线。
- O1 安全优化、中英文各三次性能复测。
- 中英文比例分层各 200 条精度验证。
- CUDA 热点、显存峰值和 PPU SDK/vLLM 兼容性调查。

剩余：

- 公开集完整精度验证。
- 从主办方获得 PPU Python 栈、Qwen3.5 镜像和隔离资源答案。

### 第 2 周：PPU 功能闭环

- 在隔离资源运行 `scripts/check_ppu_runtime.py`。
- 优先尝试主办方新版 PPU-vLLM。
- 若只有 PPU PyTorch，先建立 Transformers eager 功能/精度 baseline。
- 跑单样本、20 条、200 条，记录加载、OOM、Accuracy 和运行时。
- 用 asight/transProfiler 分离 vision prefill、text prefill 和 decode。

验收：Qwen3.5-2B 在 PPU 上完成真实图片问答，结果可被公开评测入口解析。

### 第 3 周：单变量优化

优先顺序：

1. GDN/线性注意力与 causal conv fast path。
2. Decode GEMV、GEMM 和 kernel 启动/融合。
3. KV/state cache 分配、布局和复用。
4. 图像预处理、视觉编码器与 H2D 调度。
5. 获书面许可后的 FP8/INT8/INT4 或 KV Cache 量化。

每项至少运行 3 次，保留失败结果；精度下降时先回滚并定位。

验收：至少 2 项可复现优化，其中 1 项具有明确系统级或 PPU kernel 证据。

### 第 4 周：完整评测与交付

- 固定最终 commit、环境和启动命令。
- 跑完整公开集和主办方允许的最终评测。
- 汇总 Accuracy、TTFT、吞吐、端到端耗时和峰值内存。
- 完成初赛技术方案、性能对比、复现说明和源代码清单。
- 在全新目录按提交说明复现一次。

验收：评测脚本可一次执行，报告中的每个数值能追溯到结果文件和 commit。

## 当前交接点

- 成员 A 可以继续扩充公开集样本、失败分类和实验报告。
- 成员 B 当前应先把 [主办方问题清单](questions-for-organizer.md) 发出；收到镜像和资源答复后进入 PPU 功能闭环。
- 在主办方答复前，不在共享节点安装依赖或上传比赛资产。
