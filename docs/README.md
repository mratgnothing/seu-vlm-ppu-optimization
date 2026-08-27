# 项目导航与进度总览

更新时间：2026-08-27

本页是人类成员和 Agent 的统一入口。需要快速接手时，先看“当前结论”和“模块进度”，
再按表格中的证据文档深入。

## 当前结论

- Qwen3.5-2B 已在本地 RTX 4050 6GB 上以 BF16、真实 Transformers 后端完成部署和评测。
- 中文完整公开集 Accuracy 为 83.94%（3382/4029）。
- 英文完整公开集 Accuracy 为 79.75%（3213/4029）。
- O1 相对 O0 的正式三次中位提升：
  - 中文 TTFT 8.25%，Throughput 8.82%；
  - 英文 TTFT 10.22%，Throughput 6.37%。
- Qwen3.5-2B 已在隔离 PPU-ZW810E 上形成真实多模态闭环，参数无 offload。
- 中文前 20 条 PPU 稳态基线 Accuracy 85%，平均 TTFT 约 118.5 ms，吞吐约
  48.85 token/s；O1 相对 O0 的稳态聚合提升约 10.98%/10.78%。
- PPU profile 已确认大量细碎 launch、elementwise、reduce、copy 和临时分配；
  当前 HGGC GEMV 优化核尚未超过 `torch.mv`。
- 48-edge residual-add + RMSNorm 已在真实 PPU 上消除 720 次 add/launch；CN20 两轮
  配对中位约 `1.021x`，均 20/20 全文一致，当前仍为显式 opt-in 候选。

## 模块进度

| 模块 | 状态 | 已完成 | 下一步 | 证据入口 |
|---|---|---|---|---|
| 赛事规则与接口 | 已确认 | v1.1 入口、指标公式、交付边界 | 跟踪主办方新通知 | [规则与边界](rules-and-boundaries.md) |
| 模型与本地环境 | 已完成 | 指定 revision、SHA-256、BF16 纯 GPU 加载 | 只在依赖确有需要时升级 | [当前状态](current-status.md) |
| 小样本性能基线 | 已完成 | 中英文 O0/O1 各三次、M1 首 token 口径 | PPU 上复用同一统计口径 | [M1 实验](experiments/2026-07-24-ttft-token-boundary.md) |
| 完整公开集精度 | 已完成 | 中英文各 4029 条、分块哈希与题号集合审计 | 私有集评测时保持同一 Accuracy 护栏 | [中文全量](experiments/2026-07-26-cn-full-n4029.md)、[英文全量](experiments/2026-07-26-en-full-n4029.md) |
| 输出解析鲁棒性 | 已完成 | Markdown 选项、截断结论规范化和整块复测 | 继续记录新格式边界，禁止按题号修补 | [英文全量修复记录](experiments/2026-07-26-en-full-n4029.md) |
| CUDA 热点定位 | 已完成 | GEMV/GEMM 占 self CUDA time 86.18%，显存峰值已记录 | 映射到 PPU kernel/profile | [CUDA Profile](experiments/2026-07-24-o2-cuda-profile.md) |
| PPU 工具链 | 已完成首轮闭环 | SDK、驱动、HGGC、定制 PyTorch、模型驻留、真实样本与 20 条基线 | 获取比赛 PPU-vLLM/v1.2，固定最终镜像 | [首次实验](experiments/2026-08-26-ppu-baseline-and-gemv.md)、[首次上机手册](ppu-first-validation.md) |
| PPU 关键算子 | 六类核与图融合已验证 | GDN/conv/norm/qk-RoPE、packed MLP；grouped-acBLAS GDN 两轮 20/20 exact；48-edge residual-RMSNorm 两轮约 +2.1% 中位、20/20 exact | 完整公开集与官方私有门限；继续按 profile 减少 launch | [融合实验](experiments/2026-08-26-ppu-fused-decode-kernels.md)、[packed MLP](experiments/2026-08-27-ppu-packed-mlp.md)、[packed GDN](experiments/2026-08-27-ppu-packed-gdn-projections.md)、[residual-RMSNorm](experiments/2026-08-27-ppu-residual-rmsnorm.md)、[acBLAS](experiments/2026-08-27-ppu-acblas-gemv.md) |
| 技术报告 | 初稿完成 | 方法、指标、结果和真实性边界已整理 | 补 PPU 真实实验、最终复现命令 | [初赛技术报告](preliminary-technical-report.md) |
| 源码交付 | 候选包可用 | 白名单打包、敏感扫描、可复现 ZIP | 按主办方最终目录要求定稿 | [根目录 README](../README.md) |

## 当前关键阻塞

本地不再缺少基础模型、数据、评测脚本或可复现基线。关键外部依赖是主办方提供：

1. 主办方 v1.2 评测包及与 v1.1 的差异；
2. 比赛 PPU-vLLM 是否注册 Qwen3.5 并实现 GDN/causal-conv fast path；
3. RTC cache 在最终实例中的可写性与持久化规则；
4. 量化、校准数据和混合精度允许范围；
5. 最终复现镜像、启动命令和提交目录限制。

可直接发送的内容见 [需要向主办方确认的问题](questions-for-organizer.md)。

## 快速接手

### 人类成员

1. 阅读本页和 [当前状态](current-status.md)；
2. 查看 [两人协作计划](team-plan.md)，认领一个可验收任务；
3. 从 `main` 拉出短分支；
4. 使用 [实验模板](experiment-template.md) 记录配置、提交、结果和失败；
5. 通过 Pull Request 合并，避免两人直接覆盖同一实验记录。

隔离 PPU 节点首次接入时，按 [PPU + Qwen3.5 首次上机验证手册](ppu-first-validation.md)
运行 `scripts/run_ppu_first_validation.sh`。默认只做只读预检；设备计算、微基准、
完整模型加载和真实样本必须分别显式启用，并且只允许在主办方批准的个性化隔离
资源执行。

### Agent

1. 先阅读根目录 `AGENTS.md` 和 `PROJECT_CONTEXT.md`；
2. 核对 `git status`、当前分支和目标任务；
3. 只使用显式真实后端生成正式结果；
4. 保留 Accuracy 护栏和原始证据，不混用本地 GPU、共享 PPU 与目标 PPU 结论；
5. 完成后更新本页、当前状态和对应实验记录。

## 本地常用命令

```powershell
# 无模型测试
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# 完整中英文 Accuracy，可断点续跑
.\scripts\run_full_accuracy.ps1 -Language cn
.\scripts\run_full_accuracy.ps1 -Language en

# 构建源码候选包
.\scripts\build_submission.ps1
```

本地模型、数据、原始结果和 `configs/local.psd1` 均不进入 Git。复制环境时从
`configs/local.example.psd1` 建立个人配置。

## 文档地图

- [项目目标与真实状态边界](../PROJECT_CONTEXT.md)
- [当前已验证状态](current-status.md)
- [首轮实验计划](experiment-plan.md)
- [两人协作与一个月计划](team-plan.md)
- [初赛技术报告初稿](preliminary-technical-report.md)
- [PPU 兼容性矩阵](ppu-compatibility-matrix.md)
- [需要向主办方确认的问题](questions-for-organizer.md)
- [2026-08-26 PPU 首次真实基线与算子实验](experiments/2026-08-26-ppu-baseline-and-gemv.md)
- [2026-08-27 PPU packed MLP](experiments/2026-08-27-ppu-packed-mlp.md)
- [2026-08-27 PPU 注册式 acBLAS Linear](experiments/2026-08-27-ppu-acblas-gemv.md)
- [2026-08-27 PPU GDN 输入投影打包](experiments/2026-08-27-ppu-packed-gdn-projections.md)
- [2026-08-27 PPU residual-add + RMSNorm](experiments/2026-08-27-ppu-residual-rmsnorm.md)
- [2026-08-27 PPU SwiGLU 融合负实验](experiments/2026-08-27-ppu-swiglu-negative.md)
- [PPU 资源释放前快照与恢复手册](ppu-resource-release-handoff.md)
- [PPU 后续优化路线图](ppu-future-roadmap.md)
- [实验记录目录](experiments/)
