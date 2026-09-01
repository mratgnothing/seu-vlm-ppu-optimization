# 赛道二 PPU 推理优化提交说明

## 任务与评分

本项目对应东南大学 AI+ 创新应用大赛赛道二“端侧 AI 推理优化挑战”。根据赛题页面，
评分由模型推理精度 30%、推理延迟优化 30%、吞吐量提升 20% 和系统级优化 20% 组成。
评测模型为 Qwen3.5-2B，最终成绩以主办方 PPU 环境和私有数据复测为准。

本提交不更换模型权重、不缩减视觉 Token，也不修改评测数据。最终性能栈包含 GDN、
causal-conv、RMSNorm、gated-RMSNorm、q/k RMSNorm+RoPE、packed MLP、GDN gate-prep、
residual+RMSNorm、grouped acBLAS 投影、b/a-GEMV 和 multi-row prefill 融合。

## PPU 实测结果

最终一次同实例、独立进程 CN20 ABBA 对比为：

| 指标 | 原始 eager | 当前提交 | 变化 |
|---|---:|---:|---:|
| 平均吞吐中位 | 49.2195 token/s | 133.623 token/s | 2.71484x（+171.48%） |
| 平均 TTFT 中位 | 120.059 ms | 114.313 ms | 降低 4.79% |
| Accuracy | 17/20、17/20 | 17/20、17/20 | 不变 |

20/20 解析答案和正确性在四次运行中一致，20/20 样本吞吐更快。公开 benchmark 不保存
全文 hash，因此这里只声明 Accuracy、解析答案和正确性一致。

独立的双语 prefill A/B 中，multi-row prefill 融合使 CN20/EN20 TTFT 配对中位分别
提升 4.86%/4.48%，Accuracy 和 40/40 解析答案不变。更激进的 prefill SwiGLU 路径
使性能下降，已从提交源码移除。

## 文件与启动

- `evaluation_wrapper.py`：主办方接口及优化挂载；
- `ppu/custom_ops/`：最终 HGGC/acBLAS 源码与构建入口；
- `scripts/bootstrap_ppu_env.sh`：在一次性官方镜像中创建隔离 venv 并重编扩展；
- `scripts/activate_ppu_env.sh`：激活隔离环境；
- `scripts/activate_ppu_profile.sh performance`：选择最终性能栈；
- `scripts/run_submission.sh`：按上述顺序启动公开自测；
- `environment-ppu.yml`：环境结构说明。PPU patched torch 必须使用镜像自带版本。
- `COMPETITION.md`：面向评委的完整技术演进、Qwen 架构、实验依据、失败方向、评分对齐与 VLA 应用设计。

`environment-ppu.yml` 用于记录 Python 与用户态依赖，不应在官方 PPU 镜像中直接安装
普通 PyPI/Conda torch。实际部署以 `bootstrap_ppu_env.sh` 为准：它创建可读取镜像
patched torch 的 `--system-site-packages` venv，并检查该运行时没有被覆盖。

快速启动：

```bash
bash scripts/bootstrap_ppu_env.sh
export SEU_PPU_VENV_DIR="${HOME}/.cache/seu-vlm-ppu/venv"
source scripts/activate_ppu_env.sh
source scripts/activate_ppu_profile.sh performance
bash scripts/run_submission.sh \
  /path/to/Qwen3.5-2B \
  /path/to/mmbench_dev_cn.tsv \
  result.json
```

快速冒烟可在命令前设置 `SEU_NUM_SAMPLES=2 SEU_WARMUP_SAMPLES=1`；正式复测不要设置
`SEU_NUM_SAMPLES`，脚本将运行整个数据文件。

不要把模型权重、评测数据、SSH 密钥或本地缓存放入提交目录。
