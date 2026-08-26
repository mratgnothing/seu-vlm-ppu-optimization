# 2026-08-26 PPU 首次真实基线、Profile 与 GEMV 迭代

## 结论先行

Qwen3.5-2B 已在一张 PPU-ZW810E 上形成真实多模态闭环：模型 617 个参数张量全部
驻留 `cuda:0`，无 CPU/meta/disk offload；真实 MMBench 图片、processor、视觉主干、
GDN/全注意力和自回归解码均已跑通。

固定中文前 20 条、2 条预热后，稳态 `torch.inference_mode()` 两次复测为：

- 平均 TTFT：119.171 / 117.852 ms；
- 平均解码吞吐：49.014 / 48.683 token/s；
- Accuracy：两次均为 85%，公开输出校验全部通过；
- 20 条答案和 token 数保持一致，平均每题生成 38.2 token。

Profile 表明当前主要问题是 Transformers eager 的大量小算子、分配与 launch，
而不是单个 GEMV 算力。HGGC `warp_vec2` 相对本仓库 reference 快 1.88--2.08 倍，
但三个形状仍全部慢于 PPU PyTorch `torch.mv`，因此当前版本不应接入模型。

## 实验边界

- 分支：`5070ti`；服务器隔离副本基于 commit `2085190` 加载本地未提交快照。
- 原始日志、trace、模型、数据集和环境目录均位于 Git ignored 路径，不提交仓库。
- 这是公开 dev 前 20 条工程基线，不代表主办方私有集成绩。
- 当前仍是 Transformers eager fallback，不是 PPU-vLLM，也未安装 GDN/
  causal-conv1d fast path。

## 环境

| 项目 | 实测值 |
|---|---|
| 设备 | 1 x PPU-ZW810E，98,304 MiB |
| Driver / PPU-SMI | `2.1.0-ra1f23` / `1.28` |
| SDK / HGGC | compiler `2.1.1-a5c56e` / HGGC 13.0 |
| Python | 3.12.3，独立 venv `/mnt/workspace/seu/envs/seu-vlm-ppu-20260826` |
| PyTorch | `2.11.0+v0.1.0.ppu2.1.1` |
| Triton | `3.6.0+v0.2.0.ppu2.1.1` |
| Transformers / Accelerate | 5.14.1 / 1.14.0 |
| Torchvision | `0.26.0+cpu`，只承担 CPU 图片算子 |
| 模型 | `Qwen/Qwen3.5-2B` revision `15852e8...`，BF16 |
| 参数驻留 | 617/617 张量在 `cuda:0`，2,213,241,664 elements |
| 模型 footprint | 4,426,483,648 bytes |

权重文件大小和 SHA-256 均与锁定清单匹配。当前官方小写 MMBench 文件审计结果为：

| 文件 | MD5 | 逻辑样本 | 审计 |
|---|---|---:|---|
| `mmbench_dev_cn.tsv` | `2ed5135326fed02c8e51ea50dda8222f` | 4,321 | ID、答案域、前 20 图均通过 |
| `mmbench_dev_en.tsv` | `d9ab776fc018b3d45785e9a5c23431c2` | 4,317 | ID、答案域、前 20 图均通过 |

这两个文件与仓库历史实验使用的 4,029 条版本不是同一份数据，历史 Accuracy 不做
回填或横向混算。

## 首次失败、根因与修复

真实样本第一次在视觉块 0 的 `attn.qkv` 进入 PPU GEMM RTC 时触发 `SIGABRT`。
`faulthandler` 和叶子模块边界日志给出的原生错误是：

```text
Both PPU_SDK and PPU_HOME are not exist
```

模型加载和 `32x32` BF16 冒烟不走该 RTC 路径，所以此前没有暴露。设置
`PPU_SDK=/usr/local/PPU_SDK` 后，同一 226-token 多模态 forward 完整通过，logits
为 `[1, 226, 248320]` BF16。`run_ppu_first_validation.sh` 已加入标准 SDK 与
`ppu-smi` 路径自动发现，不覆盖主办方显式变量。

首次完整 forward 还会向 `/usr/local/PPU_SDK/rtccache/PPU0010/` 写入不同形状的
GEMM/GEMV 缓存。若最终容器只读或每次清空该目录，TTFT 会重新包含编译开销。

## 20 条基线

固定参数：中文文件前 20 条、seed `20260625`、temperature 0、最多 256 新 token、
真实 Transformers 后端、2 条预热。

| Profile / run | 平均 TTFT ms | 中位 TTFT ms | P95 TTFT ms | 平均 token/s | 总时长 s | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| O1 inference r1 | 333.363 | 117.968 | 1679.646 | 49.936 | 38.745 | 85% |
| O1 inference r2 | 119.171 | 116.585 | 134.569 | 49.014 | 34.787 | 85% |
| O1 inference r3 | 117.852 | 116.372 | 127.028 | 48.683 | 34.665 | 85% |
| O0 no_grad r1 | 133.128 | 131.420 | 141.215 | 44.096 | 36.068 | 85% |

O1 r1 的均值和 P95 被新图像形状触发的额外 RTC 编译拉高；中位数已经接近稳态。
以缓存稳定后的 O1 r2/r3 中位聚合与 O0 比较，O1 平均 TTFT 降低约 10.98%，吞吐
提高约 10.78%，Accuracy 不变，因此保留 `inference_mode`。

## PPU Profile

对同一真实样本先生成 2 token 预热，再 profile 16-token 生成：prompt 为 226 token，
trace 约 151 MiB，只保留在服务器 ignored artifacts。

| 证据 | 值 | 解释 |
|---|---:|---|
| Self CPU total | 854.810 ms | Python/eager/dispatcher/分配与 launch 占比高 |
| Self accelerator total | 173.799 ms | 设备计算明显小于 CPU 调度总量 |
| `cudaLaunchKernel` | 37,293 calls / 126.816 ms CPU | 大量细碎 kernel |
| `empty_strided` | 8,832 calls / 96.078 ms CPU | 频繁临时张量分配 |
| `gemvt_op` 主组 | 1,906 calls / 29.343 ms PPU | 占 self PPU 16.88% |
| `aten::cat` | 1,017 calls / 6.974 ms PPU | 状态/KV 拼接与临时内存值得融合 |
| depthwise conv kernel | 288 calls / 1.397 ms PPU | causal conv 存在但不是单独最大项 |

另有数千次 elementwise、reduce、copy。结论是优先减少算子数、分配和 launch，
而不是只替换一个 GEMV。

实际 Transformers decode fallback 的 `torch_recurrent_gated_delta_rule` 每层每 token
依次执行 q/k L2 norm、state 衰减、`state·k` reduction、delta、`k⊗delta` 状态更新和
`state·q` reduction；18 个 GDN 层会把这些语义拆成大量小 kernel。它是比通用 GEMV
更合适的下一候选：以每个 head 为并行单元，在一个 kernel 内完成 FP32 recurrent
state 原地更新并输出 BF16，再单独验证与原实现的 state/output 误差。

## HGGC GEMV 迭代

测试三组 BF16 输入、FP32 累加/输出 kernel；使用 16 份权重轮换，32 次预热、
200 次计时、3 个独立进程复测。所有结果数值校验通过。

| N x K | reference ms | 最佳 `warp_vec2` | 自定义加速 | `torch.mv` ms | 自定义相对 torch |
|---|---:|---:|---:|---:|---:|
| 6144 x 2048 | 0.050721 | 0.024431 (64 threads) | 2.076x | 0.017053 | 慢 43.3% |
| 2048 x 6144 | 0.045066 | 0.023950 (128 threads) | 1.882x | 0.014727 | 慢 62.6% |
| 2048 x 2048 | 0.018236 | 0.009371 (128 threads) | 1.946x | 0.008029 | 慢 16.7% |

`torch.mv` 输出 BF16，而当前 HGGC kernel 输出 FP32，二者还不是可直接替换的完全同口径
实现。即便如此，自定义版本已经更慢，故不接入模型。保留 reference、warp reduction、
BF16x2 向量化和重复测试脚本，作为后续融合 kernel 的正确性基础。

## 遇到的问题与可行方案

| 问题 | 已验证处理 | 后续风险/方案 |
|---|---|---|
| 通用 torchvision wheel 缺 PPU ABI | 隔离 venv 改用 `0.26.0+cpu` | 不让它覆盖 PPU torch/triton |
| RTC 因 SDK 变量为空直接 abort | 自动发现并导出 `PPU_SDK` | 确认最终容器 SDK 与 cache 可写 |
| `ppu-smi` 不在普通 PATH | 加 `${PPU_SDK}/ppu-smi/bin` | 镜像路径变化时显式配置 |
| HGGC 链接缺符号 | 链接 `libhggcrt1.so` | 私有镜像库名变化时可用环境变量覆盖 |
| Windows 上传 shell 为 CRLF | `.gitattributes` 强制 LF，临时副本 `sed` 修复 | 提交后从 Git clone 可消除 |
| fast GDN/causal-conv 未安装 | eager fallback 可正确运行 | 性能主风险，应优先拿官方 PPU wheel/源码 |
| vLLM 与 `/opt/vllm` 均不存在 | Transformers eager 保底 | 需要主办方 PPU-vLLM/Qwen3.5/GDN 支持 |
| 主办方包仍为 v1.1 | 保持 hash 锁定 | v1.2 到手后先 diff 指标与接口再优化 |

## 待用户决策

1. **推荐：暂停继续单独优化 GEMV。** 当前版本已经输给 `torch.mv`，模型热点又显示
   launch/elementwise/分配是主要矛盾。
2. **优先路线 A：** 向主办方获取比赛 PPU-vLLM、Qwen3.5/GDN 与 causal-conv fast
   path；若已有官方实现，先做同口径基线再决定是否自写。
3. **备选路线 B：** 在 Transformers eager 中做融合候选，优先 GDN norm/gate/state
   update/causal-conv 或重复的 elementwise + reduce + copy，不先替换通用 GEMV。
4. 确认最终实例是否持久化 `/usr/local/PPU_SDK/rtccache`；否则需要提交前预编译或
   将 cache 指向可持久化目录。
5. 等 v1.2 和量化规则明确后，再决定 BF16、INT8/FP8/权重量化是否进入主路线。

## 复现入口

- 分层验证：`scripts/run_ppu_first_validation.sh`
- 原生崩溃定位：`scripts/diagnose_qwen35_ppu.py`
- 真实生成 profile：`scripts/profile_qwen35_ppu.py`
- HGGC 重复对照：`ppu/microbench/run_repeated_benchmark.sh`
- GEMV 结果汇总：`scripts/summarize_ppu_gemv.py`

原始结果保存在服务器隔离副本的 `artifacts/`；仓库只提交上述聚合证据与结论。
