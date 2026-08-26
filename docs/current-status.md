# 当前状态

更新时间：2026-08-26

## 已确认

- 正式工作目录：`D:\GitHub\seu-vlm-ppu-optimization`
- 本机 GPU：NVIDIA GeForce RTX 4050 Laptop GPU，6GB 显存
- 官方模型：`Qwen/Qwen3.5-2B`
- 模型 revision：`15852e8c16360a2fea060d615a32b45270f8a8fc`
- 官方模型完整下载约 4.6GB，单个 BF16 权重文件约 4.5GB
- 模型参数量约 2.274B
- 模型 revision 已完整下载到项目忽略目录
- 主权重 SHA-256：`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`
- 模型类：`Qwen3_5ForConditionalGeneration`
- 617 组参数全部位于 `cuda:0`，模型内存 footprint 为 4,426,483,648 bytes
- 本地公开数据：MMBench 中文/英文 dev 各 4029 条
- 官方 v1.1 评测代码已导入

## 当前 RTX 5070 Ti 开发环境

- 工作分支：`5070ti`
- 独立 Conda 环境：`G:\seu-AI\.conda-envs\seu-vlm-5070ti`
- Python 3.12.13
- PyTorch 2.13.0+cu130
- Torchvision 0.28.0+cu130
- Transformers 5.14.1
- NVIDIA GeForce RTX 5070 Ti Laptop GPU，约 11.94 GiB 显存
- CUDA runtime 13.0，驱动 591.97，BF16 可用
- 42 项无模型测试通过
- Qwen3.5-2B 锁定 revision 已通过完整性校验和真实加载冒烟；617 个参数张量均在
  `cuda:0`，模型报告内存占用 4,426,483,648 bytes（约 4.12 GiB）
- 当前 Transformers 未安装可选的 fast-path 依赖，加载时会回退到 PyTorch 实现；
  这不阻塞正确性冒烟，但不能作为最终性能环境
- PPU 首次验证入口已完成本地只读冒烟；默认不运行微基准

当前尚未在 RTX 5070 Ti 上复测下述 Accuracy、TTFT 和 Throughput。因此所有已有
性能数字仍归属于原 RTX 4050 环境，不能直接写成 5070 Ti 结果。

## 原 RTX 4050 基线运行环境

- Python 3.12.7
- PyTorch 2.13.0+cu130
- Torchvision 0.28.0+cu130
- Transformers 5.14.1
- CUDA 可用，GPU 支持 BF16
- Qwen3.5 配置、`AutoModelForImageTextToText` 和 `AutoModelForMultimodalLM` 映射均已验证

## 已完成的工程基线

- Dummy 中英文各 20 条接口冒烟完成，结果明确标记为非真实模型。
- 真实模型中英文各 1 条生成与解析验证完成。
- 中文公开集前 20 条基线：Accuracy 85%，平均 TTFT 327.451 ms，吞吐 21.813 tokens/s，公开校验通过。
- 英文公开集前 20 条基线：Accuracy 80%，平均 TTFT 409.660 ms，吞吐 28.685 tokens/s，公开校验通过。
- O1 `torch.inference_mode()` 三次中文复测均保持 Accuracy 85%、答案和 token 数不变。
- M1 首 token 口径中文 O0 三次中位数：TTFT 313.562 ms，吞吐 21.328 tokens/s。
- M1 首 token 口径中文 O1 三次中位数：TTFT 287.706 ms，吞吐 23.209 tokens/s。
- O1 中文正式提升：TTFT 8.25%，吞吐 8.82%，Accuracy 仍为 85%。
- O1 英文三次复测：TTFT 中位提升 10.22%，吞吐中位提升 6.37%，Accuracy 仍为 80%。
- 中英文跨 profile 的 question ID、答案和 token 数均完全一致，公开接口校验通过。
- 中文比例分层 200 条 Accuracy 84.5%（169/200），20 个类别覆盖，接口校验通过。
- 英文比例分层 200 条 Accuracy 82.5%（165/200），20 个类别覆盖，接口校验通过。
- 中文完整公开集 Accuracy 83.94%（3382/4029），21/21 分块、4029 个唯一题目 ID 和公开接口校验全部通过。
- 英文完整公开集原始 Accuracy 79.72%（3212/4029），有 1 条截断输出未解析；通用结论规范化经完整 200 条异常分块复测后，仅恢复该样本，最终 Accuracy 79.75%（3213/4029），公开接口校验全部通过。
- O2 单样本 CUDA profiler 已完成：GEMV/GEMM 占 self CUDA time 86.18%，elementwise/copy 调用数高。
- Profiler 口径 peak allocated/reserved 为 4.19/4.21 GiB。
- 原基线代码通过 25 项测试；当前 `5070ti` 工作分支扩展后通过 42 项测试。源码候选包生成器可自动排除模型、数据、原始结果、密钥与本地配置。
- 当前正式性能数据使用首个生成 token 计时，只覆盖固定前 20 条；中英文 Accuracy 均已扩展到完整公开集。完整集单次运行只作为精度证据，不进入正式性能表，也不代表私有评测成绩。

## PPU 状态

- 个性化隔离节点识别到 1 张 PPU-ZW810E，显存 98,304 MiB；Driver
  `2.1.0-ra1f23`、PPU-SMI 1.28、SDK/compiler `2.1.1-a5c56e`、HGGC 13.0。
- 独立 venv 复用 PPU PyTorch `2.11.0+v0.1.0.ppu2.1.1` 和定制 Triton，另装
  Transformers 5.14.1；未替换系统 PPU torch/triton。
- Qwen3.5-2B 617 个参数张量全部在 `cuda:0`，无 CPU/meta/disk offload；真实中文
  MMBench 图片的视觉、GDN、全注意力和 51-token 解码均通过。
- RTC 首次因 `PPU_SDK`/`PPU_HOME` 均未导出而在视觉 qkv 处 abort；已用叶子模块
  日志定位，并在启动脚本中自动发现 `/usr/local/PPU_SDK` 后修复。
- 中文前 20 条缓存稳定后两次 O1：平均 TTFT 119.171/117.852 ms，吞吐
  49.014/48.683 token/s，Accuracy 均为 85%，公开校验全部通过。
- O0 `no_grad` 对照为 133.128 ms、44.096 token/s；O1 稳态聚合约降低 TTFT
  10.98%、提高吞吐 10.78%，Accuracy 不变。
- 16-token PPU profile：self CPU/PPU 854.810/173.799 ms，37,293 次 kernel
  launch、8,832 次 `empty_strided`；热点是 eager 小算子/调度/分配而非单个 GEMV。
- HGGC `warp_vec2` 相对 reference 快 1.88--2.08 倍，但三个形状仍比 `torch.mv`
  慢 16.7%--62.6%，因此暂不接入模型。
- 当前没有 vLLM 或 `/opt/vllm`，Transformers 提示缺少 GDN/causal-conv fast path；
  eager 正确性可用，但不是最终性能路线。
- 完整证据见 [2026-08-26 PPU 首次真实基线、Profile 与 GEMV](experiments/2026-08-26-ppu-baseline-and-gemv.md)。

## 尚未完成

- 获取主办方 PPU-vLLM/Qwen3.5/GDN fast path，并与 eager 做同口径对照。
- 根据 PPU profile 实现并验证 GDN/causal-conv/elementwise 融合候选。
- 量化/权重变换的正式允许范围确认。
- 初赛技术报告已形成可持续更新的初稿，源码候选包可一键生成；仍缺 PPU 实测章节、最终复现说明和按主办方格式定稿。

## 当前风险

RTX 4050 的 6GB 显存仍只适合作为历史单样本环境；更长输入、更高分辨率、编译缓存
或并发请求可能 OOM。PPU 显存充足，但当前 Transformers eager 的线性注意力和
因果卷积缺少 fast path，16-token profile 暴露 37,293 次 launch 和大量临时分配。
下一步应优先取得官方 PPU-vLLM/FLA 路径或融合 recurrent GDN state update，不能再把
CUDA 上“GEMV 是第一热点”的结论直接套到 PPU。CPU offload、冷 RTC 和稳态 PPU
结果不得混算。
