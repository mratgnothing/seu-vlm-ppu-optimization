# 当前状态

更新时间：2026-08-28

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
- 46 项无模型测试通过
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
- 原基线代码通过 25 项测试；当前 `5070ti` 工作分支扩展后通过 46 项测试。源码候选包生成器可自动排除模型、数据、原始结果、密钥与本地配置。
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
- 已实现并接入五类 HGGC decode 融合核：18 层 recurrent GDN、18 层 causal-conv、
  49 个 2048 维 RMSNorm、18 个 128 维 gated RMSNorm，以及 6 层 full-attention
  q/k RMSNorm+partial RoPE；默认均需显式环境变量启用。
- 同机固定中文前 20 条：eager 为 118.493 ms / 49.737 token/s / 85%；GDN+conv
  为 117.262 ms / 63.911 token/s / 85%；all-four 为 119.677 ms /
  81.307 token/s / 85%，all-four 吞吐提升 63.47%。
- 20/20 解析答案与正确性一致，但 GDN+conv 有 3 条、all-four 有 5 条生成 token 数
  变化；因此 all-four 尚未通过完整公开集精度门禁，不能直接作为最终提交配置。
- 完整 all-four 16-token profile 的 self CPU/PPU 为 514.366/131.899 ms；其中
  18/18/49/18 个模块均已挂载，下一热点为运行时 GEMV/GEMM 和剩余
  elementwise/cat/reduce，而不是 causal-conv。
- all-five 固定中文前 20 条两次为 124.930/118.227 ms、93.918/94.889 token/s、
  85%；相对 eager 吞吐提升 88.83%/90.78%，答案与正确性 20/20 一致，token 数
  漂移仍为 5/20。51-token 单样本三次还通过 exact-text SHA-256 gate。
- all-five profile 的 self CPU/PPU 为 409.545/121.871 ms；相对 all-four，
  `cudaLaunchKernel` 19,878→17,088、`aten::cat` 747→387、`empty_strided`
  5,472→4,932。新 q/k RMSNorm+RoPE 核 90 次合计约 0.216 ms。
- 24 层 packed-MLP 把 gate/up 两次投影合并为一次且不复制常驻权重；CN20 两轮为
  96.506/96.715 token/s、85%，相对 eager 提升 94.04%/94.46%。
- 注册式 PyTorch/acBLAS extension 通过 C ABI 隔离 Torch/CUDA 与 HGGC 头，随机
  BF16 模块级达到 1.08--1.17x，并在 profile 中恰好减少 1530=102×15 次
  `aten::linear/mm`；但最终单 `.so` 版本固定 128-token 八对的成对中位仅
  0.9997x、4/8 获胜，故定性为负实验，不接入 wrapper。
- Qwen3.5 GDN 每层 qkv/z/b/a 四个同输入投影已实现共享 storage 的一次 Linear。
  最终线程隔离版固定 128-token 四对全部获胜、全文一致、成对中位 +1.82%；CN20
  平均吞吐 94.099→98.430 token/s、成对中位 +3.55%、Accuracy 均为 85%，但 19/20 全文
  一致，唯一差异为同答案多 1 token，所以保持默认关闭。
- `(2,1,1)` 精确分组恢复 CN20 20/20 全文一致，但成对中位 -1.16%，证明收益主要
  来自将四次提交压成一次，而非权重连续化本身。
- 结构专用 grouped acBLAS GDN 在一次 C++ extension 入口中保留四个原形状 GEMV。
  CN20 两轮平均吞吐为 96.409→98.028 和 95.634→99.601 token/s，成对中位
  +1.87%/+3.91%，16/20 和 17/20 获胜；Accuracy 均为 85%，两轮均 20/20 全文
  一致。固定 128-token 六对成对中位 +1.21%，但仅 3/6 获胜，故作为默认关闭的
  精度优先候选继续扩大验证。
- decoder 48 条 `residual add -> RMSNorm` 相邻边已用一个 HGGC kernel 融合；16-token
  profile 中 `[1,1,2048]` add 720→0、`cudaLaunchKernel` 16973→16253。只融合
  24 条层内边的首版固定长中位为 0.9821x，作为负实验保留；补齐跨层链后固定长
  两轮中位为 1.0159x/1.0233x，CN20 两轮平均吞吐为 100.156→101.616 和
  98.576→101.507 token/s，成对中位 1.0213x/1.0206x、均 14/20 获胜，Accuracy
  均 85% 且 20/20 全文一致。正式 wrapper 真实 PPU smoke 通过，候选仍默认关闭。
- 18 层 cached decode 的 GDN gate-prep 已把 `sigmoid(b)`、`exp(A_log)`、
  `softplus(a+dt_bias)` 等七组小 kernel 合成一次 HGGC 调用，并缓存静态
  `exp(A_log)`、thread-local 复用 `g/beta`。最终固定 128-token 六对 6/6 获胜、
  全文一致、配对中位 1.0839x；CN20 两轮为 101.651→109.275 和
  100.085→107.083 token/s，配对中位 1.0811x/1.0863x、19/20 和 17/20 获胜，
  均 20/20 全文一致、Accuracy 85%。profile 的 launch 16253→14363，Self CPU/PPU
  分别下降 11.48%/5.35%；memcheck 0 errors。中文完整公开集两路 Accuracy 均为
  3374/4029，4029/4029 完整文本、答案和 token 数一致，成对吞吐中位 1.0862x、
  3882/4029 获胜。正式开关仍需显式启用，但已是当前推荐提交配置的一部分。
- acBLASLt 四个真实 decode 形状各扫描 32 个 bit-exact heuristic；packed MLP、MLP
  down、GDN qkv 最高仅 1.0121x/1.0269x/1.0191x，止损。2048 方阵低层 1.0577x，
  配合 scratch 模块级 1.2797x，但整模固定 128-token 八对仅 0.9898x、3/8 获胜。
  Profile 显示主 `gemvt_op` 增加 270 次，故作为负实验保留，不接入 wrapper。
- 单入口 acBLAS packed-MLP 将 packed gate/up GEMV、HGGC SwiGLU 和 down GEMV 放入
  一次 C++ extension 入口，并为每层复用三个 BF16 scratch。模块级 `1.2288x`；
  固定 128-token 八对 8/8 获胜、成对中位 `1.1336x`；CN20 两轮均 20/20
  全文一致、20/20 获胜、Accuracy 85%，成对中位 `1.1212x/1.1122x`。Profile
  中 `aten::linear/mm` 各减少 720 次、launch 减少 360 次，GEMV 数保持不变；
  中文完整集两路 Accuracy 均为 3374/4029，4029/4029 文本、答案和 token 数一致，
  平均吞吐 `109.993→122.445 token/s`、成对中位 `1.1125x`，3939/4029 获胜。
  英文完整公开集也以 4029/4029 exact 通过：两路 Accuracy 均为 3214/4029，平均
  吞吐 `107.276→118.964 token/s`、成对中位 `1.1093x`，3806/4029 获胜。
  最终重编译的 memcheck 为 0 errors，正式 wrapper meta 记录 24 个新模块。
- 完成 6 个全注意力层的单入口 Attention Prep 负实验：保留 Q/K/V 三个原形状 BF16
  GEMV，并在同一 extension 入口调用既有 Q/K RMSNorm+RoPE。真实模块 Q/K/V/gate、
  prefill 回退、scratch 复用、异流拒绝和 memcheck 均通过；强化 smoke 的模块边界为
  `0.080652→0.019668 ms`（`4.1006x`）。然而固定 128-token 56 对合并中位仅
  `1.0047x`，CN20 两轮中位为 `1.0158x/0.9852x`，第二轮严格性能门禁失败。
  按规则停止 profile 与完整集，候选保持默认关闭。
- HGGC Graph Capture 在 PyTorch PPU 端已实测可用，固定 16 段 elementwise 子图
  `1.8303x` 且 exact；但 graph-backed 单入口 packed-MLP 的稳定地址收益仅 `1.0203x`，
  动态地址加 input copy 后为 `0.9316x`。按 3% 低层晋级余量止损，未改变正式配置。
- residual-RMSNorm 输出 scratch 模块级相对现有融合 `1.3373x`，bit-exact、复用地址
  和 memcheck 均通过；但当前推荐栈固定 128-token 八对中位 `0.9862x`、仅 2/8
  获胜，已停止 CN20/profile，正式 wrapper 不启用。
- raw-stream 查询直接取得当前流整数句柄，避免当前推荐栈约 127 次/token 的 Python
  Stream 对象查询。固定 128-token 与中英文 20 条双轮均 exact 且性能门禁通过；
  中文完整集 4029/4029 exact、Accuracy 均为 3374/4029，平均吞吐
  `120.383→131.107 token/s`、成对中位 `1.0906x`，3817/4029 获胜。
  英文完整集也 4029/4029 exact、Accuracy 均为 3214/4029，平均吞吐
  `118.577→129.398 token/s`、成对中位 `1.0901x`，3704/4029 获胜。
  memcheck 0 errors、profile exact、正式 wrapper/meta 均通过。
- 最终正式 wrapper 单样本 smoke 为真实 Transformers/PPU backend、公开校验通过，
  模块计数为 `18/18/49/18/6/24/24/18/24/18`；46 项无模型单元测试全部通过。
- 独立 BF16 SwiGLU HGGC 核在 `[1,1,6144]` 上四组线程均 bit-exact，但最优仅
  `0.7901x`，未通过单算子性能门禁；没有运行公开集挑样本，也没有接入 wrapper。
  后续只考虑 packed gate/up GEMM epilogue fusion。
- PPU 释放前已完成三份本地快照：实验目录 538 KiB、原始 traces 58 MiB、MMBench
  188 MiB（压缩后），SHA-256 均与远端一致；Qwen3.5-2B 4.3 GiB 权重本地已有。
  `/mnt/workspace` 与 `/mnt/cpfs` 为同一 CPFS，远端快照亦已持久化。
- 当前没有 vLLM 或 `/opt/vllm`，Transformers 提示缺少 GDN/causal-conv fast path；
  eager 正确性可用，但不是最终性能路线。
- 完整证据见 [PPU 首次真实基线、Profile 与 GEMV](experiments/2026-08-26-ppu-baseline-and-gemv.md)、
  [PPU decode 融合算子迭代](experiments/2026-08-26-ppu-fused-decode-kernels.md)、
  [packed MLP](experiments/2026-08-27-ppu-packed-mlp.md)、
  [注册式 acBLAS Linear](experiments/2026-08-27-ppu-acblas-gemv.md) 和
  [GDN 输入投影打包](experiments/2026-08-27-ppu-packed-gdn-projections.md)、
  [residual-add + RMSNorm](experiments/2026-08-27-ppu-residual-rmsnorm.md)、
  [GDN gate-prep](experiments/2026-08-28-ppu-gdn-gate-prep.md)、
  [单入口 acBLAS packed-MLP](experiments/2026-08-28-ppu-acblas-packed-mlp.md)、
  [Attention Prep 单入口融合](experiments/2026-08-28-ppu-acblas-attention-prep.md)、
  [Graph Capture 能力与止损](experiments/2026-08-28-ppu-graph-capture.md)、
  [residual-RMSNorm scratch 负实验](experiments/2026-08-28-ppu-residual-rmsnorm-scratch.md)、
  [raw stream 查询优化](experiments/2026-08-28-ppu-raw-stream-query.md)、
  [acBLASLt Matmul 负实验](experiments/2026-08-28-ppu-acblaslt-matmul.md)、
  [SwiGLU 负实验](experiments/2026-08-27-ppu-swiglu-negative.md)、
  [资源释放与恢复](ppu-resource-release-handoff.md) 和 [未来路线](ppu-future-roadmap.md)。

## 尚未完成

- 获取主办方 PPU-vLLM/Qwen3.5/GDN fast path，并与 eager 做同口径对照。
- grouped-acBLAS + residual-RMSNorm + gate-prep 的组合公开完整集已通过；仍需主办方
  私有集最终验证，并确认提交环境变量/镜像是否由评测入口保留。
- SDK 公开 acBLASLt epilogue 已确认不含 SiLU。下一轮只在厂商提供自定义
  SwiGLU epilogue、grouped/batched GEMV 或图编译接口后继续 GEMM 融合，不再盲扫
  heuristic 编号。
- 量化/权重变换的正式允许范围确认。
- 初赛技术报告已形成可持续更新的初稿，源码候选包可一键生成；仍缺 PPU 实测章节、最终复现说明和按主办方格式定稿。

## 当前风险

RTX 4050 的 6GB 显存仍只适合作为历史单样本环境；更长输入、更高分辨率、编译缓存
或并发请求可能 OOM。PPU 显存充足，但当前 Transformers eager 的线性注意力和
因果卷积原始 fast path 缺失的问题已由仓库融合候选缓解；packed-GDN 激进候选的
CN20 平均吞吐约 98.430 token/s，但它新增 1/20 文本漂移；grouped-acBLAS GDN
  两轮保持 20/20 exact，但固定长重复的胜率仍不稳定；residual-RMSNorm 虽在两轮
  CN20 都保持 20/20 exact，仍未通过完整集；all-five 也已有 5/20
生成长度漂移，说明 reduction 数值顺序仍是精度风险。最终 gate-prep 组合已经通过
公开中文 4029 条严格一致性门禁；下一步获取官方 PPU-vLLM/FLA 对照并等待主办方
私有集，公开集结果不能替代私有评测成绩。
CPU offload、冷 RTC、profiler 插桩和稳态 PPU 结果不得混算。
