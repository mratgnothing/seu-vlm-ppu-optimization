# 赛道二项目理解、学习路径与资料索引

> 本目录用于在迁移到 Linux 或更换开发环境后，保留当前项目背景、技术判断、
> 协作约定、学习路径和外部资料入口。
>
> 更新时间：2026-08-12
> 当前本地分支：`5070ti`  
> 仓库：`seu-vlm-ppu-optimization`

## 1. 比赛与项目是什么

本项目参加东南大学 AI+ 创新应用大赛赛道二“端侧 AI 推理优化挑战”。
比赛指定 Qwen3.5-2B 视觉语言模型和主办方 PPU 环境，要求在不破坏评测公平性
和模型精度的前提下：

- 提高图片单选题 Accuracy；
- 降低首个有效输出 token 的延迟 TTFT；
- 提高 decode 阶段 Throughput；
- 提供稳定、可复现、能够在主办方环境运行的工程实现；
- 保留算子、内存、调度、量化和硬件适配等系统级证据。

仓库当前记录的初赛指标权重是 Accuracy 40%、TTFT 30%、Throughput 30%，
最终以主办方最新书面规则为准。

比赛不是换模型或查表答题。当前明确禁止的行为包括：

- 更换指定模型；
- 使用未经认可的量化权重包、蒸馏模型或权重变换；
- 修改正式数据、评分逻辑或反作弊逻辑；
- 按题号、图片 hash、题干或参考答案查表；
- 跳过样本、伪造 token 数或计时；
- 调用外部模型、搜索或远程推理服务；
- 将 Accuracy 和性能评测拆成两套不同策略。

如果主办方以后开放微调、量化或权重变换，应保存书面规则并以其允许范围为准。

## 2. 当前仓库的真实状态

### 已完成

- 导入主办方 v1.1 公开评测入口；
- 锁定 `Qwen/Qwen3.5-2B` 指定 revision；
- 在本地 RTX 4050 6 GB 上完成 BF16 Transformers 真实推理；
- 中文完整公开集 Accuracy：83.94%（3382/4029）；
- 英文完整公开集 Accuracy：79.75%（3213/4029）；
- O1 使用 `torch.inference_mode()` 替换 O0 的 `torch.no_grad()`；
- 中文 O1 相对 O0：TTFT 提升 8.25%，Throughput 提升 8.82%；
- 英文 O1 相对 O0：TTFT 提升 10.22%，Throughput 提升 6.37%；
- CUDA Profile 显示 GEMV/GEMM 占 self CUDA time 的 86.18%；
- PPU SDK、驱动、HGGC 和官方 vectorAdd 基础链路已验证；
- 已准备三组 Qwen3.5-2B BF16 GEMV 参考微基准；
- 已具备分块 Accuracy、结果审计、源码白名单打包和技术报告骨架。

### 尚未完成

- Qwen3.5-2B 在目标 PPU 上的真实加载与推理；
- PPU Accuracy、TTFT 和 Throughput 基线；
- PPU 模型级 Profile；
- GDN、causal-conv1d 和矩阵算子的 PPU fast path；
- PPU 上经过验证的量化、融合或调度优化；
- 最终 PPU 优化闭环和正式复现报告。

### 当前关键阻塞

共享 PPU 节点的定制 vLLM 停留在 0.8.5，当前调查发现：

- 未注册 `Qwen3_5ForConditionalGeneration`；
- 缺少 Qwen3.5 Gated Delta Network（GDN）实现；
- 虽有 PPU FlashAttention、矩阵、causal-conv1d 和量化路径，但尚未组成
  Qwen3.5-2B 的完整可运行链路；
- 共享节点缺少可直接使用的 PyTorch、Transformers 和 vLLM Python 环境。

只有在目标 PPU 上运行真实模型并产生可复现的 Accuracy、TTFT 和 Throughput，
才能写“PPU 部署完成”。vectorAdd 或微基准通过不等于模型部署完成。

## 3. PPU 是什么

PPU 是本次比赛指定的 AI 加速硬件。它承担的角色类似 NVIDIA GPU，但使用自己的
驱动、SDK、编译器、运行时、算子实现和硬件指令。项目接触到的设备为
PPU-ZW810E，每张显存约 97.9 GB。

当前 PPU SDK 提供 HGGC 等 CUDA 风格接口，因此 CUDA 的线程层次、kernel、
显存、同步、stream、算子融合和性能分析知识可以迁移过来，但不能假定：

- CUDA 程序无需修改就能在 PPU 编译；
- NVIDIA 上最快的实现也是 PPU 上最快的；
- PyTorch/vLLM 的每个 CUDA 算子都有 PPU 实现；
- 量化格式一定有对应 PPU kernel；
- 算子能够运行就代表模型完整支持。

软件栈可以理解为：

```text
Qwen3.5-2B
  -> Transformers / vLLM
  -> PyTorch 与模型算子
  -> PPU 定制后端、运行时和编译器
  -> PPU kernel
  -> PPU-ZW810E
```

## 4. 开发环境和工作方式

### 当前本机验证状态

- RTX 5070 Ti Laptop GPU，约 11.94 GiB 显存；
- NVIDIA 驱动 591.97；
- 独立 Conda 环境：`G:\seu-AI\.conda-envs\seu-vlm-5070ti`；
- Python 3.12.13、PyTorch 2.13.0+cu130、Transformers 5.14.1；
- CUDA runtime 13.0、BF16 可用；
- 31 项无模型测试通过；
- 锁定 revision 的 Qwen3.5-2B 已完成完整性校验并在 `cuda:0` 真实加载，模型报告
  内存占用约 4.12 GiB；当前缺少可选 fast-path 依赖，性能测试前需单独处理；
- 未修改 Conda `base`；
- 旧 RTX 4050 性能数据继续作为历史基线，尚未在 5070 Ti 重跑。

### 推荐环境

- Linux：主要开发、vLLM 移植、自定义算子编译和 PPU 验证；
- Windows：Git、文档、参考实现、结果分析和少量 RTX 4050 基线；
- WSL2：Linux 脚本、Python 和无 PPU 的框架开发；
- 主办方 Linux PPU 服务器：所有 PPU 正确性和性能结论。

如果只能选择一个主要环境，应选 Linux。迁移后建议在 Linux 文件系统中重新克隆，
不要长期从 WSL 直接编译 `/mnt/g/...` 下的工程。

### 推荐闭环

```text
本地/WSL 编写参考实现与最小改动
  -> 本地单元测试和 GPU 算法验证
  -> 上传允许范围内的源码
  -> PPU 编译、正确性、Profile 和计时
  -> 下载日志与结果
  -> 本地分析、记录、提交
  -> 下一轮单变量实验
```

不是把大量代码一次性写完再上传，而是“一小步修改、一小步硬件验证”。

### Git 约定

- 当前工作分支为 `5070ti`；
- 修改暂不合并到 `main`；
- 未经明确要求不自动推送；
- 每项优化尽量单变量；
- 实验结果写入 `docs/experiments/`；
- 不提交模型、数据、密钥、SSH 私钥、逐样本结果和浏览器会话。

## 5. 当前三条 PPU 部署路线

### 路线 A：主办方新版 PPU-vLLM

首选路线。若主办方提供支持 Qwen3.5/GDN 的新版镜像，优先复用其调度、
KV Cache、FlashAttention、causal-conv1d、量化和矩阵路径。

### 路线 B：PPU PyTorch + Transformers eager

保底路线。若有 PPU 定制 PyTorch 和新版 Transformers，先建立能加载、能看图、
能回答、能计时的 BF16 功能与精度基线，再根据真实 Profile 优化。

### 路线 C：移植新版 vLLM Qwen3.5

若只有旧版 PPU-vLLM，则移植模型注册、混合注意力、GDN recurrent state、
causal-conv1d、缓存和调度逻辑，并接入现有 PPU kernel。工作量和风险最高。

## 6. 算子开发到底包含什么

写算子不只是写一个 `__global__` 函数。完整工作包括：

1. **数学语义**：明确公式、shape、dtype、layout、bias、转置和累加精度。
2. **参考实现**：使用 PyTorch/NumPy/CPU 实现清晰可信的 ground truth。
3. **真实输入画像**：从模型 Profile 收集高频 shape、调用次数和上下游算子。
4. **并行划分**：设计 grid、block、thread、tile、reduction 和每线程工作量。
5. **内存访问**：处理合并访问、对齐、向量化加载、shared memory、寄存器和布局。
6. **数值精度**：处理 BF16/FP16/FP32/INT8/INT4、scale、zero point 和误差容限。
7. **编译运行**：完成 SDK、编译参数、动态库、stream、event、错误检查和同步。
8. **算子正确性**：覆盖随机值、边界 shape、零值、极值、非整除尺寸和重复执行。
9. **算子性能**：区分 warmup、kernel 时间、端到端时间、P50/P95、GFLOP/s 和带宽。
10. **框架集成**：注册 PyTorch/vLLM custom op，处理 dispatch、fallback 和权重布局。
11. **模型验证**：依次验证算子、单层、单样本、20 条 Accuracy、分层集和完整集。

### 推荐的算子学习顺序

```text
Vector Add
  -> Reduction
  -> BF16 GEMV
  -> Fused elementwise / RMSNorm
  -> causal-conv1d update
  -> GDN recurrent update
  -> GDN + conv + gate 融合
```

### 本项目优先方向

1. **BF16 GEMV**：最适合入门，现有仓库已有参考微基准。
2. **causal-conv1d update**：研究 width=4 状态更新、环形缓存与融合。
3. **GDN recurrent update**：涉及状态、门控、卷积和矩阵运算，难度最高。
4. **Prefill/视觉路径**：主要影响 TTFT。
5. **内存与调度**：KV/GDN state、显存分配、数据搬运和 kernel 启动开销。

## 7. 本地能验证什么，什么必须上云

### 本地可以验证

- Python 接口和评测契约；
- A/B/C/D 解析和统计公式；
- 算子数学参考实现；
- shape、padding、layout、分块和边界条件；
- BF16/INT8/INT4 量化与反量化逻辑；
- RTX GPU 上的算法原型；
- 单样本和公开集 Accuracy；
- 打包、安全扫描和文档。

### 必须在 PPU 验证

- HGGC kernel 是否编译；
- PPU custom op 是否加载；
- 是否真正落到 PPU 而非 CPU fallback；
- PPU 数值行为；
- PPU kernel 延迟、带宽和指令利用；
- PPU TTFT、Throughput、显存和稳定性；
- PPU INT8/INT4 是否有真实 fast path；
- 最终比赛性能提升。

### 验证分级

```text
L1 每次修改：
  单元测试 + git diff --check + 小 shape

L2 上传前：
  参考实现对比 + 边界测试 + 单样本 + 20 条 Accuracy

L3 PPU 实验：
  编译 + 数值误差 + warmup + 重复计时 + Profile

L4 形成结论：
  完整模型 + 同口径三次以上 + Accuracy 护栏 + 中位数
```

## 8. 如果开放微调

当前不能默认微调合法。只有主办方书面开放后才进入以下路线：

1. 固定并保存官方 BF16 基线；
2. 只用主办方允许的数据，严格划分训练、验证和保留集；
3. 优先 LoRA，而不是全参数微调；
4. 显存不足再尝试 QLoRA；
5. 优先训练输出行为和薄弱能力，避免记忆公开集；
6. 同时监测中英文 Accuracy、类别 Accuracy、答案唯一性和输出长度；
7. 检查微调是否导致 TTFT、token 数和 Throughput 退化；
8. 确认最终允许提交 adapter，还是必须合并权重；
9. 保存数据版本、随机种子、超参数、checkpoint 和评测证据。

本题中微调的合理目标可能包括：

- 更稳定地输出唯一 A/B/C/D；
- 更早输出最终答案，减少冗长推理；
- 改善逻辑推理和关系理解弱项；
- 保持中英文和不同视觉题型的泛化。

## 9. 如果开放量化

推荐从低风险到高风险推进：

```text
BF16 baseline
  -> W8A16
  -> W8A8 / SmoothQuant
  -> KV Cache 或 recurrent state 量化
  -> W4A16 / AWQ / GPTQ
  -> 混合精度与 PPU 专用布局
```

量化需要同时考虑：

- 权重、激活、KV Cache/GDN state 的量化对象；
- per-tensor、per-channel、per-group 粒度；
- symmetric/asymmetric；
- scale 和 zero point；
- 校准数据是否合法、有无泄漏；
- 敏感层是否保留 BF16；
- PPU 是否有对应 kernel、packing 和矩阵指令；
- 是否因反量化而比 BF16 更慢；
- Accuracy、TTFT、Throughput、显存和加载时间。

建议保留 Embedding、输出头、Norm 和敏感视觉层的较高精度，先对大线性层做
INT8，再根据逐层敏感性尝试 INT4。量化后必须跑模型级 Accuracy，单个算子误差小
不代表最终答案不变。

## 10. 启动路径

### 本地

```powershell
cd G:\seu-AI\seu-vlm-ppu-optimization
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

此前系统 Python 测试为 17 项通过、2 项因缺少 Pillow 导入失败；安装项目依赖后
应重新运行完整测试。

### Linux 迁移

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/mratgnothing/seu-vlm-ppu-optimization.git
cd seu-vlm-ppu-optimization
git fetch
git switch 5070ti
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt
python -m unittest discover -s tests -v
```

如果 `5070ti` 仍只有本地分支，需要先在 Windows 端明确决定是否推送，或通过源码
包迁移。不要把 `.venv` 从 Windows 复制到 Linux。

### PPU 服务器第一步

先运行无第三方依赖的环境预检，而不是立即上传模型：

```bash
python3 scripts/check_ppu_runtime.py \
  --vllm-source /opt/vllm \
  --output ppu-runtime-report.json
```

仓库也提供保留环境、日志和 Git 版本证据的一键入口：

```bash
chmod +x scripts/run_ppu_first_validation.sh
scripts/run_ppu_first_validation.sh \
  --vllm-source /opt/vllm \
  --model-path /path/to/Qwen3.5-2B
```

默认只读预检。只有在主办方批准的个性化隔离节点才加入
`--run-microbench`；脚本会先做单次冒烟，再运行三组 BF16 GEMV，并把证据写入
`artifacts/ppu-first-validation/`。

然后按以下顺序验收：

```text
环境与权限
  -> vectorAdd
  -> 仓库 BF16 GEMV 三个 shape
  -> Qwen3.5 配置加载
  -> 单张图片真实问答
  -> 20 条 Accuracy/性能基线
  -> PPU Profile
  -> 选择第一个真实热点
```

## 11. 推荐学习顺序

中文第一阶段讲义见
[第一阶段：模型推理与评测入门](chinese/stage-1-model-inference.md)，中文离线资料总入口见
[中文资料离线入口](chinese/README.md)。

### 阶段 1：模型推理

- Transformer prefill/decode；
- KV Cache；
- GEMM 与 GEMV；
- Attention、RMSNorm、门控 MLP；
- Qwen3.5 混合注意力、GDN 和 causal-conv1d；
- TTFT、Throughput 和显存指标。

### 阶段 2：GPU/CUDA 基础

- SIMT、warp、grid、block、thread；
- global/shared/register memory；
- synchronization 和 reduction；
- coalesced access 和 bank conflict；
- occupancy、Roofline、算力受限与带宽受限；
- event 计时和 Profiler。

### 阶段 3：算子

- Vector Add；
- Reduction；
- GEMV/GEMM；
- 融合 Softmax、LayerNorm/RMSNorm；
- PyTorch custom operator；
- Triton 算子；
- 正确性、误差和性能基准。

### 阶段 4：PPU

- PPU SDK/HGGC；
- 设备、stream、event 和显存；
- 编译和动态库；
- PPU Profiler；
- PPU 矩阵和低精度指令；
- PyTorch/vLLM custom op 集成。

### 阶段 5：微调与量化

- SFT、LoRA、QLoRA；
- PTQ 与 QAT；
- W8A16、W8A8、W4A16；
- SmoothQuant、AWQ、GPTQ；
- calibration、packing 和硬件 kernel；
- Accuracy 与性能联合验证。

## 12. 权威网页入口

### CUDA、GPU 编程与性能

- [NVIDIA CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- [CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/)
- [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/)

中文入口：

- [NVIDIA 中国 CUDA 文档中心](https://docs.nvidia.cn/cuda/doc/index.html)：官方中文导航，底层 API 名称和核心手册正文仍应以英文最新版为准；
- [CUDA 中文手册社区翻译](https://cuda-doc.readthedocs.io/zh-cn/latest/CUDA-C-Programming-Guide/index.html)：适合第一次理解 SIMT、线程层次和内存模型；
- [新版 CUDA Programming Guide 中文翻译](https://bearneck.github.io/cuda-programming-guide-zh/)：社区维护，阅读时与 NVIDIA 英文原文交叉核对。

推荐阅读顺序：Programming Guide 的 Programming Model、Intro to CUDA C++ 和
SIMT Kernels，然后读 Best Practices 的 profiling、timing、memory optimization、
execution configuration 和 numerical accuracy。

### 算子开发

- [PyTorch Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)
- [PyTorch Custom Operators Manual](https://docs.pytorch.org/docs/stable/library.html)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton Vector Addition](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)
- [Triton Fused Softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
- [Triton Layer Normalization](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [Triton Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
- [vLLM Documentation](https://docs.vllm.ai/)

### 平头哥真武 PPU/HGGC

- [PPU SDK v2.1 Release Note](https://help.aliyun.com/zh/document_detail/3030339.html)：官方说明源码级兼容 CUDA C/C++，支持到 CUDA 13.0 API 和 Triton 3.5.x，但 PPU 与 GPU 二进制不兼容；
- [PPU SDK 快速入门](https://help.aliyun.com/zh/document_detail/3030340.html)：设备、驱动、SDK 和基础编译链检查；
- [Asight Systems 快速入门](https://help.aliyun.com/zh/document_detail/2879847.html)：PPU 系统级性能分析；
- [PPU 活动跟踪](https://help.aliyun.com/zh/document_detail/2996757.html)：HGGC、ACDNN、ACBLAS、PCCL 与 kernel/memcpy/memset 跟踪；
- [hgobjdump v2.1](https://help.aliyun.com/zh/document_detail/3031866.html)：查看 PPU device binary、汇编和 `.hggc_fatbin`；
- [hgfatbinary v2.1](https://help.aliyun.com/zh/document_detail/3031873.html)：多 PPU 架构 device code 打包格式。

目前公开帮助中心能确认 PPU SDK 以 Clang/LLVM 为基础、采用 host/device 混合
C/C++ 扩展并在源码级兼容 CUDA C/C++。完整 HGGC runtime/driver API、设备内建函数、
PPU tensor-core/PTX 指令和支持矩阵仍应以服务器上对应版本的
`/usr/local/PPU_SDK/include`、SDK samples、编译器 `--help` 以及主办方随镜像提供的
手册为准，不能仅按 CUDA 文档推断硬件语义。

### 微调

- [Hugging Face PEFT](https://huggingface.co/docs/peft/)
- [PEFT LoRA Concept Guide](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [PEFT Quantization Guide](https://huggingface.co/docs/peft/developer_guides/quantization)
- [TRL SFTTrainer](https://huggingface.co/docs/trl/sft_trainer)
- [Transformers Training](https://huggingface.co/docs/transformers/training)

### 量化

- [Transformers Quantization Overview](https://huggingface.co/docs/transformers/quantization/overview)
- [Transformers Quantization Concepts](https://huggingface.co/docs/transformers/quantization/concept_guide)
- [Selecting a Quantization Method](https://huggingface.co/docs/transformers/quantization/selecting)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [Transformers AWQ](https://huggingface.co/docs/transformers/quantization/awq)
- [Transformers GPTQ](https://huggingface.co/docs/transformers/quantization/gptq)
- [vLLM Quantization](https://docs.vllm.ai/en/latest/features/quantization/)
- [SmoothQuant Repository](https://github.com/mit-han-lab/smoothquant)
- [AWQ Repository](https://github.com/mit-han-lab/llm-awq)

## 13. 离线 PDF

PDF 保存在 [`pdf/`](pdf/)：

1. [CUDA 中文指南首页与核心概念速查](pdf/cuda-programming-guide-zh.pdf)：4 页网页离线打印；完整中文正文见 [`chinese/cuda-programming-guide-zh/`](chinese/cuda-programming-guide-zh/README.md)；
2. [PPU SDK 快速入门（官方中文）](pdf/ppu-sdk-quick-start-zh.pdf)：15 页 A4 网页离线打印，包含历史版本示例，实际环境以比赛服务器为准；
3. [CUDA Programming Guide](pdf/cuda-programming-guide.pdf)
4. [CUDA C++ Best Practices Guide](pdf/cuda-c-best-practices-guide.pdf)
5. [LoRA: Low-Rank Adaptation of Large Language Models](pdf/lora.pdf)
6. [QLoRA: Efficient Finetuning of Quantized LLMs](pdf/qlora.pdf)
7. [AWQ: Activation-aware Weight Quantization](pdf/awq.pdf)
8. [SmoothQuant: Accurate and Efficient Post-Training Quantization](pdf/smoothquant.pdf)
9. [GPTQ: Accurate Post-Training Quantization for Generative Transformers](pdf/gptq.pdf)

PDF 论文用于理解方法，不等于比赛允许直接使用这些权重格式或实现。最终路线必须同时
满足主办方规则、Qwen3.5/VLM 兼容性和 PPU kernel 支持。

## 14. 仓库内重要入口

- `AGENTS.md`：真实性和协作边界；
- `PROJECT_CONTEXT.md`：项目目标与真实状态；
- `docs/README.md`：统一进度入口；
- `docs/current-status.md`：已验证环境和结果；
- `docs/rules-and-boundaries.md`：比赛规则；
- `docs/ppu-compatibility-matrix.md`：PPU 软件栈缺口；
- `docs/questions-for-organizer.md`：需要主办方确认的问题；
- `docs/qwen35-kernel-targets.md`：Qwen3.5 关键算子尺寸；
- `docs/preliminary-technical-report.md`：技术报告初稿；
- `evaluation_wrapper.py`：主要优化入口；
- `benchmark_public.py`：公开评测入口；
- `ppu/microbench/qwen35_bf16_gemv.hg`：PPU GEMV 入门实现；
- `scripts/check_ppu_runtime.py`：PPU 环境预检。

## 15. 下一步行动清单

1. 在本地/WSL 安装依赖并恢复全部单元测试；
2. 迁移到 Linux 后复查路径、换行、脚本权限和依赖；
3. 向主办方确认隔离 PPU、镜像、上传方式和量化/微调规则；
4. 在 PPU 先运行环境预检；
5. 编译并验证现有 BF16 GEMV 微基准；
6. 建立 Qwen3.5-2B PPU 单样本 BF16 基线；
7. 取得真实 PPU Profile；
8. 根据热点选择 GEMV、causal-conv1d 或 GDN 中的一个单变量实验；
9. 每轮保存代码提交、配置、日志、原始聚合结果和 Accuracy 护栏；
10. 有效修改暂留 `5070ti`，未经明确决定不合并到 `main`。
