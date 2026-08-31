# 东南大学 AI+创新应用大赛：VLM PPU 推理优化

本仓库用于赛道二“面向 AI 芯片的 VLM 高效推理与优化”的两人协作开发。目标是在主办方指定的 Qwen3.5-2B 和 PPU 环境中，守住模型精度并降低 TTFT、提高解码吞吐，最终交付可复现代码和性能报告。

项目从初始 eager 基线、profile、五类融合算子、结构化 acBLAS 扩展到完整集门禁的
演进过程，以及所有正式性能证据和失败方向，统一整理在
[完整工作说明](docs/work-summary/README.md)。

## 当前状态

- 已导入 `dndx_participant-v1.1` 的公开评测入口和初始 wrapper。
- 本地开发环境锁定为 Python 3.12、PyTorch 2.13.0+cu130、Transformers 5.14.1。
- `5070ti` 分支已在 RTX 5070 Ti Laptop 12GB 上建立独立 Conda 环境：
  Python 3.12.13、CUDA 13.0、BF16 可用，46 项无模型测试通过；原 RTX 4050
  性能结果仍作为历史基线，不与新机器混用。
- 锁定 revision 的 Qwen3.5-2B 已在该环境通过完整性校验和纯 GPU 加载冒烟：
  617 个参数张量均位于 `cuda:0`，模型内存占用约 4.12 GiB。
- Qwen3.5-2B 指定 revision 已通过完整性校验，并在 RTX 4050 6GB 上以 BF16 纯 GPU 加载。
- 已完成中英文各 20 条公开集真实小样本基线；中文基线 Accuracy 85%。
- 严格首 token 口径下，O1 中文三次中位 TTFT 从 313.562 ms 降至 287.706 ms，吞吐从 21.328 升至 23.209 tokens/s，提升 8.25%/8.82%。
- 英文三次复测的 TTFT/吞吐中位提升为 10.22%/6.37%，中英文答案、token 数和 Accuracy 均无变化。
- 中英文比例分层 200 条 Accuracy 分别为 84.5%（169/200）和 82.5%（165/200），20 个类别全部覆盖。
- 中英文完整公开集 4029 条 Accuracy 分别为 83.94%（3382/4029）和 79.75%（3213/4029），题号集合、答案域和分块完整性审计全部通过。
- 已在隔离 PPU-ZW810E 上跑通 Qwen3.5-2B 真实多模态推理：617 个参数张量全部
  驻留 PPU，无 CPU/meta/disk offload；中文前 20 条稳态 Accuracy 85%，平均 TTFT
  约 118.5 ms，吞吐约 48.85 token/s。
- PPU profile 表明 eager 路径存在 37,293 次 kernel launch/16-token 和大量
  elementwise、reduce、copy、临时分配；当前首要瓶颈是缺少融合/fast path。
- 已实装并上机验证 recurrent GDN、causal-conv、RMSNorm、gated RMSNorm 以及
  full-attention q/k RMSNorm+RoPE 五类 HGGC decode 融合核。固定中文前 20 条
  all-five 两次为 93.918/94.889 token/s，相对 eager 49.737 提升
  88.83%/90.78%；Accuracy 均为 85%、20/20 答案一致。仍有 5/20 生成长度变化，
  因此全部融合保持显式 opt-in，待完整集验证。
- 在五类融合上新增 24 层 MLP gate/up packed projection（图布局优化，不计作第六个
  HGGC kernel）。固定中文前 20 条两次为 96.506/96.715 token/s，Accuracy 仍为
  85%；两轮答案、正确性和 token 数逐条完全一致，相对 eager 吞吐提高约
  94.04%/94.46%。它没有扩大 all-five 已有的 5/20 token 数漂移。
- 完成注册式 PyTorch/acBLAS decode Linear 调查。C ABI 隔离、随机 BF16 精度和模块级
  `1.08--1.17x` 均通过，但最终单 `.so` + 进程级 handle 版本在固定 128-token 八对
  AB/BA 中成对中位为 `0.9997x`、仅 4/8 获胜，因此不接入正式 wrapper；它作为
  “算子微基准收益不等于整模收益”的负实验保留。
- 新增 Qwen3.5 GDN 同输入投影打包实验：把每层 qkv/z/b/a 四次 decode Linear 合为
  一次。最终线程隔离版固定 128-token 四对全部获胜且全文一致，成对中位
  `1.0182x`；CN20 平均 `94.099→98.430 token/s`、成对中位 `1.0355x`、
  Accuracy 均为 85%，但仅
  19/20 全文一致（1 条多 1 token），所以当前仍是默认关闭的激进候选。
- 为保留四路 GEMV 的原始 BF16 数值路径，新增结构专用 grouped acBLAS 后端：只把
  四次 Python/ATen/pybind 入口合为一次，设备仍按 qkv/z/b/a 原顺序计算。CN20
  两轮平均吞吐分别为 `96.409→98.028` 和 `95.634→99.601 token/s`，成对中位
  `1.0187x/1.0391x`，两轮 Accuracy 均为 85% 且 20/20 全文一致。固定长六对仅
  3/6 获胜，因此它是默认关闭的精度优先候选，仍需完整集和更多重复门禁。
- 新增 Qwen3.5 decoder 的 48-edge residual-add + RMSNorm 跨层融合。它保持 BF16
  residual 舍入点和 FP32 norm reduction，并在 16-token profile 中精确消除 720 次
  `[1,1,2048]` add 与 720 次 kernel launch。CN20 两轮由 `100.156→101.616`、
  `98.576→101.507 token/s`，配对中位均约 `1.021x`、各 14/20 获胜，Accuracy
  均为 85% 且两轮 20/20 全文一致。它同样默认关闭，待完整集门禁。
- 新增 18 层 GDN gate-prep 融合：加载时缓存 FP32 `exp(A_log)`，一个 HGGC kernel
  合并 Sigmoid、两个 cast、add、Softplus、mul/neg，并用 thread-local scratch 复用
  `g/beta`。最终 128-token 六对全部获胜、全文一致，配对中位 `1.0839x`；CN20
  两轮配对中位 `1.0811x/1.0863x`、19/20 和 17/20 获胜，均 20/20 全文一致、
  Accuracy 85%。16-token profile 的 launch `16253→14363`，Self CPU/PPU
  `366.100/119.365→324.074/112.982 ms`。最终中文完整公开集两路 Accuracy 同为
  3374/4029，4029/4029 完整文本、答案和 token 数一致，成对吞吐中位 `1.0862x`。
  候选仍由显式开关启用，但已是当前推荐提交配置的一部分。
- 完成 acBLASLt 四个真实 decode 形状、每形状 32 个 heuristic 的调查。唯一方阵
  候选配合 scratch 模块级达到 `1.2797x`，但完整模型固定 128-token 八对仅
  `0.9898x`、3/8 获胜，故作为负实验止损，未进入正式 wrapper。
- GEMM 继续迭代后改为完整 MLP 提交边界：一次 extension 入口依次执行 packed
  gate/up GEMV、bit-exact HGGC SwiGLU 和 down GEMV，并复用持久 scratch。固定
  128-token 八对 8/8 获胜、成对中位 `1.1336x`；CN20 两轮均 20/20 全文一致、
  20/20 获胜、Accuracy 85%，成对中位 `1.1212x/1.1122x`。中文完整公开集
  4029/4029 文本、答案和 token 数一致，两路 Accuracy 均为 3374/4029；平均吞吐
  `109.993→122.445 token/s`，成对中位 `1.1125x`，3939/4029 获胜。最终重编译、
  正式 wrapper 和 `hggc-memcheck` 均通过。英文完整公开集同样 4029/4029 exact，
  两路 Accuracy 均为 3214/4029，平均吞吐 `107.276→118.964 token/s`，成对中位
  `1.1093x`，3806/4029 获胜。
- 完成全注意力层 Attention Prep 单入口负实验：聚合 Q/K/V 三个原形状 GEMV 与既有
  Q/K RMSNorm+RoPE。真实模块 Q/K/V/gate、prefill 回退、scratch 复用和异流保护均
  通过，memcheck 0 errors，模块边界 `4.1006x`；但固定 128-token 56 对合并中位仅
  `1.0047x`，CN20 两轮中位为 `1.0158x/0.9852x`，第二轮性能门禁失败。因此停止
  profile/完整集，默认关闭并作为“局部快不等于整模快”的负实验保留。
- 验证 HGGC Graph Capture 在当前 PyTorch PPU 运行时真实可用：16 段 elementwise
  固定子图为 `1.8303x` 且 exact；但套在已聚合的单入口 packed-MLP 上，稳定地址仅
  `1.0203x`，加入真实 input copy 后为 `0.9316x`。低于 3% 晋级余量，故不改造
  residual scratch、不进入整模门禁，等待官方固定 KV/page 或更大 decode 图边界。
- residual-RMSNorm 持久输出 scratch 在模块边界相对现有融合为 `1.3373x`，exact 且
  memcheck 0 errors；但当前完整栈固定 128-token 八对仅 `0.9862x`、2/8 获胜，
  已止损且未接正式 wrapper。
- 新增受版本能力检查保护的 PPU raw-stream 查询候选，减少约 127 次/token 的 Python
  `Stream` 对象查询。模块路径 `1.2944x`；固定 128-token 两轮中位
  `1.1055x/1.0961x`，CN20 两轮中位 `1.1026x/1.0855x`，四轮输出均完全一致、
  Accuracy 不变。中文完整集 4029/4029 exact、Accuracy 均为 3374/4029，平均吞吐
  `120.383→131.107 token/s`、成对中位 `1.0906x`，3817/4029 获胜；memcheck、
  profile 和正式入口 smoke 均通过。英文完整集同样 4029/4029 exact、Accuracy
  均为 3214/4029，平均吞吐 `118.577→129.398 token/s`、成对中位 `1.0901x`，
  3704/4029 获胜。
- 在同一 PPU 实例上用四个独立进程按 `eager A→当前栈 A→当前栈 B→eager B` 直接
  复测总加速：CN20 两次吞吐中位 `49.445→132.4265 token/s`，即 `2.6783x`、提升
  `167.83%`；四次 Accuracy 均为 85%，20/20 解析答案与正确性一致，TTFT 基本持平。
  公开结果不保存全文哈希，因此该轮不宣称全文 bit-exact，也不把 CN20 外推到完整集。
- 最终日 16-token profile 再次记录到 14,003 次 `cudaLaunchKernel`、5,705 次
  `cudaGetDeviceProperties_v2`、3,259 次 `cudaFree`；15 个 decode step 中每 token
  仍有 120 次小 BF16 GEMV。由此冻结短 elementwise 小修，转而全量验证每 token 少
  54 次 GDN GEMV 的 single-GEMV 性能档。
- 将 grouped-GDN 已连续存放的 qkv/z/b/a 四路权重从每层 4 次 GEMV 合为 1 次后，
  fixed-128 两轮均为正；最终中文 4029 条两路 Accuracy 均为 3374/4029，答案解析
  结果 4029/4029 一致，平均吞吐 `129.386→132.457 token/s`，成对中位/均值
  `1.0238x/1.0253x`、2932/4029 获胜。完整文本为 3873/4029 一致，因此该路径通过
  `SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=1` 显式启用且默认关闭，只归入允许答案级
  精度预算的性能档；精度优先档仍保留四次 GEMV。该性能档相对原始 eager 的独立进程
  CN20 ABBA 直接总加速为 `2.7689x`（`+176.89%`），四次 Accuracy 均为 85%。但
  英文 4029 中候选正确数 `3214→3213`、答案 4028/4029 一致，因此双语门禁失败，
  最终降级为 `experimental_only`，不得作为比赛推荐性能档。
- 最终保守增量只合并相邻的 b/a 两个 `[16,2048]` 投影为一个 `[32,2048]`
  GEMV。中文/英文 4029 条都达到 4029/4029 全文、答案和 token 数一致，正确数分别
  保持 3374 和 3214，成对中位 `1.00696x/1.00697x`。16-token profile 中
  `cuLaunchKernel` 减少 270 次，精确对应 18 层和 15 个 decode step。该路径已进入
  显式 `performance` 档，`precision` 档继续保留原四次 GEMV。
- 最终性能栈相对原始 eager 又运行了两个独立 ABBA block：每臂四次吞吐中位
  `49.3415→132.1895 token/s`，即 `2.67907x`、提升 `167.91%`；8 次 Accuracy
  均为 85%，20/20 解析答案和正确性一致。一次候选低值未剔除，最终采用四次中位。
- 首 Token 收尾先后验证 KV/linear-state 预分配与视觉 Token 上限。KV 候选未通过
  双语 TTFT 门槛；视觉 192-token 档保持 400/400 答案一致但英文 TTFT 中位
  `0.99803x`，176/128 档开始出现答案漂移或吞吐回退。因此正式 profile 保持两类
  实验默认关闭，不以噪声级变化冒充提升。
- 资源释放前完成独立 SwiGLU HGGC 核负实验：四组线程均 bit-exact，但最好只有
  `0.7901x`，因此未接入正式 wrapper；后续改走 packed GEMM epilogue fusion。
- 已将 PPU 源码、编译产物、小型结果、pip/设备清单、全部原始 trace 和 MMBench
  分别归档到本地 ignored `artifacts/ppu-snapshot-20260827/`，三份 SHA-256 均核对
  一致；模型权重本地已有完整副本。镜像、CPFS 和恢复步骤见资源释放手册。
- `dummy` 后端只用于接口冒烟；不得将其结果视为真实模型部署或比赛成绩。

正式性能提升同时由固定 128-token 交错八对、CN20 两轮和中英文完整公开集 paired
门禁支撑；完整集还承担 Accuracy 与逐文本一致性验证。所有本地结果均不代表主办方
私有评测成绩。可公开的聚合结果见
[results/README.md](results/README.md)。

PPU 侧已完成 SDK、真实模型闭环、20 条稳态基线、算子级 profile 和 HGGC GEMV
迭代。详见 [PPU 首次真实实验](docs/experiments/2026-08-26-ppu-baseline-and-gemv.md)、
[PPU decode 融合实验](docs/experiments/2026-08-26-ppu-fused-decode-kernels.md)、
[PPU packed-MLP 实验](docs/experiments/2026-08-27-ppu-packed-mlp.md)、
[PPU acBLAS GEMV 调查](docs/experiments/2026-08-27-ppu-acblas-gemv.md)、
[PPU GDN 输入投影打包](docs/experiments/2026-08-27-ppu-packed-gdn-projections.md)、
[PPU residual-add + RMSNorm 跨层融合](docs/experiments/2026-08-27-ppu-residual-rmsnorm.md)、
[PPU GDN gate-prep 融合](docs/experiments/2026-08-28-ppu-gdn-gate-prep.md)、
[PPU acBLASLt Matmul 负实验](docs/experiments/2026-08-28-ppu-acblaslt-matmul.md)、
[PPU 单入口 acBLAS packed-MLP](docs/experiments/2026-08-28-ppu-acblas-packed-mlp.md)、
[PPU Attention Prep 单入口融合](docs/experiments/2026-08-28-ppu-acblas-attention-prep.md)、
[PPU Graph Capture 能力与止损实验](docs/experiments/2026-08-28-ppu-graph-capture.md)、
[PPU residual-RMSNorm scratch 负实验](docs/experiments/2026-08-28-ppu-residual-rmsnorm-scratch.md)、
[PPU raw stream 查询优化](docs/experiments/2026-08-28-ppu-raw-stream-query.md)、
[PPU acBLAS 运行时主要矛盾与 single-GEMV 候选](docs/experiments/2026-08-28-ppu-acblas-runtime-overhead.md)、
[PPU 首 Token cache 负实验](docs/experiments/2026-09-01-ppu-first-token-cache.md)、
[PPU 视觉 Token / TTFT 负实验](docs/experiments/2026-09-01-ppu-visual-token.md)、
[PPU SwiGLU 融合负实验](docs/experiments/2026-08-27-ppu-swiglu-negative.md)、
[PPU decode 融合算子与问题记录](ppu/custom_ops/README.md)、
[PPU 兼容性矩阵](docs/ppu-compatibility-matrix.md) 和 [需要向主办方确认的问题](docs/questions-for-organizer.md)。
资源即将释放时，按 [PPU 镜像与快照恢复手册](docs/ppu-resource-release-handoff.md)
操作；下一阶段按 [PPU 后续优化路线图](docs/ppu-future-roadmap.md) 推进。
Qwen3.5-2B 的 GDN、MLP、全注意力与视觉层尺寸见 [关键算子与 PPU kernel 目标](docs/qwen35-kernel-targets.md)。
三组关键解码尺寸的 [PPU BF16 GEMV 微基准](ppu/microbench/README.md) 已在隔离
PPU 实测：优化核比 reference 快 1.88--2.08 倍，但仍慢于 `torch.mv`，当前不接入模型。

## PPU 服务器首次验证

### 新实例一键恢复

工程源码的权威副本只有两份：本机 Git 工作区和 GitHub `5070ti` 分支。PPU
服务器只是可随时销毁的执行节点，不把服务器/CPFS 中的代码、虚拟环境或编译产物
当作恢复来源。新官方镜像启动后，在临时工作目录从 GitHub 拉取并部署：

```bash
cd /tmp
git clone --branch 5070ti --single-branch \
  https://github.com/mratgnothing/seu-vlm-ppu-optimization.git
cd seu-vlm-ppu-optimization

# 先只读核对官方镜像是否包含 PPU SDK 和定制 torch。
bash scripts/bootstrap_ppu_env.sh --check-only

# 创建仓库外的独立 venv、安装非 Torch 依赖、重编译三个扩展并做短 smoke。
bash scripts/bootstrap_ppu_env.sh
source scripts/activate_ppu_env.sh

# 推荐性能档：b/a-GEMV 已通过中英文各 4029/4029 全文一致门禁。
source scripts/activate_ppu_profile.sh performance
# 保守复测档：保持 GDN 四次原形状 GEMV。
# source scripts/activate_ppu_profile.sh precision
# 仅复现实验：中文中位 +2.38%，但英文完整集少 1 个正确答案。
source scripts/activate_ppu_profile.sh experimental-single
```

部署脚本默认使用 `/usr/local/bin/python3`、`/usr/local/PPU_SDK` 和
`~/.cache/seu-vlm-ppu/venv`。它通过 `--system-site-packages` 复用官方镜像的
PPU 定制 Torch，且会比较部署前后的 Torch 版本与加载路径；
[`requirements-ppu.txt`](requirements-ppu.txt) 故意不包含 `torch/torchvision`，
防止 CUDA/PyPI wheel 覆盖 PPU 运行时。无外网时可先准备 wheel 目录，再传入
`--wheelhouse /path/to/wheels`。完整参数见：

```bash
bash scripts/bootstrap_ppu_env.sh --help
```

模型权重、公开数据和原始大结果仍不进 Git；使用时通过路径参数挂载/读取，本机保留
独立副本。服务器上产生的结果必须在实例释放前下载到本机，筛除敏感与巨型文件后，
仅将可公开的小型结果和文档提交 GitHub。

获得主办方允许上传代码的隔离 PPU 节点后，先运行只读预检：

```bash
chmod +x scripts/run_ppu_first_validation.sh
scripts/run_ppu_first_validation.sh \
  --vllm-source /opt/vllm \
  --model-path /path/to/Qwen3.5-2B
```

输出默认保存在忽略目录 `artifacts/ppu-first-validation/`，包括环境清单、
`runtime.json`、`runtime-summary.md`、PPU-SMI 快照、Qwen3.5 结构指纹、PPU-vLLM
源码能力和两条部署路线的阻塞项。默认不会执行 PPU 张量、编译 kernel、加载完整
模型或运行样本。

在主办方批准的个性化隔离节点，按风险逐级显式加入：

- `--run-device-smoke`：BF16 `32x32` PPU PyTorch 小张量；
- `--verify-model-hash`：读取约 4.6 GB 权重并核对 SHA-256；
- `--run-microbench`：编译并运行三组 HGGC BF16 GEMV；
- `--run-model-load`：完整加载并拒绝 CPU/meta/disk offload；
- `--run-single-sample --dataset-path PATH`：强制真实 Transformers 后端运行一条公开样本。

例如完成设备、kernel 和模型加载验证：

```bash
scripts/run_ppu_first_validation.sh \
  --model-path /path/to/Qwen3.5-2B \
  --run-device-smoke \
  --run-microbench \
  --run-model-load \
  --device 0 \
  --warmup 10 \
  --iterations 100
```

脚本会先做一次 `warmup=0, iterations=1` 的单尺寸冒烟，再运行三组 Qwen3.5
BF16 GEMV。完整分级步骤、结构指纹和故障定位见
[PPU + Qwen3.5 首次上机验证手册](docs/ppu-first-validation.md)。共享节点仍遵守
“不上传、不运行本目录”的既有边界。

两位成员的职责、四周里程碑和当前交接点见 [两人协作与一个月推进计划](docs/team-plan.md)。
已验证内容已同步整理到 [初赛技术报告初稿](docs/preliminary-technical-report.md)，未在 PPU 实测的部分均保留显式边界。

## 仓库结构

```text
.
├─ benchmark_public.py       # 主办方 v1.1 公开评测入口
├─ evaluation_wrapper.py     # 主要优化入口
├─ requirements.txt          # 主办方基础依赖
├─ requirements-ppu.txt      # PPU 用户态依赖；故意不含官方定制 Torch
├─ README_ORGANIZER.md       # 主办方 v1.1 说明
├─ configs/                  # 可复现实验配置
├─ data/                     # 本地数据说明，数据文件不入库
├─ docs/                     # 规则、协作和实验记录
├─ models/                   # 本地模型说明，权重不入库
├─ ppu/custom_ops/           # PPU decode 融合核、ctypes 接入与 smoke
├─ ppu/microbench/           # HGGC GEMV/GDN 微基准
├─ results/                  # 可公开的小型汇总，原始结果不入库
├─ scripts/                  # 运行脚本
└─ tests/                    # 无模型依赖的接口测试
```

## 第一次运行

当前 Windows 5070 Ti 开发环境位于仓库外：

```text
G:\seu-AI\.conda-envs\seu-vlm-5070ti
```

可直接使用环境内 Python，避免修改 Conda `base`：

```powershell
$python = "G:\seu-AI\.conda-envs\seu-vlm-5070ti\python.exe"
& $python .\scripts\check_environment.py
& $python -m unittest discover -s tests -v
```

1. 建立本地 CUDA 环境：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\bootstrap_local.ps1
   ```

2. 下载并校验已锁定 revision 的官方模型：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\download_model.ps1
   ```

3. 把公开数据放在仓库外，或放入已被忽略的 `data/`。
4. 先做 dummy 冒烟：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\run_benchmark.ps1 `
     -DatasetPath "D:\path\to\mmbench_dev_cn.tsv" `
     -Backend dummy `
     -NumSamples 2
   ```

5. 再做真实模型小样本基线：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\run_benchmark.ps1 `
     -DatasetPath "D:\path\to\mmbench_dev_cn.tsv" `
     -ModelPath "E:\models\Qwen3.5-2B" `
     -Backend transformers `
     -NumSamples 20
   ```

真实实验必须检查结果中的 `backend` 字段，禁止把自动回退的 dummy 结果当成真实基线。

运行真实 benchmark 前，先做不生成文本的模型加载门禁：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\smoke_model_load.ps1
```

## 协作规则

- `main` 始终保持可复现。
- 每项实验使用独立分支：`baseline/...`、`opt/...`、`ppu/...`、`docs/...`。
- 通过 Pull Request 合并；PR 必须附 Accuracy、TTFT、Throughput 的前后对比。
- 不在一个提交中混合无关优化。
- 实验配置、环境和失败结果同样需要记录。

## 安全边界

以下内容不得提交，即使仓库是私有的：

- `key.pem`、SSH 私钥、访问令牌和 `.env`。
- Qwen3.5-2B 权重及其他大模型文件。
- 群聊原始记录、测试环境截图、培训 PDF 和原始压缩包。
- 未确认允许再分发的评测数据。
- 含私有测试答案或反作弊敏感内容的产物。

详细规则见 [docs/rules-and-boundaries.md](docs/rules-and-boundaries.md)。

## 提交候选包

在主办方最终提交格式公布前，可以生成经过白名单和 SHA-256 复核的源码候选包：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_submission.ps1
```

候选包默认写入 Git 忽略的 `artifacts/submission-source.zip`，不会包含模型、数据、
原始结果、密钥或本地配置。详细边界见 [submission/README.md](submission/README.md)。

## 完整公开集精度

中英文 4029 条完整公开集使用可断点续跑的分块入口：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\run_full_accuracy.ps1 `
  -Language cn
```

默认每 200 条保存一个原始结果；再次执行会跳过结构和 profile 均匹配的已完成分块。
全部分块完成后才生成严格聚合结果，避免一次长进程中断后从头运行。
