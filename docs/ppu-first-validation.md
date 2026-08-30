# PPU + Qwen3.5-2B 首次上机验证手册

本手册对应 `scripts/run_ppu_first_validation.sh`。目标是在获得隔离 PPU 的 SSH
权限后，以最小风险确认硬件、SDK、HGGC、PPU PyTorch、Transformers、vLLM、
Qwen3.5 模型结构和真实多模态推理是否形成闭环。

## 真实性边界

- 默认运行只做环境、源码、模型文件和配置检查，不执行 kernel 或加载权重。
- `--run-device-smoke`、`--run-microbench`、`--run-model-load` 和
  `--run-single-sample` 必须显式开启。
- 只允许在主办方批准的个性化隔离节点上传模型、数据和比赛代码。
- 预检通过不等于模型部署完成；单样本生成通过也不等于取得性能优化。
- 只有真实 PPU 上的完整模型、公开评测入口、Accuracy 护栏和同口径重复计时，
  才能形成 PPU baseline 或优化结论。

## 验证对象

脚本会采集或检查：

1. OS、CPU、内存、磁盘、主机名和 Git commit/工作区状态；
2. `/dev/alixpu*`、`ppu-smi`、可见设备环境变量；
3. PPU SDK 根目录、`clang++ -x hggc`、HGGC 头文件/库；
4. `hgcc`、MemCheck、PPU-GDB、Asight Systems、ACU、hgobjdump；
5. Python、PyTorch、Transformers、vLLM、Triton、FlashAttention、
   FlashInfer 和 causal-conv1d 版本；
6. PPU PyTorch 是否暴露 CUDA 兼容设备，以及设备名称和显存；
7. PPU-vLLM 是否包含 PPU、Qwen3.5、GDN 和 causal-conv1d 源码标记；
8. 模型文件、锁定 revision、权重尺寸和可选 SHA-256；
9. Qwen3.5-2B 的 Transformer 结构指纹；
10. Transformers eager 与 vLLM 两条路线的机器可读 blocker；
11. 已知迁移风险、主办方待确认问题和下一阶段建议。

## Qwen3.5 结构指纹

`check_ppu_runtime.py` 不依赖 Transformers 就可以直接读取 `config.json`，检查：

| 模块 | 关键结构 |
|---|---|
| 总体 | `qwen3_5`、`Qwen3_5ForConditionalGeneration`、BF16 |
| 文本主干 | hidden 2048、MLP 6144、24 层 |
| 混合注意力 | 18 层 linear attention/GDN、6 层 full attention，每 4 层一次 |
| 全注意力 | 8 Q heads、2 KV heads、head dim 256 |
| GDN | 16 key/value heads、head dim 128、causal-conv width 4 |
| 视觉主干 | hidden 1024、MLP 4096、24 层、16 heads、patch 16 |
| 词表 | 248320，输出头尺寸 `2048 -> 248320` |

任意字段变化都先作为 blocker 处理，因为它可能表示模型版本错误、配置不完整，或
主办方已经更换 revision。不能在未知模型结构上继续做性能比较。

## SSH 接通后的执行顺序

### L0：只读预检

```bash
cd ~/projects/seu-vlm-ppu-optimization
git status --short
git branch --show-current
chmod +x scripts/run_ppu_first_validation.sh

scripts/run_ppu_first_validation.sh \
  --vllm-source /opt/vllm \
  --model-path /path/to/Qwen3.5-2B
```

重点先读：

```text
artifacts/ppu-first-validation/runtime-summary.md
artifacts/ppu-first-validation/runtime.json
artifacts/ppu-first-validation/ppu-smi.txt
```

### L1：PPU PyTorch 小张量

确认节点是个性化隔离资源后：

```bash
scripts/run_ppu_first_validation.sh \
  --vllm-source /opt/vllm \
  --model-path /path/to/Qwen3.5-2B \
  --run-device-smoke
```

该步骤只计算一个 BF16 `32x32` 单位矩阵乘。脚本发现不到 PPU 时会拒绝在本地
NVIDIA GPU 上误跑，以免把 CUDA 结果写成 PPU 结果。

### L2：锁定权重与 HGGC kernel

```bash
scripts/run_ppu_first_validation.sh \
  --model-path /path/to/Qwen3.5-2B \
  --verify-model-hash \
  --run-device-smoke \
  --run-microbench \
  --device 0 \
  --warmup 10 \
  --iterations 100
```

`--verify-model-hash` 会读取约 4.6 GB 权重；第一次或模型来源变化时运行即可。
微基准先以一次无预热执行冒烟，再测 Qwen3.5 的三组 BF16 GEMV：

- `N=6144, K=2048`；
- `N=2048, K=6144`；
- `N=2048, K=2048`。

只有数值误差通过才能记录延迟、GFLOP/s 和有效带宽。

### L3：完整模型加载

```bash
scripts/run_ppu_first_validation.sh \
  --model-path /path/to/Qwen3.5-2B \
  --run-device-smoke \
  --run-model-load
```

`model-load.json` 会记录模型类、参数张量/元素的设备分布、设备名称、模型内存和
offload 状态。启用该阶段后，CPU、meta、disk offload 或无加速设备都会失败。

### L4：单张真实多模态样本

```bash
scripts/run_ppu_first_validation.sh \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_cn.tsv \
  --run-device-smoke \
  --run-model-load \
  --run-single-sample
```

单样本强制使用 `--backend transformers`，不会回退到 dummy。它同时覆盖：

- 图片解码与 processor；
- chat template/tokenizer；
- 视觉主干和视觉到文本投影；
- GDN linear attention、causal-conv1d 和 recurrent state；
- full attention、KV Cache 与解码；
- 首个生成 token 计时和 A/B/C/D 输出解析。

### L5：正式小基线

L0-L4 全部通过后，再使用主办方公开入口固定前 20 条，至少重复三次。每次记录：

- Git commit、官方代码版本和模型 revision；
- PPU/SDK/Driver/HGGC/Python/PyTorch/Transformers/vLLM；
- 样本 ID、生成参数和后端；
- Accuracy、首 token TTFT、decode throughput；
- 参数设备分布、PPU-SMI 和 Profile；
- 是否存在输出/token 数变化或 CPU fallback。

## 常见问题定位

| 现象 | 首要检查 | 不能得出的结论 |
|---|---|---|
| `ppu-smi` 不存在 | 镜像、PATH、驱动/设备挂载 | 不能开始 SDK 或模型验证 |
| 有 `ppu-smi`，无 `/dev/alixpu*` | 容器设备透传和权限 | 不能认为 Python 可使用 PPU |
| 缺 `hggc_runtime.h` | 是否只装 runtime、SDK 路径 | 不能编译自定义 kernel |
| CUDA 源码编译失败 | PPU 不支持 API、inline PTX、nvcc 参数 | 不能用 NVIDIA 二进制替代 |
| `torch.cuda.is_available()==False` | 是否安装 PPU 定制 PyTorch | 不要安装通用 NVIDIA wheel 硬顶 |
| GEMM RTC 直接 `SIGABRT` 且提示 SDK/HOME 不存在 | 导出 `PPU_SDK=/usr/local/PPU_SDK` | 模型加载成功不代表 RTC 路径可用 |
| 通用 torchvision 报 `__cudaGetKernel` 未定义 | wheel 与 PPU torch ABI 不匹配；图片路径可用 CPU wheel | 不要覆盖 PPU torch/triton |
| Transformers 无 `qwen3_5` | Transformers 版本/定制补丁 | 不能用旧 Qwen 类冒充 |
| vLLM 无模型注册 | PPU-vLLM 版本 | 先走 eager 保底或移植注册 |
| vLLM 无 GDN | GDN prefill/decode、state cache | FlashAttention 存在也不能覆盖 GDN |
| 能生成但 PPU 利用率低 | 参数设备、CPU fallback、kernel trace | 正确性通过不等于硬件部署完成 |
| 单样本首 token 很慢 | 视觉前处理、prefill、JIT、首次分配 | 首次冷启动不能与稳态混算 |
| decode 很慢 | GEMV、GDN update、causal conv、launch 数 | GPU 热点不能直接当作 PPU 热点 |
| BF16 误差过大 | 累加精度、布局、越界和同步 | 不得先报性能再补正确性 |
| Accuracy 变化 | dtype、kernel、输出 token、解析和版本 | 不得只比较速度 |

## 仍需外部确认

1. 私有和最终镜像的完整版本矩阵；
2. PPU-vLLM 是否已经支持 Qwen3.5/GDN；
3. 是否允许携带或现场编译自定义 wheel/so；
4. 量化、校准数据和权重变换边界；
5. 主办方 v1.2 与当前 v1.1 的逐文件差异；
6. 网络、安装、模型加载和总运行时间限制；
7. 最终提交包目录、启动命令和持久化目录。

## 输出文件

默认输出到 Git 忽略目录 `artifacts/ppu-first-validation/`：

```text
manifest.txt                  本次参数、Git commit、设备和迭代数
system-info.txt               OS/CPU/内存/磁盘
python-packages.json          Python 包版本
runtime.json                  完整机器可读预检
runtime-summary.md            人类可读问题和下一步
runtime.stdout.log            终端原始输出
ppu-smi.txt                   设备快照
model-load.json               完整模型加载与设备分布（可选）
single-sample.json            单样本真实结果（可选）
microbench-build.log          HGGC 编译日志（可选）
microbench-smoke.log          单次算子冒烟（可选）
microbench-suite.log          三组 GEMV 结果（可选）
```

这些原始文件不提交 Git。确认可公开后，只把去敏的环境摘要、聚合数值和结论写入
`docs/experiments/`。
