# Qwen3.5-2B PPU decode 融合算子

本目录是在 PPU-ZW810E、HGGC 13.0 上实测的 Transformers decode fast path。
它不修改系统 PyTorch/Transformers，不覆盖主办方环境；`build_gdn_shared.sh` 只生成
一个本仓库内的共享库，`evaluation_wrapper.py` 只有在显式设置环境变量后才挂载算子。

## 当前包含的算子

| 算子 | 固定 decode 形状/语义 | 模型调用数 |
|---|---|---:|
| recurrent GDN | q/k/v BF16 `[B,1,16,128]`，state FP32 `[B,16,128,128]` | 18/token |
| GDN gate-prep | raw a/b BF16 `[B,1,16]`，cached A FP32，输出 g FP32/beta BF16 | 18/token |
| causal-conv update | BF16 `[B,6144,1]`，state `[B,6144,4]`，SiLU | 18/token |
| RMSNorm | BF16 `[B,1,2048]`，FP32 reduction | 49/token |
| residual-add + RMSNorm | BF16 `[B,1,2048]` residual/branch，FP32 reduction | 48/token |
| gated RMSNorm | BF16 `[16,128]` + gate，FP32 reduction/SiLU | 18/token |
| q/k RMSNorm+RoPE | q `[B,1,8,256]`、k `[B,1,2,256]`、partial rotary 64 | 6/token |

GDN 在一个 kernel 内完成 BF16 q/k L2Norm、state 衰减、`state·k`、delta、
rank-1 state 更新和 `state·q`；causal-conv 在一个 kernel 内完成 4-tap depthwise
卷积、state 左移、SiLU 和 BF16 输出。两个 RMSNorm kernel 合并原先的 cast、square、
mean、add、rsqrt、weight/gate/SiLU 与输出 cast。
q/k 融合核在一个 256-thread block/head 内完成 RMSNorm、BF16 舍入、64 维
partial RoPE 和目标布局写回，替代 full-attention decode 中的两次 norm 与多组
neg/mul/add/cat。

residual-RMSNorm 路径进一步利用 decoder 图中 48 条相邻的 `residual add -> norm`
边：一个 kernel 先产生与 eager 相同 BF16 舍入点的 residual sum，再做 FP32
RMS reduction 和 weight scaling。每层内部的 attention 边直接融合，MLP 边通过
thread-local、输入对象身份校验的缓存连接下一层 input norm（最后一层连接 final
norm）；prefill 和不匹配的 dtype/device/shape 均回退原 forward。

GDN gate-prep 在加载时缓存 FP32 `exp(A_log)`，一个 kernel 合并
`sigmoid(b)`、两个 cast、`a+dt_bias`、Softplus、乘法和取负。每层使用
thread-local scratch 复用 `g/beta`，只覆盖 eval、BF16、batch=1、seq=1 且已有
recurrent cache 的 decode；prefill 和其他契约全部回退。

目录还保留一个独立的 BF16 SwiGLU 负实验核。它在 `[1,1,6144]` 上与
`F.silu(gate) * up` bit-exact，但 128/256/512/1024 线程的最好速度仅为 Torch
两核的 `0.7901x`，因此没有接入模型 wrapper。后续应争取 acBLAS GEMM epilogue
fusion，直接避免 gate/up 中间张量，而不是增加独立 HGGC launch。

后续 acBLASLt 调查确认公开 epilogue 只有 Bias/ReLU/GELU，没有 SiLU。四个真实
decode 形状的 32 heuristic 扫描中，只有 2048 方阵超过 3%；它配合 scratch 的
模块级为 `1.2797x`，但整模固定 128-token 八对仅 `0.9898x`、3/8 获胜，故同样
不接入正式 wrapper。详见
[`2026-08-28-ppu-acblaslt-matmul.md`](../../docs/experiments/2026-08-28-ppu-acblaslt-matmul.md)。

随后把优化边界提升到完整 decode MLP：一次 C++ extension 入口按原顺序提交 packed
gate/up GEMV、bit-exact HGGC SwiGLU 和 down GEMV，并复用每层三个 scratch。它在
固定 128-token 八对中 8/8 获胜、成对中位 `1.1336x`；CN20 两轮均 20/20 全文一致
和 20/20 获胜，成对中位 `1.1212x/1.1122x`。该候选已接入显式 wrapper 开关，详见
[`2026-08-28-ppu-acblas-packed-mlp.md`](../../docs/experiments/2026-08-28-ppu-acblas-packed-mlp.md)。

继续把相同边界策略应用到 6 个全注意力层：单入口依次提交 Q/K/V 三个原形状
acBLAS GEMV，再调用已验证的 Q/K RMSNorm+RoPE 核。真实模块 Q/K/V/gate 与 prefill
均 bit-exact，强化 smoke 模块边界 `4.1006x`，memcheck 0 errors；固定长 56 对合并中位
仅 `1.0047x`，CN20 两轮中位 `1.0158x/0.9852x`，第二轮门禁失败，故默认关闭并作为
负实验保留。
详见 [`2026-08-28-ppu-acblas-attention-prep.md`](../../docs/experiments/2026-08-28-ppu-acblas-attention-prep.md)。

此外，本目录提供一个不新增 HGGC kernel 的 packed-MLP decode 路径：24 个 MLP 的
`gate_proj` 和 `up_proj` 权重拼成共享存储 `[12288, 2048]`，decode 时把两次
`2048→6144` 线性投影合成一次 `2048→12288`，再 split、SiLU、逐元素乘和
`down_proj`。prefill 仍调用原始 forward，gate/up Parameter 只是 packed storage 的
两个 view，不常驻第二份 1.2 GiB 权重。

另有三条图/运行时实验路径。`ppu_gdn_projection_pack.py` 把 GDN 的 qkv/z/b/a 四个
同输入 decode 投影合为一次；四份 Parameter 共享 `[8224,2048]` storage，prefill
回退。`ppu_acblas_gdn_projection.py` 则只合并 Python/ATen/pybind 调度，在一个 C++
入口内仍按原顺序执行四个原形状 acBLAS GEMV，以保留 BF16 数值路径。通用注册式
PyTorch/acBLAS 路径替换 102 个 bias-free decode Linear，并用 C ABI 隔离
Torch/CUDA 与 PPU 半精度头；通用替换保留作负实验，结构专用 grouped GDN 作为
默认关闭的精度优先候选接入 wrapper。

## 构建和单算子验收

```bash
cd ppu/custom_ops
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export LD_LIBRARY_PATH="$PPU_SDK/lib:$PPU_SDK/lib64:${LD_LIBRARY_PATH:-}"
./build_gdn_shared.sh

python smoke_gdn_integration.py --tiles-per-head 4 --warmup 50 --iters 1000
python smoke_gdn_gate_prep_integration.py \
  --library build/libseu_ppu_gdn.so --warmup 50 --iters 1000 --repeats 5
python smoke_causal_conv_integration.py --threads 96 --warmup 50 --iters 1000
python smoke_rmsnorm_integration.py --threads 512 --warmup 50 --iters 1000
python smoke_swiglu_integration.py \
  --library build/libseu_ppu_gdn.so --threads 128 --warmup 50 --iters 1000
python smoke_gated_rmsnorm_integration.py --threads 128 --warmup 50 --iters 1000
python smoke_qk_rmsnorm_rope_integration.py --warmup 50 --iters 1000
python smoke_packed_mlp_integration.py --warmup 32 --iters 400
python smoke_packed_gdn_projections.py --warmup 32 --iters 400
python build_acblas_linear_extension.py
python smoke_acblas_linear_module.py \
  --build-dir build/acblas_linear_extension \
  --input-features 2048 --output-features 2048
python smoke_acblas_gdn_projection.py \
  --build-dir build/acblas_linear_extension \
  --warmup 32 --iters 400
python build_acblas_packed_mlp_extension.py
python smoke_acblas_packed_mlp_module.py \
  --build-dir build/acblas_packed_mlp_extension \
  --warmup 10 --iters 100
SEU_PPU_GDN_LIBRARY="$PWD/build/libseu_ppu_gdn.so" \
  python build_acblas_attention_prep_extension.py
python smoke_acblas_attention_prep_module.py \
  --model-path /path/to/Qwen3.5-2B \
  --gdn-library "$PWD/build/libseu_ppu_gdn.so" \
  --build-dir build/acblas_attention_prep_extension \
  --warmup 20 --iters 400
```

每个 smoke 都先与当前 Transformers eager 逐元素比较，再计时；数值失败时返回非零。
当前 PPU 实测中，causal-conv、RMSNorm 和 gated RMSNorm 的随机 BF16 输出均
bit-exact，GDN 最大 state/output 误差为 `5.96e-8 / 0`。
五个 smoke 还分别在 `hggc-memcheck` 下以 `warmup=0, iters=1` 执行，均报告
`ERROR SUMMARY: 0 errors`；memcheck 插桩时延不能用于性能比较。

| 算子 | eager ms | fused ms | 单算子加速 |
|---|---:|---:|---:|
| recurrent GDN，4 tiles | 0.19490 | 0.031362 | 6.21x |
| GDN gate-prep | 0.028592 | 0.020680 | 1.3826x |
| causal-conv，96 threads | 0.03677 | 0.015997 | 2.30x |
| RMSNorm，512 threads | 0.05651 | 0.012120 | 4.66x |
| gated RMSNorm，128 threads | 0.06298 | 0.014572 | 4.32x |
| q/k RMSNorm+RoPE，真实 query stride | 0.20616 | 0.026680 | 7.73x |
| packed MLP gate/up projection | 0.04537 | 0.03997 | 1.135x |
| 单入口 packed MLP（两 GEMV + SwiGLU） | 0.041666 | 0.033907 | 1.229x |
| 单入口 Attention Prep（Q/K/V + QK Norm/RoPE） | 0.080652 | 0.019668 | 4.101x |
| graph-backed 单入口 packed-MLP（稳定输入地址） | 0.044214 | 0.043336 | 1.020x |
| graph-backed 单入口 packed-MLP（含 input copy） | 0.044214 | 0.047462 | 0.932x |
| residual-RMSNorm 持久输出 scratch | 0.018729 | 0.014005 | 1.337x |

packed-MLP smoke 的 decode/prefill 均 bit-exact，gate/up 两个参数均确认复用 packed
storage，重打包后的常驻显存增量为 20 KiB（allocator 元数据/对齐量级）。

## 模型接入

默认不开任何自定义算子。保守候选只开 GDN + causal-conv：

```bash
export SEU_PPU_GDN_LIBRARY="$PWD/ppu/custom_ops/build/libseu_ppu_gdn.so"
export SEU_PPU_GDN_TILES=4
export SEU_PPU_CONV_ENABLE=1
export SEU_PPU_CONV_THREADS=96
```

高性能候选再显式加入两个 norm 和 q/k RMSNorm+RoPE；它会产生少量自回归文本长度
漂移，提交前必须扩大精度验证：

```bash
export SEU_PPU_RMSNORM_ENABLE=1
export SEU_PPU_RMSNORM_THREADS=512
export SEU_PPU_GATED_RMSNORM_ENABLE=1
export SEU_PPU_GATED_RMSNORM_THREADS=128
export SEU_PPU_QK_ROPE_ENABLE=1
export SEU_PPU_PACK_MLP_ENABLE=1
# 跨层 residual-add + RMSNorm；仍需完整集精度门禁，默认关闭。
export SEU_PPU_RESIDUAL_RMSNORM_ENABLE=1
# 缓存 A 并融合 GDN gate-prep；只覆盖 batch=1 cached decode。
export SEU_PPU_GDN_GATE_PREP_ENABLE=1
# 可选：直接查询当前 raw stream handle，减少每次 ctypes 提交的 Python 开销。
# 依赖当前 PPU PyTorch 的私有 API；不支持时显式启用会立即报错。
export SEU_PPU_RAW_STREAM_QUERY_ENABLE=1
# 激进候选；CN20 有 1/20 完整文本漂移，默认必须保持关闭。
export SEU_PPU_PACK_GDN_PROJECTIONS_ENABLE=1
export SEU_PPU_PACK_GDN_PROJECTIONS_GROUPS=4
```

若优先保持原四个 GEMV 的逐位数值路径，不要设置上面两个 packed-GDN 变量，改用：

```bash
export SEU_PPU_ACBLAS_GDN_BUILD_DIR="$PWD/ppu/custom_ops/build/acblas_linear_extension"
# 可选；默认 -1 让 SDK 选择算法。
export SEU_PPU_ACBLAS_GDN_ALGORITHM=-1
# Accuracy-budget 候选：将四个连续权重拼成一次 8224x2048 GEMV。
# CN100 答案 100/100 一致、Accuracy 93%→93%，但 1/100 完整文本漂移；
# 因此默认关闭，只有接受非 bit-exact 数值路径时才显式设为 1。
export SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=0
```

在精度优先 grouped-GDN/gate-prep 栈上启用单入口 packed-MLP：

```bash
export SEU_PPU_ACBLAS_PACKED_MLP_BUILD_DIR="$PWD/ppu/custom_ops/build/acblas_packed_mlp_extension"
export SEU_PPU_ACBLAS_PACKED_MLP_SWIGLU_THREADS=128
```

实验性启用 Attention Prep（当前只允许单请求、patch-time stream 串行 decode）：

```bash
export SEU_PPU_ACBLAS_ATTENTION_PREP_BUILD_DIR="$PWD/ppu/custom_ops/build/acblas_attention_prep_extension"
export SEU_PPU_ACBLAS_ATTENTION_PREP_ALGORITHM=-1
```

两种 GDN projection backend 互斥；同时设置时 wrapper 会立即报错。

随后照常运行 `benchmark_public.py`。`GenerationResult.meta` 会记录实际挂载的
GDN/conv/RMSNorm/gated-RMSNorm/qk-RoPE/packed-MLP/packed-GDN 模块数，预期分别为
`18/18/49/18/6/24/18`。启用跨层融合时另应得到 24 个 decoder patch；启用
gate-prep 时还应得到 18 个 gate-prep module patch。
启用单入口 packed-MLP 时还应得到 24 个 `ppu_acblas_packed_mlp_modules`。
启用 Attention Prep 时还应得到 6 个 `ppu_acblas_attention_prep_modules`。
grouped-acBLAS + residual-RMSNorm 正式单样本冒烟已得到完整计数，且
`ppu_gdn_projection_backend` 为 `acblas-grouped`、公开校验无错误。

固定中文前 20 条、同一实例/模型/seed、2 条预热的实测：

| 路径 | 平均 TTFT ms | 平均 token/s | Accuracy | 相对 eager 吞吐 |
|---|---:|---:|---:|---:|
| eager | 118.493 | 49.737 | 85% | - |
| GDN only | 119.460 | 61.350 | 85% | +23.35% |
| GDN + causal-conv | 117.262 | 63.911 | 85% | +28.50% |
| all-four | 119.677 | 81.307 | 85% | +63.47% |
| all-five r1 | 124.930 | 93.918 | 85% | +88.83% |
| all-five r2 | 118.227 | 94.889 | 85% | +90.78% |
| all-five + packed-MLP r1 | 119.401 | 96.506 | 85% | +94.04% |
| all-five + packed-MLP r2 | 115.916 | 96.715 | 85% | +94.46% |
| all-five + packed-MLP + packed-GDN，CN20 paired | 122.652 | 98.430 | 85% | +97.90% |
| all-five + packed-MLP + grouped-acBLAS GDN r1 | - | 98.028 | 85% | +97.10% |
| all-five + packed-MLP + grouped-acBLAS GDN r2 | - | 99.601 | 85% | +100.26% |
| 上项 + 48-edge residual-RMSNorm r1 | 118.654 | 101.616 | 85% | +104.31% |
| 上项 + 48-edge residual-RMSNorm r2 | 120.807 | 101.507 | 85% | +104.09% |
| 上项 + GDN gate-prep r1 | 121.477 | 109.275 | 85% | +119.71% |
| 上项 + GDN gate-prep r2 | 120.096 | 107.083 | 85% | +115.31% |
| 上项 + 单入口 packed-MLP r1 | 118.023 | 122.350 | 85% | +145.99% |
| 上项 + 单入口 packed-MLP r2 | - | 121.297 | 85% | +143.88% |

20 条中各路径的解析答案和正确性均一致。GDN-only 有 3 条 token 数变化，
all-four 有 5 条；因此这些结果证明小样本无 Accuracy 回退，不等于已经证明完整公开集
或私有集无回退。

all-five 两次逐样本答案、正确性和 token 数完全一致；相对 eager 的 token 数漂移仍为
5/20，没有因 q/k 融合扩大。单样本三次严格 A/B 还实现了 51-token 文本 SHA-256
完全一致。

packed-MLP 两轮之间，以及相对 all-five 的对应 20 条，解析答案、正确性和 token 数
均 20/20 一致；单样本三次 A/B 的 51-token 文本 SHA-256 也与 eager 完全一致。

同一 226-token prompt、2-token warmup、16-token 的完整 all-four profile 中，
18/18/49/18 个模块均成功挂载，Self CPU/PPU 为 `514.366/131.899 ms`；最大设备
热点已转为运行时 GEMV/GEMM，其中 `gemvt_op` 为 1,906 次、29.359 ms。

all-five profile 的 Self CPU/PPU 为 `409.545/121.871 ms`。相对 all-four，
`cudaLaunchKernel` 从 19,878 降到 17,088，`aten::cat` 从 747 降到 387，
`empty_strided` 从 5,472 降到 4,932；新 q/k 核 90 次合计约 0.216 ms。

all-five + packed-MLP profile 的 Self CPU/PPU 为 `427.177/119.956 ms`。其中
decode MLP 的 720 次 `[1,2048]×[2048,6144]` 变为 360 次
`[1,2048]×[2048,12288]`，`aten::linear` 和 `aten::mm` 均减少 360 次；剩余
270 次 2048→6144 属于 GDN qkv。底层 `cudaLaunchKernel` 仍为 17,088 次，说明
PPU GEMV 后端会拆分大矩阵，收益来自减少 ATen 调度和改善大投影效率，而不是减少
底层 launch 数。

最终线程隔离版 packed-GDN 的同模型 CN20 逐样本 AB/BA 平均吞吐从 `94.099`
提升到 `98.430 token/s`，成对中位 `1.0355x`、20 条赢 15 条；Accuracy 都是
85%，但只有 19/20 完整文本一致。固定 128-token 四对 4/4 获胜且全文一致。`(2,1,1)`
精确分组恢复 20/20 exact，却降到成对中位 `0.9884x`。

结构专用 grouped-acBLAS GDN 保留原 qkv/z/b/a 四个 GEMV，只把四次主机入口合为
一次。CN20 两轮平均吞吐分别为 `96.409→98.028` 和 `95.634→99.601 token/s`，
成对中位为 `1.0187x/1.0391x`，16/20 和 17/20 条获胜，Accuracy 均为 85%，
且两轮都是 20/20 全文一致。固定 128-token 六对成对中位 `1.0121x`，但只有 3/6
获胜，因此仍需扩大验证，不能直接成为默认配置。

对应 profile 中 `aten::linear/mm` 均从 `2730/2632` 降至 `1650/1552`，即各减少
`18×4×15=1080` 次；`gemvt_op` 和 `cudaLaunchKernel` 仍为 `1906/16973`。这证明
收益来自主机调度、mutex/handle/stream 设置合并，而不是减少设备 GEMV。

acBLAS profile 虽恰好减少 `102×15=1530` 次 `aten::linear/mm`，最终固定 128-token
八对的成对中位只有 `0.9997x`、4/8 获胜。它证明 dispatcher 被绕过，但不构成整模
wall-clock 优化。

## 遇到的问题、根因和方案

| 问题 | 根因 | 当前处理 | 后续方案 |
|---|---|---|---|
| 模型加载后原生 abort | RTC 找不到 `PPU_SDK/PPU_HOME` | wrapper 在标准路径存在时自动补齐 | 最终镜像仍应由启动脚本显式导出，并保证 RTC cache 可写 |
| 首次整模型 GDN 报 beta dtype | `sigmoid(b)` 实际仍是 BF16，eager 才转 FP32 | kernel 直接读取 BF16 beta | 若模型版本改变 dtype，shape/dtype 门禁会拒绝运行 |
| 误以为只有 49 个 RMSNorm | 另有 12 个 256 维 full-attention q/k norm | 只挂载 weight=2048 的 49 个模块 | 可另做 256 维核，但先由 profile 证明收益 |
| q/k smoke 通过但首轮整模型答案 A→C | 模型 query 来自 `torch.chunk`，head stride=512；首版 smoke 错用了连续 stride=256 | C ABI 显式传 q/k head stride，smoke 改用真实非连续 view；修正后整模型 exact-text 通过 | 所有新核 smoke 必须从真实模块捕获 shape/dtype/stride，不能只复刻 shape |
| GDN 4 tiles 缩到 32 threads 反而慢 | block 数增加但访存并行度下降 | 保留 64 blocks × 128 threads | 用设备计数器确认 occupancy/带宽后再改 |
| profiler 中自定义 GDN 为约 53 us，事件微基准约 31 us | profiler 插桩本身有开销 | 跨版本只比较同口径 | 端到端结论使用无 profiler 的 benchmark |
| 线程内异常会让 streamer 等待 | 生成工作线程异常时没有结束标记 | 所有 shape/dtype/模块数先做 smoke 门禁 | 后续给 wrapper 增加异常向主线程传播和 streamer 关闭 |
| GDN/RMSNorm 极小归约差异被自回归放大 | FP32 reduction 顺序不保证 bit-exact | norm 保持显式 opt-in；20 条逐答案对比 | 跑完整公开集、官方 Accuracy 门限和多类别漂移报告后再决定提交配置 |
| Torch CUDA 头与 PPU HGGC 头重复定义 half/BF16 | 两套兼容头不能进入同一 C++ 翻译单元 | Torch extension 与 acBLAS bridge 通过窄 C ABI 隔离，显式传当前 stream | 通用 102-Linear 替换保留负实验；结构专用 grouped GDN 继续扩大验证 |
| packed-GDN CN20 有 1/20 多生成 1 token | 合并 GEMV 改变 BF16 数值路径，触发停止边界 | 默认关闭；记录完整文本哈希并做连续分组消融 | 写保持原累加顺序的 fused multi-output GEMV |
| `$ORIGIN` 被 Ninja 当变量吞掉 | PyTorch extension 的 Ninja 生成层处理 `$` | 构建时写入确定的绝对 build rpath | 打包阶段改用转义后的相对 rpath 或 wheel 修复工具 |
| 首次 packed-MLP A/B 在 `empty_cache` 报 `torch` 未定义 | 验证脚本新增调用但漏导入 torch | 补 `import torch`，语法和单测后重传 | CI 保持脚本 import/`--help` 冒烟 |
| 首轮公开 20 条只有 eager 速度 | 使用了错误的环境变量名，实际挂载数全为 0 | 以结果 `meta` 的 `18/18/49/18/6/24` 作为有效性门禁后重跑 | 启动脚本统一导出 `SEU_PPU_GDN_LIBRARY`、`SEU_PPU_CONV_ENABLE` 等正式名称 |
| profile 用 `--device ppu` 被 Transformers 拒绝 | SDK 通过 PyTorch CUDA 兼容层暴露设备 | 改为 `--device cuda:0`，设备名仍报告 PPU-ZW810E | 文档和脚本默认值统一使用兼容层设备名 |
| packed 后底层 launch 数未下降 | 12288 输出 GEMV 被 PPU 后端内部切分 | 以端到端两轮和 mm 形状变化验收，保留候选 | 后续若有专用 batched/packed GEMV API，再验证能否真正合并设备 launch |
| 自定义 SwiGLU kernel 比框架组合慢 | 6144 元素计算太短，ctypes/HGGC 启动成本超过少一次 elementwise launch 的收益 | 128/256/512 threads 与完整 MLP 路径均实测后删除 ABI | 优先寻找 GEMV epilogue 融合，而非独立小 kernel |
| 原地 MLP 激活仅有噪声级收益 | 20 条仅 +0.12%/+0.16%，profile 的分配、copy 和 launch 数未下降 | 回退执行分支，只保留实验数据 | 保留门槛要求两轮端到端提升并有结构性 profile 证据 |

## 当前决策边界

- GDN + causal-conv 是较保守的默认候选：20 条 Accuracy 不变，causal-conv 本身
  bit-exact；GDN 仍有 3/20 生成长度漂移。
- all-five + packed-MLP + packed-GDN 是激进性能候选：同模型 CN20 paired 相对
  eager 约 +97.90%，相对 packed 基线成对中位 +3.55%；但它新增 1/20 文本漂移，
  只能作为默认关闭的激进候选。
- grouped-acBLAS GDN 是当前精度优先候选：CN20 两轮均 20/20 exact、方向一致，
  但固定长只有 3/6 获胜，保持默认关闭并等待完整集与更多固定长重复。
- 通用 102-Linear acBLAS 和 `(2,1,1)` 精确 packed-GDN 均未通过性能门槛。
- 这些 kernel 锁定 Qwen3.5-2B 当前维度、BF16、batch=1 decode 主路径；shape/dtype
  不匹配会失败或回退，不能静默套到其他模型。
- 下一热点已转为运行时 GEMV/GEMM、剩余 elementwise/cat/reduce；仓库已有 HGGC
  通用 GEMV 仍慢于 `torch.mv`，不应强行替换。
