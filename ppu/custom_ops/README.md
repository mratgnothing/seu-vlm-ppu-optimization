# Qwen3.5-2B PPU decode 融合算子

本目录是在 PPU-ZW810E、HGGC 13.0 上实测的 Transformers decode fast path。
它不修改系统 PyTorch/Transformers，不覆盖主办方环境；`build_gdn_shared.sh` 只生成
一个本仓库内的共享库，`evaluation_wrapper.py` 只有在显式设置环境变量后才挂载算子。

## 当前包含的算子

| 算子 | 固定 decode 形状/语义 | 模型调用数 |
|---|---|---:|
| recurrent GDN | q/k/v BF16 `[B,1,16,128]`，state FP32 `[B,16,128,128]` | 18/token |
| causal-conv update | BF16 `[B,6144,1]`，state `[B,6144,4]`，SiLU | 18/token |
| RMSNorm | BF16 `[B,1,2048]`，FP32 reduction | 49/token |
| gated RMSNorm | BF16 `[16,128]` + gate，FP32 reduction/SiLU | 18/token |
| q/k RMSNorm+RoPE | q `[B,1,8,256]`、k `[B,1,2,256]`、partial rotary 64 | 6/token |

GDN 在一个 kernel 内完成 BF16 q/k L2Norm、state 衰减、`state·k`、delta、
rank-1 state 更新和 `state·q`；causal-conv 在一个 kernel 内完成 4-tap depthwise
卷积、state 左移、SiLU 和 BF16 输出。两个 RMSNorm kernel 合并原先的 cast、square、
mean、add、rsqrt、weight/gate/SiLU 与输出 cast。
q/k 融合核在一个 256-thread block/head 内完成 RMSNorm、BF16 舍入、64 维
partial RoPE 和目标布局写回，替代 full-attention decode 中的两次 norm 与多组
neg/mul/add/cat。

此外，本目录提供一个不新增 HGGC kernel 的 packed-MLP decode 路径：24 个 MLP 的
`gate_proj` 和 `up_proj` 权重拼成共享存储 `[12288, 2048]`，decode 时把两次
`2048→6144` 线性投影合成一次 `2048→12288`，再 split、SiLU、逐元素乘和
`down_proj`。prefill 仍调用原始 forward，gate/up Parameter 只是 packed storage 的
两个 view，不常驻第二份 1.2 GiB 权重。

## 构建和单算子验收

```bash
cd ppu/custom_ops
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export LD_LIBRARY_PATH="$PPU_SDK/lib:$PPU_SDK/lib64:${LD_LIBRARY_PATH:-}"
./build_gdn_shared.sh

python smoke_gdn_integration.py --tiles-per-head 4 --warmup 50 --iters 1000
python smoke_causal_conv_integration.py --threads 96 --warmup 50 --iters 1000
python smoke_rmsnorm_integration.py --threads 512 --warmup 50 --iters 1000
python smoke_gated_rmsnorm_integration.py --threads 128 --warmup 50 --iters 1000
python smoke_qk_rmsnorm_rope_integration.py --warmup 50 --iters 1000
python smoke_packed_mlp_integration.py --warmup 32 --iters 400
```

每个 smoke 都先与当前 Transformers eager 逐元素比较，再计时；数值失败时返回非零。
当前 PPU 实测中，causal-conv、RMSNorm 和 gated RMSNorm 的随机 BF16 输出均
bit-exact，GDN 最大 state/output 误差为 `5.96e-8 / 0`。
五个 smoke 还分别在 `hggc-memcheck` 下以 `warmup=0, iters=1` 执行，均报告
`ERROR SUMMARY: 0 errors`；memcheck 插桩时延不能用于性能比较。

| 算子 | eager ms | fused ms | 单算子加速 |
|---|---:|---:|---:|
| recurrent GDN，4 tiles | 0.19490 | 0.031362 | 6.21x |
| causal-conv，96 threads | 0.03677 | 0.015997 | 2.30x |
| RMSNorm，512 threads | 0.05651 | 0.012120 | 4.66x |
| gated RMSNorm，128 threads | 0.06298 | 0.014572 | 4.32x |
| q/k RMSNorm+RoPE，真实 query stride | 0.20616 | 0.026680 | 7.73x |
| packed MLP gate/up projection | 0.04537 | 0.03997 | 1.135x |

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
```

随后照常运行 `benchmark_public.py`。`GenerationResult.meta` 会记录实际挂载的
GDN/conv/RMSNorm/gated-RMSNorm/qk-RoPE/packed-MLP 模块数，预期分别为
`18/18/49/18/6/24`。

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
| 自定义库依赖 ctypes | 当前镜像没有现成 PPU PyTorch extension 模板 | 显式传当前 PPU stream，保留 Python dtype/stride 门禁 | 决赛版迁移为注册的 PyTorch/vLLM custom op，补 ABI/生命周期测试 |
| 首次 packed-MLP A/B 在 `empty_cache` 报 `torch` 未定义 | 验证脚本新增调用但漏导入 torch | 补 `import torch`，语法和单测后重传 | CI 保持脚本 import/`--help` 冒烟 |
| 首轮公开 20 条只有 eager 速度 | 使用了错误的环境变量名，实际挂载数全为 0 | 以结果 `meta` 的 `18/18/49/18/6/24` 作为有效性门禁后重跑 | 启动脚本统一导出 `SEU_PPU_GDN_LIBRARY`、`SEU_PPU_CONV_ENABLE` 等正式名称 |
| profile 用 `--device ppu` 被 Transformers 拒绝 | SDK 通过 PyTorch CUDA 兼容层暴露设备 | 改为 `--device cuda:0`，设备名仍报告 PPU-ZW810E | 文档和脚本默认值统一使用兼容层设备名 |
| packed 后底层 launch 数未下降 | 12288 输出 GEMV 被 PPU 后端内部切分 | 以端到端两轮和 mm 形状变化验收，保留候选 | 后续若有专用 batched/packed GEMV API，再验证能否真正合并设备 launch |

## 当前决策边界

- GDN + causal-conv 是较保守的默认候选：20 条 Accuracy 不变，causal-conv 本身
  bit-exact；GDN 仍有 3/20 生成长度漂移。
- all-five + packed-MLP 是当前最高性能候选：20 条两次吞吐提升 94.04%/94.46%，但仍有 5/20
  生成长度漂移，
  尚不能直接宣称最终可提交。
- 这些 kernel 锁定 Qwen3.5-2B 当前维度、BF16、batch=1 decode 主路径；shape/dtype
  不匹配会失败或回退，不能静默套到其他模型。
- 下一热点已转为运行时 GEMV/GEMM、剩余 elementwise/cat/reduce；仓库已有 HGGC
  通用 GEMV 仍慢于 `torch.mv`，不应强行替换。
