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

GDN 在一个 kernel 内完成 BF16 q/k L2Norm、state 衰减、`state·k`、delta、
rank-1 state 更新和 `state·q`；causal-conv 在一个 kernel 内完成 4-tap depthwise
卷积、state 左移、SiLU 和 BF16 输出。两个 RMSNorm kernel 合并原先的 cast、square、
mean、add、rsqrt、weight/gate/SiLU 与输出 cast。

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
```

每个 smoke 都先与当前 Transformers eager 逐元素比较，再计时；数值失败时返回非零。
当前 PPU 实测中，causal-conv、RMSNorm 和 gated RMSNorm 的随机 BF16 输出均
bit-exact，GDN 最大 state/output 误差为 `5.96e-8 / 0`。
四个 smoke 还分别在 `hggc-memcheck` 下以 `warmup=0, iters=1` 执行，均报告
`ERROR SUMMARY: 0 errors`；memcheck 插桩时延不能用于性能比较。

| 算子 | eager ms | fused ms | 单算子加速 |
|---|---:|---:|---:|
| recurrent GDN，4 tiles | 0.19490 | 0.031362 | 6.21x |
| causal-conv，96 threads | 0.03677 | 0.015997 | 2.30x |
| RMSNorm，512 threads | 0.05651 | 0.012120 | 4.66x |
| gated RMSNorm，128 threads | 0.06298 | 0.014572 | 4.32x |

## 模型接入

默认不开任何自定义算子。保守候选只开 GDN + causal-conv：

```bash
export SEU_PPU_GDN_LIBRARY="$PWD/ppu/custom_ops/build/libseu_ppu_gdn.so"
export SEU_PPU_GDN_TILES=4
export SEU_PPU_CONV_ENABLE=1
export SEU_PPU_CONV_THREADS=96
```

高性能候选再显式加入两个 norm；它会产生少量自回归文本长度漂移，提交前必须扩大
精度验证：

```bash
export SEU_PPU_RMSNORM_ENABLE=1
export SEU_PPU_RMSNORM_THREADS=512
export SEU_PPU_GATED_RMSNORM_ENABLE=1
export SEU_PPU_GATED_RMSNORM_THREADS=128
```

随后照常运行 `benchmark_public.py`。`GenerationResult.meta` 会记录实际挂载的
GDN/conv/RMSNorm/gated-RMSNorm 模块数，预期分别为 `18/18/49/18`。

固定中文前 20 条、同一实例/模型/seed、2 条预热的实测：

| 路径 | 平均 TTFT ms | 平均 token/s | Accuracy | 相对 eager 吞吐 |
|---|---:|---:|---:|---:|
| eager | 118.493 | 49.737 | 85% | - |
| GDN only | 119.460 | 61.350 | 85% | +23.35% |
| GDN + causal-conv | 117.262 | 63.911 | 85% | +28.50% |
| all-four | 119.677 | 81.307 | 85% | +63.47% |

20 条中四条路径的解析答案和正确性均一致。GDN-only 有 3 条 token 数变化，
all-four 有 5 条；因此这些结果证明小样本无 Accuracy 回退，不等于已经证明完整公开集
或私有集无回退。

同一 226-token prompt、2-token warmup、16-token 的完整 all-four profile 中，
18/18/49/18 个模块均成功挂载，Self CPU/PPU 为 `514.366/131.899 ms`；最大设备
热点已转为运行时 GEMV/GEMM，其中 `gemvt_op` 为 1,906 次、29.359 ms。

## 遇到的问题、根因和方案

| 问题 | 根因 | 当前处理 | 后续方案 |
|---|---|---|---|
| 模型加载后原生 abort | RTC 找不到 `PPU_SDK/PPU_HOME` | wrapper 在标准路径存在时自动补齐 | 最终镜像仍应由启动脚本显式导出，并保证 RTC cache 可写 |
| 首次整模型 GDN 报 beta dtype | `sigmoid(b)` 实际仍是 BF16，eager 才转 FP32 | kernel 直接读取 BF16 beta | 若模型版本改变 dtype，shape/dtype 门禁会拒绝运行 |
| 误以为只有 49 个 RMSNorm | 另有 12 个 256 维 full-attention q/k norm | 只挂载 weight=2048 的 49 个模块 | 可另做 256 维核，但先由 profile 证明收益 |
| GDN 4 tiles 缩到 32 threads 反而慢 | block 数增加但访存并行度下降 | 保留 64 blocks × 128 threads | 用设备计数器确认 occupancy/带宽后再改 |
| profiler 中自定义 GDN 为约 53 us，事件微基准约 31 us | profiler 插桩本身有开销 | 跨版本只比较同口径 | 端到端结论使用无 profiler 的 benchmark |
| 线程内异常会让 streamer 等待 | 生成工作线程异常时没有结束标记 | 所有 shape/dtype/模块数先做 smoke 门禁 | 后续给 wrapper 增加异常向主线程传播和 streamer 关闭 |
| GDN/RMSNorm 极小归约差异被自回归放大 | FP32 reduction 顺序不保证 bit-exact | norm 保持显式 opt-in；20 条逐答案对比 | 跑完整公开集、官方 Accuracy 门限和多类别漂移报告后再决定提交配置 |
| 自定义库依赖 ctypes | 当前镜像没有现成 PPU PyTorch extension 模板 | 显式传当前 PPU stream，保留 Python dtype/stride 门禁 | 决赛版迁移为注册的 PyTorch/vLLM custom op，补 ABI/生命周期测试 |

## 当前决策边界

- GDN + causal-conv 是较保守的默认候选：20 条 Accuracy 不变，causal-conv 本身
  bit-exact；GDN 仍有 3/20 生成长度漂移。
- all-four 是当前最高性能候选：20 条吞吐提升 63.47%，但有 5/20 生成长度漂移，
  尚不能直接宣称最终可提交。
- 这些 kernel 锁定 Qwen3.5-2B 当前维度、BF16、batch=1 decode 主路径；shape/dtype
  不匹配会失败或回退，不能静默套到其他模型。
- 下一热点已转为运行时 GEMV/GEMM、剩余 elementwise/cat/reduce；仓库已有 HGGC
  通用 GEMV 仍慢于 `torch.mv`，不应强行替换。
