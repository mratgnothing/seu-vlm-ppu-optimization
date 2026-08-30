# 2026-08-27 PPU Qwen3.5 GDN 输入投影打包

## 为什么做

Qwen3.5-2B 的 18 个 `Qwen3_5GatedDeltaNet` 层会对同一个 decode hidden state
依次调用四个无 bias Linear：

| 投影 | 权重形状 | 作用 |
|---|---:|---|
| `in_proj_qkv` | 6144×2048 | 生成 GDN 的 q/k/v |
| `in_proj_z` | 2048×2048 | 输出门控 z |
| `in_proj_b` | 16×2048 | beta 门控 |
| `in_proj_a` | 16×2048 | 衰减门控 |

batch=1、单 token decode 下，这四次都是同输入 GEMV。优化把四份权重按输出维拼成
`[8224,2048]`，一次 `F.linear` 后按 `(6144,2048,16,16)` 切成 view。四个原
Parameter 改为打包 buffer 的 view，不保留第二份常驻权重；多 token prefill 和异常
调用顺序回退原 forward。候选完全由模型图结构决定，不读取公开集标签或题目内容。

## 随机 BF16 与模块级结果

PPU-ZW810E 随机 BF16 测试中：decode/prefill 均逐位一致，四份 Parameter 与打包
buffer 共享 storage；线程隔离版四投影合计从 `0.036405 ms` 降到 `0.022420 ms`，
即 `1.6237x`。双线程 smoke 在两条线程完成 qkv 后用 barrier 强制交错，再分别读取
z/b/a，结果也逐位一致，证明 `threading.local()` 缓存不会跨请求串数据。

## 整模固定长解码

基线为五类 HGGC 融合加 packed-MLP。固定同一输入生成 128 token，先预热两条路径，
最终线程隔离版再做 4 对交替 AB/BA：

| 路径 | 中位 TTFT | 中位 token/s |
|---|---:|---:|
| optimized baseline | 119.352 ms | 94.992 |
| 四投影一次完成 | 122.408 ms | 97.211 |

4/4 对候选更快，成对速度比中位/均值为 `1.0182x/1.0287x`；两条路径生成的
128-token 文本 SHA-256 完全一致。原始结果：
[`results/ppu-packed-gdn-ab128-20260827.json`](../../results/ppu-packed-gdn-ab128-20260827.json)。

## CN20 精度与吞吐门禁

中文固定前 20 条逐样本交替 AB/BA：

| 路径 | 平均 TTFT | 平均 token/s | Accuracy | 全文一致 |
|---|---:|---:|---:|---:|
| optimized baseline | 124.992 ms | 94.099 | 85% | - |
| 四投影一次完成 | 122.652 ms | 98.430 | 85% | 19/20 |

成对速度比中位/均值为 `1.0355x/1.0484x`，15/20 条候选更快。唯一不一致样本
两条路径答案均为 D 且正确，但基线生成 43 token、候选生成 44 token；8 对复测中
该差异稳定复现，因此不是计时噪声，而是合并后 GEMV 形状改变所带来的 BF16 数值路径
差异触发了停止边界。原始结果：
[`results/ppu-packed-gdn-cn20-20260827.json`](../../results/ppu-packed-gdn-cn20-20260827.json)。

## Profile 机制证据

同一 226-token prompt、16-token profile（首 token 为 prefill 后输出，因此有 15 个
纯 decode step）得到：

| 事件 | 基线 | 四投影一次完成 | 差值 |
|---|---:|---:|---:|
| `aten::linear` | 2730 | 1920 | -810 |
| `aten::mm` | 2632 | 1822 | -810 |
| `gemvt_op`（两类模板合计） | 1906 | 1636 | -270 |
| profiler 区间 | 655.062 ms | 598.756 ms | -56.306 ms |

`810 = 18 layers × 3 eliminated calls × 15 decode steps`；原 qkv/z/b/a 四个
`mm` 形状被 270 个 `[1,2048]×[2048,8224]` 取代。设备端只减少 270 个 GEMV，
说明 8224 输出被后端拆成约 3 个 `gemvt_op`，但仍比原四路少一次/层/step。这同时
证明了 ATen 调度减少和设备工作减少。小型汇总见：

- [`results/ppu-packed-gdn-profile-ab-20260827.json`](../../results/ppu-packed-gdn-profile-ab-20260827.json)
- [`results/ppu-packed-gdn-profile-baseline-summary-20260827.json`](../../results/ppu-packed-gdn-profile-baseline-summary-20260827.json)
- [`results/ppu-packed-gdn-profile-candidate-summary-20260827.json`](../../results/ppu-packed-gdn-profile-candidate-summary-20260827.json)

大型 Chrome trace 只保存在隔离服务器，不进入 Git。

## 连续分组消融

为区分“发射次数收益”和“权重连续化收益”，还测试了连续分组。`(2,1,1)` 只合并
qkv+z，b/a 保持原调用：CN20 达到 20/20 全文一致，但平均吞吐
`96.539→94.806 token/s`，成对中位 `0.9884x`，只有 8/20 获胜。`(2,2)` 在问题
样本上全文一致但无稳定收益；`(1,3)`、`(3,1)` 仍出现同一 token 漂移。

这说明当前可见收益主要来自把四次 dispatcher/kernel submission 压成一次。仅打包
存储或保留两到三次 Linear 不能兑现整模收益。精确分组原始结果：
[`results/ppu-packed-gdn-exact-cn20-20260827.json`](../../results/ppu-packed-gdn-exact-cn20-20260827.json)。

## 保持四路 GEMV 的 grouped acBLAS

为同时保留逐位稳定性和减少主机调度，进一步实现了一个结构专用的 C++ extension：
Python 只调用一次 `gdn_projections_bf16`，C ABI bridge 在同一个 mutex、acBLAS handle
和当前 PyTorch stream 下，仍按 qkv、z、b、a 的原顺序提交四个 `acblasGemvEx`。
权重仍共享 `[8224,2048]` storage，但没有把四个矩阵变成一个 8224 行 GEMV，因此
数学形状、输出切片和 BF16 累加路径都保持不变。这是模型结构优化，不依赖公开集
题目、标签或答案。

随机 BF16 smoke 的 decode/prefill 均逐位一致，四投影耗时从 `0.039343 ms` 降至
`0.030659 ms`，即 `1.2832x`。固定 128-token 六对结果为：

| 路径 | 中位 token/s | 成对中位/均值 | 获胜对数 | 全文一致 |
|---|---:|---:|---:|---:|
| optimized baseline | 99.880 | - | - | - |
| grouped acBLAS | 101.896 | `1.0121x/1.0122x` | 3/6 | 是 |

CN20 在同一代码上独立运行两轮：

| 轮次 | baseline token/s | grouped acBLAS token/s | 成对中位/均值 | 获胜 | Accuracy | 全文一致 |
|---|---:|---:|---:|---:|---:|---:|
| r1 | 96.409 | 98.028 | `1.0187x/1.0173x` | 16/20 | 85%/85% | 20/20 |
| r2 | 95.634 | 99.601 | `1.0391x/1.0426x` | 17/20 | 85%/85% | 20/20 |

两轮方向一致且所有文本逐字相同，说明它比“一次 8224 行 Linear”的激进路径更适合
精度优先配置；但固定长六对只有 3/6 获胜，完整公开集和官方私有集仍是最终门禁。

正式 `benchmark_public.py` 单样本冷启动冒烟也已通过：backend 为真实
`transformers`，公开校验无错误，GDN/conv/RMSNorm/gated-RMSNorm/qk-RoPE/
packed-MLP/grouped-GDN 挂载数依次为 `18/18/49/18/6/24/18`，结果元数据明确记录
`ppu_gdn_projection_backend=acblas-grouped`。冷启动 TTFT 不进入性能统计。

Profile 进一步区分了主机和设备侧变化：

| 事件 | 基线 | grouped acBLAS | 差值 |
|---|---:|---:|---:|
| `aten::linear` | 2730 | 1650 | -1080 |
| `aten::mm` | 2632 | 1552 | -1080 |
| `gemvt_op` | 1906 | 1906 | 0 |
| `cudaLaunchKernel` | 16973 | 16973 | 0 |

`1080 = 18 layers × 4 calls × 15 decode steps`。设备仍执行同样的四路 GEMV 和同样
数量的 kernel，收益来自把四次 Python/ATen/pybind 入口合并为一次，并复用一次
mutex、handle 和 stream 设置。单次 profiler 计时受插桩和短样本噪声影响，不作为
吞吐结论；性能只引用上面的无 profiler 成对 A/B。

原始小型结果：

- [`results/ppu-acblas-grouped-gdn-ab128-20260827.json`](../../results/ppu-acblas-grouped-gdn-ab128-20260827.json)
- [`results/ppu-acblas-grouped-gdn-cn20-r1-20260827.json`](../../results/ppu-acblas-grouped-gdn-cn20-r1-20260827.json)
- [`results/ppu-acblas-grouped-gdn-cn20-r2-20260827.json`](../../results/ppu-acblas-grouped-gdn-cn20-r2-20260827.json)
- [`results/ppu-acblas-grouped-gdn-profile-ab-20260827.json`](../../results/ppu-acblas-grouped-gdn-profile-ab-20260827.json)
- [`results/ppu-acblas-grouped-gdn-profile-baseline-summary-20260827.json`](../../results/ppu-acblas-grouped-gdn-profile-baseline-summary-20260827.json)
- [`results/ppu-acblas-grouped-gdn-profile-candidate-summary-20260827.json`](../../results/ppu-acblas-grouped-gdn-profile-candidate-summary-20260827.json)
- [`results/ppu-acblas-grouped-gdn-formal-wrapper-smoke-20260827.json`](../../results/ppu-acblas-grouped-gdn-formal-wrapper-smoke-20260827.json)

## 当前决策

- 精度优先候选使用 `SEU_PPU_ACBLAS_GDN_BUILD_DIR=<extension-build-dir>`；它保持
  四个原始 GEMV，CN20 两轮均 20/20 全文一致，但仍默认关闭；
- 性能优先候选使用 `SEU_PPU_PACK_GDN_PROJECTIONS_ENABLE=1` 和
  `SEU_PPU_PACK_GDN_PROJECTIONS_GROUPS=4`；它改变 GEMV 形状，CN20 为 19/20 exact；
- 两个 backend 互斥，wrapper 会在同时配置时直接报错；
- `2,1,1` 可复现实验性精确 Torch 分组，但没有性能价值；
- 不能把 85% Accuracy 不变表述为严格无损；完整公开集和官方私有集仍是最终门禁；
- 下一步应继续寻找厂商 grouped-GEMV/batched-GEMV 或 HGGC multi-output GEMV，
  争取在保持原四路数值路径的同时真正减少设备 kernel launch。

## 复现入口

- `ppu/custom_ops/ppu_gdn_projection_pack.py`：权重别名、分组 closure 与 prefill 回退；
- `ppu/custom_ops/ppu_acblas_gdn_projection.py`：一次 pybind 入口、四路原形状 GEMV；
- `ppu/custom_ops/smoke_acblas_gdn_projection.py`：随机 BF16 grouped acBLAS 门禁；
- `ppu/custom_ops/smoke_packed_gdn_projections.py`：随机 BF16 数值/存储/耗时门禁；
- `scripts/benchmark_ppu_packed_gdn_ab.py`：固定长解码与问题样本重复 AB/BA；
- `scripts/benchmark_ppu_packed_gdn_multisample_ab.py`：CN20 逐样本成对门禁。
