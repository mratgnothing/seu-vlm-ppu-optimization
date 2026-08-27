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

## 当前决策

- `SEU_PPU_PACK_GDN_PROJECTIONS_ENABLE=1` 才启用，默认关闭；
- `SEU_PPU_PACK_GDN_PROJECTIONS_GROUPS=4` 是高性能候选；可用 `2,1,1` 复现实验性
  精确分组，但它没有性能价值；
- 不能把 85% Accuracy 不变表述为严格无损；完整公开集和官方私有集仍是最终门禁；
- 下一步应实现一个保持原四个输出累加顺序的 HGGC fused multi-output GEMV，或获得
  厂商 grouped-GEMV 接口，再争取同时满足一次发射和逐位稳定。

## 复现入口

- `ppu/custom_ops/ppu_gdn_projection_pack.py`：权重别名、分组 closure 与 prefill 回退；
- `ppu/custom_ops/smoke_packed_gdn_projections.py`：随机 BF16 数值/存储/耗时门禁；
- `scripts/benchmark_ppu_packed_gdn_ab.py`：固定长解码与问题样本重复 AB/BA；
- `scripts/benchmark_ppu_packed_gdn_multisample_ab.py`：CN20 逐样本成对门禁。
