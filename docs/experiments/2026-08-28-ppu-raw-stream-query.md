# 2026-08-28 PPU raw stream 查询优化

## 问题

`PPUGDNLibrary` 的每次 ctypes 算子提交原先都执行
`torch.cuda.current_stream(device).cuda_stream`。在 batch=1 单 token decode 中，一个
token 会经过多层 recurrent、conv、RMSNorm、residual-RMSNorm、gate-prep、SwiGLU
与 Q/K norm+RoPE；重复构造/查询 Python `Stream` 对象形成可观的 host 端固定开销。
在当前推荐栈的 cached decode 中，按 18 次 recurrent、18 次 conv、48 次 fused
residual-RMSNorm、1 次末端 RMSNorm、18 次 gated-RMSNorm、18 次 gate-prep 和
6 次 Q/K norm+RoPE 估算，每个生成 token 约有 127 次这类 Python stream 查询。

当前 PPU PyTorch 运行时提供 `_cuda_getCurrentRawStream(device_index)`。候选路径直接
取当前流的整数句柄，仍把同一句柄传入原 HGGC ctypes ABI，不改变 kernel、输入、输出、
执行顺序或数值计算。该接口属于 PyTorch 私有 API，因此实现包含能力检测，正式开关
`SEU_PPU_RAW_STREAM_QUERY_ENABLE` 默认关闭；不具备该 API 的运行时在显式启用时会
立即报错，不静默换语义。

## 模块验证

PPU-ZW810E 上 residual-RMSNorm 真实提交路径：

| 路径 | 平均耗时 | 相对现有路径 |
|---|---:|---:|
| 现有 Stream 对象查询 | 0.016212 ms | `1.0000x` |
| raw stream 句柄查询 | 0.012524 ms | `1.2944x` |

两种查询得到同一 stream handle，residual 与 normalized 输出 bit-exact。

## 当前完整优化栈的小样本门禁

A/B 基线固定为 grouped-acBLAS GDN + residual-RMSNorm + GDN gate-prep + 单入口
acBLAS packed-MLP，仅切换 stream 查询方法。

| 数据 | exact | Accuracy | 成对中位/均值 | 候选获胜 | 门禁 |
|---|---:|---:|---:|---:|---:|
| 固定 128 token，8 对 | 8/8 | 不适用 | `1.1055x / 1.1007x` | 8/8 | 通过 |
| 固定 128 token，16 对 | 16/16 | 不适用 | `1.0961x / 1.0833x` | 15/16 | 通过 |
| CN20，第 1 轮 | 20/20 | 85% / 85% | `1.1026x / 1.0901x` | 17/20 | 通过 |
| CN20，第 2 轮 | 20/20 | 85% / 85% | `1.0855x / 1.0930x` | 19/20 | 通过 |
| EN20，第 1 轮 | 20/20 | 90% / 90% | `1.0691x / 1.0502x` | 16/20 | 通过 |
| EN20，第 2 轮 | 20/20 | 90% / 90% | `1.0795x / 1.0775x` | 18/20 | 通过 |

固定长度和真实多样本各两轮均满足全文一致、准确率不下降、成对中位和均值均大于
`1.0x`。因此候选晋级完整中文公开集、profile、memcheck 与正式 wrapper smoke。

## 完整集与最终门禁

中文 MMBench 4029 样本完整 paired A/B 已通过：

| 指标 | 当前推荐栈 | + raw stream |
|---|---:|---:|
| Accuracy | 3374/4029（83.7429%） | 3374/4029（83.7429%） |
| 平均 TTFT | 119.722 ms | 119.867 ms |
| 平均吞吐 | 120.383 token/s | 131.107 token/s |

- 4029/4029 完整文本、答案和 token 数一致；
- 成对吞吐中位 `1.0906x`、均值 `1.0905x`；3817/4029 获胜；
- P05/P95 为 `0.9987x/1.1828x`，212 个单样本受运行噪声影响回落；
- 平均 TTFT 轻微波动 `+0.12%`，候选收益明确来自 decode，不宣称改善 TTFT。

英文完整 4029 样本正在运行；完成前只把中文完整集视为正式证据。

## Profile、安全与正式入口

- 16-token profile 两路输出 exact；`cudaLaunchKernel` 次数均为 14,118，说明候选
  没有改变设备 kernel 图或提交数量，只减少 Python/运行时查询开销。
- profile 中 `aten::to/_to_copy/empty_strided` 次数不变，累计 CPU 时间分别从
  `35.87/34.98/46.33 ms` 降至 `13.19/12.31/24.32 ms`；该插桩数据只解释趋势，
  性能结论以 4029 对未插桩评测为准。
- `hggc-memcheck` 返回 0，报告 `ERROR SUMMARY: 0 errors`；raw handle 与原
  Stream 对象句柄一致，输出 bit-exact。
- 正式 `benchmark_public.py` 单样本真实 Transformers/PPU smoke 通过公开校验；
  meta 记录 raw-stream 为 true，并得到 18 GDN、18 conv、49 RMSNorm、18 gated
  RMSNorm、6 Q/K-RoPE、24 MLP、18 grouped-GDN、24 residual-RMSNorm、18
  gate-prep、24 acBLAS packed-MLP 模块。

正式配置新增：

```bash
export SEU_PPU_RAW_STREAM_QUERY_ENABLE=1
```

开关默认关闭并带运行时能力检查；当前锁定的 PPU PyTorch 通过全部门禁后推荐显式启用。

## 证据

- `results/raw-stream-query-ab128-r1-20260828.json`
- `results/raw-stream-query-ab128-r2-20260828.json`
- `results/raw-stream-query-cn20-r1-20260828.json`
- `results/raw-stream-query-cn20-r2-20260828.json`
- `results/raw-stream-query-en20-r1-20260828.json`
- `results/raw-stream-query-en20-r2-20260828.json`
- `results/raw-stream-query-cn4029-summary-20260828.json`
- `results/raw-stream-query-profile-ab-20260828.json`
- `results/raw-stream-query-profile-baseline-summary-20260828.json`
- `results/raw-stream-query-profile-candidate-summary-20260828.json`
- `results/raw-stream-query-memcheck-20260828.txt`
- `results/raw-stream-query-formal-wrapper-smoke-20260828.json`
