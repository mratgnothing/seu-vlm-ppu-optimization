# acBLAS 运行时开销与主要矛盾验证

日期：2026-08-28

## 结论

当前完整优化栈的主要矛盾已经收敛到 **单 token 仍需 120 次 BF16 小 GEMV**，而不是
Python stream 查询、acBLAS workspace 大小或单个输出张量分配。16-token baseline trace
包含 `5705` 次 `cudaGetDeviceProperties_v2`、`3259` 次 `cudaFree` 和 `14003` 次
`cudaLaunchKernel`。其中 15 个 decode step × 120 GEMV/step 与 acBLAS 内部重复设备查询
高度吻合。

本轮所有候选均全文 bit-exact，但没有候选通过两轮整模性能门禁，因此正式路径保持不变。
三个候选的最小真实张量路径均通过 `hggc-memcheck --tool memcheck`，每项
`ERROR SUMMARY: 0 errors`；memcheck 下的计时只用于触发访问，不作为性能结论。

## 候选一：持久 acBLAS workspace

实现方式：由 PyTorch `uint8` PPU tensor 持有设备内存，通过
`acblasSetWorkspace_v2` 分别绑定 grouped-GDN 与 packed-MLP 的进程级 handle；默认关闭。

模块扫描：

- grouped-GDN：4/16/64 MiB 的 workspace speedup 分别为
  `0.9914x / 1.0873x / 1.0060x`，仅 16 MiB 单轮为正，尺寸趋势不一致；
- packed-MLP：分别为 `0.9983x / 0.9996x / 0.9640x`，没有收益；
- 所有模块输出 exact。

固定 128-token、8 对交错 A/B：

- 中位 `0.9846x`，均值 `1.0027x`，3/8 胜；
- profiler：设备查询 `5705→5705`、释放 `3259→3259`、kernel `14003→14003`。

判定：workspace 没有改变 GemvEx 的内部查询/释放路径，停止，不跑公开集。

## 候选二：GDN b/a strided-batched GEMM

`in_proj_b` 与 `in_proj_a` 均为 `[16,2048]`，权重和输出在 packed buffer 中连续，因此
可用一次 `acblasGemmStridedBatchedEx(batch=2, strideB=0)` 代替两次 GemvEx。

- 模块：exact，`1.0241x`；
- 固定 128-token、12 对：中位 `0.9852x`，均值 `0.9870x`，4/12 胜；
- profiler：设备查询 `5705→4895`，释放 `3259→2989`，但 kernel 数仍为
  `14003→14003`。

判定：确实减少 270 次 acBLAS 调用对应的内部 host/runtime 事件，但 batched 路径仍发出
同等数量 device kernel，整模反而下降约 1.5%。这证明“只减少 host API 次数”不是当前
主要矛盾。

## 候选三：grouped-GDN 持久输出 scratch

每个 GDN 层持有一个 `[1,1,8224]` BF16 输出 buffer，由新的
`gdn_projections_bf16_into` 写入；候选限制在 patch 时的单一 stream，默认关闭。

- 模块：exact，`1.0432x`；
- 固定第一轮 12 对：中位 `1.0105x`、均值 `1.0046x`；
- 固定第二轮 16 对：中位 `0.9934x`、均值 `0.9899x`；
- profiler 的设备查询、释放和 kernel 数均未改变。

判定：两轮方向相反，属于模块微基准假阳性，停止，不跑公开集。

## 真正下一步

阿里云官方 SDK 2.1 文档明确列出 acext 的 A16W8/A16W4 与小 batch
`WeightOnlyBatchedGemv`，这类 kernel 能把主要 MLP/GDN 权重流量降为 1/2 或 1/4，才有
机会实质改变 120 个 memory-bound GEMV 的成本：

- <https://help.aliyun.com/zh/document_detail/3030284.html>
- <https://help.aliyun.com/zh/document_detail/3030339.html>

但当前比赛镜像中不存在 `libacext.so`、`acext` 头文件或 wheel；公开 PyPI 也没有
`acext`/`ppu-acext` 分发包，官方源码地址指向不可公开访问的内网仓库。因此下一轮主线
应先向主办方获取以下任一项：

1. 与 SDK 2.1.1 匹配的 acext wheel/头文件/动态库；
2. 已编译 `BUILD_WEIGHTONLY_KERNELS=1` 的 PPU-vLLM；
3. 官方 Qwen3.5 PPU 推理镜像及 weight-only 量化工具。

在依赖到位前，不再继续堆叠逐层 scratch 或用通用 batched GEMM 替换 GEMV。所有本轮
A/B 开关只用于复现负实验，正式评测配置保持关闭。
