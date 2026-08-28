# acBLAS 运行时开销与主要矛盾验证

日期：2026-08-28

## 结论

当前完整优化栈的主要矛盾已经收敛到 **单 token 仍需 120 次 BF16 小 GEMV**，而不是
Python stream 查询、acBLAS workspace 大小或单个输出张量分配。16-token baseline trace
包含 `5705` 次 `cudaGetDeviceProperties_v2`、`3259` 次 `cudaFree` 和 `14003` 次
`cudaLaunchKernel`。其中 15 个 decode step × 120 GEMV/step 与 acBLAS 内部重复设备查询
高度吻合。

本轮前三个候选均全文 bit-exact，但没有通过两轮整模性能门禁。随后新增的单-GEMV
候选通过了性能与答案准确率门禁，但改变了 BF16 累加/tiling 路径，因此作为默认关闭的
accuracy-budget 开关保留，不能称为 bit-exact 优化。
四个候选的最小真实张量路径均通过 `hggc-memcheck --tool memcheck`，每项
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

## 候选四：四路 GDN projection 合为一次 GEMV

四个权重已经按 `qkv/z/b/a = 6144/2048/16/16` 连续存成 `[8224,2048]`，且四路
输入相同。因此候选直接提交一次 `8224×2048` BF16 GEMV，再将连续输出切片，数学表达
不变，但 SDK 会因矩阵行数变化选择不同 tile/归约顺序。

- 模块：相对当前 four-GEMV grouped 路径 `0.02750→0.01894 ms`，约 `1.45x`；随机
  smoke 的四个输出逐元素 exact；
- fixed-128 r1（12 对）：中位 `1.01835x`、均值 `1.02681x`、10/12 胜、全文 exact；
- fixed-128 r2（16 对）：中位 `1.01123x`、均值 `1.01586x`、11/16 胜、全文 exact；
- 16-token trace：`gemvt_op 2446→1636`，恰少 `810=18×3×15`；设备查询
  `5705→4085`、`cudaFree 3259→2449`；
- CN20：Accuracy `0.85→0.85`、答案 20/20 一致、全文 19/20 一致；
- CN100：Accuracy `0.93→0.93`、答案 100/100 一致、全文 99/100 一致；成对中位
  `1.02610x`、均值 `1.02435x`、68/100 胜。

漂移样本 `3001005` 重复 10 对后，基线固定为 43 token，候选固定为 44 token，二者答案
始终为 D，证明这是确定性数值路径差异。保留 qkv、只合并 z/b/a 的两-GEMV折中仍产生
相同漂移，因此停止继续拆小分组。正式入口为
`SEU_PPU_ACBLAS_GDN_SINGLE_GEMV_ENABLE=1`，默认 0；它只适用于允许答案级精度预算的
提交配置，精度优先配置仍使用原 four-GEMV grouped 路径。正式 `evaluation_wrapper`
入口已在真实模型上验证：环境变量启用后报告 `True / 18 / acblas-grouped`，即 18 个
GDN 层全部完成 patch；single-GEMV 最小张量路径 memcheck 为 0 errors。

## 低精度可行性探针

SDK 头文件虽然声明 INT8/FP8 和 acBLASLt scale 属性，真实 heuristic 查询显示：

- BF16×BF16→BF16：16 个算法；
- INT8×BF16→BF16（A16W8）：0 个算法，`ACBLAS_STATUS_INTERNAL_ERROR`；
- FP8×BF16→BF16：0 个算法；
- INT8×INT8→INT32：16 个算法。

A8W8 的裸 INT8 GEMV 仅在最大投影上有约 `1.3x`，在 `16×2048`、`2048×2048`
等小投影反而慢于 BF16；动态激活量化和反量化尚未计入。因此不实现全模型 A8W8，
weight-only 仍需厂商 acext/PPU-vLLM 后端。

## 候选五：只合并 GDN b/a

为避免 8224 行 single-GEMV 的文本漂移，进一步只把 packed buffer 中相邻的
`in_proj_b/in_proj_a` 两个 `[16,2048]` 权重作为一次 `[32,2048]` GEMV；qkv 与 z
仍保持原始 `6144/2048` 行 GEMV，因此每层从 4 次降为 3 次。

- 模块随机输入 decode/prefill 逐元素 exact；相对 four-GEMV grouped 路径约
  `1.52x`；
- fixed-128 r1（12 对）：中位 `1.00107x`、均值 `1.00918x`、6/12 胜、全文 exact；
- fixed-128 r2（16 对）：中位 `1.00512x`、均值 `1.01755x`、12/16 胜、全文 exact；
- CN100：Accuracy `0.93→0.93`、100/100 完整文本一致，中位 `1.00685x`、均值
  `1.00828x`、58/100 胜。

该候选收益小但两轮 fixed 与 CN100 方向一致，正式入口为默认关闭的
`SEU_PPU_ACBLAS_GDN_BA_GEMV_ENABLE=1`，且与 single-GEMV 性能档互斥。当前只把它列为
精度优先小增量；完整中英文 4029 门禁前不宣称全量无损。
正式 wrapper 已验证 18/18 个 GDN 层启用，模块 memcheck 为 0 errors。

## 8224 行算法与 GemmEx 替代负实验

对 single-GEMV 的 `8224×2048` 扫描 `-3/-2/-1/0..23/99` 共 28 个算法，全部 exact，
默认 `-1` 为 `0.020365 ms`，最佳算法 20 为 `0.020292 ms`，只差 `1.0036x`，不进入
整模。将同一向量乘改写为 `acblasGemmEx(m=8224,n=1,k=2048)` 后，延迟仍约
`0.02031 ms`；300 次 profiler 中两条路径都产生 600 次设备属性查询和 300 次释放。
因此算法枚举和 GemmEx-for-GemvEx 均未改变主要矛盾，停止。

## 后续主线

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

在依赖到位前，不再继续堆叠逐层 scratch 或用通用 batched GEMM 替换 GEMV。单-GEMV
作为显式 accuracy-budget 候选保留；其他本轮 A/B 开关只用于复现负实验。
