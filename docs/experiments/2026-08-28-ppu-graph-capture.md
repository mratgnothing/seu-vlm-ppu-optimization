# 2026-08-28 PPU Graph Capture 能力与 packed-MLP 止损实验

## 问题

SDK 2.1.1 的 HGGC 头文件包含完整 graph/stream-capture 类型，但这不能证明当前
PyTorch PPU 端已经接通，也不能证明 graph replay 能加速真实模型边界。本实验分两级：

1. 固定 BF16 张量上的 16 段 SiLU 链，验证 capture、输入更新、replay 和数值；
2. 对已通过中英文 4029 门禁的单入口 acBLAS packed-MLP 做 capture，比较现有入口与
   graph replay，并单独计入动态输入地址所需的 copy 成本。

所有测试都只使用随机/规则生成张量，不读取公开集答案。

## 结果

PPU-ZW810E、PyTorch `2.11.0+v0.1.0.ppu2.1.1`：

| 边界 | eager/现有路径 | graph | 速度比 | 数值 |
|---|---:|---:|---:|---|
| 65,536 元素 × 16 段 SiLU | 0.153274 ms | 0.083743 ms | `1.8303x` | exact |
| 单入口 packed-MLP，稳定输入地址 | 0.044214 ms | 0.043336 ms | `1.0203x` | exact |
| 单入口 packed-MLP，含 input copy | 0.044214 ms | 0.047462 ms | `0.9316x` | exact |

生产式 smoke 还验证了：更新稳定输入内容后 replay 正确、动态指针回退正确、prefill
回退正确、非绑定 stream 被拒绝，最大绝对误差为 0。低开销 stream 守卫使用当前 PPU
PyTorch 已暴露的 `_cuda_getCurrentRawStream`，避免每 token 构造 Stream Python 对象。

## 结论

HGGC Graph 在当前实例上确实可用，不是只有头文件；对包含许多细碎算子的固定子图可
显著减少 launch/dispatcher 开销。但 packed-MLP 已经被压缩为一次 C++ extension 入口，
再套 graph 只剩约 2% 的模块级收益，低于 3% 晋级余量；一旦输入地址不稳定而增加一次
copy，候选反而慢约 6.8%。

因此不为 graph 强行改造 24 层 residual-RMSNorm scratch，不运行整模公开集，也不改变
当前推荐配置。下一次只有在官方 PPU-vLLM/FLA 提供固定 KV/page 地址，或能捕获完整
decoder layer/多层 decode 子图时，才重新评估 graph；不能用本次 1.83x 的合成链结果
外推模型吞吐。

## 复现

```bash
python ppu/microbench/probe_ppu_graph_capture.py --repeats 100

cd ppu/custom_ops
SEU_PPU_GDN_LIBRARY=/path/to/libseu_ppu_gdn.so \
  python smoke_acblas_packed_mlp_graph.py \
  --build-dir build/acblas_packed_mlp_extension --repeats 500
```

证据：

- `results/ppu-graph-capture-probe-20260828.json`
- `results/acblas-packed-mlp-graph-smoke-20260828.json`
