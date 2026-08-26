# Qwen3.5-2B PPU BF16 GEMV 微基准

本目录准备了一个正确性优先的 HGGC 参考 kernel，用于在主办方提供的隔离
PPU 环境中测量 Qwen3.5-2B 解码阶段关键 BF16 矩阵向量乘尺寸：

| `N` | `K` | 对应模型位置 |
|---:|---:|---|
| 6144 | 2048 | GDN `in_proj_qkv`、MLP `gate_proj/up_proj` |
| 2048 | 6144 | MLP `down_proj` |
| 2048 | 2048 | GDN `z_proj/out_proj` 等 |

## 已验证结论（2026-08-26）

已在隔离 PPU-ZW810E、SDK 2.1.1 上编译并运行。`warp_vec2` 使用 BF16x2 加载和
warp 级归约；16 份权重轮换、32 次预热、200 次计时、3 次复测均通过数值校验。

| N x K | reference ms | 最佳 `warp_vec2` ms | 加速 | `torch.mv` ms |
|---|---:|---:|---:|---:|
| 6144 x 2048 | 0.050721 | 0.024431 | 2.076x | 0.017053 |
| 2048 x 6144 | 0.045066 | 0.023950 | 1.882x | 0.014727 |
| 2048 x 2048 | 0.018236 | 0.009371 | 1.946x | 0.008029 |

优化核仍慢于 PPU PyTorch `torch.mv`，且当前自定义输出 FP32、PyTorch 输出 BF16，
因此不接入模型。完整边界和 profile 见
[首次 PPU 实验](../../docs/experiments/2026-08-26-ppu-baseline-and-gemv.md)。

## 边界

- 这是单请求、单向量、BF16 输入、FP32 累加/输出的 reference kernel。
- 它用于建立正确性、延迟和访存下界，不是生产级融合 kernel，也不是现成优化收益。
- 代码已在个性化隔离 PPU 验证；这些数字不代表私有评测或端到端收益。
- reference/warp 结果只证明微基准正确性与优化方向，不能替代模型 Accuracy/TTFT。

## 构建

```bash
cd ppu/microbench
chmod +x build.sh run_qwen35_suite.sh
./build.sh
```

默认 SDK 路径为 `/usr/local/PPU_SDK`，需要时可覆盖：

```bash
PPU_SDK_ROOT=/path/to/PPU_SDK ./build.sh
```

构建脚本使用已在官方 `vectorAdd` 样例中验证过的标准链接器写法
`-Wl,-rpath,/usr/local/PPU_SDK/lib`。
当前镜像还需要链接 `libhggcrt1.so`；可用 `HGGC_RUNTIME_LIBRARY` 覆盖库名。

## recurrent GDN 微基准

`qwen35_gdn_recurrent.hg` 锁定 Qwen3.5-2B decode 的 `16×128×128` FP32 state，
`build_gdn.sh` 生成独立正确性/计时程序；`torch_gdn_recurrent_baseline.py` 提供
同公式 PyTorch 对照并可读取 HGGC dump 做逐元素比较：

```bash
./build_gdn.sh
./build/qwen35_gdn_recurrent \
  --threads 128 --warmup 50 --iters 500 --dump-prefix /tmp/gdn
python torch_gdn_recurrent_baseline.py \
  --warmup 50 --iters 500 --hggc-dump-prefix /tmp/gdn
```

首版单 block/head 融合核最佳约 `0.03279 ms`，PyTorch eager 为
`0.151--0.157 ms`。生产接入版进一步复刻 BF16 L2Norm 舍入点并使用 4 tiles/head，
见 [ppu/custom_ops/README.md](../custom_ops/README.md)。

## 运行

```bash
./run_qwen35_suite.sh
```

可通过环境变量调整设备、预热和迭代次数：

```bash
DEVICE=0 WARMUP=20 ITERATIONS=200 ./run_qwen35_suite.sh
```

推荐重复冷工作集对照：

```bash
PYTHON=/path/to/ppu-python \
REPEATS=3 MATRIX_COPIES=16 WARMUP=32 ITERATIONS=200 \
./run_repeated_benchmark.sh | tee repeated.log

python ../../scripts/summarize_ppu_gemv.py repeated.log
```

也可以单独运行任意尺寸：

```bash
./build/qwen35_bf16_gemv --n 6144 --k 2048 --warmup 10 --iters 100
```

每个尺寸会输出：

- 平均 kernel 延迟；
- GFLOP/s；
- 按权重、输入和输出最低字节数计算的有效带宽；
- 与 CPU BF16 参考结果的最大绝对/相对误差；
- 一行便于解析的 `RESULT {...}` JSON。

进程只有在全部输出通过误差阈值时才返回 0。

## 隔离 PPU 上的验收顺序

1. `./build.sh` 编译通过。
2. 先用 `--iters 1 --warmup 0` 做冒烟测试。
3. 使用 `hggc-memcheck` 检查越界、未初始化访问和同步问题。
4. 运行三组默认尺寸，保存完整终端输出和环境快照。
5. 使用 `asys`/`acu` 获取访存、占用率和 kernel 热点。
6. 再与 Acompute/模型运行时的同尺寸算子对照；任何优化必须重新检查数值误差。

生产优化可从向量化加载、warp 级归约、权重布局、多个输出行/块、bias/激活融合
和 PPU 矩阵指令开始，但应由真实 profile 决定先后顺序。
