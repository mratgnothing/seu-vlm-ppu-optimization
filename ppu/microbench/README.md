# Qwen3.5-2B PPU BF16 GEMV 微基准

本目录准备了一个正确性优先的 HGGC 参考 kernel，用于在主办方提供的隔离
PPU 环境中测量 Qwen3.5-2B 解码阶段关键 BF16 矩阵向量乘尺寸：

| `N` | `K` | 对应模型位置 |
|---:|---:|---|
| 6144 | 2048 | GDN `in_proj_qkv`、MLP `gate_proj/up_proj` |
| 2048 | 6144 | MLP `down_proj` |
| 2048 | 2048 | GDN `z_proj/out_proj` 等 |

## 边界

- 这是单请求、单向量、BF16 输入、FP32 累加/输出的 reference kernel。
- 它用于建立正确性、延迟和访存下界，不是生产级融合 kernel，也不是现成优化收益。
- 代码根据 PPU SDK 2.1 的 HGGC/CUDA 兼容接口编写，但尚未在隔离 PPU
  资源上编译运行；不能把仓库中的“已准备”写成“已验证”。
- 不在共享节点上传或运行本目录。等待主办方明确提供个性化隔离资源后再执行。

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

## 运行

```bash
./run_qwen35_suite.sh
```

可通过环境变量调整设备、预热和迭代次数：

```bash
DEVICE=0 WARMUP=20 ITERATIONS=200 ./run_qwen35_suite.sh
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
