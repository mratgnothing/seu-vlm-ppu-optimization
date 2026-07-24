# 共享 PPU 环境核验

核验时间：2026-07-24

## 资源

- 节点形态：阿里云 PAI DSW 共享容器
- 容器系统：Ubuntu 24.04.4 LTS
- CPU：44 vCPU
- 内存：440GiB
- 根目录可用空间：约 1.1TiB
- PPU：4 × PPU-ZW810E
- 单卡显存：97920MiB
- PPU-SMI：1.28
- Driver：1.3.2-d7f5a2
- HGGC：13.0

## SDK

- PPU SDK：2.1.0-a5f865
- SDK 路径：`/usr/local/PPU_SDK`
- CUDA 兼容工具链：12.6
- 编译/诊断工具包括 `clang++ -x hggc`、`hgcc`、`hggc-memcheck`、`ppu-gdb`
- 分析工具包括 `asys`、`acu`、`transProfiler_tool`
- 系统 Python：3.12.3
- 基础环境未发现 PyTorch、Transformers、vLLM 或 SGLang

## 官方样例实测

节点自带 `vectorAdd.hg` 已在临时目录完成编译和运行，输出 `Test PASSED`。

原始 `samples/build.sh` 使用 `-rpath`，当前宿主 g++ 会报：

```text
g++: error: unrecognized command-line option '-rpath'
```

改为标准链接器参数后成功：

```bash
-Wl,-rpath,/usr/local/PPU_SDK/lib
```

该结果证明 PPU 驱动、HGGC 编译和基础 kernel 执行链可用；尚未证明 PyTorch/Transformers 或 Qwen3.5-2B 已在 PPU 上部署。

## 安全边界

- 当前是公开共享节点，只运行主办方自带样例和只读环境探测。
- 不上传模型权重、公开数据、正式代码或其他密钥。
- 正式模型部署等待个性化资源或主办方明确允许的隔离目录。

