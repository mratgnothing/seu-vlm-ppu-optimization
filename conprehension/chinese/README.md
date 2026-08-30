# 中文资料离线入口

更新时间：2026-08-19

本目录把比赛需要的中文资料按“学习—移植—优化—调试”整理在一起。外部翻译和
官方网页的离线副本由 `.gitignore` 排除，不会随仓库再分发；本项目原创讲义可以
正常进入版本控制。CUDA 中文翻译遇到 API、硬件行为或版本差异时，以 NVIDIA 最新
英文原文为最终依据。

## Ubuntu 一键下载

切换到 `5070ti` 分支，在仓库根目录执行：

```bash
chmod +x scripts/download_chinese_tutorials.sh
./scripts/download_chinese_tutorials.sh
```

脚本需要 `git` 和 `curl`：

```bash
sudo apt-get update
sudo apt-get install -y git curl
```

脚本会下载完整 CUDA 中文编程指南、CUDA 中文性能优化资料，以及 PPU SDK v2.1.x
官方中文核心开发文档。重复运行会覆盖网页副本并更新 Git 形式的 CUDA 指南。

## 本地资料在哪里

### 1. 项目和模型推理入门

- [第一阶段：模型推理与评测入门](stage-1-model-inference.md)：本项目原创中文讲义，
  第一次接触模型部署时先读它。
- `../README.md`：比赛理解、技术路线、本地与服务器边界、环境配置和已有实验入口。

### 2. CUDA 中文教程与开发参考

- [完整 CUDA Programming Guide 中文源码](cuda-programming-guide-zh/README.md)：正文位于
  `cuda-programming-guide-zh/docs/`，可直接用 VS Code 阅读。这是社区维护、AI 辅助翻译
  并经人工审校的完整离线资料，不是 NVIDIA 官方译本。
- [CUDA Best Practices 中文目录](external/cuda/best-practices-index.html)：旧版社区译本，
  当前中文内容主要集中在内存优化。
- [CUDA 内存优化中文章节](external/cuda/best-practices-memory-optimization.html)：包含页锁定
  内存、异步传输与计算重叠、零拷贝、合并访存、共享内存和 bank conflict。
- [NVIDIA CUDA 中文入门](external/cuda/nvidia-cuda-intro-cn.html)：NVIDIA 开发者中文站的
  入门文章；同目录另有编程模型和接口两篇中文文章。
- [CUDA 中文指南首页 PDF](../pdf/cuda-programming-guide-zh.pdf)：只有目录和核心概念速查，
  不是完整指南；系统学习请用上面的 Markdown 源码。

推荐顺序：编程模型与线程层级 → CUDA C++ 入门 → SIMT kernel → 全局/共享/寄存器内存
→ 同步与异步拷贝 → 正确性测试 → 合并访存、共享内存和占用率优化。

如果希望把完整指南作为网页阅读，在资料目录运行：

```bash
cd conprehension/chinese/cuda-programming-guide-zh
python -m pip install mkdocs mkdocs-material mkdocs-minify-plugin
mkdocs serve
```

浏览器打开终端显示的本地地址即可；不需要联网。

### 3. PPU 官方中文教程与开发参考

PPU 网页副本都在 `external/ppu-sdk-v2.1/`。文件名前缀就是建议使用顺序：

| 阶段 | 先看这些文件 | 解决的问题 |
|---|---|---|
| 登录服务器、确认环境 | `02-release-notes.html`、`03-quick-start.html`、`30-ppu-environment-check.html`、`31-ppu-smi.html` | SDK/驱动版本、设备是否可见、样例是否能跑 |
| 评估 CUDA 代码能否直接迁移 | `04-compatibility-index.html`、`10`～`22` 兼容性文件 | CUDA API、编译选项、cuBLAS/cuDNN/NCCL 等库支持范围 |
| 编写或移植算子 | `05-programming-guide-v1.4.html`、`06-compiler-guide.html`、`07-hgrtc-api.html` | CUDA 思路迁移、`.cu`/`.hggc` 编译、HGGC/ppu-clang、运行时编译 |
| 找整网性能瓶颈 | `40`～`43` Asight Systems 文件 | CPU/PPU 时间线、kernel launch、拷贝、同步和空洞 |
| 优化热点算子 | `44`～`46` Asight Compute 文件 | 单 kernel 指标、访存/计算瓶颈、`acu` 命令行采集和 GUI 分析 |
| 查错误与非法访存 | `50-gdb.html`、`51-memcheck.html` | 断点、调用栈、越界、未初始化访问和同步问题 |
| 高级工具链分析 | `60`～`63` 文件 | 反汇编、裁剪、fat binary、链接和 JIT link |
| 多卡与通信 | `70`～`74` 文件 | PCCL 异常定位、P2P 测试、拓扑顺序、带宽与 DeepEP |
| 服务器监控 | `32-dcgm.html` | 设备遥测和运行状态 |

官方目录里的量化、OpenCV、DALI、Open3D、视频、Firmware/KMD、虚拟化和容器隔离等
专项资料也已完整保存在 `23`～`39`、`75`～`76` 文件中。它们当前不是单卡大模型
算子优化的第一阅读优先级，但需要对应功能时无需再联网查找。

本地已有的 [PPU SDK 快速入门 PDF](../pdf/ppu-sdk-quick-start-zh.pdf) 适合随手浏览；
`external/ppu-sdk-v2.1/03-quick-start.html` 是脚本重新抓取的当前官方网页副本。

## 在比赛工程里具体怎么用

1. **先跑通**：登录主办方服务器，按 PPU `release notes → quick start → environment
   check → PPU-SMI` 检查环境，不要自行覆盖主办方预装 SDK。
2. **先查兼容性**：看到项目里的 CUDA API、cuBLAS/cuDNN/NCCL 或 `nvcc` 参数时，先在
   `10`～`22` 文件中查是否支持，再决定直接编译、替换 API，还是写自定义算子。
3. **再写算子**：CUDA 中文指南负责线程组织、张量索引、访存和同步；PPU compiler
   guide 负责把同一思路落到 `.cu`/`.hggc` 和 PPU 工具链。
4. **先验证正确性**：用小张量和 PyTorch/CPU 参考结果比较，再用 GDB/Memcheck 查越界。
5. **最后优化**：先用 Asight Systems 找到模型级热点，再用 Asight Compute 分析那个
   kernel；此时再回查 CUDA Best Practices 的合并访存、共享内存和传输重叠章节。

LoRA、QLoRA、AWQ、GPTQ、SmoothQuant 等论文保持英文原版，位于 `../pdf/`，不在本次
中文资料下载范围内。

## 在线入口与版本边界

- CUDA 中文社区指南：<https://bearneck.github.io/cuda-programming-guide-zh/>
- CUDA 中文指南源码：<https://github.com/bearneck/cuda-programming-guide-zh>
- CUDA Best Practices 中文旧译：<https://cuda-doc.readthedocs.io/zh-cn/latest/CUDA-C-Best-Practices-Guide/index.html>
- NVIDIA CUDA 中文入门：<https://developer.nvidia.cn/blog/cuda-intro-cn/>
- NVIDIA 最新 CUDA 英文原文：<https://docs.nvidia.com/cuda/cuda-programming-guide/>
- PPU SDK v2.1.x 官方中文目录：<https://help.aliyun.com/zh/document_detail/3029921.html>
- PPU SDK 快速入门：<https://help.aliyun.com/zh/document_detail/3030340.html>
- PPU CUDA 兼容性目录：<https://help.aliyun.com/zh/document_detail/3029924.html>

下载日期为 2026-08-19。PPU 文档和比赛镜像可能继续更新，服务器实测时以主办方镜像、
`/usr/local/PPU_SDK`、实际驱动版本和当期官方文档为准。PPU 源码兼容不代表 GPU
预编译二进制可以直接在 PPU 上运行。
