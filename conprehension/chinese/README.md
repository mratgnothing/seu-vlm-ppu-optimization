# 中文资料离线入口

更新时间：2026-08-12

本目录保存可以离线阅读的中文学习材料。外部翻译不属于本项目原创内容；涉及
CUDA API、硬件行为和版本差异时，以 NVIDIA 英文原文为最终依据。

第三方中文教程副本和网页 PDF 仅保存在本机，并由 `.gitignore` 排除，避免随项目
仓库再分发；本项目原创的第一阶段讲义可以正常进入版本控制。

## 从这里开始

1. 先读 [第一阶段：模型推理与评测入门](stage-1-model-inference.md)。这是针对本项目
   编写的中文讲义，不要求预先懂 CUDA。
2. 再读 [CUDA Programming Guide 中文源码](cuda-programming-guide-zh/README.md)。
   这是 `bearneck/cuda-programming-guide-zh` 的离线副本，正文在 `docs/`，共 112 个
   Markdown/资源文件；在线版本声明使用 AI 辅助翻译并经人工审校。
3. 需要快速浏览时，打开
   [CUDA 中文指南首页 PDF](../pdf/cuda-programming-guide-zh.pdf)。该 PDF 只是目录和
   核心概念速查，不是完整 112 章内容。
4. 准备进入主办方服务器时，读
   [PPU SDK 快速入门 PDF](../pdf/ppu-sdk-quick-start-zh.pdf)。这是阿里云官方中文页面
   的离线打印版。

## CUDA 中文源码怎么读

不需要先安装网站工具，直接用编辑器打开 `cuda-programming-guide-zh/docs/` 下的
Markdown 文件即可。推荐顺序：

1. 第一部分的简介和编程模型；
2. 第二部分的 CUDA C++ 入门；
3. 编写 CUDA SIMT 内核；
4. 技术附录中的内存模型和执行模型；
5. 等开始写算子后，再按问题查高级章节。

如果希望还原成网页，在该资料目录执行：

```bash
python -m pip install mkdocs mkdocs-material mkdocs-minify-plugin
mkdocs serve
```

## 来源与边界

- CUDA 中文源码来源：<https://github.com/bearneck/cuda-programming-guide-zh>
- CUDA 中文在线版：<https://bearneck.github.io/cuda-programming-guide-zh/>
- NVIDIA 英文原文：<https://docs.nvidia.com/cuda/cuda-programming-guide/>
- PPU SDK 官方中文原文：<https://help.aliyun.com/zh/document_detail/3030340.html>
- 本地获取日期：2026-08-12

PPU SDK 快速入门页面中含有旧版 SDK 的示例版本和历史安装命令。实际比赛服务器
必须以主办方镜像、服务器 `/usr/local/PPU_SDK`、当前驱动和配套文档为准，不要直接
照抄旧版下载地址覆盖服务器环境。
