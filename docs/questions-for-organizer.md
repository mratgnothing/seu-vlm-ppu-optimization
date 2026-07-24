# 需要向主办方确认的问题

更新时间：2026-07-24

以下问题已经压缩为会直接改变技术路线的最小集合，可原样发送：

> 1. 当前共享测试节点的 PPU SDK 可以运行 kernel，但 Python 环境没有安装 PyTorch、Transformers 和 vLLM。复赛个性化资源是否会提供完整的 PPU 定制 PyTorch/推理镜像？对应镜像名称或版本是什么？
>
> 2. 当前节点保留的 PPU-vLLM 源码版本为 0.8.5+cu126，未注册 `Qwen3_5ForConditionalGeneration`，也未发现 Qwen3.5 的 Gated Delta Network 实现。主办方是否会提供支持 Qwen3.5-2B 的新版 PPU-vLLM、参考实现或补丁？
>
> 3. 个性化资源何时开放？是否允许参赛队上传 Qwen3.5-2B 权重、公开评测数据和自定义 C++/HGGC/Triton kernel，并在隔离目录中安装 Python 依赖？
>
> 4. 量化/剪枝方面，正式评测允许哪些权重变换和格式？运行时 BF16/FP8/INT8/INT4、KV Cache 量化、AWQ/GPTQ、TorchAO 或自定义量化分别是否允许？
>
> 5. 初赛统一复现的标准环境是否与当前 PPU 镜像一致？参赛代码的启动命令、依赖安装时限、模型加载时限、网络权限和持久化目录限制是什么？

## 为什么必须确认

- 问题 1-2 决定走现成 PPU-vLLM、Transformers eager，还是自行移植 GDN。
- 问题 3 决定何时能够建立 PPU 真实 baseline。
- 问题 4 决定能否投入量化路线。
- 问题 5 决定提交包和一键复现脚本的形式。

