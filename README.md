# 东南大学 AI+创新应用大赛：VLM PPU 推理优化

本仓库用于赛道二“面向 AI 芯片的 VLM 高效推理与优化”的两人协作开发。目标是在主办方指定的 Qwen3.5-2B 和 PPU 环境中，守住模型精度并降低 TTFT、提高解码吞吐，最终交付可复现代码和性能报告。

## 当前状态

- 已导入 `dndx_participant-v1.1` 的公开评测入口和初始 wrapper。
- 本地开发环境锁定为 Python 3.12、PyTorch 2.13.0+cu130、Transformers 5.14.1。
- 尚未提交模型权重、评测数据、真实 baseline 或 PPU 优化结果。
- `dummy` 后端只用于接口冒烟；不得将其结果视为真实模型部署或比赛成绩。

## 仓库结构

```text
.
├─ benchmark_public.py       # 主办方 v1.1 公开评测入口
├─ evaluation_wrapper.py     # 主要优化入口
├─ requirements.txt          # 主办方基础依赖
├─ README_ORGANIZER.md       # 主办方 v1.1 说明
├─ configs/                  # 可复现实验配置
├─ data/                     # 本地数据说明，数据文件不入库
├─ docs/                     # 规则、协作和实验记录
├─ models/                   # 本地模型说明，权重不入库
├─ results/                  # 可公开的小型汇总，原始结果不入库
├─ scripts/                  # 运行脚本
└─ tests/                    # 无模型依赖的接口测试
```

## 第一次运行

1. 建立本地 CUDA 环境：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\bootstrap_local.ps1
   ```

2. 下载并校验已锁定 revision 的官方模型：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\download_model.ps1
   ```

3. 把公开数据放在仓库外，或放入已被忽略的 `data/`。
4. 先做 dummy 冒烟：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\run_benchmark.ps1 `
     -DatasetPath "D:\path\to\mmbench_dev_cn.tsv" `
     -Backend dummy `
     -NumSamples 2
   ```

5. 再做真实模型小样本基线：

   ```powershell
   powershell.exe -NoProfile -ExecutionPolicy Bypass `
     -File .\scripts\run_benchmark.ps1 `
     -DatasetPath "D:\path\to\mmbench_dev_cn.tsv" `
     -ModelPath "E:\models\Qwen3.5-2B" `
     -Backend transformers `
     -NumSamples 20
   ```

真实实验必须检查结果中的 `backend` 字段，禁止把自动回退的 dummy 结果当成真实基线。

运行真实 benchmark 前，先做不生成文本的模型加载门禁：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\smoke_model_load.ps1
```

## 协作规则

- `main` 始终保持可复现。
- 每项实验使用独立分支：`baseline/...`、`opt/...`、`ppu/...`、`docs/...`。
- 通过 Pull Request 合并；PR 必须附 Accuracy、TTFT、Throughput 的前后对比。
- 不在一个提交中混合无关优化。
- 实验配置、环境和失败结果同样需要记录。

## 安全边界

以下内容不得提交，即使仓库是私有的：

- `key.pem`、SSH 私钥、访问令牌和 `.env`。
- Qwen3.5-2B 权重及其他大模型文件。
- 群聊原始记录、测试环境截图、培训 PDF 和原始压缩包。
- 未确认允许再分发的评测数据。
- 含私有测试答案或反作弊敏感内容的产物。

详细规则见 [docs/rules-and-boundaries.md](docs/rules-and-boundaries.md)。
