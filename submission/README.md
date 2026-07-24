# 提交候选包

当前尚未拿到主办方最终提交格式和标准环境限制，因此这里只生成“源码候选包”，
不把它直接称为最终提交包。

在项目根目录运行：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\build_submission.ps1
```

默认输出：

```text
artifacts/submission-source.zip
```

打包器使用显式白名单，只包含运行代码、配置模板、依赖文件、技术文档、测试和 PPU
微基准。以下内容不会进入 ZIP：

- 模型权重和模型缓存；
- MMBench TSV、派生数据和图片；
- 逐样本结果、日志、profile 和 artifacts；
- `.env`、`key.pem`、`configs/local.psd1` 等凭据或本地路径；
- Git 历史、虚拟环境、缓存和编译产物；
- PDF、压缩包和其他主办方原始附件。

ZIP 内包含 `MANIFEST.sha256`。生成后脚本会重新打开 ZIP，验证路径集合、每个文件的
SHA-256 和固定时间戳，再输出整个 ZIP 的 SHA-256。

收到主办方最终提交说明后，需要再确认：

1. 是否允许或要求提交测试、实验文档和 PPU 微基准；
2. 模型权重是由评测环境预置，还是需要单独提交；
3. 数据路径、启动命令和依赖安装方式；
4. 报告/论文的文件格式和命名；
5. 源码包大小、目录层级和压缩格式限制。
