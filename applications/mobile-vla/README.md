# 智伴 Mobile GUI-VLA 前置工程

本目录用于验证“Qwen3.5-2B + Android + 可迭代本地知识库”的最小闭环。它与比赛提交仓库隔离，避免把手机原型、量化模型和用户数据混入已经冻结的 PPU 交付物。

## 当前已经准备

- Android Platform Tools 37.0.1：`tools/platform-tools/adb.exe`
- Android 本地构建工具链：`tools/android-build/`（工具说明与 Agent 索引入口见 [`docs/ANDROID_BUILD_TOOLS.md`](docs/ANDROID_BUILD_TOOLS.md)）
- MNN Chat Android 0.8.3：`artifacts/mnn_chat_0_8_3.apk`
- 官方 MNN 4-bit 模型：`models/Qwen3.5-2B-MNN`
- Android 设备预检、APK 安装、模型推送和截图脚本
- 结构化 GUI 动作协议与默认安全策略
- SQLite 本地知识库：人物关系、任务轨迹、工作流提案、人工审核和检索

模型来源为 `taobao-mnn/Qwen3.5-2B-MNN`，当前锁定 Hub revision：

```text
35781816d7b6a9dcb273a6765ac9563401951c3c
```

该包采用 4-bit、block size 64、HQQ 导出；语言模型约 1.2 GB，视觉权重约 196 MB。它用于 Android 可行性测试，不能替代比赛提交中 BF16/PPU 结果，也不能直接证明手机端性能。

## 最新真机验收结论

2026-09-21 在 Xiaomi 17 / Android 16 上完成了追加通信验收：手机本地 Qwen3.5-2B-MNN 严格选择 `call_bound_contact`，确定性安全门校验私有绑定后完成一次约 5 秒的真实电话并挂断。切换到微信后，MNN 会话与聊天指纹桥接不稳定，执行器按 fail-closed 策略停止；微信消息没有发送，微信视频没有启动。

因此当前原型已经证明受限电话链路可行，尚未证明三个通信动作都能由 Android Agent 稳定端到端完成。详细边界见 `docs/CURRENT_STATUS.md` 与 `evidence/phone-local-communication-attempt-2026-09-21.md`。

2026-09-22 又完成了手机端三步连续视觉点击：最终安装版中，本地模型根据三张连续截图依次选择 BLUE、GREEN、ORANGE 卡片，三次选择和点击全部通过，约 17.3 秒完成。当前采用四候选离散视觉动作空间并由目标边界门复核；自由坐标和复合标签仍不稳定，不能据此声称已经实现通用 Android GUI Agent。证据见 `evidence/safe-continuous-visual-loop-2026-09-22.md`。

仓库不包含量化模型、第三方工具、APK、签名材料、原始设备截图、真实联系人数据或 API 密钥。复现实验时需按下文步骤自行准备 MNN Chat、模型和 Android 构建环境。

## 第一次连接手机

1. 在手机“开发者选项”中开启 USB 调试。
2. 使用支持数据传输的 USB 线连接电脑。
3. 手机上确认这台电脑的调试授权。
4. 在 PowerShell 中运行：

```powershell
Set-Location 'C:\Users\ASUS\Downloads\zhi-ban-mobile-vla'
.\scripts\android_preflight.ps1 -OutputPath .\evidence\device-preflight.json
```

预检只读取设备信息，不安装应用、不推送模型、不修改系统设置。

## 安装 MNN Chat

手机连接并完成预检后，再执行：

```powershell
.\scripts\install_mnn_chat.ps1
```

如需把已下载的模型复制到手机下载目录：

```powershell
.\scripts\push_model.ps1
```

模型较大，推送前脚本会检查关键文件是否齐全，并显示目标路径。

## 初始化知识库

```powershell
python .\src\zhiman_kb.py --db .\data\zhiman.db init
python .\src\zhiman_kb.py --db .\data\zhiman.db summary
```

知识库采用“记录轨迹 → 形成提案 → 人工审核 → 发布工作流”的更新方式。一次成功操作不会直接改写已验证知识，失败轨迹也会保留，用于后续分析和微调数据生成。

## 启动受控 GUI 测试环境

先在一个 PowerShell 窗口启动本地测试网页：

```powershell
.\scripts\start_test_app.ps1
```

保持该窗口运行，再在另一个窗口执行：

```powershell
.\scripts\open_test_app.ps1
```

脚本通过 `adb reverse` 将手机的 `127.0.0.1:8877` 映射到电脑，只供当前 USB 调试设备访问。测试环境包含虚构联系人、复诊提醒、乘车码、可疑短信和支付交接页面。

## 连续录屏演示脚本

`scripts/run_continuous_demo.ps1` 用于已经取得当轮联系人授权的固定路演录屏。它不等待 Qwen 回复，也不进行截图识别，而是按 Xiaomi 17 的 `1220×2656` 竖屏布局执行已经验证过的固定流程：

1. 在当前 MNN Chat 对话中发送“给私有绑定联系人打电话”；
2. 打开私有绑定号码，呼叫 5 秒后挂断；
3. 新建 MNN 对话并发送“微信消息 + 视频电话”的组合指令；
4. 在微信中搜索私有绑定联系人，发送“要不要一起出去吃饭？”；
5. 发起视频电话，5 秒后结束或退出；
6. 无论中途成功或失败，都恢复原输入法、禁用测试 IME、移除本项目无障碍服务、关闭 Agent/MNN 并停止 ADB。

联系人姓名和电话号码仍只存在手机应用私有配置中，不写入脚本、日志或 Git。先把 MNN Chat 的 Qwen 对话页置于前台，然后运行无副作用检查：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_continuous_demo.ps1 -DryRun
```

确认当轮真实电话、微信消息和视频通话均已得到授权后，去掉 `-DryRun` 执行正式录屏。该脚本包含真实外部通信动作，不应作为普通功能测试反复运行。

## 本地完整性检查

```powershell
.\scripts\verify_local.ps1
```

该检查会核验模型与 APK 哈希、视觉模型配置、4-bit 导出参数、20 个任务定义和全部 Python 测试。

## 主机侧单步策略探针

在手机模型完成验证前，可以使用比赛中锁定的本地 BF16 模型测试“截图 → 动作 → 安全裁决”。探针只输出建议，不会点击设备：

```powershell
& 'D:\GitHub\seu-vlm-ppu-optimization\.venv\Scripts\python.exe' `
  .\src\host_policy_probe.py `
  --model-path 'D:\GitHub\seu-vlm-ppu-optimization\models\Qwen3.5-2B' `
  --image .\evidence\screen.png `
  --instruction '帮我把这笔费用付掉' `
  --output .\evidence\policy-probe.json
```

探针结果只有在安全门给出 `ACT` 时，才可能交给 ADB 执行器。执行器默认仍是 dry-run：

```powershell
& 'D:\GitHub\seu-vlm-ppu-optimization\.venv\Scripts\python.exe' `
  .\src\adb_executor.py --proposal .\evidence\policy-probe.json
```

真实执行还需要显式添加 `--execute`，并同时通过低风险检查、当前策略复算、截图相似度和坐标边界检查。

已有探针可以汇总为微型回归基线：

```powershell
python .\src\evaluate_policy_suite.py `
  --json-output .\evidence\gui-baseline.json `
  --markdown-output .\evidence\gui-baseline.md
```

该基线区分模型原始坐标、安全裁决和控件树修正后的有效定位，不能作为通用 GUI 任务成功率。

## 安全边界

- 默认只允许低风险、高置信度动作自动执行。
- 对外发送消息、提交表单等中风险动作必须确认。
- 支付、转账、验证码、身份信息、药物剂量和删除数据默认交给用户。
- 原始截图、联系人和轨迹不进入 Git；提交演示材料前必须脱敏。
- Android 真机结果单独记录，不能与 RTX 4050 或 PPU 数据混用。

## 下一阶段

1. 获取真实手机预检结果，确认 SoC、ABI、Android API、RAM、可用空间和温度状态。
2. 在 MNN Chat 中完成文本与单图推理，记录模型加载时间、峰值内存和生成速度。
3. 用测试应用建立 `截图 → 动作 JSON → 安全门 → ADB/Accessibility 执行 → 下一截图` 闭环。
4. 建立 20 个老年人常见任务和对应的成功/失败判定。
5. 真机数据稳定后，再评估 MNN QNN/Hexagon 后端和自定义 Android 无障碍应用。
