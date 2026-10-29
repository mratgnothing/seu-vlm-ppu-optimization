# ZhiBan Accessibility Agent

这是将现有“手机本地 Qwen + 主机 ADB Skill Executor”迁移为普通 Android 端侧执行器的原型。

## 目标运行链

```text
用户意图
  → 手机 MNN / Qwen3.5-2B
  → 单 Skill 规划
  → 确定性安全门
  → AccessibilityService / IME
  → 执行后观察
  → 本地审计
```

## 组件

- `AgentAccessibilityService`
  - 读取 `AccessibilityNodeInfo`
  - `FLAG_REPORT_VIEW_IDS`
  - `FLAG_INCLUDE_NOT_IMPORTANT_VIEWS`
  - 语义点击与 `ACTION_SET_TEXT`
  - `dispatchGesture` 作为自绘 UI 的点击通道
  - `takeScreenshot` 直接在手机端获取视觉输入
  - `GLOBAL_ACTION_HOME` 等全局动作
- `AgentInputMethodService`
  - 用户显式选择后作为无界面 IME
  - 使用 `InputConnection.commitText` 向当前输入框提交 Unicode 文本
  - 用于微信这类不暴露可编辑 Accessibility 节点的页面
- `PlannerClient`
  - 只连接 `127.0.0.1:8080`
  - 本地 Qwen 只做受限 Skill 选择
- `VisionVerifier`
  - AccessibilityService 自己截图
  - 在设备内裁剪、放大微信关键区域
  - 只做约束式身份 / 页面 YES-NO 确认
- `ProbeReceiver`
  - 仅保存脱敏的 Accessibility 节点统计
  - 可武装 Settings、IME 和通信规划的无主机证明
  - 不提供任何真实通信触发入口
- `ShellBridgeReceiver`
  - 只允许持有系统 `android.permission.DUMP` 的 ADB shell 调用
  - 为固定录屏演示提交两段 MNN 指令、固定晚餐邀请，并打开私有绑定拨号页
  - 不返回联系人姓名、电话号码、消息内容或 API key
- `CommunicationSkillExecutor`
  - 执行三个单次授权 Skill：电话、微信固定消息、微信短视频呼叫
  - 原始号码、微信名和消息只从应用私有绑定解析，模型不可见
  - 电话先做完整号码哈希核验；微信先做登记标题指纹与本地视觉联合核验
- `VisualFingerprint`
  - 对用户手动登记的微信聊天标题区域计算灰度量化 SHA-256
  - 只保存指纹，不保存聊天截图
  - 不执行通信动作
- `VisualActionPlanner` / `VisualTaskActivity`
  - 在不导出的本地实验页面中运行三步连续视觉任务
  - 每步把新截图和当前文本目标交给手机本地 Qwen
  - 模型从四个可见编号候选中选择一个动作
  - 前台包、唯一目标边界、选择一致性和状态转移全部通过后才点击
  - 模型误选、格式冲突或页面变化时立即 fail-closed

## 微信实测结论

2026-09-21 在 Xiaomi 17 / Android 16 上，微信搜索页与聊天页的 `uiautomator dump`
都只有 407-byte 空根节点。因此不能把“Accessibility 能读微信好友姓名/按钮”作为产品假设。

微信路径采用：

```text
稳定绑定
  → Accessibility 手势
  → 端侧截图
  → 本地 Qwen-VL 约束确认
  → Headless IME 输入
  → 执行后视觉/Activity 状态确认
```

普通 Settings、系统拨号器等页面仍优先用结构化 Accessibility 节点。

追加通信验收中，电话 Skill 已完成一次约 5 秒的真实运行；微信绑定指纹在跨应用切换后核验失败，因此文本发送和视频呼叫均安全停止。该结果表明 fail-closed 路径有效，同时说明微信执行链仍需修复后重新申请单次授权验收。

## 构建

本机 Android 构建工具链位于项目内的 `tools/android-build/`（该目录由 `.gitignore` 排除，属于本机可重建工具，不是仓库源码）。C 盘逻辑路径通过 NTFS 目录联接指向 `E:\CDriveMigration\zhi-ban-mobile-vla\tools\android-build`。构建工具目录说明见 [`docs/ANDROID_BUILD_TOOLS.md`](../../docs/ANDROID_BUILD_TOOLS.md)，本机工具目录入口见 [`tools/android-build/README.md`](../../tools/android-build/README.md)。

从项目根目录运行以下命令准备/构建：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_android_accessibility_agent.ps1
```

脚本会复用已安装组件；缺少时联网下载并校验 Android Command Line Tools/JDK/Gradle、接受 SDK 许可证、安装 `android-36` / Build Tools `36.0.0`，然后构建。它会修改本机工具目录和 SDK 状态，不是只读检查。输出：

```text
android\accessibility-agent\app\build\outputs\apk\debug\app-debug.apk
```

## 首次授权

安装 APK 后：

1. 打开 ZhiBan Accessibility Agent。
2. 点 “Open Accessibility settings”，显式开启 ZhiBan Accessibility Service。
3. 点 “Open keyboard settings” 启用 ZhiBan Headless IME。
4. 点 “Show input method picker” 并选择 ZhiBan Headless IME。
5. 配置 MNN 本地 API key。
6. 用 “Refresh runtime status” 确认：
   - `accessibility_connected=true`
   - `ime_selected=true`
   - `ime_connected=true`
   - `mnn_api_key_configured=true`

## 第一正式验收

第一轮不做真实通信，只做无副作用的 Settings 测试：

1. MNN Chat 本机加载 Qwen 并启动回环 API。
2. 在 Agent 中点击 “Arm 15s offline Settings proof”。
3. 主机立即执行 `adb kill-server`。
4. 手机自行：
   - 调 Qwen 选择 `open_android_settings`
   - 安全门放行
   - 打开 Settings
   - AccessibilityService 读取 Settings 结构化树
   - AccessibilityService 执行 Home
   - 写入私有 `offline-proof.json`
5. 主机之后恢复 ADB，只读取证据。

只有这轮通过，才进入第二轮“真实微信消息 → 视频 → 电话”的新一次显式授权测试。

## 通信 Skill 的使用顺序

1. 在可见 Agent 页面填写并保存私有电话、微信显示名和一条批准消息；输入框随后清空。
2. 点击“Enroll current WeChat chat title”；Agent 通过私有绑定搜索唯一结果并打开直接聊天，只保存标题区域灰度量化 SHA-256，不发送消息。
3. 确认 MNN 本地 API、Accessibility 和 Headless IME 都已就绪。
4. 每个真实动作分别点击一次授权按钮，并在系统对话框中核对完整目标与动作。
5. 本地 Qwen 选择一个 Skill；确定性 Gate 精确匹配本次授权后才运行。

三个真实按钮均为一次性授权，不能由调试广播触发。未完成微信标题登记、目标不唯一、标题指纹变化、视觉核验失败或显示分辨率变化时都会拒绝执行。

路演录屏另有开发者专用的 `scripts/run_continuous_demo.ps1`。它只适用于已经取得当轮外部通信授权的 `1220×2656` 真机，通过受 `android.permission.DUMP` 保护的窄桥读取私有绑定，并使用已经验证的固定坐标；它不属于产品侧自主 Agent 路径。脚本的 `finally` 会恢复原输入法、禁用测试 IME、移除本项目无障碍服务、关闭 Agent/MNN 并停止 ADB。

## 安全边界

- 真实通信按钮位于可见 Agent 页面，并且每次都必须经过系统确认对话框；导出的调试广播不暴露真实通信入口。
- 联系人身份不能仅由视觉 OCR 决定。
- 微信视觉结果必须与预绑定目标一致。
- 每次真实通信需新的一次性授权。
- MNN API key 不写入项目证据；正式产品应改用 Android Keystore。
