# 微信 Accessibility / UiAutomation 语义探测

日期：2026-09-21\
设备：Xiaomi 17，Android 16\
微信包：`com.tencent.mm`

## 结论

当前微信版本在已验证的搜索页和聊天页上，都没有向系统 UiAutomation 暴露可用的控件语义树。两次 `uiautomator dump` 均只得到一个空根节点，XML 大小均为 407 字节，根节点 `package=com.tencent.mm`，但 `text/resource-id/class/content-desc` 全空，`bounds=[0,0][0,0]`。

因此：

- 不能假定 AccessibilityService 能直接读取微信联系人姓名、输入框和视频按钮；
- `FLAG_INCLUDE_NOT_IMPORTANT_VIEWS` 对普通 View 有帮助，但无法凭空恢复微信没有暴露的语义；
- 微信执行链需要“稳定联系人绑定 + 端侧截图视觉确认 + Accessibility 手势 + IME 文本注入”的混合方案；
- 对 Settings、系统拨号器等能正常暴露语义树的页面，仍应优先走 AccessibilityNodeInfo，避免视觉 OCR。

## 实测页

### 搜索页

- Activity：`com.tencent.mm.plugin.fts.ui.FTSMainUI`
- 搜索词：使用已验证的非敏感绑定搜索键；证据不记录真实姓名或号码
- dump 大小：407 bytes
- 节点：仅 1 个空根节点

### 聊天页

- Activity：`com.tencent.mm.ui.chatting.ChattingUI`
- dump 大小：407 bytes
- 节点：仅 1 个空根节点

## 对 Android 原型的调整

新增 `android/accessibility-agent`：

1. `AgentAccessibilityService`：结构化 UI、语义点击、全局动作、`dispatchGesture`、端侧 `takeScreenshot`。
2. `AgentInputMethodService`：作为用户显式启用的无界面 IME，通过 `InputConnection.commitText` 向当前编辑器输入文本。
3. `VisionVerifier`：在手机内裁剪/放大微信关键区域，通过本机 MNN/Qwen3.5 回环接口做约束式 YES/NO 身份确认。
4. `PlannerClient`：本地 Qwen 只选择受限 Skill，确定性安全门保留最终执行权。

这套架构的目标是把此前的主机 ADB 截图、点击和 `input text` 全部移到手机端。
