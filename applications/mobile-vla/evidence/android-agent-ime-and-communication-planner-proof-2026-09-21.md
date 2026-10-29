# Android Agent 端侧输入与通信 Skill 规划证明

日期：2026-09-21\
设备：Xiaomi 17（1220×2656，Android 16/API 36）\
模型：Qwen3.5-2B-MNN（MNN Chat 0.8.3，本机 `127.0.0.1:8080`）

## 结论

Android Agent 已在正式断开电脑执行器的窗口中完成两类验证：

1. Headless IME 通过 `InputConnection.commitText` 提交中文文本，并由 Accessibility 以 SHA-256 精确匹配验证；
2. 手机本地 Qwen 分别选择电话、微信消息、微信视频三个受限 Skill，确定性安全门逐字段验证匿名联系人别名、固定消息 ID 和五秒时限。

所有正式窗口都在 `adb kill-server` 后确认 Windows 中 `adb.exe` 进程数为 0。通信 Skill 规划证明设置了 `external_action_attempted=false`，没有拨号、发送消息或发起视频通话。

## Headless IME 正式结果

| 字段 | 结果 |
|---|---|
| `result` | `PASS` |
| `host_executor_required` | `false` |
| `ime_selected` | `true` |
| `ime_connected` | `true` |
| `commit_text` | `true` |
| `unicode_text_verified` | `true` |
| SHA-256 精确匹配节点数 | 1 |
| Accessibility Home 动作 | `true` |
| 手机端执行耗时 | 2447 ms |

第一次尝试因手机熄屏、测试 EditText 未被系统交给输入法而安全失败；没有发生对外动作。亮屏并保持 USB 供电唤醒后，在同样的 ADB 断开条件下复测通过。

## 三个通信 Skill 的端侧规划结果

| 预期 Skill | 本地模型结果 | 参数 | 规划耗时 | Gate | 对外动作 |
|---|---|---|---:|---|---|
| `call_bound_contact` | 完全匹配 | 匿名别名 + 5 秒 | 3717 ms | `ACT_DRY_RUN_ONLY` | 未尝试 |
| `send_wechat_message` | 完全匹配 | 匿名别名 + 固定消息 ID | 1443 ms | `ACT_DRY_RUN_ONLY` | 未尝试 |
| `start_wechat_video_call` | 完全匹配 | 匿名别名 + 5 秒 | 1806 ms | `ACT_DRY_RUN_ONLY` | 未尝试 |

模型没有接收电话号码、微信账号、微信显示名或实际消息文本。真实绑定只允许保存在 Android 应用私有数据目录中。

## 已落入 Android Agent 的执行保护

- 真实通信只能由可见 Activity 中的单次确认对话框武装；导出的调试广播不能触发真实通信。
- 本地 Qwen 每次只能规划一个 Skill；Skill 名、字段集合、匿名别名、固定消息 ID 和时限均需精确匹配。
- 电话在点击呼叫前，必须由 Accessibility 对拨号盘完整号码做 SHA-256 精确匹配。
- 微信绑定需由用户手动打开目标直接聊天完成一次标题区域灰度量化 SHA-256 登记；不保存截图。
- 微信运行时同时要求：唯一搜索结果视觉检查、`ChattingUI` Activity、登记标题 SHA-256 精确匹配和本地视觉标题检查全部通过。
- 当前设备显示配置必须精确为 1220×2656，否则坐标路径拒绝执行。
- 任何缺少绑定、模型格式变化、身份验证失败、页面状态变化或超时都会 fail closed。
- 审计 JSON 不写原始电话号码、联系人姓名、消息正文、截图或 API key。

## 尚未宣称通过的边界

Android 端真实通信执行器已经实现和安装，但本轮没有新的目标/内容单次授权，因此没有进行新的真实拨号、微信发送或微信视频呼叫。只有在用户重新明确给出目标、消息内容并确认三次外部动作后，才能把它们记为“Android Agent 在断开 ADB 后真实执行通过”。
