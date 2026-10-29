# Accessibility Agent 无主机正式运行证明

日期：2026-09-21\
设备：Xiaomi 17 / Android 16 (API 36)\
Agent：`ai.zhiman.accessibility` 0.1.0\
APK SHA-256：`0179599aefb3dac1fdc12fa93f9859e1a933bf4e869bce8d5e80eb85f1ed01e4`

## 结论

本轮正式窗口验证了：

```text
手机本地 Qwen3.5-2B-MNN
  → 单 Skill 规划
  → 确定性安全门
  → Android AccessibilityService
  → 打开 Settings
  → 读取结构化 Accessibility 树
  → GLOBAL_ACTION_HOME
```

上述正式运行窗口内没有主机 ADB 进程，也没有 ADB forward/reverse 参与动作执行。

这证明的是**预部署完成后的运行时独立性**。模型预加载、MNN 本地 API 启动、Agent 安装和 Accessibility 授权仍在正式窗口之前由电脑辅助完成，因此不能表述为“从安装到执行完全不需要电脑”。

## 正式时间线

- Agent ARM：`1789976242731`
- 主机执行 `adb kill-server`：`1789976256272`
- 杀掉后主机 `adb.exe` 进程数：0
- 手机端正式执行开始：`1789976257732`
- 本地 Qwen 规划耗时：938 ms
- Skill：`open_android_settings`
- Gate：`ACT`
- 手机端执行完成：`1789976260522`
- 主机重连前再次检查 `adb.exe`：0
- 主机正式窗口等待结束：`1789976286725`

正式执行开始时间晚于 `adb kill-server` 约 1.46 秒，因此 Qwen 推理和后续 Accessibility 动作均发生在主机 ADB 已不存在之后。

## Accessibility 结果

Settings 页面结构化快照：

- `package=com.android.settings`
- 根节点存在
- 节点总数：125
- 含文本节点：21
- 有 view id 的节点：87
- clickable 节点：18
- editable 节点：0

随后 `GLOBAL_ACTION_HOME` 返回成功，事后重连确认最终前台为 MIUI Launcher。

## 调试过程中发现的问题

### 1. Activity 延时线程不适合作为正式触发器

最初将 15 秒倒计时放在 `MainActivity` 中。ADB 断开时两次未生成 proof，因此将正式 runner 迁入系统绑定的 `AgentAccessibilityService`，并由 `ProbeReceiver` 接收 `ARM_OFFLINE_PROOF` 广播。ARM 时立即写入 `result=ARMED`，执行过程中持续覆盖 proof。

### 2. MNN 0.8.3 同会话第二轮可能卡住

连接状态下预演曾停在 `result=RUNNING`。这与此前已观测到的 MNN Chat 0.8.3 同会话第二轮 Thinking/请求卡住一致。正式策略继续采用“每个任务使用新鲜模型会话，单意图、单首轮推理”。

### 3. Android 16 后台 Activity 启动限制

第一次 AccessibilityService 正式 runner 能执行 Qwen，但从后台 service 拉起 Settings 时，系统日志明确给出：

```text
Abort background activity starts from 10494
```

因此最终正式窗口在断 ADB 前把 ZhiBan Agent Activity 保持在前台，而 MNN/Qwen 保持后台已加载。此后 service 发起 Settings 成功，正式结果 PASS。

这意味着产品化时不应依赖后台 AccessibilityService 随意拉起 Activity；应采用用户可见前台 Agent / 合规通知入口 / 用户触发任务等方式满足 Android 后台启动规则。

## 正式 proof

```json
{
  "result": "PASS",
  "armed_epoch_ms": 1789976242731,
  "run_start_epoch_ms": 1789976257732,
  "runner": "accessibility_service",
  "service_pid": 27080,
  "host_executor_required": false,
  "planner_location": "device_loopback_mnn",
  "planner_skill": "open_android_settings",
  "planner_elapsed_ms": 938,
  "gate_decision": "ACT",
  "settings_verified": true,
  "accessibility_home_action": true,
  "finished_epoch_ms": 1789976260522
}
```

完整设备私有 proof 还包含脱敏后的 Settings 节点统计，不包含 API key、联系人、电话号码或聊天内容。

## 尚未证明

- 微信真实消息 / 视频 / 电话在同一个无主机正式窗口内执行；
- 微信目标身份可仅依赖 Accessibility 安全验证；
- Headless IME 在微信自绘输入框中的正式端侧 Unicode 输入；
- 端侧视觉截图 → Qwen-VL → 手势的连续真实微信闭环；
- fresh install 后完全无电脑配置。

下一步应先做**无外部副作用的微信草稿测试**：端侧进入已绑定聊天、通过 Headless IME 输入固定测试文本但不发送；验证截图、视觉确认、手势和 Unicode IME 都能在 ADB 断开后完成。确认后，再为真实消息/视频/电话获取一次新的单次授权。
