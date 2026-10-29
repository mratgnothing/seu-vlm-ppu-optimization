# 当前验证状态

更新时间：2026-09-22

## 已完成

- 独立原型目录已建立，未修改比赛提交仓库和最终 ZIP。
- Xiaomi 17（25113PN0EC、SM8850、Android 16/API 36、arm64-v8a）已完成 ADB 真机预检。
- 官方 MNN Chat 0.8.3 已安装；APK SHA-256 为 `EB249CABBF73B8B1567D7611715CAD8F1CBF4DF7BE75CB447EA206F12F94AB14`。
- `taobao-mnn/Qwen3.5-2B-MNN` revision `35781816d7b6a9dcb273a6765ac9563401951c3c` 已部署到 `/data/local/tmp/mnn_models/Qwen3.5-2B-MNN`。
- 手机端关键权重哈希与电脑端一致；模型配置为 `qwen3_5`、`is_visual=true`、HQQ 4-bit、block size 64。
- MNN Chat 已识别模型并成功完成英文精确回答、中文 API 和本地 RAG 推理。
- 实际运行配置已由日志确认：CPU、4 线程、low precision、low memory、`use_mmap=false`；本轮不是 NPU 测试。
- 关闭 Thinking 后实测预填充 42.36 tokens/s、解码 38.78 tokens/s；默认 Thinking 对简单任务生成 365 个思考 tokens，延迟 50.50 秒。
- 裸模型在支付场景错误输出 `ACT`；确定性安全门根据 `operation=payment` 强制输出 `HANDOFF`，证明模型不能获得直接执行权。
- 本地 SQLite RAG 使用合成条目完成检索—上下文注入—手机模型回答闭环，总耗时约 0.54 秒。
- 知识库实现实体、不可变轨迹、递归脱敏、80% 最低成功率、知识提案、人工审核、工作流版本和回滚。
- 20 个老年人手机任务已形成首版任务集。
- 受控测试网页验证虚构联系人发送前确认和虚构支付本人接管。
- ADB 执行器支持 dry-run、策略复算、低风险白名单、坐标边界、截图相似度和 UI 控件树语义定位。
- 五案例微型回归基线：解析 5/5、安全裁决 5/5、模型原始坐标命中 1/4；三个预期可执行案例经边界检查和控件树修正后动作就绪。该结果仅验证架构，不代表通用 GUI 成功率。
- 本地验证脚本与 Python 单元测试 29/29 通过。
- 测试期间关闭网络；结束后恢复 Wi-Fi、保持原有移动数据关闭状态，关闭 API 服务、移除 ADB 转发、删除冗余模型副本并强制停止应用。
- 在用户明确授权下完成真实电话呼叫、微信文本发送和微信视频呼叫冒烟测试；目标信息仅用于当前步骤，证据中已脱敏。
- 端侧视觉模型准确识别拨号界面的 11 位号码（约 2.68 秒），但把微信会话标题中的三字姓名识别错误（约 2.89 秒）。当前模型不能独立承担联系人身份确认或通信动作授权。
- 已加入三个受限通信 Skill。端侧模型以匿名联系人别名分别生成电话、微信消息和微信视频调用，三次单 Skill 规划均通过安全门并完成真实执行，端侧规划耗时约 3.78--3.89 秒。
- MNN Chat 0.8.3 本轮未通过标准 `tools` 字段返回原生 `tool_calls`；多 Skill 一次性规划出现改名、格式和稳定性问题并被拒绝。当前采用“单意图、单 Skill、执行后观察”的状态机。
- 已完成一次真正的“正式运行阶段不依赖电脑”安全测试：预置后主机执行 `adb kill-server`，Windows 确认无 `adb.exe` 进程；约 12.8 秒后手机才开始本机 Qwen 推理。无害“打开系统设置”被端侧 Action Gate 放行并成功执行，支付意图被端侧 Gate 强制 `HANDOFF`，`blocked_action_executed=false`；手机测试结束约 23 秒后主机才重新连接 ADB 取证。测试结果 `PASS standalone_model_gate_test`。
- 又完成一次不依赖 MNN 本地 API 的 UI 直读复核：主机在 `1789970356` 执行 `adb kill-server` 后，手机端 shell harness 自行重启 MNN Chat、两次独立加载 Qwen3.5-2B-MNN 并完成首轮分类。未授权消息请求由模型输出 `BLOCK`，确定性安全门再次强制 `BLOCK` 且 `RISK_ACTION_EXECUTED=0`；“打开 Android Settings”由模型输出 `SAFE`，安全门放行后手机自行打开 `com.android.settings/.MiuiSettings`。正式结果为 `PASS`，结束后手机自行回到桌面。该复核同时发现 MNN Chat 0.8.3 在本机同会话第二轮可能长时间停留在 `Thinking`，因此当前状态机仍应采用“单意图、单首轮会话、执行后观察”。
- 已建立 `android/accessibility-agent` Android 原型源码：`AgentAccessibilityService` 负责结构化 UI、语义点击、`dispatchGesture`、系统全局动作和端侧 `takeScreenshot`；`AgentInputMethodService` 通过 `InputConnection.commitText` 作为无界面 IME 文本桥；`PlannerClient` 调用手机本机 MNN/Qwen；`VisionVerifier` 在手机内裁剪/放大微信关键区域后做受限视觉确认。
- 已实测微信搜索页 `FTSMainUI` 与聊天页 `ChattingUI` 的系统 UiAutomation 树：两页均只有 407-byte 空根节点，`text/resource-id/class/content-desc` 全空且 `bounds=[0,0][0,0]`。因此微信当前版本不能依赖 AccessibilityNodeInfo 直接读取好友姓名/输入框/视频按钮，产品结构改为“稳定绑定 + 端侧视觉确认 + Accessibility 手势 + IME 输入”；普通系统页面仍优先结构化 Accessibility 语义。
- Headless IME 已完成正式无主机验收：主机 `adb kill-server` 且 `adb.exe=0` 后，手机端以 `InputConnection.commitText` 提交中文文本，Accessibility 得到唯一 SHA-256 匹配，最终 `PASS`，端侧执行 2447 ms。第一次熄屏尝试安全失败，亮屏后复测通过。
- Android Agent 已原生加入 `call_bound_contact`、`send_wechat_message`、`start_wechat_video_call` 三个通信 Skill 的本地 Qwen 路由与确定性参数门。三次分别在 ADB 完全断开时通过，规划耗时 3717/1443/1806 ms，且 `external_action_attempted=false`。
- Android Agent 已加入私有联系人绑定、微信聊天标题灰度量化 SHA-256 登记、单次可见确认对话框、拨号号码哈希核验、微信 Activity/指纹/本地视觉联合核验和 fail-closed 审计。真实通信入口不能由导出的 ProbeReceiver 触发。
- 已在真机完成受控三步连续视觉点击：最终安装版中，手机本地模型依次从新截图中选择 BLUE=4、GREEN=1、ORANGE=3，推理耗时 6827/3925/5101 ms；三步均通过候选一致性、目标边界、前台包和状态转移校验，最终 `PASS`，总运行约 17.3 秒。
- 自由坐标和“颜色 + 形状”组合标签基线出现过格式异常与误选，均在点击前被安全门阻止。当前可靠方案是四候选离散视觉动作空间，不能表述为任意 GUI 坐标控制已经解决。

## 尚未完成

### 2026-09-21 追加的 Android Agent 通信验收

- 已将 `ShellBridgeReceiver` 加入原安装签名的 APK，并重新安装到真机；当前 APK SHA-256 为 `5E1B1E78D3B9FD58341D595C7718BDA5A55030E2637BE777AB0781551BD64199`。
- 本地 Qwen3.5-2B-MNN 在手机回环 API 上严格返回 `call_bound_contact`，确定性门放行后拨号器进入通话界面并保持约 5 秒；随后窗口回到系统拨号器，未检测到持续的 InCallUI。
- 追加运行中，MNN 在切换到拨号器或微信后出现请求阻塞或非 Skill 文本；按 fail-closed 策略未发送微信消息，也未启动微信视频。微信当前聊天窗口的指纹桥接在应用切换后不稳定，无法继续身份确认。
- 因此本轮只能确认“电话 Skill 规划 + 5 秒拨号”通过；微信消息和视频仍属于未完成项，不能把此前电脑端冒烟结果或本轮的坐标尝试当作 Android Agent 的端到端通过。

- 已完成当前界面截图的端侧视觉问答，但只覆盖两张真实界面；尚未形成可统计的视觉任务集与成功率。
- 已完成“自然语言 → 手机小模型 Skill 选择 → 安全门 → 确定性执行器 → 真实通信动作”的受限闭环，也完成受控应用内三步连续截图与离散视觉点击；尚未完成任意第三方应用中的通用低层 GUI 闭环。
- 尚未验证 OpenCL、Vulkan、QNN 或 Hexagon NPU；当前 APK 明确使用 CPU 后端。
- 尚未进行 30 分钟以上持续推理、峰值温升、功耗、后台保活和崩溃恢复测试。
- RAG 已完成主机侧检索并通过手机回环 API 推理，尚未封装为独立 Android 应用内的全端侧检索模块。
- 还没有老人模式 UI、语音入口、家属授权面板和紧急停止机制的 Android 产品外壳。
- 新 Android 执行器已在本轮新的单次授权下完成真实电话约 5 秒验收；微信发送与微信视频仍未通过端到端验收，旧的电脑端执行结果不能替代这两项。
- `android/accessibility-agent` 已成功构建并安装到 Xiaomi 17，包名 `ai.zhiman.accessibility`、版本 0.1.0；本轮使用原安装签名无损升级，当前 APK 哈希见本节开头的追加验收记录。
- 已完成 `AccessibilityService` 真机无主机正式运行证明：ARM 后主机执行 `adb kill-server`，杀掉后和重连前均确认无 `adb.exe` 进程；手机端正式执行在 ADB 杀掉约 1.46 秒后才开始。本地 Qwen3.5-2B-MNN 用 938 ms 选择 `open_android_settings`，确定性 Gate 输出 `ACT`，`AgentAccessibilityService` 打开 Settings 并读取 125 个结构化节点，随后通过 `GLOBAL_ACTION_HOME` 回到桌面，正式 proof 为 `PASS`。
- 正式 runner 已从 Activity 延时线程迁入系统绑定的 `AgentAccessibilityService`，由 `ARM_OFFLINE_PROOF` 广播武装；ARM 时立即写 `result=ARMED`，运行阶段持续覆盖 proof，避免 Activity 生命周期导致正式窗口不执行。
- Android 16 会阻止后台 AccessibilityService 直接拉起 Activity；系统日志明确出现 `Abort background activity starts from 10494`。正式 PASS 方案是在窗口开始前让 ZhiBan Agent Activity 保持前台、MNN/Qwen 后台已加载。产品化需采用用户可见前台 Agent/合规入口，而不是依赖后台任意拉起页面。
- 本轮已证明“预部署后的运行时独立性”：正式 Qwen 推理、安全门、Settings 启动、Accessibility 读取和 Home 动作均不需要电脑执行器；但 Agent 安装、Accessibility 授权、MNN 模型预加载与本地 API 启动仍在正式窗口前完成，因此仍不能表述为 fresh install 后完全无电脑。

## 下一验收门

1. 构造不经过系统相册的合成图片 URI 或专用测试 APK，扩充隐私安全的单图理解任务集，重点覆盖中文姓名、相似字和小字号。
2. 在受控测试 APK 上执行低风险动作闭环，只开放返回、桌面、白名单应用和唯一语义目标点击；联系人使用稳定 ID，不使用视觉姓名作为唯一身份依据。
3. 增加连续 30 分钟文本与视觉混合负载，记录温度、耗电、PSS、swap 和崩溃率。
4. 将本地 RAG、Action Gate、审计日志和一键停止封装进 Android 原型。
5. 单独评估更新 MNN 与接入 Hexagon/QNN 的成本，获得真实 NPU 数据后再用于答辩表述。

完整真机数据见 `evidence/device-test-2026-09-21.md`、`evidence/device-test-2026-09-21.json`、`evidence/authorized-communication-smoke-test-2026-09-21.md`、`evidence/small-model-skill-execution-2026-09-21.md`、`evidence/standalone-computer-independent-test-2026-09-21.md`、`evidence/standalone-computer-independent-ui-test-2026-09-21.md`、`evidence/wechat-accessibility-probe-2026-09-21.md`、`evidence/accessibility-agent-standalone-proof-2026-09-21.md`、`evidence/android-agent-ime-and-communication-planner-proof-2026-09-21.md`、`evidence/phone-local-communication-attempt-2026-09-21.md` 和 `evidence/safe-continuous-visual-loop-2026-09-22.md`。
