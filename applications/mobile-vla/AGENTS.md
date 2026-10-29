# Agent 工作入口

## Android 构建工具

- 开始 Android Accessibility Agent 的构建、依赖排查或工具迁移前，先读 [`docs/ANDROID_BUILD_TOOLS.md`](docs/ANDROID_BUILD_TOOLS.md)。
- 工具目录的逻辑入口始终是 `tools/android-build/`；它被 `.gitignore` 排除，存放本机生成/下载的 JDK、Android SDK、Gradle 与缓存，不属于源码提交内容。
- 具体工具调用与安装边界见 `tools/android-build/README.md`。优先复用现有工具；只有用户明确要求准备/构建且组件缺失时，才运行 `scripts/bootstrap_android_accessibility_agent.ps1`。该脚本可能联网、接受 SDK 许可证并安装组件。
- `tools/android-build/` 可能通过目录联接存放在 E 盘。始终通过项目内逻辑路径调用，不要绕过联接直接改物理目录，也不要删除工具目录。
- 构建 APK 不等于真机验收。不要因构建任务自动安装到手机、开启无障碍服务、修改输入法或发送真实通信；这些需按任务单独授权。
