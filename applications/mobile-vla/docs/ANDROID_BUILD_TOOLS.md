# Android 构建工具目录

本项目使用本机工具目录 `tools/android-build/`，提供以下版本化工具：

- Eclipse Temurin JDK 17：`temurin-17/`
- Android SDK：`sdk/`（含 Command Line Tools、Platform Tools、Android 36、Build Tools 36.0.0 与模拟器系统镜像）
- Gradle 9.6.0：`gradle-9.6.0/`
- 已校验的安装包和下载缓存：`downloads/`

## Agent 索引入口

仓库根目录的 `AGENTS.md` 将本文件注册为 Android 构建任务的先读文档。本机目录中的 `tools/android-build/README.md` 再链接回本文件。工具目录被 `.gitignore` 排除，因此文档入口不能只放在工具目录里。

## 工具路径与调用

始终通过项目内路径 `tools/android-build/` 使用工具。该目录可以是指向 E 盘的 NTFS 目录联接；不要在命令、脚本或项目文档中硬编码联接的物理目标。

本机工具准备/构建脚本为：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap_android_accessibility_agent.ps1
```

脚本会复用已存在的组件；缺失时会联网下载并校验 JDK/Gradle/Android Command Line Tools、接受 SDK 许可证、安装 Android SDK packages，并调用 Gradle 构建 debug APK。它会修改本机工具/SDK状态，不是只读命令。

构建输出：

```text
android/accessibility-agent/app/build/outputs/apk/debug/app-debug.apk
```

构建仅证明 APK 编译完成；安装、连接手机、开启无障碍服务、切换输入法和真实通信都不包含在该命令里，须由用户分别明确授权。

## 维护约束

- 工具目录是可重建依赖，但下载和 SDK 组件体积较大；除非用户明确要求，否则不要删除、移动或重新下载。
- 迁移时保留 `tools/android-build/` 逻辑路径，并在切换前独立核对文件清单、字节数和哈希。
- 不将 JDK、SDK、Gradle、模型或 APK 添加到 Git；修改 `.gitignore` 需要单独说明原因。
