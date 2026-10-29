# 本轮素材与关键口径

日期：2026-10-29。

## 新增来源

- `C:\Users\ASUS\Downloads\GUI-VLA项目汇报-注释版(1).pptx`：5 页。P2 在线主链，P3 RSI，P4 实机视频与阶段结果，P5 未来工作。全部文本、备注、媒体关系保存在 gui_report_inventory.json；渲染在 renders\gui_report。
- `C:\Users\ASUS\Downloads\README.md`：2026-10-28 的 Mobile GUI VLA 架构与进展交接。快照 sources\Mobile_GUI_VLA_README_20261028.md；包括两幅 Mermaid 图，原文件没有 Markdown 图片引用。
- `C:\Users\ASUS\AppData\Local\Temp\codex-clipboard-c07c2997-be01-4894-9818-87eb6a0d1d5d.png`：用户提供的参考评测和技术目标截图。已复制到 sources\gui_reference_benchmark_153.png；人工转录数字保存为 sources\benchmark_153.json。
- 既有计划及数据路径：`C:\Users\ASUS\Downloads\Qwen35_融合版PPT规划_20261028\逐页详细计划.md` 与同目录素材索引。

## 三页保留内容

源：`C:\Users\ASUS\Downloads\一吴是处_决赛路演_含附录20页_v2.pptx`。

| 源页 | 新位置 | 内容要求 |
|---|---|---|
| 16 RSI | 主讲20 | 原标题、短长链、角色、连接、后训练路线完整保留 |
| 19 GUI-VLA完整架构 | A1 | 原模块/文字/连线/模型与契约完整保留 |
| 20 Qwen完整架构 | A2 | 原计算结构/维度/旁路/公式/标记完整保留 |

图像：sources\locked_originals\sun_P16.png、sun_P19.png、sun_P20.png。逐页文字、备注及源 XML SHA-256：同目录 content_manifest.json。后续只做视觉适配，逐项对照保留清单。

## 视频留空

原始内嵌视频已提取到 sources\gui_report\media1.mp4，848×1200，25fps，1157 帧，46.28 秒。占位比例为 848:1200（约 0.707:1）。视频没有重编码。元数据位于 video_metadata.json。

留空建议：页面高度的约 65%—75%，保持视频原比例；具体位置由 ImageGen 决定。空白中不加播放符号、文字、框内装饰或生成的手机界面，供用户后续直接粘贴原视频。原视频的任务说明和指标栏也应保留在画面中。

视频末帧 clock_frame_final.png 显示：进度 17/17、慢模型平均输出 46.2 tokens、快路径平均响应 517 ms。17/17 是视频中的步骤进度显示。

新增 PPT P4 与备注：决策时延中位数约 682 ms，六成低于 800 ms，输出约 46 tokens、阈值目标 <180 tokens。README 给出历史 API 时钟任务的独立七项检查、20 决策、19 动作、44.354 秒，决策 P50 681.76 ms、12/20 小于 800 ms；模型为旧 Max + 旧 Flash。

制作时按各来源注明不同统计对象，不把快路径均值、总体决策 P50、视频时长、运行时长、任务步骤进度合并为一个数字。

## 参考评测的用途

截图包含 54 指定场景与 99 泛化场景，共 153 例；五个模型的百分比原样记录。它用于展示领域基线。图下 success >95%、decision <800ms、frequent share >80%、output <180 tokens、clarification >95%、handover >95%、false alarm <5% 属于技术目标。

当前截图没有本系统同协议成绩。主讲将系统优势落在任务感知观察、按需快慢协同、可核验执行和持续改进机制上；量化结果分别使用时钟、高德、首包覆盖等已有记录，并保留任务条件。

## README 的实验摘录

- 六页面 v4 首包：8149/8184/8047/8119/8006/8131 bytes，预算 8192 bytes；保留源字段与必要状态。v3 爱奇艺两页必要组超预算，v4 首包纳入对应搜索/输入目标。
- 高德单步：185 nodes、8111 bytes、1 API、1 ACTION_SET_TEXT，无追加查询/无图，输入6578/输出58Token、API2.633s/整轮18.650s。只对应写入搜索框这一任务。
- 同轮10次尝试共5 API、33096Token，包含动态稳定性、歧义、stale_or_exhausted等失败；首包覆盖和输入案例分别有自己的分母。
- 四部分：Data Lab 组织会话/监督/数据，Harness 组织证据/上下文/提案，Platform提供真实观察/重绑/执行/轨迹，私有信息策略研究任务首包。网页前端主要属于 Data Lab；正文按职责解释。
- RSI 原图保留，备注可用 README 补充固定任务比较、工程候选、独立后训练、回到真实任务验证。

## 当前交付

本轮交付为《逐页内容规划_v2.md》及素材参考。后续收到内容反馈再开始逐页生成。
