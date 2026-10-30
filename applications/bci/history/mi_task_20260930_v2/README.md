# 真实脑电运动想象 · 第二轮训练

任务根目录位于原E盘数据集的 training/mi_expanded_20260930_v2。原始数据和首轮训练结果保持不变。

## 查看结果

运行 `view_results.ps1`，打开本地交互结果页。服务只监听127.0.0.1，不上传脑电数据。端口占用时自动选择下一个空闲端口，不停止其他进程。使用完后Ctrl+C结束服务。

离线便携版为 `dashboard.html`，内嵌全部图表数据，文件较大。推荐使用本地服务以获得稳定浏览体验。

页面展示两个协议和两个模型、整群bootstrap区间、混淆矩阵、逐人BA、验证选中候选的学习曲线，以及所有留出试次的真实C3/Cz/C4信号。可选择受试者、预测模型、正确/错误试次。回放没有手机动作或在线推理。

## 复现训练

运行 `run_training.ps1`。使用首轮任务既有 .venv，新运行保存到独立时间戳目录，不覆盖已有实验。若原首轮环境移动，需要重新配置解释器。

`train_expanded.py` 复用首轮 `train.py` 的训练函数及 `eegnet.py`。`source_hashes.json`记录此次实际代码哈希。模型共有1730个参数；本次仅用本地RTX4050 Laptop，无云端费用。

`prepare_visualization.py` 从 `latest_run.json`定位完成的实验，从测试CSV手算混淆矩阵和BA，与训练报告对照，再生成 reviewed_snapshot.json。例：

```powershell
& '..\mi_pilot_20260930_v1\.venv\Scripts\python.exe' .\prepare_visualization.py $PSScriptRoot .\results
```

可视化源码在 `visualization/src/content/dashboard/`；审阅数据与定义在 `visualization/src/data.json`。构建使用本机Data Analytics插件的预构建React运行时；构建后的 `visualization/dist`不需要额外安装npm包。

## 边界

实验只包含明确提示的左/右手运动想象。预测概率未经校准；缺少休息检测、跨天与在线验证。不得将片段回放理解为实时脑控，不得直接授权手机操作，也不能据此声称医疗有效。

选择候选只使用验证集；本轮测试已解封，后续方法探索需要另外设计独立评估，不反复对这份测试集调参。
