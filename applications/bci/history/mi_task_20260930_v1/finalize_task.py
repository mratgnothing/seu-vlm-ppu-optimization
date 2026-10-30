"""Make a concise entry report from completed and independently verified run artifacts."""
import json
from pathlib import Path


root = Path(__file__).parent
latest = json.loads((root / "latest_run.json").read_text(encoding="utf-8"))
assert latest["status"] == "COMPLETE"
run_dir = Path(latest["run_dir"])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
verification = json.loads((run_dir / "checkpoint_verification.json").read_text(encoding="utf-8"))
assert verification["status"] == "PASS"
lines = ["# 训练任务：首轮完成", "", "本任务已真实训练 CSP-LDA 与 EEGNet，并从保存模型冷加载复现全部测试预测。", "",
         f"结果目录：`{run_dir}`", "", "## 首轮结果", "",
         "S001—S010，run 4/8/12 分别用于训练/验证/测试，每组 150 个试次。这是同人跨 run 评估，不是跨人泛化。", "",
         "| 模型 | 测试准确率 | 测试平衡准确率 | ROC-AUC |", "|---|---:|---:|---:|"]
for name,result in summary["test_metrics"].items():
    lines.append(f"| {name} | {result['accuracy']:.2%} | {result['balanced_accuracy']:.2%} | {result['roc_auc']:.3f} |")
lines += ["", "CSP-LDA 有有限分类信号，但个体差异很大。EEGNet 的成绩接近随机水平，且偏向左手预测。训练曲线中训练分数上升、验证分数持续接近 50%，提示过拟合或跨 run 分布差异；首轮没有分离这些因素，不能确诊原因。", "",
          "单人的测试集只有 15 个试次，不能将高分个例外推到长期使用。当前结果不适合直接作为手机控制授权。", "",
          f"EEGNet 仅 {summary['eegnet_resources']['parameters']} 个参数，实际训练 {summary['eegnet_resources']['epochs_ran']} epochs；选定 checkpoint 来自第 {summary['eegnet_resources']['best_epoch']} 个 epoch。",
          f"EEGNet 训练阶段约 {summary['eegnet_resources']['fit_seconds']:.2f} 秒，GPU 最大 tensor 分配约 {summary['eegnet_resources']['peak_cuda_allocated_mib']:.1f} MiB，CUDA 最大缓存预留 {summary['eegnet_resources']['peak_cuda_reserved_mib']:.1f} MiB，训练进程 RSS 最大约 {summary['eegnet_resources']['peak_training_process_rss_mib']:.0f} MiB。",
          f"本轮流程内部计时约 {summary['total_seconds']:.1f} 秒，不包含环境安装。显存指标是 PyTorch 统计，不包含全部驱动/上下文开销。", "",
          f"检查：{verification['trial_keys_unique']} 个试次主键无重复，{verification['holdout_predictions_reproduced']} 条模型测试预测均可复现。", "",
          "## 查看与继续", "",
          f"- [详细实验报告](runs/{run_dir.name}/实验报告.md)",
          f"- [学习曲线](runs/{run_dir.name}/learning_curves.png)",
          f"- [逐受试者结果](runs/{run_dir.name}/subject_metrics.csv)",
          f"- [结果 JSON](runs/{run_dir.name}/summary.json)",
          "- `run_training.ps1` 会用当前固定配置创建新结果目录；修改配置前保留首轮结果。",
          "- `predict_trial.py` 可对保存的测试试次进行离线模型推理；不操作手机。", "",
          "下一轮建议使用尚未查看测试结果的新受试者，比较 pooled 与个体化校准，再扩大训练数据。对休息/无操作状态、伪迹和置信度校准单独实验，随后考虑流式回放。", ""]
(root / "任务结果.md").write_text("\n".join(lines),encoding="utf-8")
print(root / "任务结果.md")
