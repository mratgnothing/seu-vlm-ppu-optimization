# 左右手运动想象训练任务

任务根目录：`E:\BCI\datasets\physionet-eegmmidb-1.0.0\training\mi_pilot_20260930_v1`。

此任务实际训练两个模型：CSP-LDA 与 EEGNet-8,2。只读取数据根目录的 `raw/`、`metadata/`，原始文件保持不变。

## 第一轮固定协议

- S001—S010；run 4 训练、run 8 验证、run 12 测试。
- 模型在这 10 人的训练记录上共同训练；评估同一批人的另一段 run。
- FC3、FCz、FC4、C3、Cz、C4、CP3、CPz、CP4 共 9 个中央区电极。
- 读取真实采样率；需要时重采样到 160 Hz。
- 9 电极平均参考，8—30 Hz、四阶因果 Butterworth IIR。
- 提示后 1—4 秒，每个试次一个完整窗口，480 点；不切出重叠训练/测试窗口。
- 只使用 T1/T2（左右手运动想象）；任务区间短于 4 秒的试次不纳入；固定峰峰值阈值 250 µV。
- CSP 分量只按 run 8 选择；EEGNet checkpoint 与早停只按 run 8 选择。
- run 12 最后评价，保存完整预测；不按测试结果反复更改第一轮配置。
- 这是初步离线实验，尚未做跨人、跨天、休息拒识或真实硬件验证。

## 环境

`.venv/`、`.uv-cache/` 在任务目录，MNE、scikit-learn 等为任务独立安装。本机已验证的 CUDA PyTorch 从 `D:\GitHub\seu-vlm-ppu-optimization\.venv\Lib\site-packages` 只读复用，由 `borrowed_torch.pth` 加入路径；不修改该已有环境。它的实际位置和版本记录在每次运行的 `environment.json`。

如果迁移到另一台机器或移除已有 PyTorch 环境，需在任务 `.venv` 内安装适合该机器的 CUDA PyTorch 并移除 `borrowed_torch.pth`。不会把本机环境桥接误称为自包含安装包。

安装任务环境：

```powershell
& 'E:\BCI\datasets\physionet-eegmmidb-1.0.0\training\mi_pilot_20260930_v1\setup_environment.ps1'
```

执行实际训练：

```powershell
& 'E:\BCI\datasets\physionet-eegmmidb-1.0.0\training\mi_pilot_20260930_v1\run_training.ps1'
```

每次运行生成新的 `runs/<UTC时间>/`，不覆盖先前模型。`latest_run.json` 保存最新任务状态和结果路径。

## 交付结果

- `实验报告.md`：中文结果与证据边界。
- `summary.json`、`subject_metrics.csv`：总体与逐受试者指标。
- `test_predictions.csv`：每个测试试次的真值、预测、模型概率。
- `confusion_matrices.png`、`learning_curves.png`、`subject_performance.png`：真实实验图。
- `csp_lda.joblib`、`eegnet_best.pt`：选定模型。
- `confidence_validation.csv`：验证集概率阈值诊断，概率尚未校准。
- `permutation_scores.csv`：100 次 CSP 训练标签打乱对照；不是 EEGNet 的显著性检验。
- `prepared_trials.npz`、`split_manifest.csv`、`data_quality.csv`：预处理试次与质量记录。
- `environment.json`、`source_hashes.json`、`training.log`：复现信息和运行日志。
- `checkpoint_verification.json`：冷加载模型后全部测试预测复现检查。

使用模型判断一个预处理好的测试试次：

```powershell
& '<任务根目录>\.venv\Scripts\python.exe' '<任务根目录>\predict_trial.py' --run-dir '<某次结果目录>' --trial-index 0
```

复核已保存模型：

```powershell
& '<任务根目录>\.venv\Scripts\python.exe' '<任务根目录>\verify_run.py' --run-dir '<某次结果目录>'
```

EEGNet 是依据论文结构独立实现的 PyTorch 复现（时间卷积、深度空间卷积、可分离时间卷积、max-norm 约束），不宣称与作者 Keras 实现逐数值一致。

## 来源

- 数据与标签：https://physionet.org/content/eegmmidb/1.0.0/
- EEGNet：https://arxiv.org/abs/1611.08024
- 结构参考：https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py
- CSP 示例：https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html
