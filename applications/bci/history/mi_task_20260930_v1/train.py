from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata as metadata
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import psutil
import torch
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eegnet import EEGNet


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
    def flush(self):
        self.original.flush()
        self.log.flush()


def metric(y, probability):
    prediction = probability.argmax(axis=1)
    return {"n":len(y), "accuracy":float(accuracy_score(y, prediction)),
            "balanced_accuracy":float(balanced_accuracy_score(y, prediction)),
            "roc_auc":float(roc_auc_score(y, probability[:, 1])),
            "confusion_matrix":confusion_matrix(y, prediction, labels=[0, 1]).tolist()}


def cluster_interval(y, probability, trial_rows, seed):
    subjects = sorted({row["subject"] for row in trial_rows})
    indices = {subject:np.array([i for i,row in enumerate(trial_rows) if row["subject"] == subject]) for subject in subjects}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(1000):
        selected = rng.choice(subjects, len(subjects), replace=True)
        idx = np.concatenate([indices[subject] for subject in selected])
        scores.append(balanced_accuracy_score(y[idx], probability[idx].argmax(1)))
    return [float(x) for x in np.quantile(scores, [0.025, 0.975])]


def load_data(config, run_dir):
    root = Path(config["dataset_root"])
    official = {}
    for line in (root / "metadata/SHA256SUMS.official.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        official[name.lstrip("*").removeprefix("./")] = digest
    arrays, labels, records, qc = {}, {}, {}, []
    for split, run in config["split_runs"].items():
        split_x, split_y, split_rows = [], [], []
        for subject in config["subjects"]:
            name = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            path = root / "raw" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != official[name]:
                raise RuntimeError(f"Input checksum mismatch: {name}")
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            source_sfreq = raw.info["sfreq"]
            mne.datasets.eegbci.standardize(raw)
            raw.pick(config["channels"])
            raw.set_montage("standard_1005")
            raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            if source_sfreq != config["target_sfreq"]:
                raw.resample(config["target_sfreq"], verbose="ERROR")
            raw.filter(*config["bandpass_hz"], method="iir", phase="forward",
                       iir_params={"order":4,"ftype":"butter","output":"sos"}, verbose="ERROR")
            events, _ = mne.events_from_annotations(raw, event_id={"T1":1,"T2":2}, verbose="ERROR")
            start, end = config["epoch_seconds"]
            sfreq = raw.info["sfreq"]
            event_duration = {int(round(ann["onset"]*sfreq)):ann["duration"] for ann in raw.annotations if ann["description"] in ("T1","T2")}
            valid_events = np.array([ev for ev in events if event_duration.get(int(ev[0]), 0) + 1/sfreq >= end], dtype=int)
            if valid_events.size == 0:
                raise RuntimeError(f"No valid task intervals: {name}")
            epochs = mne.Epochs(raw, valid_events, event_id={"left":1,"right":2},
                                tmin=start, tmax=end-1/sfreq, baseline=None,
                                preload=True, reject={"eeg":config["peak_to_peak_reject_uv"]*1e-6}, verbose="ERROR")
            x = epochs.get_data(copy=True).astype(np.float32) * 1e6
            y = epochs.events[:,2].astype(np.int64) - 1
            if len(x) == 0 or set(y.tolist()) != {0,1} or not np.isfinite(x).all():
                raise RuntimeError(f"Invalid trials or missing class: {name}")
            trial_rows = [{"subject":subject, "run":run, "split":split,
                           "onset_sample":int(ev[0]), "label":int(label), "file":name}
                          for ev,label in zip(epochs.events, y)]
            split_x.append(x)
            split_y.append(y)
            split_rows.extend(trial_rows)
            row = {"subject":subject,"run":run,"split":split,"file":name,
                   "source_sfreq":source_sfreq,"target_sfreq":sfreq,"cued_trials":len(events),
                   "short_task_intervals":len(events)-len(valid_events),"accepted_trials":len(y),
                   "rejected_trials":len(valid_events)-len(y),"left":int((y==0).sum()),"right":int((y==1).sum()),
                   "samples_per_trial":x.shape[2],"sha256":official[name]}
            qc.append(row)
            print(f"DATA {split} S{subject:03d} R{run:02d}: {len(y)} trials, rejected {row['rejected_trials']}", flush=True)
        arrays[split], labels[split], records[split] = np.concatenate(split_x), np.concatenate(split_y), split_rows
    all_keys = [(row["subject"],row["run"],row["onset_sample"]) for rows in records.values() for row in rows]
    if len(all_keys) != len(set(all_keys)):
        raise RuntimeError("Trial overlap between splits")
    if len({tuple(x.shape[1:]) for x in arrays.values()}) != 1:
        raise RuntimeError("Inconsistent model inputs")
    write_csv(run_dir / "data_quality.csv", qc)
    write_csv(run_dir / "split_manifest.csv", [row for rows in records.values() for row in rows])
    np.savez_compressed(run_dir / "prepared_trials.npz", **{f"X_{split}":x for split,x in arrays.items()}, **{f"y_{split}":y for split,y in labels.items()})
    return arrays, labels, records, qc


def csp_pipeline(components):
    return make_pipeline(CSP(n_components=components, reg=0.1, log=True, norm_trace=False, rank="full"),
                         LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def train_csp(x, y, config, run_dir):
    candidates, fitted = [], []
    started = time.perf_counter()
    for count in config["csp_components"]:
        pipeline = csp_pipeline(count)
        pipeline.fit(x["train"], y["train"])
        result = metric(y["validation"], pipeline.predict_proba(x["validation"]))
        candidates.append({"components":count, **result})
        fitted.append(pipeline)
    best = max(range(len(candidates)), key=lambda i:candidates[i]["balanced_accuracy"])
    pipeline = fitted[best]
    joblib.dump(pipeline, run_dir / "csp_lda.joblib")
    save_json(run_dir / "csp_validation.json", candidates)
    print(f"CSP selected {candidates[best]['components']} components using validation only", flush=True)
    return pipeline, {"selected_components":candidates[best]["components"],"fit_seconds":time.perf_counter()-started}


def train_eegnet(x, y, config, run_dir, device):
    settings = config["eegnet"]
    mean = x["train"].mean(axis=(0,2), keepdims=True)
    std = x["train"].std(axis=(0,2), keepdims=True).clip(1e-3)
    scaled = {split:(data-mean)/std for split,data in x.items()}
    tensor_x = {split:torch.from_numpy(data).to(device) for split,data in scaled.items()}
    tensor_y = {split:torch.from_numpy(labels).to(device) for split,labels in y.items()}
    model_settings = {key:settings[key] for key in ("F1","D","F2","dropout","kernel_length")}
    model = EEGNet(x["train"].shape[1], x["train"].shape[2], **model_settings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    rng = torch.Generator(device=device).manual_seed(config["seed"])
    history, best_state, best_score, best_loss, best_epoch = [], None, -1.0, float("inf"), 0
    started = time.perf_counter()
    rss_peak = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, settings["max_epochs"]+1):
        model.train()
        order = torch.randperm(len(y["train"]), generator=rng, device=device)
        losses = []
        for batch in order.split(settings["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tensor_x["train"][batch]), tensor_y["train"][batch])
            loss.backward()
            optimizer.step()
            model.constrain_weights()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            logits = model(tensor_x["validation"])
            val_loss = float(loss_fn(logits, tensor_y["validation"]).cpu())
            probability = logits.softmax(1).cpu().numpy()
            train_probability = model(tensor_x["train"]).softmax(1).cpu().numpy()
        score = metric(y["validation"], probability)["balanced_accuracy"]
        train_score = metric(y["train"], train_probability)["balanced_accuracy"]
        row = {"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_loss":val_loss,
               "train_balanced_accuracy":train_score,"validation_balanced_accuracy":score}
        history.append(row)
        rss_peak = max(rss_peak, psutil.Process().memory_info().rss)
        if score > best_score + 1e-9 or (abs(score-best_score)<1e-9 and val_loss<best_loss):
            best_score, best_loss, best_epoch = score, val_loss, epoch
            best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"EEGNET epoch {epoch}: train_BA={train_score:.3f} val_BA={score:.3f} val_loss={val_loss:.4f}", flush=True)
            write_csv(run_dir / "eegnet_history.csv", history)
        if epoch >= settings["minimum_epochs"] and epoch-best_epoch >= settings["patience"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.load_state_dict(best_state)
    model.eval()
    checkpoint = {"model_state":best_state,"model_settings":model_settings,"channels":config["channels"],
                  "samples":x["train"].shape[2],"channel_mean_uv":torch.from_numpy(mean),"channel_std_uv":torch.from_numpy(std),
                  "config":config,"classes":["left_hand_imagery","right_hand_imagery"],"best_epoch":best_epoch}
    torch.save(checkpoint, run_dir / "eegnet_best.pt")
    probabilities = {}
    with torch.no_grad():
        for split in x:
            probabilities[split] = model(tensor_x[split]).softmax(1).cpu().numpy()
    write_csv(run_dir / "eegnet_history.csv", history)
    resource = {"parameters":sum(p.numel() for p in model.parameters()),"best_epoch":best_epoch,"epochs_ran":len(history),
                "fit_seconds":time.perf_counter()-started,"peak_training_process_rss_mib":rss_peak/1024**2,
                "peak_cuda_allocated_mib":torch.cuda.max_memory_allocated()/1024**2 if device.type=="cuda" else 0,
                "peak_cuda_reserved_mib":torch.cuda.max_memory_reserved()/1024**2 if device.type=="cuda" else 0}
    return probabilities, history, resource


def subject_metrics(y, probability, rows, model):
    output = []
    for subject in sorted({row["subject"] for row in rows}):
        idx = np.array([i for i,row in enumerate(rows) if row["subject"]==subject])
        scores = metric(y[idx], probability[idx])
        output.append({"model":model,"subject":subject,"n":len(idx),"accuracy":scores["accuracy"],"balanced_accuracy":scores["balanced_accuracy"],"roc_auc":scores["roc_auc"]})
    return output


def generate_figures(run_dir, y, probabilities, history, by_subject, config):
    fig, axes = plt.subplots(1,2,figsize=(10,4), layout="constrained")
    for ax,(name,p) in zip(axes, probabilities.items()):
        matrix = confusion_matrix(y, p.argmax(1), labels=[0,1])
        ax.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(2):
            for j in range(2):
                ax.text(j,i,str(matrix[i,j]),ha="center",va="center",fontsize=15)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Left","Right"],yticklabels=["Left","Right"],xlabel="Predicted",ylabel="True",title=name)
    fig.suptitle("Held-out run 12: left/right motor imagery")
    fig.savefig(run_dir / "confusion_matrices.png", dpi=180)
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs,[row["train_loss"] for row in history],label="Train")
    axes[0].plot(epochs,[row["validation_loss"] for row in history],label="Validation")
    axes[0].set(xlabel="Epoch",ylabel="Cross-entropy",title="EEGNet loss")
    axes[1].plot(epochs,[row["train_balanced_accuracy"] for row in history],label="Train")
    axes[1].plot(epochs,[row["validation_balanced_accuracy"] for row in history],label="Validation")
    axes[1].axhline(.5,color="gray",linestyle="--")
    axes[1].set(xlabel="Epoch",ylabel="Balanced accuracy",ylim=(0,1),title="EEGNet validation")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.savefig(run_dir / "learning_curves.png",dpi=180)
    plt.close(fig)
    fig,ax = plt.subplots(figsize=(10,4),layout="constrained")
    subjects = config["subjects"]
    for offset,name in ((-.18,"CSP-LDA"),(.18,"EEGNet")):
        values = [next(row["balanced_accuracy"] for row in by_subject if row["model"]==name and row["subject"]==subject) for subject in subjects]
        ax.bar(np.arange(len(subjects))+offset,values,width=.36,label=name)
    ax.axhline(.5,color="gray",linestyle="--",label="Chance reference")
    ax.set(xticks=np.arange(len(subjects)),xticklabels=[f"S{x:03d}" for x in subjects],ylim=(0,1),ylabel="Balanced accuracy",title="Held-out run performance by subject")
    ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.savefig(run_dir / "subject_performance.png",dpi=180)
    plt.close(fig)


def build_report(run_dir, config, summary, by_subject, qc):
    scores = summary["test_metrics"]
    lines = ["# 左右手运动想象：首轮真实训练结果", "", "## 实验定义", "",
             f"- 受试者：{', '.join(f'S{x:03d}' for x in config['subjects'])}。",
             "- 同一批受试者：run 4 训练，run 8 验证，run 12 测试；这是同人跨 run，不是跨人或跨天验证。",
             "- 标签：T1=左手运动想象，T2=右手运动想象；不涉及确认/取消语义或自由思想解码。",
             f"- 电极：{', '.join(config['channels'])}；平均参考；8—30 Hz 四阶因果 IIR；提示后 1—4 秒，每试次 480 点。",
             "- 固定峰峰值阈值 250 µV；采样率按实际文件读取，需要时重采样至 160 Hz。",
             "- 两个模型使用相同输入、划分；CSP 分量与 EEGNet checkpoint 仅按验证集选择。", "",
             "## 数据质量", "", f"- 本轮读取 {len(qc)} 个官方 EDF，输入 SHA-256 全部匹配。",
             f"- 训练/验证/测试试次数：{summary['split_counts']}。",
             f"- 剔除短任务区间 {sum(row['short_task_intervals'] for row in qc)} 个，按幅度/边界剔除 {sum(row['rejected_trials'] for row in qc)} 个。",
             "- 试次主键无交叉；一个完整试次只进入一个集合；尚未执行系统性眼动/肌电伪迹审计。", "",
             "## 测试集结果", "", "| 模型 | Accuracy | Balanced accuracy | ROC-AUC | 按受试者重采样的 BA 95% 区间 |", "|---|---:|---:|---:|---|" ]
    for name,score in scores.items():
        ci = score["subject_cluster_bootstrap_ba_95ci"]
        lines.append(f"| {name} | {score['accuracy']:.2%} | {score['balanced_accuracy']:.2%} | {score['roc_auc']:.3f} | {ci[0]:.2%}—{ci[1]:.2%} |")
    lines += ["", "两个模型的逐受试者结果在 `subject_metrics.csv`；小测试集与受试者差异会带来较大不确定性。区间仅描述这批数据，不代表任意新用户。", "",
              "## 对照和置信度", "",
              f"- CSP 训练标签打乱 {config['permutation_repeats']} 次，保持配置和测试集固定。对照结果：{summary['csp_permutation']}",
              "- 这是 CSP 的标签打乱对照，不是 EEGNet 的显著性检验；验证集选分量未纳入置换过程，p 值仅作探索性诊断。",
              "- 置信度阈值的覆盖率/错误率在 `confidence_validation.csv`；只在验证集分析，未据测试集调阈值。",
              "- 两个模型的概率尚未校准，也未训练休息/无操作检测；高概率不能直接授权手机执行。", "",
              "## 资源与复现", "", f"- EEGNet：{summary['eegnet_resources']}。", f"- CSP-LDA：{summary['csp_resources']}。",
              "- `config.json`、`environment.json`、`source_hashes.json` 记录运行配置、库版本与源码哈希。",
              "- `split_manifest.csv`、`data_quality.csv`、`prepared_trials.npz` 保存本轮划分、质量检查和预处理试次。",
              "- `csp_lda.joblib` 与 `eegnet_best.pt` 可用于这套预处理后的试次推理。", "",
              "## 可以据此说明什么", "",
              "本实验衡量在固定 EEG 范式下，从真实记录区分左右手运动想象的离线能力。是否稳定高于随机水平应结合对照、区间与逐受试者结果判断。",
              "尚不能据此声称新用户无需校准、全天候脑控、在线无误触发或医疗有效。下一轮优先做跨人留出、少量个人校准、休息拒识与因果流式回放。", "",
              "模型结构参考：[EEGNet 原论文](https://arxiv.org/abs/1611.08024)；信号与传统基线：[MNE 官方 CSP 示例](https://mne.tools/stable/auto_examples/decoding/decoding_csp_eeg.html)。", ""]
    (run_dir / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def run(config_path, cpu):
    task_root = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = task_root / "runs" / run_id
    run_dir.mkdir(parents=True)
    log = (run_dir / "training.log").open("w",encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(sys.stdout,log), Tee(sys.stderr,log)
    started = time.perf_counter()
    status = {"run_id":run_id,"run_dir":str(run_dir),"status":"RUNNING","stage":"preprocessing"}
    save_json(task_root / "latest_run.json",status)
    save_json(run_dir / "status.json",status)
    try:
        mne.set_log_level("ERROR")
        random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
        torch.set_num_threads(4)
        device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config["seed"])
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
        print(f"RUN {run_dir}\nDEVICE {device}",flush=True)
        save_json(run_dir / "config.json",config)
        versions = {name:metadata.version(name) for name in ("torch","numpy","mne","scipy","scikit-learn","matplotlib","joblib","psutil")}
        environment = {"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"versions":versions,
                       "cuda_runtime":torch.version.cuda,"device":str(device),"gpu":torch.cuda.get_device_name() if device.type=="cuda" else None,
                       "ram_total_gib":psutil.virtual_memory().total/1024**3,"torch_module":torch.__file__}
        save_json(run_dir / "environment.json",environment)
        save_json(run_dir / "source_hashes.json",{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in task_root.glob("*.py")})
        x,y,rows,qc = load_data(config,run_dir)
        status.update(stage="training")
        save_json(run_dir / "status.json",status)
        csp,csp_resources = train_csp(x,y,config,run_dir)
        eeg_prob,history,eeg_resources = train_eegnet(x,y,config,run_dir,device)
        csp_prob = {split:csp.predict_proba(data) for split,data in x.items()}
        test_probs = {"CSP-LDA":csp_prob["test"],"EEGNet":eeg_prob["test"]}
        test_metrics,by_subject,predictions = {},[],[]
        for name,probability in test_probs.items():
            test_metrics[name] = metric(y["test"],probability)
            test_metrics[name]["subject_cluster_bootstrap_ba_95ci"] = cluster_interval(y["test"],probability,rows["test"],config["seed"])
            by_subject.extend(subject_metrics(y["test"],probability,rows["test"],name))
            predictions.extend([{**row,"model":name,"prediction":int(p.argmax()),"p_left":float(p[0]),"p_right":float(p[1])} for row,p in zip(rows["test"],probability)])
        validation_rows = []
        for name,probability in (("CSP-LDA",csp_prob["validation"]),("EEGNet",eeg_prob["validation"])):
            for threshold in config["confidence_thresholds"]:
                accepted = probability.max(1) >= threshold
                accuracy = float((probability[accepted].argmax(1)==y["validation"][accepted]).mean()) if accepted.any() else None
                validation_rows.append({"model":name,"threshold":threshold,"accepted":int(accepted.sum()),"total":len(accepted),"coverage":float(accepted.mean()),"accepted_accuracy":accuracy})
        print("Running CSP label-permutation diagnostic...",flush=True)
        null_scores = []
        rng = np.random.default_rng(config["seed"])
        for repeat in range(config["permutation_repeats"]):
            shuffled = y["train"].copy()
            # Shuffle within each subject to preserve each subject's class balance.
            for subject in config["subjects"]:
                idx = [i for i,row in enumerate(rows["train"]) if row["subject"]==subject]
                shuffled[idx] = rng.permutation(shuffled[idx])
            null = csp_pipeline(csp_resources["selected_components"])
            null.fit(x["train"],shuffled)
            null_scores.append(float(balanced_accuracy_score(y["test"],null.predict(x["test"]))))
        observed = test_metrics["CSP-LDA"]["balanced_accuracy"]
        permutation = {"repeats":len(null_scores),"null_mean_ba":float(np.mean(null_scores)),"null_95percent_range":[float(v) for v in np.quantile(null_scores,[.025,.975])],"exploratory_p_ge_observed":float((1+sum(score>=observed for score in null_scores))/(1+len(null_scores)))}
        write_csv(run_dir / "permutation_scores.csv",[{"repeat":i+1,"balanced_accuracy":score} for i,score in enumerate(null_scores)])
        write_csv(run_dir / "subject_metrics.csv",by_subject)
        write_csv(run_dir / "test_predictions.csv",predictions)
        write_csv(run_dir / "confidence_validation.csv",validation_rows)
        summary = {"run_id":run_id,"protocol":"same-subject cross-run; pooled models; run04/train run08/validation run12/test",
                   "split_counts":{split:len(labels) for split,labels in y.items()},"test_metrics":test_metrics,
                   "csp_resources":csp_resources,"eegnet_resources":eeg_resources,"csp_permutation":permutation,
                   "total_seconds":time.perf_counter()-started}
        save_json(run_dir / "summary.json",summary)
        generate_figures(run_dir,y["test"],test_probs,history,by_subject,config)
        build_report(run_dir,config,summary,by_subject,qc)
        status.update(status="COMPLETE",stage="finished",summary=str(run_dir / "summary.json"))
        save_json(run_dir / "status.json",status)
        save_json(task_root / "latest_run.json",status)
        print("COMPLETE "+json.dumps(summary,ensure_ascii=False),flush=True)
    except Exception:
        status.update(status="FAILED",error=traceback.format_exc())
        save_json(run_dir / "status.json",status)
        save_json(task_root / "latest_run.json",status)
        traceback.print_exc()
        raise
    finally:
        sys.stdout,sys.stderr = original_stdout,original_stderr
        log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("config.json"))
    parser.add_argument("--cpu",action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(),args.cpu)
