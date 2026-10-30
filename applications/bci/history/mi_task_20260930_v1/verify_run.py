"""Verify split separation and reproduce every held-out prediction from checkpoints."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from eegnet import EEGNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = list(csv.DictReader((run_dir / "split_manifest.csv").open(encoding="utf-8-sig")))
    keys = [(row["subject"],row["run"],row["onset_sample"]) for row in manifest]
    assert len(keys)==len(set(keys)), "trial leakage"
    for split,run in (("train","4"),("validation","8"),("test","12")):
        assert {row["run"] for row in manifest if row["split"]==split}=={run}
    with np.load(run_dir / "prepared_trials.npz") as data:
        x,y = data["X_test"],data["y_test"]
    csp = joblib.load(run_dir / "csp_lda.joblib")
    csp_probability = csp.predict_proba(x)
    checkpoint = torch.load(run_dir / "eegnet_best.pt",map_location="cpu",weights_only=True)
    model = EEGNet(len(checkpoint["channels"]),checkpoint["samples"],**checkpoint["model_settings"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    standardized = (torch.from_numpy(x)-checkpoint["channel_mean_uv"])/checkpoint["channel_std_uv"]
    with torch.no_grad():
        eeg_probability = model(standardized).softmax(1).numpy()
    predictions = list(csv.DictReader((run_dir / "test_predictions.csv").open(encoding="utf-8-sig")))
    maximum_difference = {}
    for name,p in (("CSP-LDA",csp_probability),("EEGNet",eeg_probability)):
        rows = [row for row in predictions if row["model"]==name]
        assert len(rows)==len(y)
        recorded = np.array([[float(row["p_left"]),float(row["p_right"])] for row in rows])
        difference = float(np.max(np.abs(recorded-p)))
        assert difference<1e-4,(name,difference)
        assert np.array_equal(p.argmax(1),recorded.argmax(1))
        assert abs(balanced_accuracy_score(y,p.argmax(1))-summary["test_metrics"][name]["balanced_accuracy"])<1e-12
        maximum_difference[name]=difference
    result = {"status":"PASS","trial_keys_unique":len(keys),"holdout_predictions_reproduced":2*len(y),
              "max_cpu_reload_probability_difference":maximum_difference}
    (run_dir / "checkpoint_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
