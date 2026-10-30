"""Run a saved EEGNet on a prepared held-out trial. No phone execution."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eegnet import EEGNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--trial-index", type=int, default=0)
    args = parser.parse_args()
    checkpoint = torch.load(args.run_dir / "eegnet_best.pt", map_location="cpu", weights_only=True)
    model = EEGNet(len(checkpoint["channels"]), checkpoint["samples"], **checkpoint["model_settings"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with np.load(args.run_dir / "prepared_trials.npz") as data:
        x = torch.from_numpy(data["X_test"][args.trial_index:args.trial_index+1])
        true_label = int(data["y_test"][args.trial_index])
    if len(x) != 1:
        raise ValueError("trial-index out of range")
    x = (x-checkpoint["channel_mean_uv"])/checkpoint["channel_std_uv"]
    with torch.no_grad():
        probabilities = model(x).softmax(1)[0].numpy()
    result = {"trial_index":args.trial_index,"class":checkpoint["classes"][int(probabilities.argmax())],
              "probabilities":dict(zip(checkpoint["classes"], probabilities.astype(float))),
              "true_class":checkpoint["classes"][true_label],"probability_calibrated":False,
              "phone_execution_authorized":False,"mode":"offline prepared EEG trial"}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
