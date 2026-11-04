"""Saving results: every experiment run writes its own CSV file to results/.

File name: <date>_<time>_<models>_<attacks>_<dataset settings>-epochs<N>.csv, for example

    results/2026-11-04_19-30-12_EEGNet+TIDNet_loss+lira_mobi-stride275-epochs50.csv

Each row is one trained target model. Every row also holds the run's settings (dataset,
stride, epochs, learning rate, ...) and where it ran (computer, device, package versions,
git commit), so a single row tells you how its numbers were made. Files from different
runs can be combined with pandas.concat.
"""

import datetime
import os
import platform
import socket
import subprocess
import sys

import braindecode
import numpy as np
import pandas as pd
import torch

import config


def results_path(model_names, attack_names, dataset, epochs, results_dir=config.RESULTS_DIR):
    """A new, unused file path for this run's results."""
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = "_".join([now, "+".join(model_names), "+".join(attack_names), f"{dataset.label}-epochs{epochs}"])
    path = os.path.join(results_dir, f"{name}.csv")
    n = 2
    while os.path.exists(path):   # two runs started in the same second
        path = os.path.join(results_dir, f"{name}-{n}.csv")
        n += 1
    return path


def run_info(experiment, dataset, device, epochs):
    """Settings and environment shared by every row of one run."""
    return {
        "experiment": experiment,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        **dataset.settings(),
        "epochs": epochs,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "dropout": config.DROPOUT,
        "patience": config.PATIENCE,
        "computer": socket.gethostname(),
        "device": str(device),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "braindecode": braindecode.__version__,
        "numpy": np.__version__,
        "command": " ".join(sys.argv),
    }


def save(rows, path):
    """Write all rows so far. Called after every trained model, so a crash keeps finished ones."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def git_commit():
    """Short hash of the code's git commit, plus "-dirty" if tracked files have uncommitted changes."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=config.PROJECT_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def print_environment(device):
    print(f"python {platform.python_version()}, torch {torch.__version__}, "
          f"braindecode {braindecode.__version__}, numpy {np.__version__}, device: {device}")
