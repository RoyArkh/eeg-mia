"""Data-quality report for a prepared dataset: one row per recording, plus a printed summary.

    uv run python inspect_data.py sleepedf
    uv run python inspect_data.py tuab --stride 500

Per recording: duration, usable minutes of each label, windows of each label per split
(train/val/test, before balancing), the per-channel amplitude scale in microvolts, and
the share of clipped samples. Recordings that look unusual are listed at the end: a flat
or very noisy channel (scale under 1/4 or over 4x the median for that channel), more than
1% clipped samples, or a participant without val or test windows of one label. Saved to results/data_report_<dataset>-stride<N>.csv.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config
import datasets
from datasets.segments import CLIP, SPLITS, map_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=[name for name in datasets.DATASETS if name != "mobi"])
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.dataset, data_dir=args.data_dir, stride=args.stride)

    rows = []
    for subject in dataset.participants:
        split_of = {}    # (recording name, start) -> split
        if dataset.split_mode != "subject":
            for split, windows in dataset.time_split(subject).items():
                split_of.update({(rec.name, start): split for rec, start, _ in windows})
        for rec in dataset.recordings[subject]:
            with np.load(os.path.join(dataset.data_dir, rec.name + ".npz")) as npz:
                labels = map_labels(npz["labels"], dataset.label_map)
                scale = npz["scale_uv"]
            fs = dataset.sampling_rate
            row = {"subject": subject, "recording": rec.name,
                   "minutes": rec.signal.shape[1] / fs / 60,
                   "minutes_label0": (labels == 0).sum() / fs / 60,
                   "minutes_label1": (labels == 1).sum() / fs / 60,
                   "clipped_fraction": float((np.abs(rec.signal) >= CLIP).mean())}
            if dataset.split_mode != "subject":
                splits_here = [split_of.get((rec.name, start)) for start in rec.starts]
                for split in SPLITS:
                    for label in (0, 1):
                        row[f"{split}_label{label}"] = sum(1 for sp, lab in zip(splits_here, rec.labels)
                                                           if sp == split and lab == label)
            else:
                for label in (0, 1):
                    row[f"windows_label{label}"] = int((rec.labels == label).sum())
            for channel, value in zip(dataset.channels, scale):
                row[f"scale_uv_{channel}"] = float(value)
            rows.append(row)
    report = pd.DataFrame(rows)

    scale_columns = [c for c in report if c.startswith("scale_uv_")]
    ratio = report[scale_columns] / report[scale_columns].median()
    report["odd_channels"] = [", ".join(c[len("scale_uv_"):] for c in scale_columns
                                        if not 0.25 <= r[c] <= 4) for _, r in ratio.iterrows()]
    problems = report["odd_channels"] != ""
    problems |= report["clipped_fraction"] > 0.01
    if dataset.split_mode != "subject":   # per participant: no val or test windows of one label
        columns = ["val_label0", "val_label1", "test_label0", "test_label1"]
        per_subject = report.groupby("subject")[columns].transform("sum")
        problems |= (per_subject == 0).any(axis=1)

    print(f"{dataset.label}: {len(dataset.participants)} participants, {len(report)} recordings, "
          f"{report['minutes'].sum() / 60:.1f} hours, {dataset.n_channels} channels at "
          f"{dataset.sampling_rate:g} Hz, window {dataset.window} samples")
    print(f"usable minutes: label 0 = {report['minutes_label0'].sum():.0f}, "
          f"label 1 = {report['minutes_label1'].sum():.0f}")
    window_columns = [c for c in report if c.endswith(("_label0", "_label1")) and not c.startswith("minutes")]
    print("windows before balancing: " + ", ".join(f"{c} {report[c].sum()}" for c in window_columns))
    print("channel scale (microvolts, median over recordings): "
          + ", ".join(f"{c[len('scale_uv_'):]} {report[c].median():.1f}" for c in scale_columns))
    print(f"clipped samples: median {report['clipped_fraction'].median():.4%}, "
          f"max {report['clipped_fraction'].max():.4%}")

    rng = np.random.default_rng(0)
    splits = dataset.splits(dataset.participants, rng)
    print("windows after balancing, all participants: "
          + ", ".join(f"{split} {len(y)}" for split, (_, y) in splits.items()))

    if problems.any():
        print(f"\n{problems.sum()} recordings to look at:")
        columns = ["subject", "recording", "minutes", "clipped_fraction", "odd_channels"]
        if dataset.split_mode != "subject":
            columns += ["val_label0", "val_label1", "test_label0", "test_label1"]
        print(report.loc[problems, columns].to_string(index=False))
    else:
        print("\nno unusual recordings")

    path = os.path.join(config.RESULTS_DIR, f"data_report_{dataset.label}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"\nsaved to {path}")


if __name__ == "__main__":
    main()
