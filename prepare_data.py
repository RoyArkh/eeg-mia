"""Download (or read) a raw dataset and write one .npz file per recording to data/<dataset>/.

Run once per dataset, on the machine that runs the experiments:

    uv run python prepare_data.py sleepedf                    # downloads ~8 GB from PhysioNet
    uv run python prepare_data.py eegmmidb                    # downloads ~0.8 GB from PhysioNet
    uv run python prepare_data.py tuab --raw /path/to/tuh_eeg_abnormal/v3.0.1/edf

Add `--subjects 0 1 2` for a quick test with a few participants. Raw downloads go to
data/raw/<dataset>/ and can be deleted afterwards. The .npz format is described in
datasets/segments.py.
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import mne
import numpy as np

import config
from datasets import eegmmidb, sleepedf, tuab
from datasets.segments import robust_normalize

mne.set_log_level("ERROR")


def save(out_dir, name, signal_uv, labels, subject, fs, channels, **extra):
    signal, center, scale = robust_normalize(signal_uv)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{name}.npz"), signal=signal, labels=labels.astype(np.int8),
             subject=subject, fs=fs, channels=np.array(channels), center_uv=center, scale_uv=scale,
             **extra)


# ---------------------------------------------------------------- Sleep-EDF

SLEEP_STAGE_CODES = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
                     "Sleep stage 4": 3, "Sleep stage R": 4}   # "?" and "Movement time" stay -1
SLEEP_CHANNELS = ["Fpz-Cz", "Pz-Oz"]
WAKE_MARGIN_SECONDS = 30 * 60


def sleep_stage_labels(annotations, n_samples, fs):
    """One stage per sample (sleepedf.STAGES coding, -1 = not scored) from hypnogram annotations."""
    labels = np.full(n_samples, -1, dtype=np.int8)
    for onset, duration, description in zip(annotations.onset, annotations.duration,
                                             annotations.description):
        if description in SLEEP_STAGE_CODES:
            begin = int(round(onset * fs))
            end = min(n_samples, int(round((onset + duration) * fs)))
            labels[max(0, begin):end] = SLEEP_STAGE_CODES[description]
    return labels


def trim_to_sleep(labels, fs, margin_seconds=WAKE_MARGIN_SECONDS):
    """(begin, end) sample range from `margin` before the first to `margin` after the last sleep epoch."""
    asleep = np.flatnonzero(np.isin(labels, [1, 2, 3, 4]))
    if len(asleep) == 0:
        return 0, 0
    margin = int(margin_seconds * fs)
    return max(0, asleep[0] - margin), min(len(labels), asleep[-1] + 1 + margin)


def prepare_sleepedf(out_dir, raw_dir, subjects):
    from mne.datasets.sleep_physionet.age import fetch_data
    subjects = subjects if subjects is not None else list(range(83))
    os.makedirs(raw_dir, exist_ok=True)
    files = fetch_data(subjects=subjects, recording=[1, 2], path=raw_dir, on_missing="warn")
    for psg_file, hypnogram_file in files:
        match = re.search(r"SC4(\d\d)(\d)", os.path.basename(psg_file))
        subject, night = int(match.group(1)), int(match.group(2))
        raw = mne.io.read_raw_edf(psg_file, preload=True, infer_types=True)
        raw.set_annotations(mne.read_annotations(hypnogram_file), emit_warning=False)
        fs = raw.info["sfreq"]
        channels = [next(ch for ch in raw.ch_names if ch.endswith(name)) for name in SLEEP_CHANNELS]
        signal = raw.get_data(picks=channels) * 1e6     # volts -> microvolts
        labels = sleep_stage_labels(raw.annotations, signal.shape[1], fs)
        begin, end = trim_to_sleep(labels, fs)
        if end <= begin:
            print(f"  subject {subject} night {night}: no sleep scored, skipped")
            continue
        save(out_dir, f"S{subject:02d}_N{night}", signal[:, begin:end], labels[begin:end],
             subject, fs, SLEEP_CHANNELS)
        stages = np.bincount(labels[begin:end][labels[begin:end] >= 0], minlength=5) / fs / 60
        print(f"  subject {subject:2d} night {night}: {(end - begin) / fs / 3600:.1f} h, minutes "
              + " ".join(f"{s}={m:.0f}" for s, m in zip(sleepedf.STAGES, stages)))


# ---------------------------------------------------------------- EEGMMIDB

def prepare_eegmmidb(out_dir, raw_dir, subjects):
    from mne.datasets import eegbci
    subjects = subjects if subjects is not None else list(range(1, 110))
    subjects = [s for s in subjects if s not in eegmmidb.EXCLUDED_SUBJECTS]
    os.makedirs(raw_dir, exist_ok=True)
    for subject in subjects:
        files = eegbci.load_data(subject, eegmmidb.RUNS, path=raw_dir, update_path=False)
        for run, path in zip(eegmmidb.RUNS, files):
            raw = mne.io.read_raw_edf(path, preload=True)
            eegbci.standardize(raw)
            fs = raw.info["sfreq"]
            if fs != 160:
                print(f"  subject {subject} run {run}: {fs} Hz, skipped")
                continue
            raw.filter(1.0, 40.0)
            signal = raw.get_data(picks=eegmmidb.CHANNELS) * 1e6
            labels = np.full(signal.shape[1], -1, dtype=np.int8)
            codes = {"T1": eegmmidb.LEFT, "T2": eegmmidb.RIGHT}
            for onset, duration, description in zip(raw.annotations.onset, raw.annotations.duration,
                                                     raw.annotations.description):
                if description in codes:
                    begin = int(round(onset * fs))
                    labels[begin:begin + int(round(duration * fs))] = codes[description]
            save(out_dir, f"S{subject:03d}_R{run:02d}", signal, labels, subject, fs, eegmmidb.CHANNELS)
        print(f"  subject {subject:3d}: runs {eegmmidb.RUNS}")


# ---------------------------------------------------------------- TUAB

def tuab_bipolar(raw):
    """The 16 TCP bipolar channels (microvolts) from a TUH raw with channels like 'EEG FP1-REF'."""
    names = {}
    for ch in raw.ch_names:
        match = re.match(r"EEG ([A-Z0-9]+)-(REF|LE)$", ch.upper())
        if match:
            names[match.group(1)] = ch
    needed = sorted({e for pair in tuab.BIPOLAR for e in pair})
    missing = [e for e in needed if e not in names]
    if missing:
        raise ValueError(f"missing electrodes {missing}")
    data = raw.get_data(picks=[names[e] for e in needed]) * 1e6
    index = {e: i for i, e in enumerate(needed)}
    return np.stack([data[index[a]] - data[index[b]] for a, b in tuab.BIPOLAR])


def tuab_process(raw):
    """Crop, convert to bipolar, band-pass 0.5-45 Hz and resample to 100 Hz. Returns [16, time]."""
    fs = raw.info["sfreq"]
    end = min(raw.times[-1], tuab.SKIP_SECONDS + tuab.KEEP_SECONDS)
    raw.crop(tmin=tuab.SKIP_SECONDS, tmax=end, include_tmax=False).load_data()
    signal = tuab_bipolar(raw)
    signal = mne.filter.filter_data(signal, fs, 0.5, 45.0)
    return mne.filter.resample(signal, up=tuab.SAMPLING_RATE, down=fs)


def prepare_tuab(out_dir, raw_dir, subjects, split="train"):
    root = Path(raw_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"{root} not found. Pass --raw .../tuh_eeg_abnormal/v3.0.1/edf")
    files = defaultdict(list)    # patient -> [(label, path)]
    for path in sorted(root.rglob("*.edf")):
        label = tuab.ABNORMAL if "abnormal" in path.parts else tuab.NORMAL
        files[path.name.split("_")[0]].append((label, path))
    both = [p for p, fl in files.items() if len({label for label, _ in fl}) > 1]
    patients = [p for p in sorted(files) if p not in both]
    if subjects is not None:
        patients = [p for p in patients if p in {str(s) for s in subjects}]
    print(f"  {len(files)} patients in {root}, {len(both)} with both labels left out, "
          f"preparing {len(patients)}")
    skipped = 0
    for i, patient in enumerate(patients, 1):
        label, path = files[patient][0]     # first session, first file
        try:
            raw = mne.io.read_raw_edf(path, preload=False)
            if raw.times[-1] < tuab.SKIP_SECONDS + 60:
                raise ValueError("shorter than 2 minutes")
            signal = tuab_process(raw)
        except Exception as error:   # a few files have unusual channel sets
            skipped += 1
            print(f"  {path.name}: skipped ({error})")
            continue
        labels = np.full(signal.shape[1], label, dtype=np.int8)
        save(out_dir, patient, signal, labels, patient, float(tuab.SAMPLING_RATE),
             [f"{a}-{b}" for a, b in tuab.BIPOLAR], source=path.name)
        if i % 100 == 0:
            print(f"  {i}/{len(patients)} patients done")
    print(f"  done, {skipped} skipped")


# ----------------------------------------------------------------

PREPARE = {"sleepedf": prepare_sleepedf, "eegmmidb": prepare_eegmmidb, "tuab": prepare_tuab}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(PREPARE))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="only these participants (numbers; TUAB: patient ids)")
    parser.add_argument("--raw", default=None, help="raw data folder (default: data/raw/<dataset>)")
    parser.add_argument("--out", default=None, help="output folder (default: data/<dataset>)")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(config.DATA_ROOT, args.dataset)
    raw_dir = args.raw or os.path.join(config.DATA_ROOT, "raw", args.dataset)
    subjects = args.subjects
    if subjects is not None and args.dataset != "tuab":
        subjects = [int(s) for s in subjects]
    print(f"{args.dataset}: raw data in {raw_dir}, writing to {out_dir}")
    PREPARE[args.dataset](out_dir, raw_dir, subjects)


if __name__ == "__main__":
    main()
