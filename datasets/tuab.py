"""TUH Abnormal EEG Corpus (TUAB): normal (label 0) vs abnormal (label 1) clinical EEG.

Hospital EEGs from Temple University Hospital. prepare_data.py takes one session per
patient from the official train folder (patients recorded as both normal and abnormal are
left out), skips the first minute, keeps the next 5 minutes, converts to the 16 bipolar
channels of the TCP montage (as BIOT and PyHealth), band-passes 0.5-45 Hz (this also
removes the 60 Hz mains) and resamples to 100 Hz.

A window is 10 s (1000 samples). Stride 1000 = no overlap, 500 = 50% overlap.
The label belongs to the whole recording, so the split is by PATIENT: members are windows of
training patients, non-members windows of held-out patients (70/15/15, stratified by label).
A time split inside each recording would let the model get test windows right just by
recognizing the patient.
Source: isip.piconepress.com/projects/nedc/html/tuh_eeg (free, registration required).
"""

from datasets.segments import PreparedDataset

NAME = "tuab"
NORMAL, ABNORMAL = 0, 1
BIPOLAR = [("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
           ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
           ("FP1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("FP2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")]
SAMPLING_RATE = 100
SKIP_SECONDS = 60          # the start of a clinical EEG is often noisy
KEEP_SECONDS = 300


class TUAB(PreparedDataset):
    name = NAME
    window = 1000            # 10 s at 100 Hz
    default_stride = 1000
    label_map = {NORMAL: 0, ABNORMAL: 1}
    split_mode = "subject"
    default_counts = [20, 50, 100, 200]
