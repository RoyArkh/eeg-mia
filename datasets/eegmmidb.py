"""EEG Motor Movement/Imagery Dataset (PhysioNet eegmmidb, PyHealth's EEGBCIDataset).

Imagined opening and closing of the left (label 0) or right (label 1) fist, runs 4, 8 and
12 of each of 105 people (subjects 88, 92, 100 and 104 are left out, as is usual: their
sampling rate or trial timing differs). 16 channels over the motor cortex, 160 Hz,
band-passed 1-40 Hz. Each trial lasts about 4.1 s.

A window is the first 4 s of a trial (640 samples), so there is exactly one window per
trial and no overlap.

Subjects 51 and 89 are also left out (EXCLUDED_AFTER_CHECK): in the data check
(inspect_data.py, 2026-11-11) the amplitude of all or almost all of their channels was more
than 4x away from the median over all recordings, in every run, so the recording itself
looks off. Their .npz files are still written; they are skipped when loading. A person's three runs are put one after the other and split in
time order (70/15/15 with a gap), so the test trials come from the end of run 12. (A run
has only ~15 trials, too few to split on its own.)
Source: physionet.org/content/eegmmidb/1.0.0 (open access).
"""

from datasets.segments import PreparedDataset

NAME = "eegmmidb"
RUNS = [4, 8, 12]          # imagined left / right fist
EXCLUDED_SUBJECTS = [88, 92, 100, 104]
CHANNELS = ["FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
            "CP3", "CP1", "CP2", "CP4"]
EXCLUDED_AFTER_CHECK = [51, 89]
LEFT, RIGHT = 0, 1         # stored labels (T1 = left fist, T2 = right fist; rest T0 = -1)


class EEGMMIDB(PreparedDataset):
    name = NAME
    window = 640             # 4 s at 160 Hz
    default_stride = 640
    label_map = {LEFT: 0, RIGHT: 1}
    split_mode = "time"
    time_split_unit = "participant"
    excluded = EXCLUDED_AFTER_CHECK
    default_counts = [2, 5, 10, 25, 50, 100]
