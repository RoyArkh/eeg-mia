"""Sleep-EDF Expanded, Sleep Cassette study: N2 vs REM sleep, 78 healthy people aged 25-101.

Two EEG channels (Fpz-Cz, Pz-Oz) at 100 Hz, two nights per person (a few have one).
Every 30 s epoch was scored by an expert (R&K stages; 3 and 4 are merged into N3).
prepare_data.py keeps each night from 30 min before the first sleep epoch to 30 min after
the last one, and stores the stage of every sample.

A window is one 30 s epoch (3000 samples). Stride 3000 = no overlap, 1500 = 50% overlap
(a 50% window spans two epochs of the same stage). Each night is cut into 10-minute blocks
that go to train, val and test (70/15/15) in a random order fixed per night, so members
and non-members come from the same nights. (MoBI's split, with the last 15% of the night as
test, would give val and test very different stages: deep sleep comes early in the night
and REM late, and a 1-hour block can hold no REM at all.)
Source: physionet.org/content/sleep-edfx/1.0.0 (open access).
"""

from datasets.segments import PreparedDataset

NAME = "sleepedf"
STAGES = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}   # coding of the stored labels


class SleepEDF(PreparedDataset):
    name = NAME
    window = 3000            # 30 s at 100 Hz
    default_stride = 3000
    label_map = {STAGES["N2"]: 0, STAGES["REM"]: 1}
    split_mode = "blocks"
    block = 60000            # 10 minutes at 100 Hz
    default_counts = [2, 5, 10, 20, 40, 70]
