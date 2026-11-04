"""Default settings. They follow the original HEALTHINF code (mia_bci/exp1_run.ipynb)."""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data", "mobi")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# MoBI dataset: the 11 participants with both rest and game recordings
PARTICIPANTS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13]
SAMPLING_RATE = 110   # Hz
WINDOW = 275          # samples (2.5 s)
STRIDE = 275          # 275 = no overlap, 1 = 99.6% overlap (the paper's main MoBI setting)

# windowing and splitting
MAX_BAD_FRACTION = 0.30    # drop windows where >30% of samples have poor electrode contact
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15        # the remaining 15% is test
MIN_WINDOWS_TO_SPLIT = 8   # recordings with fewer windows go entirely to train
Z_CLIP = 4.0               # z-scored windows are clipped to +-4

# training
MODELS = ["EEGNet", "ShallowConvNet", "Conv1DModel"]
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.2
PATIENCE = 8               # early stopping: epochs without validation improvement
MIN_SPLIT_SIZE = 10        # skip a run if train, val or test has fewer windows

SEED = 0
