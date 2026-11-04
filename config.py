"""Default settings for training, attacks and results.

Dataset settings (window length, stride, participants, ...) are in the dataset's own
file in datasets/. Training settings follow the original HEALTHINF code
(mia_bci/exp1_run.ipynb).
"""

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_DIR, "data")         # each dataset has a subfolder
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")    # one CSV per experiment run

# what runs by default (all choices: see datasets/, models/ and attacks/)
DATASET = "mobi"
MODELS = ["EEGNet", "ShallowConvNet", "Conv1DModel"]
# LiRA is not run by default because it trains extra shadow models (64 by default) per target model
ATTACKS = ["loss", "confidence", "max_confidence", "modified_entropy"]

# training
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
DROPOUT = 0.2
PATIENCE = 8               # early stopping: epochs without validation improvement
MIN_SPLIT_SIZE = 10        # skip a run if train, val or test has fewer windows

# attacks
TARGET_FPR = 0.01          # TPR is reported at this false positive rate

SEED = 0
