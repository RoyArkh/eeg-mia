# eeg-mia

Membership inference attacks (MIA) on EEG models: a clean rewrite of the MoBI
experiments behind the HEALTHINF paper ("Understanding Membership Inference
Attacks on EEG Models"). The original code is in `../mia_bci` (see
`../mia_bci/guide.md`).

The task: a CNN learns to tell whether a 2.5 s EEG window was recorded while a
child was **resting** (label 0) or **playing a video game** (label 1). The LOSS
attack then tries to tell the model's training windows (members) apart from
held-out test windows (non-members), using only the model's loss on each window.

## Setup

```bash
cd eeg-mia-roya
uv sync                     # creates .venv with Python 3.12 and installs everything
mkdir -p data/mobi
cp ../mia_bci/preprocessed/*.npz data/mobi/
```

`uv sync` writes `uv.lock` with the exact package versions. Commit it, so every
result can be traced to the same environment.

`data/` and `results/` are git-ignored: the EEG files hold participant data
(age, sex) and should not go into git.

## Check that everything works

```bash
uv run pytest                    # about a minute
uv run python models.py          # parameter count and output shape of each model
```

The tests check that:
- the windows are identical to the original code's (same count per participant
  and split, at all 5 overlaps),
- no train window shares a sample with a val or test window,
- every model returns one logit per window,
- the attack gives AUC 1.0 when members and non-members are fully separable,
  and about 0.5 when they are not,
- one short training run works end to end.

## Run

All commands run from the `eeg-mia-roya` folder. Every script accepts `--help`.

**One model** (the best place to start and to debug):

```bash
uv run python run_single.py --model EEGNet
uv run python run_single.py --model ShallowConvNet --participants 1 2 3 4 5
```

It prints the number of windows, train/val/test loss and accuracy, the
generalization gaps, and the attack's AUC and TPR at 1% FPR.

**Participant sweep** (paper Experiment 1: train on 2–10 participants, 5 random
participant combinations each, all 3 models):

```bash
uv run python run_participant_sweep.py --stride 275
```

Results are written to `results/participant_sweep_stride275.csv`, one row per
trained model. The CSV is re-saved after every run, so a crash keeps what was
already done.

**Compare with the original results:**

```bash
uv run python summarize.py results/participant_sweep_stride275.csv \
    --reference ../mia_bci/experiment1_out/exp1_runs_stride275.csv
```

This prints, for both runs side by side: mean attack AUC per number of
participants, means per model, and the correlation of attack AUC with the loss
gap, the accuracy gap and the number of participants (paper Figs 1 and 2).

## Overlap (`--stride`)

| `--stride` | overlap | where to run |
|---|---|---|
| 275 | 0% | laptop (seconds per model) |
| 206 | 25% | laptop |
| 138 | 50% | laptop |
| 69 | 75% | laptop |
| 1 | 99.6% (paper's main MoBI setting) | GPU recommended (the original code took up to ~30 min per model on CPU) |

Use `--device mps` to try the Apple GPU, or `--device cuda` on a cloud GPU. The
default is CUDA if available, otherwise CPU.

## What to expect

The original runs (`../mia_bci/experiment1_out/`) give these averages:

| model | stride 275: test acc | stride 275: attack AUC | stride 1: test acc | stride 1: attack AUC |
|---|---|---|---|---|
| EEGNet | 0.49 | 0.60 | 0.72 | 0.64 |
| ShallowConvNet | 0.51 | 0.65 | 0.76 | 0.65 |
| Conv1DModel | 0.58 | 0.64 | 0.68 | 0.74 |

- **At stride 275 the models barely learn the task.** Test accuracy is near 0.5
  because there are only 250 training windows. The attack still works
  (AUC ≈ 0.6) because the models memorise those few windows. Near-chance
  accuracy here is expected and does not mean the code is broken.
- **At stride 275 the attack sets are tiny** (e.g. 46 test windows for all 11
  participants), so single-run AUCs are noisy and TPR at 1% FPR is close to
  meaningless. Compare averages over many runs.
- **Numbers won't match the original exactly.** The participant combinations,
  window sampling and weight initialization use different random draws. Means
  should be close. `--combos 20` matches the number of combinations in the
  original stride-275 run.

## Files

| file | what it does |
|---|---|
| `config.py` | all default settings in one place |
| `data.py` | loads `SubjectN.npz`, finds windows with good electrode contact, splits each recording by time into train/val/test, balances rest/game, cuts and normalizes windows |
| `models.py` | EEGNet and ShallowConvNet (from braindecode) and Conv1DModel |
| `train.py` | training loop with early stopping; per-window loss and accuracy |
| `attacks.py` | LOSS attack, AUC and TPR at 1% FPR |
| `pipeline.py` | `run_one()`: windows → train → evaluate → attack, for one model and one participant set |
| `run_single.py` | one run, printed in detail |
| `run_participant_sweep.py` | the participant-count experiment, saved as CSV |
| `summarize.py` | summary tables and correlations; optional comparison with the original CSV |
| `tests/test_pipeline.py` | the checks listed above |

## Same as the original code

- 2.5 s windows (275 samples at 110 Hz). Windows with more than 30% poor
  electrode contact are dropped.
- Each recording is split in time order into 70% train, 15% val and 15% test,
  with a one-window gap between blocks.
- Equal numbers of rest and game windows per participant and split.
- Each window is z-scored per channel and clipped to ±4.
- Training: Adam (lr 1e-3), batch 128, binary cross-entropy, up to 50 epochs,
  dropout 0.2, early stopping on val loss (patience 8), best weights restored.
- Attack: members are train windows and non-members are test windows, with
  equal numbers of each.

## Different from the original code

- Windows are cut into arrays once per run instead of being sliced and
  normalized on every batch. The windows and normalization are the same; it
  is just faster.
- Model settings are passed explicitly, so a wrong setting name raises an
  error. The original silently dropped settings that braindecode didn't
  accept: EEGNet's `third_kernel_size` and ShallowConvNet's `pool_nonlin` never
  took effect, so they are not passed here either.
- braindecode is pinned to 1.6.1. The original runs used a bundled 1.6.0dev0
  copy (`../mia_bci/braindecode/`), which has the same model signatures.
- Participant combinations are drawn with NumPy instead of Python's `random`,
  so they differ from the original runs.
- No mixed precision or DataLoader workers. Runs use CPU by default on the Mac.
- Only the LOSS attack for now. The paper found the confidence and modified
  entropy attacks nearly identical to it on this binary task.
