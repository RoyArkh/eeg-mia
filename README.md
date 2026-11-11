# eeg-mia

Membership inference attacks (MIA) on EEG models: a clean rewrite of the MoBI
experiments behind the HEALTHINF paper ("Understanding Membership Inference
Attacks on EEG Models"). The original code is in `../mia_bci` (see
`../guide.md`).

The task: a CNN learns to tell whether a 2.5 s EEG window was recorded while a
child was **resting** (label 0) or **playing a video game** (label 1). The
attacks then try to tell the model's training windows (members) apart from
held-out test windows (non-members), using only the model's output on each
window.

The dataset, the models and the attacks can each be chosen from the command
line. Each one lives in its own file:

```
datasets/   mobi.py  sleepedf.py  eegmmidb.py  tuab.py  (segments.py: shared code of the last three)
models/     eegnet.py  shallow_convnet.py  conv1d.py  eegnet2tl.py  tidnet.py
            eeg_inception.py  test_network.py
attacks/    loss.py  confidence.py  max_confidence.py  modified_entropy.py  lira.py
```

See `HOW_IT_WORKS.md` for diagrams of how the pieces fit together.

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
uv run pytest                    # about 10 seconds
```

The tests check that:
- the MoBI windows are identical to the original code's (same count per
  participant and split, at all 5 overlaps),
- no train window shares a sample with a val or test window,
- every model returns one logit per window and has exactly the original
  model's number of parameters,
- every simple attack gives AUC 1.0 when members and non-members are fully
  separable, and about 0.5 when they are not,
- loss, confidence and modified entropy give identical results (as they must on
  a binary model), while max confidence does not,
- LiRA beats the loss attack on made-up data where it should,
- one short training run with every attack (LiRA with 6 shadow models) works end
  to end, and results files are named and written correctly.

## Run

All commands run from the `eeg-mia-roya` folder. Every script accepts `--help`.

**One model** (the best place to start and to debug):

```bash
uv run python run_single.py --model EEGNet
uv run python run_single.py --model TIDNet --participants 1 2 3 4 5 --stride 69
uv run python run_single.py --model EEGNet --attacks loss max_confidence
```

It prints the number of windows, train/val/test loss and accuracy, the
generalization gaps, and each attack's AUC and TPR at 1% FPR.

**Participant sweep** (paper Experiment 1: train on 2–10 participants, 5 random
participant combinations each):

```bash
uv run python run_participant_sweep.py --stride 275
uv run python run_participant_sweep.py --models all --attacks all --stride 1
```

**What you can choose** (both scripts):

| flag | choices | default |
|---|---|---|
| `--dataset` | `mobi`, `sleepedf`, `eegmmidb`, `tuab` | `mobi` |
| `--model` / `--models` | `EEGNet`, `ShallowConvNet`, `Conv1DModel`, `EEGNet2TL`, `TIDNet`, `EEGInception`, `TestNetwork` (sweep also: `all`) | single: `EEGNet`; sweep: `EEGNet ShallowConvNet Conv1DModel` |
| `--attacks` | `loss`, `confidence`, `max_confidence`, `modified_entropy`, `lira`, or `all` | all but `lira` |
| `--stride` | window stride in samples (see below) | 275 |
| `--epochs` | maximum training epochs (early stopping on val loss) | 50 |
| `--shadows` | shadow models for LiRA | 64 |
| `--device` | `auto`, `cpu`, `cuda`, `mps` | `auto` (CUDA if available) |

## Other datasets: Sleep-EDF, EEGMMIDB, TUAB

Besides MoBI, three public datasets can be used. Each is converted once with
`prepare_data.py` into one `.npz` file per recording in `data/<dataset>/`. The
experiment scripts then take `--dataset sleepedf`, `eegmmidb` or `tuab`.

| | `sleepedf` | `eegmmidb` | `tuab` |
|---|---|---|---|
| data | Sleep-EDF Expanded, Sleep Cassette: 78 healthy people aged 25–101, 2 nights each | PhysioNet EEG Motor Movement/Imagery: 105 people | TUH Abnormal EEG: ~2,000 hospital patients |
| binary task | N2 (0) vs REM (1) sleep | imagined left (0) vs right (1) fist, runs 4, 8, 12 | normal (0) vs abnormal (1) |
| channels | Fpz-Cz, Pz-Oz, 100 Hz | 16 over the motor cortex, 160 Hz, 1–40 Hz | 16 bipolar (TCP montage), 0.5–45 Hz, resampled to 100 Hz |
| window | 30 s = one scored epoch (3000 samples) | the first 4 s of a trial (640) | 10 s (1000) |
| `--stride` | 3000 = no overlap (default), 1500 = 50% | always one window per trial, no overlap | 1000 = no overlap (default), 500 = 50% |
| members / non-members | each night cut into 10-min blocks that go to train/val/test (70/15/15) in a fixed random order, so every split covers the whole night | each person's 3 runs one after the other, split in time order | **split by patient**: members are training patients, non-members held-out patients |
| default sweep counts | 2 5 10 20 40 70 | 2 5 10 25 50 100 | 20 50 100 200 |
| download | open, ~8 GB, automatic | open, ~0.8 GB, automatic | free registration, ~60 GB, by hand |

Differences from MoBI:
- **Normalization per recording, not per window.** Each channel of a whole night,
  run or session is centered on its median, divided by its IQR-based standard
  deviation and clipped to ±10. Z-scoring each window, as for MoBI, would erase
  what these tasks depend on: the amplitude of sleep waves, abnormal slowing, and
  the power difference between left and right motor cortex.
- **A window has a single label.** Windows start where a stretch of one label
  starts (a sleep epoch, a trial), so they line up with the scoring.
- **Sleep-EDF is split in 10-minute blocks**, not in one block per split: deep sleep
  comes early in the night and REM late, so with MoBI's split (last 15% = test) val
  and test held very different stages, and val sometimes no REM at all.
- **TUAB is split by patient**, because its label belongs to the whole recording:
  with a time split, the model could get test windows right by recognizing the
  patient.
- Model sizes depend on the window length (layers that span the whole window grow),
  so the models are no longer all ~10K parameters; `n_params` in the results says
  how big each one was.

**Prepare the data** (once, on the machine that runs the experiments):

```bash
uv run python prepare_data.py eegmmidb            # ~10 min
uv run python prepare_data.py sleepedf            # downloads ~8 GB; can take an hour or more
uv run python prepare_data.py tuab --raw /path/to/tuh_eeg_abnormal/v3.0.1/edf
uv run python prepare_data.py sleepedf --subjects 0 1   # a quick test with 2 people
```

Raw downloads go to `data/raw/<dataset>/` and can be deleted afterwards.

**Check the data** before training:

```bash
uv run python inspect_data.py sleepedf
```

It prints hours, usable minutes per label, windows per split, channel amplitude
in microvolts, the share of clipped samples, and recordings that look unusual.
Per-recording numbers go to `results/data_report_<dataset>-stride<N>.csv`.

## Sample run commands

All commands run from the `eeg-mia-roya` folder (`~/eeg-mia` on the
workstation).

**Quick checks**

```bash
uv run pytest                                  # all tests, about 10 s
uv run python run_single.py --model EEGNet     # one model at stride 275, ~30 s on a laptop CPU
uv run python run_single.py --help             # every option
```

**One model**

```bash
# another model, fewer participants, 75% overlap
uv run python run_single.py --model TIDNet --participants 1 2 3 4 5 --stride 69

# only some attacks
uv run python run_single.py --model EEGNet --attacks loss max_confidence

# the paper's main MoBI setting (99.6% overlap): use the GPU
uv run python run_single.py --model EEGNet --stride 1

# TestNetwork (from the command line it has its default size: 40 filters, 2 layers)
uv run python run_single.py --model TestNetwork

# a quick LiRA test with few shadow models and short training
uv run python run_single.py --model Conv1DModel --attacks loss lira --shadows 8 --epochs 5
```

**Participant sweeps**

```bash
# the default: 3 models, every attack except LiRA, no overlap
uv run python run_participant_sweep.py

# all 7 models at the paper's main overlap
uv run python run_participant_sweep.py --models all --stride 1

# only the six models of the paper
uv run python run_participant_sweep.py --models EEGNet ShallowConvNet Conv1DModel EEGNet2TL TIDNet EEGInception

# a small, fast sweep to test that things work
uv run python run_participant_sweep.py --models Conv1DModel --counts 3 5 --combos 2 --epochs 3

# 20 participant combinations per count, like the original stride-275 run
uv run python run_participant_sweep.py --combos 20

# LiRA on top of the other attacks (64 shadow models per target: GPU, and slow)
uv run python run_participant_sweep.py --attacks all --stride 275
```

**Every model at every overlap, every attack except LiRA**

```bash
for stride in 275 206 138 69 1; do
    uv run python run_participant_sweep.py --models all --stride $stride \
        --attacks loss confidence max_confidence modified_entropy
done 2>&1 | tee -a sweep_all.log
```

Each stride writes its own results file. `tee` also keeps the printed output in
`sweep_all.log`.

**Long runs on the workstation**

```bash
ssh labws                        # on your Mac
tmux new -s sweep                # the run keeps going after you disconnect
cd ~/eeg-mia && git pull
# ... start the run here ...
# detach: Ctrl-b, then d    reattach later: tmux attach -t sweep
nvidia-smi                       # in another window: your python process should show type C
```

**Looking at results**

```bash
uv run python summarize.py results/<file>.csv
uv run python summarize.py results/*stride275*.csv          # several files combined
scp -r labws:~/eeg-mia/results ./results_workstation        # on your Mac: copy results back
```

## Results

Every run writes a **new file** in `results/`, named

```
<date>_<time>_<models>_<attacks>_<dataset>-stride<N>-epochs<N>.csv
e.g. 2026-11-04_19-30-12_EEGNet+TIDNet_loss+lira_mobi-stride275-epochs50.csv
```

One row per trained model. Each row holds:
- the run's settings: dataset, stride, window, epochs, batch size, learning
  rate, weight decay, dropout, patience, the full command,
- where it ran: computer, device, Python/torch/braindecode/numpy versions, git
  commit (`-dirty` means the code had uncommitted changes),
- the results: participants, seed, window counts, epochs run, train/val/test loss
  and accuracy, loss and accuracy gaps, and `<attack>_auc` and `<attack>_tpr` for
  every attack.

A sweep re-saves its file after every trained model, so a crash keeps what was
already done.

**Summarize** one or more results files, optionally next to the original results:

```bash
uv run python summarize.py results/<file>.csv
uv run python summarize.py results/<file>.csv \
    --reference ../mia_bci/experiment1_out/exp1_runs_stride275.csv
```

This prints, for each attack: mean AUC per number of participants and model,
means per model, and the correlation of the AUC with the loss gap, the accuracy
gap and the number of participants (paper Figs 1 and 2). Files from before the
`attacks/` folder existed (with an `attack_auc` column) are read as loss-attack
results.

## The attacks

| attack | score (higher = "member") | notes |
|---|---|---|
| `loss` | −BCE loss | Yeom et al. 2018 |
| `confidence` | probability of the **true** class | what `mia_bci/attacks.py` computed |
| `max_confidence` | probability of the **top** class | HEALTHINF Eq. 4; ignores the label |
| `modified_entropy` | −modified entropy | Song & Mittal 2021 |
| `lira` | z-score of the target's output against shadow models that did not train on the window | Carlini et al. 2022, offline; HEALTHINF Eq. 6 |

On a model with one output for two classes, `loss`, `confidence` and
`modified_entropy` all rank windows by the probability of the true class, so
they give the **same** AUC and TPR. (`confidence` can differ in the 4th decimal:
when the model is extremely sure, the probability rounds to exactly 1 and those
windows tie.) Only `max_confidence` and `lira` can really differ from `loss`.

**LiRA** trains `--shadows` extra models (64 by default) for **every** target
model, so a run takes dozens of times longer. Each shadow model trains like the
target on a random half of the train + test windows. See the docstring at the
top of `attacks/lira.py` for how it differs from `mia_bci/LiRA.ipynb` and for a
known limitation at high overlap.

## Overlap (`--stride`)

| `--stride` | overlap | where to run |
|---|---|---|
| 275 | 0% | laptop (seconds per model) |
| 206 | 25% | laptop |
| 138 | 50% | laptop |
| 69 | 75% | laptop |
| 1 | 99.6% (paper's main MoBI setting) | GPU recommended (the original code took up to ~30 min per model on CPU) |

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

| file or folder | what it does |
|---|---|
| `config.py` | default training, attack and results settings |
| `datasets/` | one file per dataset: loading, windowing, train/val/test split, normalization. `mobi.py` holds the MoBI settings (window, stride, participants); `segments.py` is shared by the other three |
| `prepare_data.py` | downloads or reads a raw dataset and writes `data/<dataset>/*.npz` |
| `inspect_data.py` | data-quality report for a prepared dataset |
| `models/` | one file per model, each with a `build()` function |
| `attacks/` | one file per attack, each with a `scores()` function; `__init__.py` runs them and computes AUC and TPR |
| `train.py` | training loop with early stopping; per-window logits, loss and accuracy |
| `pipeline.py` | `run_one()`: windows → train → evaluate → attacks, for one model and one participant set; also trains LiRA's shadow models |
| `recording.py` | results file names and contents |
| `run_single.py` | one run, printed in detail |
| `run_participant_sweep.py` | the participant-count experiment |
| `summarize.py` | summary tables and correlations; optional comparison with the original CSV |
| `tests/` | the checks listed above, one file per folder plus an end-to-end test |

**Adding a dataset, model or attack:** write one new file in the folder and add
one line to the dictionary in that folder's `__init__.py`. The docstring at the
top of each `__init__.py` says what the new file must provide.

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
- The six models and TestNetwork have exactly the original parameter counts.

## Different from the original code

- Windows are cut into arrays once per run instead of being sliced and
  normalized on every batch. The windows and normalization are the same; it
  is just faster.
- Model settings are passed explicitly, so a wrong setting name raises an
  error. The original silently dropped settings that braindecode didn't
  accept: EEGNet's `third_kernel_size` and ShallowConvNet's `pool_nonlin` never
  took effect, so they are not passed here either.
- braindecode is pinned to 1.6.1. The original runs used a bundled 1.6.0dev0
  copy (`../mia_bci/braindecode/`), which has the same model signatures and
  defaults.
- Participant combinations are drawn with NumPy instead of Python's `random`,
  so they differ from the original runs.
- No mixed precision or DataLoader workers.
- The attack scores are computed from the logits without clipping
  probabilities (the original clipped them to [1e-7, 1 − 1e-7]), so very
  confident windows are no longer tied.
- LiRA attacks the same target model and windows as the other attacks, and
  uses the logit-scaled confidence of HEALTHINF Eq. 6 by default (the notebook
  used the loss). See `attacks/lira.py`.
