# How the code works

The diagrams use [Mermaid](https://mermaid.js.org/). GitHub draws them
automatically. In VS Code, install the "Markdown Preview Mermaid Support"
extension and open the preview (Cmd+Shift+V).

## 1. Which file calls which

```mermaid
flowchart LR
    npz[("data/mobi/<br/>SubjectN.npz")]
    single["<b>run_single.py</b><br/>one run, printed"]
    sweep["<b>run_participant_sweep.py</b><br/>many runs"]
    pipe["<b>pipeline.py</b><br/>run_one()"]
    datasets["<b>datasets/</b><br/>mobi.py"]
    models["<b>models/</b><br/>one file per CNN"]
    train["<b>train.py</b><br/>training and evaluation"]
    attacks["<b>attacks/</b><br/>one file per attack"]
    rec["<b>recording.py</b><br/>file name + settings"]
    csv[("results/<br/>date_time_models_<br/>attacks_settings.csv")]
    summ["<b>summarize.py</b><br/>tables and correlations"]

    npz --> datasets
    single --> pipe
    sweep --> pipe
    pipe --> datasets
    pipe --> models
    pipe --> train
    pipe --> attacks
    single --> rec
    sweep --> rec
    rec --> csv --> summ
```

The scripts pick the dataset, models and attacks by name (`--dataset`,
`--models`, `--attacks`). Each folder's `__init__.py` holds a dictionary from
name to file, so adding one means one new file plus one line. Defaults for
training are in `config.py`. Dataset settings (window, stride, participants)
are in the dataset's own file.

## 2. One run, step by step (`pipeline.run_one`)

```mermaid
flowchart TD
    A[("<b>SubjectN.npz</b><br/>rest: 4 ch × 20–75 s<br/>game: 4 ch × 10–20 min<br/>+ poor-contact flags")]

    subgraph D ["datasets/mobi.py"]
        B["<b>good_window_starts()</b><br/>slide a 2.5 s window<br/>along each recording,<br/>moving --stride samples;<br/>drop windows with<br/>>30% poor contact"]
        C["<b>split_by_time()</b><br/>split each recording<br/>in time order:<br/>train 70%, val 15%,<br/>test 15%, gaps between"]
        Dd["<b>build_splits()</b><br/>equal rest and game<br/>windows per participant<br/>and split (random pick)"]
        E["<b>windows_to_arrays()</b><br/>X [n, 4, 275], y [n]<br/>z-score each channel,<br/>clip to ±4"]
    end

    subgraph T ["models/ + train.py"]
        F["<b>build_model()</b><br/>the --model chosen,<br/>window → 1 logit<br/>(rest = 0, game = 1)"]
        G["<b>train_model()</b><br/>Adam, cross-entropy,<br/>batch 128, ≤ 50 epochs;<br/>stop after 8 epochs<br/>without val improvement;<br/>keep the best weights"]
        H["<b>evaluate()</b><br/>logit and loss of<br/>every window,<br/>accuracy per split"]
    end

    subgraph AT ["attacks/"]
        P["<b>pick_balanced()</b><br/>members = train windows<br/>non-members = test windows<br/>equally many of each"]
        I["<b>run_attacks()</b><br/>each --attacks file<br/>scores every window;<br/>→ AUC, TPR at 1% FPR<br/>per attack"]
    end

    J["<b>result</b><br/>window counts, losses,<br/>accuracies, gaps,<br/>AUC and TPR per attack"]
    K[("<b>results/…csv</b><br/>one row per model<br/>+ run settings")]

    A --> B --> C --> Dd --> E
    E -->|train + val| G
    F --> G --> H
    E -->|all three splits| H
    H -->|train + test logits| P --> I --> J
    H -->|accuracies, losses| J
    J -->|recording.save| K
```

## 3. What each split is used for

| split | trains the model | early stopping | attacks |
|---|---|---|---|
| train (70%) | yes | | **members** |
| val (15%) | | yes | not used, because early stopping already looked at it |
| test (15%) | | | **non-members** |

## 4. How one recording is split (`datasets/mobi.py: split_by_time`)

Each recording (one participant, rest or game) is split in **time order**, not
at random:

```
start of recording                                                  end
|──────────────── train (70%) ────────────────|gap|── val ──|gap|── test ──|
```

The **gap** is wide enough that no train window shares a sample with a val or
test window. Otherwise, with overlapping windows, the model would already
have seen part of every test window during training.

- `--stride 275`: windows don't overlap, so the gap is 1 window.
- `--stride 1`: each window starts 1 sample after the previous one (99.6%
  overlap), so the gap is 275 windows.

Rest recordings are much shorter than game recordings, so the **balancing
step** in `build_splits()` keeps only as many game windows as there are rest
windows. For example, Subject 1 at stride 1 has 4,897 rest and 64,433 game
train windows. The balanced train set uses 4,897 of each.

## 5. What the attacks do (`attacks/`)

A model usually fits the windows it trained on better than new ones. Every
attack turns the target model's output on a window into a **score**, where
higher means "probably a member":

```
score of each window:   low ◄──────────────────────────► high

non-members (test)      ▁▂▃▅▆▇▇▆▅▃▂▁
members (train)                ▁▂▃▅▆▇▇▇▆▅▄▃
                                         └────┬────┘
                           the attack says "member" here
```

- **AUC 0.5**: the two curves lie on top of each other, so the attacker can't
  tell members apart. There is no privacy leak.
- **AUC 1.0**: the curves don't overlap, so the attacker always knows which
  windows were in training.
- **TPR at 1% FPR**: the share of members the attacker finds while wrongly
  flagging only 1% of non-members.

| attack | score |
|---|---|
| `loss` | −loss |
| `confidence` | probability of the true class |
| `max_confidence` | probability of the top class (ignores the label) |
| `modified_entropy` | −modified entropy |
| `lira` | how far the target's output is above what shadow models that did **not** train on the window give |

All attacks are scored on the **same** windows of the **same** target model, so
their AUCs are directly comparable. With one output for two classes, `loss`,
`confidence` and `modified_entropy` rank windows identically and always give the
same AUC.

**LiRA** needs more than the target's output:

```mermaid
flowchart LR
    pool["pool = train windows + test windows"]
    pool -->|"random half 1"| s1["shadow model 1"]
    pool -->|"random half 2"| s2["shadow model 2"]
    pool -->|"…"| sn["shadow model 64"]
    s1 & s2 & sn --> g["for each window: mean and sd<br/>of the shadow models that<br/>did NOT train on it"]
    t["target model's output<br/>on the window"] --> z
    g --> z["score = (target − mean) / sd"]
```

Each shadow model is trained exactly like the target (`pipeline.train_shadow`).
A window that every model finds easy gets a high mean, so a low loss alone no
longer makes it look like a member.

## 6. The participant sweep (`run_participant_sweep.py`)

```mermaid
flowchart TD
    L["load all 11<br/>participants once"] --> N
    N["for n = 2, 3, …, 10<br/>participants"] --> C
    C["pick 5 random<br/>combinations of<br/>n participants"] --> M
    M["for each --models<br/>model"] --> R
    R["pipeline.run_one()<br/>with the --attacks"] --> W
    W["add a row to this run's<br/>results file and save it<br/>(a crash keeps finished runs)"]
    W -->|next run| M
    W -->|all done| S["<b>summarize.py</b><br/>per attack: mean AUC<br/>per n and per model;<br/>correlation of AUC with<br/>loss gap, accuracy gap, n"]
```

The defaults give 9 values of n × 5 combinations × 3 models = **135 runs**.
Each run gets its own seed (`seed + 1000·n + combination`), so a run can be
repeated on its own.
