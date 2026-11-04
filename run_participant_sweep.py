"""Attack strength vs number of training participants.

Cobilean et al. Experiment 1, HEALTHINF section 4.2 (Figs 2 and 4). The saved
train/test gaps also give HEALTHINF Fig 1 (attack AUC vs generalization gap).
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd

import config
import data
import pipeline
import train


def choose_combinations(participants, n, max_combos, rng):
    """All combinations of n participants, or `max_combos` random ones if there are more."""
    all_combos = list(itertools.combinations(participants, n))
    if len(all_combos) <= max_combos:
        return all_combos
    picked = rng.choice(len(all_combos), max_combos, replace=False)
    return [all_combos[i] for i in sorted(picked)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=config.MODELS, choices=config.MODELS)
    parser.add_argument("--counts", type=int, nargs="+", default=list(range(2, 11)),
                        help="numbers of participants to train on")
    parser.add_argument("--combos", type=int, default=5, help="participant combinations per count")
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--out", default=None, help="output CSV (default: results/participant_sweep_stride<N>.csv)")
    args = parser.parse_args()

    out_path = args.out or os.path.join(config.RESULTS_DIR, f"participant_sweep_stride{args.stride}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = train.get_device(args.device)
    pipeline.print_environment(device)

    subjects = {}
    for subject_id in config.PARTICIPANTS:
        subjects[subject_id] = data.load_subject(args.data_dir, subject_id, args.stride)

    rng = np.random.default_rng(args.seed)
    plan = []
    for n in args.counts:
        for combo_id, combo in enumerate(choose_combinations(config.PARTICIPANTS, n, args.combos, rng)):
            plan.append((n, combo_id, list(combo)))
    total = len(plan) * len(args.models)
    print(f"stride {args.stride}: {len(plan)} participant combinations x {len(args.models)} models = {total} runs\n")

    rows = []
    done = 0
    for n, combo_id, combo in plan:
        seed = args.seed + 1000 * n + combo_id
        for model_name in args.models:
            done += 1
            result = pipeline.run_one(model_name, combo, subjects, rng, seed, device, args.epochs)
            if result is None:
                print(f"[{done}/{total}] n={n} combo={combo_id} {model_name}: skipped (too few windows)")
                continue

            result["stride"] = args.stride
            result["combo_id"] = combo_id
            rows.append(result)
            pd.DataFrame(rows).to_csv(out_path, index=False)   # saved after every run
            print(f"[{done}/{total}] n={n} combo={combo_id} {model_name:<15} "
                  f"test_acc={result['test_acc']:.3f} attack_auc={result['attack_auc']:.3f} "
                  f"({result['seconds']} s)")

    if not rows:
        print("\nNo successful runs.")
        return
    print(f"\nSaved {len(rows)} runs to {out_path}")
    print(f"Summarize with: uv run python summarize.py {out_path}")


if __name__ == "__main__":
    main()
