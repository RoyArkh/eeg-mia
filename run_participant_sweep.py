"""Attack strength vs number of training participants.

Cobilean et al. Experiment 1, HEALTHINF section 4.2 (Figs 2 and 4). The saved
train/test gaps also give HEALTHINF Fig 1 (attack AUC vs generalization gap).
Results are saved as a new file in results/, re-written after every trained model.
"""

import argparse
import itertools

import numpy as np

import attacks
import config
import datasets
import models
import pipeline
import recording
import train
from attacks import lira


def choose_combinations(participants, n, max_combos, rng):
    """All combinations of n participants, or `max_combos` random ones if there are more."""
    all_combos = list(itertools.combinations(participants, n))
    if len(all_combos) <= max_combos:
        return all_combos
    picked = rng.choice(len(all_combos), max_combos, replace=False)
    return [all_combos[i] for i in sorted(picked)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=config.DATASET, choices=list(datasets.DATASETS))
    parser.add_argument("--models", nargs="+", default=config.MODELS,
                        help=f"any of: {', '.join(models.MODELS)}; or all")
    parser.add_argument("--attacks", nargs="+", default=config.ATTACKS,
                        help=f"any of: {', '.join(attacks.ATTACKS)}; or all")
    parser.add_argument("--counts", type=int, nargs="+", default=None,
                        help="numbers of participants to train on (default: 2 to all but one)")
    parser.add_argument("--combos", type=int, default=5, help="participant combinations per count")
    parser.add_argument("--stride", type=int, default=None,
                        help="window stride in samples (default: the dataset's)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--shadows", type=int, default=lira.N_SHADOWS, help="number of shadow models for LiRA")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--data-dir", default=None, help="default: data/<dataset>")
    args = parser.parse_args()
    model_names = models.resolve_names(args.models)
    attack_names = attacks.resolve_names(args.attacks)

    device = train.get_device(args.device)
    recording.print_environment(device)
    dataset = datasets.load_dataset(args.dataset, data_dir=args.data_dir, stride=args.stride)
    path = recording.results_path(model_names, attack_names, dataset, args.epochs)
    info = recording.run_info("participant_sweep", dataset, device, args.epochs)

    participants = dataset.participants
    counts = args.counts or list(range(2, len(participants)))
    rng = np.random.default_rng(args.seed)
    plan = []
    for n in counts:
        for combo_id, combo in enumerate(choose_combinations(participants, n, args.combos, rng)):
            plan.append((n, combo_id, list(combo)))
    total = len(plan) * len(model_names)
    print(f"{dataset.label}: {len(plan)} participant combinations x {len(model_names)} models = {total} runs")
    print(f"attacks: {', '.join(attack_names)}")
    print(f"saving to {path}\n")

    rows = []
    done = 0
    for n, combo_id, combo in plan:
        seed = args.seed + 1000 * n + combo_id
        for model_name in model_names:
            done += 1
            result = pipeline.run_one(model_name, combo, dataset, attack_names, rng, seed, device, args.epochs,
                                      attack_options={"lira": {"n_shadows": args.shadows}})
            if result is None:
                print(f"[{done}/{total}] n={n} combo={combo_id} {model_name}: skipped (too few windows)")
                continue

            rows.append({**info, "combo_id": combo_id, **result})
            recording.save(rows, path)
            aucs = " ".join(f"{name}={result[name + '_auc']:.3f}" for name in attack_names)
            print(f"[{done}/{total}] n={n} combo={combo_id} {model_name:<15} "
                  f"test_acc={result['test_acc']:.3f} AUC: {aucs} ({result['seconds']} s)")

    if not rows:
        print("\nNo successful runs.")
        return
    print(f"\nSaved {len(rows)} runs to {path}")
    print(f"Summarize with: uv run python summarize.py {path}")


if __name__ == "__main__":
    main()
