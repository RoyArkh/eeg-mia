"""Train one model, run the attacks and print everything. Start here when debugging.

The result is also saved as a new file in results/.
"""

import argparse

import numpy as np

import attacks
import config
import datasets
import models
import pipeline
import recording
import train
from attacks import lira


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=config.DATASET, choices=list(datasets.DATASETS))
    parser.add_argument("--model", default="EEGNet", choices=list(models.MODELS))
    parser.add_argument("--attacks", nargs="+", default=config.ATTACKS,
                        help=f"any of: {', '.join(attacks.ATTACKS)}; or all")
    parser.add_argument("--participants", type=int, nargs="+", default=None,
                        help="default: all participants of the dataset")
    parser.add_argument("--stride", type=int, default=None,
                        help="window stride in samples (default: the dataset's)")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--shadows", type=int, default=lira.N_SHADOWS, help="number of shadow models for LiRA")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--data-dir", default=None, help="default: data/<dataset>")
    args = parser.parse_args()
    attack_names = attacks.resolve_names(args.attacks)

    device = train.get_device(args.device)
    recording.print_environment(device)
    dataset = datasets.load_dataset(args.dataset, data_dir=args.data_dir, stride=args.stride,
                                    participants=args.participants)
    path = recording.results_path([args.model], attack_names, dataset, args.epochs)
    info = recording.run_info("single", dataset, device, args.epochs)

    rng = np.random.default_rng(args.seed)
    result = pipeline.run_one(args.model, dataset.participants, dataset, attack_names, rng, args.seed,
                              device, args.epochs, attack_options={"lira": {"n_shadows": args.shadows}})
    if result is None:
        print("Not enough windows for these participants at this stride.")
        return

    print(f"\n{args.model} | {dataset.label} | participants {result['participants']}")
    print(f"windows: train {result['n_train_windows']}, val {result['n_val_windows']}, "
          f"test {result['n_test_windows']}")
    print(f"parameters: {result['n_params']:,}, epochs run: {result['epochs_run']}")
    print(f"\n{'':8}{'loss':>8}{'acc':>8}")
    for split in pipeline.SPLITS:
        print(f"{split:8}{result[split + '_loss']:8.3f}{result[split + '_acc']:8.3f}")
    print(f"\nloss gap (test - train): {result['loss_gap']:.3f}")
    print(f"acc gap (train - test):  {result['acc_gap']:.3f}")

    tpr_label = f"TPR at {config.TARGET_FPR:.0%} FPR"
    print(f"\n{'attack':<18}{'AUC':>7}{tpr_label:>18}   ({result['n_attack']} windows)")
    for name in attack_names:
        print(f"{name:<18}{result[name + '_auc']:7.3f}{result[name + '_tpr']:18.3f}")
    print(f"\ntook {result['seconds']} s")

    recording.save([{**info, **result}], path)
    print(f"saved to {path}")


if __name__ == "__main__":
    main()
