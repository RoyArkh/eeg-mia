"""Train one model, run the LOSS attack and print everything. Start here when debugging."""

import argparse

import numpy as np

import config
import data
import pipeline
import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="EEGNet", choices=config.MODELS)
    parser.add_argument("--participants", type=int, nargs="+", default=config.PARTICIPANTS)
    parser.add_argument("--stride", type=int, default=config.STRIDE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    args = parser.parse_args()

    device = train.get_device(args.device)
    pipeline.print_environment(device)

    subjects = {}
    for subject_id in args.participants:
        subjects[subject_id] = data.load_subject(args.data_dir, subject_id, args.stride)

    rng = np.random.default_rng(args.seed)
    result = pipeline.run_one(args.model, args.participants, subjects, rng, args.seed, device, args.epochs)
    if result is None:
        print("Not enough windows for these participants at this stride.")
        return

    print(f"\n{args.model} | stride {args.stride} | participants {result['participants']}")
    print(f"windows: train {result['n_train_windows']}, val {result['n_val_windows']}, "
          f"test {result['n_test_windows']}")
    print(f"parameters: {result['n_params']:,}, epochs run: {result['epochs_run']}")
    print(f"\n{'':8}{'loss':>8}{'acc':>8}")
    for split in data.SPLITS:
        print(f"{split:8}{result[split + '_loss']:8.3f}{result[split + '_acc']:8.3f}")
    print(f"\nloss gap (test - train): {result['loss_gap']:.3f}")
    print(f"acc gap (train - test):  {result['acc_gap']:.3f}")
    print(f"LOSS attack: AUC {result['attack_auc']:.3f}, TPR at 1% FPR {result['attack_tpr']:.3f} "
          f"({result['n_attack']} windows)")
    print(f"took {result['seconds']} s")


if __name__ == "__main__":
    main()
