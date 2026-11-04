r"""Summarize a participant-sweep CSV, optionally next to the original HEALTHINF results.

Example:
    uv run python summarize.py results/participant_sweep_stride275.csv \
        --reference ../mia_bci/experiment1_out/exp1_runs_stride275.csv
"""

import argparse

import pandas as pd
from scipy import stats


def load_reference(path, models):
    """Load an original exp1_runs_stride*.csv and rename its columns to ours."""
    reference = pd.read_csv(path)
    reference = reference.rename(columns={
        "train_test_acc_gap": "acc_gap",
        "attack_tpr_at_1pct_fpr": "attack_tpr",
    })
    reference["loss_gap"] = reference["test_loss"] - reference["train_loss"]
    return reference[reference["model"].isin(models)]


def print_summary(df, title):
    print(f"\n===== {title} ({len(df)} runs) =====")

    print("\nMean attack AUC by number of participants:")
    auc_table = df.pivot_table(index="n_participants", columns="model", values="attack_auc", aggfunc="mean")
    print(auc_table.round(3).to_string())

    print("\nMeans per model:")
    columns = ["test_acc", "acc_gap", "loss_gap", "attack_auc", "attack_tpr"]
    print(df.groupby("model")[columns].mean().round(3).to_string())

    print("\nPearson correlation with attack AUC (all runs):")
    for column in ["loss_gap", "acc_gap", "n_participants"]:
        clean = df[[column, "attack_auc"]].dropna()
        r, p = stats.pearsonr(clean[column], clean["attack_auc"])
        print(f"  {column:<15} r = {r:+.3f}  (p = {p:.2g}, n = {len(clean)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", help="CSV written by run_participant_sweep.py")
    parser.add_argument("--reference", help="original exp1_runs_stride*.csv to compare with")
    args = parser.parse_args()

    ours = pd.read_csv(args.results)
    print_summary(ours, f"This code: {args.results}")
    if args.reference:
        reference = load_reference(args.reference, ours["model"].unique())
        print_summary(reference, f"Original code: {args.reference}")


if __name__ == "__main__":
    main()
