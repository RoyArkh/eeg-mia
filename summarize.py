r"""Summarize results files, optionally next to the original HEALTHINF results.

Example:
    uv run python summarize.py results/2026-11-04_19-30-12_EEGNet_loss_mobi-stride275-epochs50.csv \
        --reference ../mia_bci/experiment1_out/exp1_runs_stride275.csv

Several results files can be given at once; they are combined.
"""

import argparse

import pandas as pd
from scipy import stats


def load_results(paths):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        # files written before the attacks/ folder existed hold only the loss attack
        frame = frame.rename(columns={"attack_auc": "loss_auc", "attack_tpr": "loss_tpr"})
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_reference(path, models):
    """Load an original exp1_runs_stride*.csv and rename its columns to ours."""
    reference = pd.read_csv(path)
    reference = reference.rename(columns={
        "train_test_acc_gap": "acc_gap",
        "attack_auc": "loss_auc",
        "attack_tpr_at_1pct_fpr": "loss_tpr",
    })
    reference["loss_gap"] = reference["test_loss"] - reference["train_loss"]
    return reference[reference["model"].isin(models)]


def attack_names(df):
    """Attacks with results in `df` (columns <attack>_auc and <attack>_tpr)."""
    return [column[:-len("_auc")] for column in df.columns
            if column.endswith("_auc") and column[:-len("_auc")] + "_tpr" in df.columns]


def print_summary(df, title):
    attacks = attack_names(df)
    print(f"\n===== {title} ({len(df)} runs) =====")

    for attack in attacks:
        print(f"\nMean {attack} attack AUC by number of participants:")
        table = df.pivot_table(index="n_participants", columns="model", values=f"{attack}_auc", aggfunc="mean")
        print(table.round(3).to_string())

    print("\nMeans per model:")
    columns = ["test_acc", "acc_gap", "loss_gap"] + [f"{a}_{m}" for a in attacks for m in ("auc", "tpr")]
    print(df.groupby("model")[columns].mean().round(3).T.to_string())

    print("\nPearson correlation with attack AUC (all runs):")
    for attack in attacks:
        for column in ["loss_gap", "acc_gap", "n_participants"]:
            clean = df[[column, f"{attack}_auc"]].dropna()
            if len(clean) < 3 or clean[column].nunique() < 2:
                continue
            r, p = stats.pearsonr(clean[column], clean[f"{attack}_auc"])
            print(f"  {attack + '_auc':<22} vs {column:<15} r = {r:+.3f}  (p = {p:.2g}, n = {len(clean)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results CSV file(s) from results/")
    parser.add_argument("--reference", help="original exp1_runs_stride*.csv to compare with")
    args = parser.parse_args()

    ours = load_results(args.results)
    print_summary(ours, f"This code: {', '.join(args.results)}")
    if args.reference:
        reference = load_reference(args.reference, ours["model"].unique())
        print_summary(reference, f"Original code: {args.reference}")


if __name__ == "__main__":
    main()
