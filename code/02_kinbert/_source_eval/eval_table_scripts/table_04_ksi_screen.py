"""
Table for the KSI screening benchmark figure.

Source notebook: ``eval_3_yeast_human_benchmarks.ipynb`` (cells 11-18).

Figures covered
---------------
  eval_figs/eval_ksi_bar.png    eval_ksi_roc.png

This benchmark scores the four models against the lysate kinase-substrate
screening set.  It is the one benchmark that includes the geometry-augmented
model, which reads a separate structure-aware prediction file rather than a
pickle of model outputs.

``eval_3`` also builds a four-model yeast figure, but the yeast figure now in
eval_figs was written later by ``eval_2_v2test_dark_nonhuman.ipynb`` with only
three models; table_03 covers that one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from eval_table_utils import (
    Checker, dedup, load_kolossus, load_pairs_aug, load_phosformer,
    load_pkl_pred, load_struct_pred, run_benchmark, to_pairs_pred,
    write_benchmark_tables, write_verification,
)

# From the ROC_AUC / PR_AUC table printed in cell 15 of the notebook.
FIGURE_VALUES = {
    "Phosformer":              {"ROC_AUC": 0.494744, "PR_AUC": 0.759362},
    "Kolossus":                {"ROC_AUC": 0.400984, "PR_AUC": 0.691548},
    "KINBERT:TAPE":            {"ROC_AUC": 0.758473, "PR_AUC": 0.879821},
    "KINBERT:TAPE + Geometry": {"ROC_AUC": 0.724861, "PR_AUC": 0.852653},
}


def build_ksi():
    pairs = load_pairs_aug("pairs_aug_ksi.csv")

    phosformer = load_phosformer("pairs_aug_ksi.csv")
    kolossus = load_kolossus("pairs_aug_ksi.csv", pairs)
    tape = load_pkl_pred("exp_tape_bertbase_sep_v2__ksi__1b97f9a11dd9.pkl")
    geometry = load_struct_pred("tape_struct_3epoch/bert_struct_eval_pairs_ksi.csv")

    pairs_pred = to_pairs_pred(pairs)
    prediction_dfs = {
        "Phosformer": dedup(phosformer),
        "Kolossus": dedup(kolossus),
        "KINBERT:TAPE": dedup(tape),
        "KINBERT:TAPE + Geometry": dedup(geometry),
    }
    return pairs_pred, prediction_dfs


def main():
    print("\n=== eval_ksi ===")
    pairs_pred, prediction_dfs = build_ksi()

    # Saved with the default F1-optimised thresholds.
    metrics_df, bar_data, roc_data = run_benchmark(
        "eval_ksi", pairs_pred, prediction_dfs,
        bar_plot_metrics=["Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"])
    write_benchmark_tables("eval_ksi", metrics_df, bar_data, roc_data)

    c = Checker("eval_ksi_bar.png / eval_ksi_roc.png")
    c.check_metrics(metrics_df, FIGURE_VALUES, tol=5e-6)
    write_verification([c.report()], "VERIFICATION_04_ksi_screen.csv")


if __name__ == "__main__":
    main()
