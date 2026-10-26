"""
Tables for the model-architecture benchmark figures.

Source notebook: ``eval_2_v2test_dark_nonhuman.ipynb`` (cells 3-68).

Figures covered
---------------
  eval_figs/evalv2_test_bar.png                 evalv2_test_roc.png
  eval_figs/evalv2_test2_bar.png                evalv2_test2_roc.png
  eval_figs/eval_phosformer_train_bar.png       eval_phosformer_train_roc.png
  eval_figs/eval_phosformer_test_bar.png        eval_phosformer_test_roc.png
  eval_figs/benchmark_model_structure_bar.png   benchmark_model_structure_roc.png
  (and the .pdf twins of the last pair)

Every figure above compares the same six models on a different test set.
The combined ``benchmark_model_structure`` figure facets the four test sets
side by side, so its table is just the four per-benchmark tables stacked,
exactly as the notebook stacked them in ``collected_data``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from plotnine import *  # noqa: F401,F403  (the notebook built its plots this way)

from eval_table_utils import (
    Checker, dedup, load_frozen_pred, load_kolossus, load_pairs_aug,
    load_phosformer, load_pkl_pred, require, run_benchmark, table_path,
    to_pairs_pred, write_benchmark_tables, write_verification,
)

MODEL_ORDER = [
    "Phosformer", "Kolossus", "KINBERT:TAPE", "KINBERT:ESM",
    "KINBERT:ESM-CONCAT", "KINBERT:ESM-FROZEN",
]

# Values read back from the stored outputs of the notebook cells that saved
# each figure.  They are what the figure actually shows.
FIGURE_VALUES = {
    # From the LaTeX summary table printed in cell 67 (3 decimal places).
    "evalv2_test": {
        "Phosformer":         {"ROC_AUC": 0.695, "PR_AUC": 0.614},
        "Kolossus":           {"ROC_AUC": 0.779, "PR_AUC": 0.720},
        "KINBERT:TAPE":       {"ROC_AUC": 0.973, "PR_AUC": 0.970},
        "KINBERT:ESM":        {"ROC_AUC": 0.972, "PR_AUC": 0.968},
        "KINBERT:ESM-CONCAT": {"ROC_AUC": 0.968, "PR_AUC": 0.964},
        "KINBERT:ESM-FROZEN": {"ROC_AUC": 0.940, "PR_AUC": 0.924},
    },
    # From the metrics_df printed in cell 34 (6 decimal places).
    "evalv2_test2": {
        "Phosformer":         {"Accuracy": 0.779925, "Precision": 0.703566, "Recall": 0.513199,
                               "F1_Score": 0.593490, "ROC_AUC": 0.854428, "PR_AUC": 0.687963,
                               "MCC": 0.458101, "Specificity": 0.901468, "True_Positive": 1322},
        "Kolossus":           {"Accuracy": 0.707133, "Precision": 0.526051, "Recall": 0.650621,
                               "F1_Score": 0.581742, "ROC_AUC": 0.757808, "PR_AUC": 0.606848,
                               "MCC": 0.365105, "Specificity": 0.732885, "True_Positive": 1676},
        "KINBERT:TAPE":       {"Accuracy": 0.840807, "Precision": 0.729182, "Recall": 0.781832,
                               "F1_Score": 0.754590, "ROC_AUC": 0.887023, "PR_AUC": 0.779873,
                               "MCC": 0.637843, "Specificity": 0.867681, "True_Positive": 2014},
        "KINBERT:ESM":        {"Accuracy": 0.834002, "Precision": 0.743952, "Recall": 0.716227,
                               "F1_Score": 0.729826, "ROC_AUC": 0.870370, "PR_AUC": 0.753633,
                               "MCC": 0.610314, "Specificity": 0.887670, "True_Positive": 1845},
        "KINBERT:ESM-CONCAT": {"Accuracy": 0.781869, "Precision": 0.652242, "Recall": 0.649457,
                               "F1_Score": 0.650846, "ROC_AUC": 0.821179, "PR_AUC": 0.662344,
                               "MCC": 0.492239, "Specificity": 0.842208, "True_Positive": 1673},
        "KINBERT:ESM-FROZEN": {"Accuracy": 0.769838, "Precision": 0.630952, "Recall": 0.637811,
                               "F1_Score": 0.634363, "ROC_AUC": 0.819339, "PR_AUC": 0.685722,
                               "MCC": 0.466447, "Specificity": 0.830002, "True_Positive": 1643},
    },
    # From the metrics_df printed in cell 48.
    "eval_phosformer_train": {
        "Phosformer":         {"Accuracy": 0.721099, "Precision": 0.802200, "Recall": 0.565463,
                               "F1_Score": 0.663342, "ROC_AUC": 0.844772, "PR_AUC": 0.816484,
                               "MCC": 0.456746, "Specificity": 0.868210, "True_Positive": 8314},
        "Kolossus":           {"Accuracy": 0.676152, "Precision": 0.663249, "Recall": 0.677549,
                               "F1_Score": 0.670323, "ROC_AUC": 0.736863, "PR_AUC": 0.728726,
                               "MCC": 0.352249, "Specificity": 0.674831, "True_Positive": 9962},
        "KINBERT:TAPE":       {"Accuracy": 0.903629, "Precision": 0.883567, "Recall": 0.923349,
                               "F1_Score": 0.903020, "ROC_AUC": 0.946971, "PR_AUC": 0.920850,
                               "MCC": 0.808116, "Specificity": 0.884989, "True_Positive": 13576},
        "KINBERT:ESM":        {"Accuracy": 0.903232, "Precision": 0.895267, "Recall": 0.906958,
                               "F1_Score": 0.901074, "ROC_AUC": 0.942864, "PR_AUC": 0.914149,
                               "MCC": 0.806445, "Specificity": 0.899711, "True_Positive": 13335},
        "KINBERT:ESM-CONCAT": {"Accuracy": 0.836903, "Precision": 0.856809, "Recall": 0.797660,
                               "F1_Score": 0.826177, "ROC_AUC": 0.907772, "PR_AUC": 0.865564,
                               "MCC": 0.674456, "Specificity": 0.873995, "True_Positive": 11728},
        "KINBERT:ESM-FROZEN": {"Accuracy": 0.744927, "Precision": 0.820795, "Recall": 0.607767,
                               "F1_Score": 0.698398, "ROC_AUC": 0.851204, "PR_AUC": 0.821573,
                               "MCC": 0.502299, "Specificity": 0.874574, "True_Positive": 8936},
    },
    # From the metrics_df printed in cell 60.
    "eval_phosformer_test": {
        "Phosformer":         {"Accuracy": 0.652413, "Precision": 0.736191, "Recall": 0.426660,
                               "F1_Score": 0.540230, "ROC_AUC": 0.781937, "PR_AUC": 0.738503,
                               "MCC": 0.319458, "Specificity": 0.859649, "True_Positive": 893},
        "Kolossus":           {"Accuracy": 0.704093, "Precision": 0.678747, "Recall": 0.724797,
                               "F1_Score": 0.701017, "ROC_AUC": 0.761814, "PR_AUC": 0.750765,
                               "MCC": 0.409611, "Specificity": 0.685088, "True_Positive": 1517},
        "KINBERT:TAPE":       {"Accuracy": 0.906929, "Precision": 0.879388, "Recall": 0.933588,
                               "F1_Score": 0.905678, "ROC_AUC": 0.949572, "PR_AUC": 0.918508,
                               "MCC": 0.815405, "Specificity": 0.882456, "True_Positive": 1954},
        "KINBERT:ESM":        {"Accuracy": 0.899154, "Precision": 0.879247, "Recall": 0.914955,
                               "F1_Score": 0.896745, "ROC_AUC": 0.934311, "PR_AUC": 0.881751,
                               "MCC": 0.798878, "Specificity": 0.884649, "True_Positive": 1915},
        "KINBERT:ESM-CONCAT": {"Accuracy": 0.848617, "Precision": 0.855793, "Recall": 0.822265,
                               "F1_Score": 0.838694, "ROC_AUC": 0.915631, "PR_AUC": 0.871169,
                               "MCC": 0.696684, "Specificity": 0.872807, "True_Positive": 1721},
        "KINBERT:ESM-FROZEN": {"Accuracy": 0.763320, "Precision": 0.822561, "Recall": 0.644529,
                               "F1_Score": 0.722743, "ROC_AUC": 0.864987, "PR_AUC": 0.830626,
                               "MCC": 0.533353, "Specificity": 0.872368, "True_Positive": 1349},
    },
}

# Per-test-set means printed by cell 64, shown as the four facets of the
# combined figure.
COMBINED_MEANS = {
    "DB + LYSATE Test": {"PR_AUC": 0.860, "ROC_AUC": 0.888},
    "DB Test":          {"PR_AUC": 0.696, "ROC_AUC": 0.835},
    "Phosformer Test":  {"PR_AUC": 0.832, "ROC_AUC": 0.868},
    "Phosformer Train": {"PR_AUC": 0.845, "ROC_AUC": 0.872},
}

TEST_NAMES = {
    "evalv2_test": "DB + LYSATE Test",
    "evalv2_test2": "DB Test",
    "eval_phosformer_train": "Phosformer Train",
    "eval_phosformer_test": "Phosformer Test",
}


def build_v2_split(split):
    """The DB+LYSATE test split (``test``) or the DB-only split (``test2``)."""
    pairs = load_pairs_aug("pairs_aug_v2.csv")
    pred_range = pd.read_csv(require(f"/ksi_ml/data/splits_v2/{split}.csv"))
    pred_range.columns = ["kinase_domain_seq", "substrate_seq_15aa", "label"]
    pairs = pd.merge(pairs, pred_range).drop_duplicates()

    phosformer = load_phosformer("pairs_aug_v2.csv")
    kolossus = load_kolossus("pairs_aug_v2.csv", pairs)
    esm = load_pkl_pred("exp_esm2_t12_fullfinetune_eos_v2__db_lysate__a543af984148.pkl")
    if split == "test":
        tape = load_pkl_pred("exp_tape_bertbase_sep_v2_long__db_lysate__53ceb023016e.pkl")
    else:
        tape = load_pkl_pred("exp_tape_bertbase_sep_v2__db_lysate__1b97f9a11dd9.pkl")
    concat = load_pkl_pred("exp3_esm2t12_unfrozen_v2__db_lysate__2b97b6229963.pkl")
    frozen = load_frozen_pred("exp_esmt12_frozen_v2__pairs_v2.csv")

    pairs_pred = to_pairs_pred(pairs)
    prediction_dfs = {
        "Phosformer": dedup(phosformer),
        "Kolossus": dedup(kolossus),
        "KINBERT:TAPE": dedup(tape),
        "KINBERT:ESM": dedup(esm),
        "KINBERT:ESM-CONCAT": dedup(concat),
        "KINBERT:ESM-FROZEN": dedup(frozen),
    }
    return pairs_pred, prediction_dfs


def build_phosformer_split(which):
    """The Phosformer train or test split."""
    pairs = load_pairs_aug(f"pairs_aug_phosphomer_{which}.csv")

    phosformer = load_phosformer(f"pairs_aug_phosphomer_{which}.csv")
    kolossus = load_kolossus(f"pairs_aug_phosphomer_{which}.csv", pairs)
    esm = load_pkl_pred(
        f"exp_esm2_t12_fullfinetune_eos_v2__phosformer_{which}__a543af984148.pkl")
    tape = load_pkl_pred(
        f"exp_tape_bertbase_sep_v2__phosformer_{which}__1b97f9a11dd9.pkl")
    concat = load_pkl_pred(
        f"exp3_esm2t12_unfrozen_v2__phosformer_{which}__2b97b6229963.pkl")
    frozen = load_frozen_pred(f"exp_esmt12_frozen_v2__pairs_phosphomer_{which}.csv")

    pairs_pred = to_pairs_pred(pairs)
    prediction_dfs = {
        "Phosformer": dedup(phosformer),
        "Kolossus": dedup(kolossus),
        "KINBERT:TAPE": dedup(tape),
        "KINBERT:ESM": dedup(esm),
        "KINBERT:ESM-CONCAT": dedup(concat),
        "KINBERT:ESM-FROZEN": dedup(frozen),
    }
    return pairs_pred, prediction_dfs


BENCHMARKS = [
    # benchmark_id, builder, bar_plot_metrics passed to the figure code
    ("evalv2_test", lambda: build_v2_split("test"),
     ["Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]),
    ("evalv2_test2", lambda: build_v2_split("test2"),
     ["Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]),
    ("eval_phosformer_train", lambda: build_phosformer_split("train"),
     ["Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]),
    # This one was saved without a bar_plot_metrics argument, so the figure
    # shows the function's six default metrics.
    ("eval_phosformer_test", lambda: build_phosformer_split("test"), None),
]


def main(combine_only=False):
    checks = []
    collected = []
    collected_roc = []

    for benchmark_id, builder, bar_metrics in BENCHMARKS:
        print(f"\n=== {benchmark_id} ===")
        if combine_only:
            # Reuse the per-benchmark tables written by an earlier full run.
            metrics_df = pd.read_csv(table_path(f"{benchmark_id}_metrics_full.csv"))
            roc_data = pd.read_csv(table_path(f"{benchmark_id}_roc.csv"))
            print("  reusing tables from a previous run")
        else:
            pairs_pred, prediction_dfs = builder()
            kwargs = dict(threshold_mode="fixed", fixed_threshold=0.5)
            if bar_metrics is not None:
                kwargs["bar_plot_metrics"] = bar_metrics

            metrics_df, bar_data, roc_data = run_benchmark(
                benchmark_id, pairs_pred, prediction_dfs, **kwargs)
            write_benchmark_tables(benchmark_id, metrics_df, bar_data, roc_data)

        c = Checker(f"{benchmark_id}_bar.png / {benchmark_id}_roc.png")
        c.check_metrics(metrics_df, FIGURE_VALUES[benchmark_id],
                        tol=5e-4 if benchmark_id == "evalv2_test" else 5e-6)
        checks.append(c.report())

        m = metrics_df.copy()
        m["Model"] = m["Model"].astype(str)
        m["Test_Dataset"] = TEST_NAMES[benchmark_id]
        m["Benchmark_ID"] = benchmark_id
        collected.append(m)

        r = roc_data.copy()
        r["Model"] = r["Model"].astype(str)
        r["Test_Dataset"] = TEST_NAMES[benchmark_id]
        r["Benchmark_ID"] = benchmark_id
        collected_roc.append(r)

    # ---------------------------------------------------------------
    # Combined figure: the four benchmarks faceted side by side
    # ---------------------------------------------------------------
    print("\n=== benchmark_model_structure (combined) ===")
    combined_metrics = pd.concat(collected, ignore_index=True)

    bar_long = combined_metrics[["Model", "ROC_AUC", "PR_AUC", "Test_Dataset"]].melt(
        id_vars=["Model", "Test_Dataset"], var_name="Metric", value_name="Score")
    p = table_path("benchmark_model_structure_bar.csv")
    bar_long.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}")

    combined_roc = pd.concat(collected_roc, ignore_index=True)
    p = table_path("benchmark_model_structure_roc.csv")
    combined_roc.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}")

    # A wide ROC_AUC / PR_AUC view matching the LaTeX table the notebook printed.
    wide = (combined_metrics.set_index(["Test_Dataset", "Model"])[["ROC_AUC", "PR_AUC"]]
            .unstack("Model").sort_index(axis=1, level=0))
    p = table_path("benchmark_model_structure_summary_wide.csv")
    wide.to_csv(p)
    print(f"  wrote {os.path.basename(p)}")

    c = Checker("benchmark_model_structure_bar.png (facet means)")
    means = bar_long.groupby(["Test_Dataset", "Metric"])["Score"].mean()
    for dataset, want in COMBINED_MEANS.items():
        for metric, value in want.items():
            c.check(f"{dataset} / mean {metric}",
                    float(means.loc[(dataset, metric)]), value, 5e-4)
    checks.append(c.report())

    write_verification(checks, "VERIFICATION_02_model_structure.csv")


if __name__ == "__main__":
    main(combine_only="--combine-only" in sys.argv)
