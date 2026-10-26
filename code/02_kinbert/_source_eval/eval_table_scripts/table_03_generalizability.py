"""
Tables for the generalizability benchmark figures.

Source notebook: ``eval_2_v2test_dark_nonhuman.ipynb`` (cells 72-122).

Figures covered
---------------
  eval_figs/eval_dark_test_bar.png              eval_dark_test_roc.png
  eval_figs/eval_nonhuman_test_bar.png          eval_nonhuman_test_roc.png
  eval_figs/eval_yeast_bar.png                  eval_yeast_roc.png
  eval_figs/benchmark_generalizability_bar.png  benchmark_generalizability_roc.png
  (and the .pdf twins of the last pair)

Each holdout asks whether a model that never saw a class of kinase during
training still scores it.  ``KINBERT:TAPE`` is the model trained on
everything; ``KINBERT:TAPE-No Dark`` and ``KINBERT:TAPE-Non Human`` are the
ablated models, and each appears only in its own holdout, which is why the
combined table has blanks.

Note on ``eval_yeast``: the yeast benchmark is built by two notebooks.  The
figure currently in eval_figs is the one from this notebook (three models,
F1-optimised thresholds).  The four-model variant in
``eval_3_yeast_human_benchmarks.ipynb`` is handled by table_04.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from eval_table_utils import (
    Checker, dedup, load_kolossus, load_pairs_aug, load_phosformer,
    load_pkl_pred, require, run_benchmark, table_path, to_pairs_pred,
    write_benchmark_tables, write_verification,
)

# From the LaTeX summary table printed in cell 120 (3 decimal places).
FIGURE_VALUES = {
    "eval_dark_test": {
        "Phosformer":           {"ROC_AUC": 0.681, "PR_AUC": 0.635},
        "Kolossus":             {"ROC_AUC": 0.730, "PR_AUC": 0.658},
        "KINBERT:TAPE":         {"ROC_AUC": 0.996, "PR_AUC": 0.996},
        "KINBERT:TAPE-No Dark": {"ROC_AUC": 0.959, "PR_AUC": 0.961},
    },
    "eval_nonhuman_test": {
        "Phosformer":             {"ROC_AUC": 0.851, "PR_AUC": 0.693},
        "Kolossus":               {"ROC_AUC": 0.762, "PR_AUC": 0.612},
        "KINBERT:TAPE":           {"ROC_AUC": 0.976, "PR_AUC": 0.949},
        "KINBERT:TAPE-Non Human": {"ROC_AUC": 0.874, "PR_AUC": 0.750},
    },
    "eval_yeast": {
        "Phosformer":   {"ROC_AUC": 0.695, "PR_AUC": 0.602},
        "Kolossus":     {"ROC_AUC": 0.636, "PR_AUC": 0.569},
        "KINBERT:TAPE": {"ROC_AUC": 0.713, "PR_AUC": 0.674},
    },
}

# Per-facet means printed by cell 116 of the same notebook.
COMBINED_MEANS = {
    "Dark Kinase Holdout":       {"PR_AUC": 0.813, "ROC_AUC": 0.842},
    "Non-Human Kinase Holdout":  {"PR_AUC": 0.751, "ROC_AUC": 0.866},
    "Yeast x Human PK":          {"PR_AUC": 0.615, "ROC_AUC": 0.681},
}

# Label balance of the yeast reference set, printed by cell 106.
YEAST_LABEL_COUNTS = {0: 3846, 1: 3215}

TEST_NAMES = {
    "eval_dark_test": "Dark Kinase Holdout",
    "eval_nonhuman_test": "Non-Human Kinase Holdout",
    "eval_yeast": "Yeast x Human PK",
}

BAR_METRICS = ["Accuracy", "F1_Score", "ROC_AUC", "PR_AUC"]


def build_holdout(holdout):
    """``holdout`` is 'dark' or 'nonhuman'.

    The reference table is the full v2 pair set restricted to the rows that
    appear in the holdout file, so the models are scored only on kinases the
    ablated model never trained on.
    """
    pairs = load_pairs_aug("pairs_aug_v2.csv")
    pred_range = (pd.read_csv(require(f"/ksi_ml/data/pairs_aug_v2_{holdout}.csv"))
                  .iloc[:, 1:].drop_duplicates()
                  .loc[:, ["kinase_domain_seq", "substrate_seq_15aa", "label"]])
    pairs = pd.merge(pairs, pred_range).drop_duplicates()

    phosformer = load_phosformer("pairs_aug_v2.csv")
    kolossus = load_kolossus("pairs_aug_v2.csv", pairs)
    tape = load_pkl_pred("exp_tape_bertbase_sep_v2__db_lysate__1b97f9a11dd9.pkl")
    if holdout == "dark":
        ablated_name = "KINBERT:TAPE-No Dark"
        ablated = load_pkl_pred("exp_tape_bertbase_sep_dark__db_lysate__0b837b4b2af3.pkl")
    else:
        ablated_name = "KINBERT:TAPE-Non Human"
        ablated = load_pkl_pred("exp_tape_bertbase_sep_human__db_lysate__63f1bc4d5cfe.pkl")

    pairs_pred = to_pairs_pred(pairs)
    prediction_dfs = {
        "Phosformer": dedup(phosformer),
        "Kolossus": dedup(kolossus),
        "KINBERT:TAPE": dedup(tape),
        ablated_name: dedup(ablated),
    }
    return pairs_pred, prediction_dfs


def build_yeast():
    """Human kinases scored against yeast substrates."""
    pairs = load_pairs_aug("pairs_aug_yeast.csv")

    phosformer = load_phosformer("pairs_aug_yeast.csv")
    kolossus = load_kolossus("pairs_aug_yeast.csv", pairs)
    tape = load_pkl_pred("exp_tape_bertbase_sep_v2_long__yeast_human__53ceb023016e.pkl")

    pairs_pred = to_pairs_pred(pairs)
    prediction_dfs = {
        "Phosformer": dedup(phosformer),
        "Kolossus": dedup(kolossus),
        "KINBERT:TAPE": dedup(tape),
    }
    return pairs_pred, prediction_dfs


BENCHMARKS = [
    # benchmark_id, builder, threshold kwargs used when the figure was saved
    ("eval_dark_test", lambda: build_holdout("dark"),
     dict(threshold_mode="fixed", fixed_threshold=0.5)),
    ("eval_nonhuman_test", lambda: build_holdout("nonhuman"),
     dict(threshold_mode="fixed", fixed_threshold=0.5)),
    # The yeast figure was saved with the default F1-optimised thresholds.
    ("eval_yeast", build_yeast, dict()),
]


def main(combine_only=False):
    checks = []
    collected = []
    collected_roc = []

    for benchmark_id, builder, thr_kwargs in BENCHMARKS:
        print(f"\n=== {benchmark_id} ===")
        if combine_only:
            metrics_df = pd.read_csv(table_path(f"{benchmark_id}_metrics_full.csv"))
            roc_data = pd.read_csv(table_path(f"{benchmark_id}_roc.csv"))
            print("  reusing tables from a previous run")
        else:
            pairs_pred, prediction_dfs = builder()

            if benchmark_id == "eval_yeast":
                counts = pairs_pred.groupby("label").size().to_dict()
                c0 = Checker("eval_yeast reference set")
                for label, want in YEAST_LABEL_COUNTS.items():
                    c0.check(f"pairs with label={label}",
                             float(counts.get(label, 0)), float(want), 0)
                checks.append(c0.report())

            metrics_df, bar_data, roc_data = run_benchmark(
                benchmark_id, pairs_pred, prediction_dfs,
                bar_plot_metrics=BAR_METRICS, **thr_kwargs)
            write_benchmark_tables(benchmark_id, metrics_df, bar_data, roc_data)

        c = Checker(f"{benchmark_id}_bar.png / {benchmark_id}_roc.png")
        c.check_metrics(metrics_df, FIGURE_VALUES[benchmark_id], tol=5e-4)
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
    # Combined figure: the three holdouts faceted side by side
    # ---------------------------------------------------------------
    print("\n=== benchmark_generalizability (combined) ===")
    combined_metrics = pd.concat(collected, ignore_index=True)

    bar_long = combined_metrics[["Model", "ROC_AUC", "PR_AUC", "Test_Dataset"]].melt(
        id_vars=["Model", "Test_Dataset"], var_name="Metric", value_name="Score")
    p = table_path("benchmark_generalizability_bar.csv")
    bar_long.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}")

    combined_roc = pd.concat(collected_roc, ignore_index=True)
    p = table_path("benchmark_generalizability_roc.csv")
    combined_roc.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}")

    # Wide view matching the LaTeX table the notebook printed.  Blanks are
    # real: an ablated model appears only in its own holdout.
    wide = (combined_metrics.set_index(["Test_Dataset", "Model"])[["ROC_AUC", "PR_AUC"]]
            .unstack("Model").sort_index(axis=1, level=0))
    p = table_path("benchmark_generalizability_summary_wide.csv")
    wide.to_csv(p)
    print(f"  wrote {os.path.basename(p)}")

    c = Checker("benchmark_generalizability_bar.png (facet means)")
    means = bar_long.groupby(["Test_Dataset", "Metric"])["Score"].mean()
    for dataset, want in COMBINED_MEANS.items():
        for metric, value in want.items():
            c.check(f"{dataset} / mean {metric}",
                    float(means.loc[(dataset, metric)]), value, 5e-4)
    checks.append(c.report())

    write_verification(checks, "VERIFICATION_03_generalizability.csv")


if __name__ == "__main__":
    main(combine_only="--combine-only" in sys.argv)
