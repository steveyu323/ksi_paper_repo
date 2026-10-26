"""
Shared helpers for regenerating, as CSV tables, the exact numbers plotted in
the figures under ``evaluation/eval_figs``.

Design rule
-----------
Every metric written by these scripts is produced by calling the *same*
function that produced the figures (``eval_plot_utils.output_evaluation_outputs``)
on the *same* inputs.  The tables are then dumped straight off the plot
objects it returns (``p2.data`` for a bar figure, ``p4.data`` for a ROC
figure), so a table cannot drift from its figure by construction.

Path remapping
--------------
The notebooks were authored on a machine where the project lived at
``/ksi_ml`` and an older copy at ``/kinase_ml``.  On the restored tree the
same content sits under a single root, laid out as ``<root>/code/code``,
``<root>/data/data``, ``<root>/runs/runs`` and ``<root>/configs/configs``.
``rp()`` translates the notebook-era absolute paths to the restored layout.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------

# <root>/code/code/evaluation/eval_table_scripts/eval_table_utils.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
RESTORE_ROOT = os.path.abspath(os.path.join(EVAL_DIR, "..", "..", ".."))

TABLE_DIR = os.path.join(EVAL_DIR, "eval_tables")
FIG_DIR = os.path.join(EVAL_DIR, "eval_figs")

# Make ``eval_plot_utils`` importable and make relative paths such as
# './eval_outputs/...' resolve the way they did inside the notebooks.
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

_PREFIX_MAP = {
    "/ksi_ml/code": os.path.join(RESTORE_ROOT, "code", "code"),
    "/ksi_ml/data": os.path.join(RESTORE_ROOT, "data", "data"),
    "/ksi_ml/runs": os.path.join(RESTORE_ROOT, "runs", "runs"),
    "/ksi_ml/configs": os.path.join(RESTORE_ROOT, "configs", "configs"),
    "/kinase_ml/code": os.path.join(RESTORE_ROOT, "code", "code"),
    "/kinase_ml/data": os.path.join(RESTORE_ROOT, "data", "data"),
    "/kinase_ml/runs": os.path.join(RESTORE_ROOT, "runs", "runs"),
}


def rp(path: str) -> str:
    """Translate a notebook-era path into its location on the restored tree."""
    path = str(path)
    for prefix, target in _PREFIX_MAP.items():
        if path == prefix:
            return target
        if path.startswith(prefix + "/"):
            return os.path.join(target, path[len(prefix) + 1:])
    if path.startswith("./"):
        return os.path.join(EVAL_DIR, path[2:])
    if not os.path.isabs(path):
        return os.path.join(EVAL_DIR, path)
    return path


def require(path: str) -> str:
    """Resolve ``path`` and fail loudly if the input is not on this machine."""
    resolved = rp(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"required input missing: {path}\n  looked in: {resolved}"
        )
    return resolved


def table_path(name: str) -> str:
    os.makedirs(TABLE_DIR, exist_ok=True)
    return os.path.join(TABLE_DIR, name)


# --------------------------------------------------------------------------
# Loading the prediction files the notebooks used
# --------------------------------------------------------------------------

def load_pairs_aug(name: str, columns=("kinase_seq", "kinase_domain_seq",
                                       "substrate_seq_15aa", "label")):
    """Load a ``pairs_aug_*.csv`` reference table the way the notebooks did."""
    df = pd.read_csv(require(f"/ksi_ml/data/{name}")).iloc[:, 1:].drop_duplicates()
    return df.loc[:, list(columns)]


def to_pairs_pred(pairs):
    """Drop the full kinase sequence, leaving the domain-keyed reference table.

    Mirrors the notebooks' ``pairs_pred.drop('kinase_seq', axis=1)`` step,
    which must happen only after every prediction file has been merged.
    """
    out = pairs.drop("kinase_seq", axis=1).drop_duplicates()
    out.columns = ["kinase_seq", "substrate_seq", "label"]
    return out


def load_phosformer(name: str):
    df = pd.read_csv(require(f"/ksi_ml/code/evaluation/eval_outputs/phosformer/{name}")).iloc[:, 1:]
    df = df.loc[:, ["kinase_domain_seq", "substrate_seq_15aa", "phosformer_logit"]]
    df.columns = ["kinase_seq", "substrate_seq", "probs"]
    return df


def load_kolossus(name: str, pairs):
    """Kolossus scores are keyed on the full kinase sequence, so they are
    re-keyed onto the kinase domain through the reference table."""
    df = pd.read_csv(require(f"/ksi_ml/code/evaluation/eval_outputs/kolossus/{name}")).iloc[:, 1:]
    df = pd.merge(
        df, pairs,
        left_on=["kinase_seq", "substrate_seq"],
        right_on=["kinase_seq", "substrate_seq_15aa"],
    ).loc[:, ["kinase_domain_seq", "substrate_seq_15aa", "value"]].drop_duplicates()
    df.columns = ["kinase_seq", "substrate_seq", "probs"]
    return df


def load_pkl_pred(name: str):
    from eval_plot_utils import parse_yml_pred
    return parse_yml_pred(require(f"/ksi_ml/code/evaluation/eval_outputs/{name}")).drop_duplicates()


def load_struct_pred(name: str):
    df = pd.read_csv(require(f"/ksi_ml/code/evaluation/eval_outputs/{name}")).iloc[:, 1:].drop_duplicates()
    df = df.loc[:, ["kinase_seq", "substrate_seq", "prob_positive"]].drop_duplicates()
    df.columns = ["kinase_seq", "substrate_seq", "probs"]
    return df


def load_frozen_pred(name: str):
    df = pd.read_csv(require(f"./eval_outputs/{name}")).iloc[:, 1:]
    df = df.iloc[:, [0, 1, 4]]
    df.columns = ["kinase_seq", "substrate_seq", "probs"]
    return df


def dedup(df):
    return df.drop_duplicates(["kinase_seq", "substrate_seq"])


# --------------------------------------------------------------------------
# Running a benchmark and writing its tables
# --------------------------------------------------------------------------

def run_benchmark(benchmark_id, pairs_pred, prediction_dfs, *, quiet=True, **kwargs):
    """Evaluate one benchmark through the figure-producing code path.

    Returns ``(metrics_df, bar_data, roc_data)`` where ``bar_data`` and
    ``roc_data`` are lifted directly off the plot objects that were saved as
    ``eval_figs/<benchmark_id>_bar.png`` and ``<benchmark_id>_roc.png``.
    """
    from eval_plot_utils import output_evaluation_outputs

    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        metrics_df, p2, p4, p5, p6 = output_evaluation_outputs(
            pairs_pred, prediction_dfs, **kwargs
        )

    bar_data = p2.data.copy()
    roc_data = p4.data.copy()
    return metrics_df, bar_data, roc_data


def write_benchmark_tables(benchmark_id, metrics_df, bar_data, roc_data,
                           *, note=None):
    """Write the bar, ROC and full-metrics tables for one benchmark."""
    written = []

    bar = bar_data.copy()
    bar["Model"] = bar["Model"].astype(str)
    bar["Metric"] = bar["Metric"].astype(str)
    p = table_path(f"{benchmark_id}_bar.csv")
    bar.to_csv(p, index=False)
    written.append(p)

    roc = roc_data.copy()
    roc["Model"] = roc["Model"].astype(str)
    p = table_path(f"{benchmark_id}_roc.csv")
    roc.to_csv(p, index=False)
    written.append(p)

    full = metrics_df.copy()
    full["Model"] = full["Model"].astype(str)
    p = table_path(f"{benchmark_id}_metrics_full.csv")
    full.to_csv(p, index=False)
    written.append(p)

    for w in written:
        print(f"  wrote {os.path.relpath(w, EVAL_DIR)}")
    if note:
        print(f"  note: {note}")
    return written


# --------------------------------------------------------------------------
# Verification against the numbers recorded in the notebooks
# --------------------------------------------------------------------------

class Checker:
    """Collects pass/fail comparisons against values stored in the notebooks."""

    def __init__(self, label):
        self.label = label
        self.rows = []

    def check(self, what, got, expected, tol):
        ok = (got is not None) and (expected is not None) and abs(got - expected) <= tol
        self.rows.append({
            "figure": self.label,
            "quantity": what,
            "regenerated": got,
            "figure_value": expected,
            "abs_diff": (abs(got - expected) if got is not None and expected is not None else None),
            "tolerance": tol,
            "match": bool(ok),
        })
        return ok

    def check_metrics(self, metrics_df, expected, tol=5e-4):
        """``expected`` maps model name -> {metric column -> value}."""
        indexed = metrics_df.set_index(metrics_df["Model"].astype(str))
        for model, metrics in expected.items():
            for col, want in metrics.items():
                got = float(indexed.loc[model, col]) if model in indexed.index else None
                self.check(f"{model} / {col}", got, want, tol)

    def report(self):
        df = pd.DataFrame(self.rows)
        n_ok = int(df["match"].sum())
        print(f"  verification: {n_ok}/{len(df)} values match the figure")
        bad = df[~df["match"]]
        if len(bad):
            print(bad.to_string(index=False))
        return df


def write_verification(all_rows, name="VERIFICATION_summary.csv"):
    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    p = table_path(name)
    df.to_csv(p, index=False)
    print(f"\nwrote {os.path.relpath(p, EVAL_DIR)}  "
          f"({int(df['match'].sum())}/{len(df)} values match)")
    return df
