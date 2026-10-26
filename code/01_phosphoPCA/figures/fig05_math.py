#!/usr/bin/env python3
"""Figure 05 (MATH) - how the decoupling model turns counts into binding scores.

Panels, and where each comes from:

  B  mock Boltzmann partition -> barcode log2FC
     mock_boltzmann_to_logfc.build_panel()
  C  hit selection in the (b_sk, lfsr) plane
     decoupling_supp_figures.fig_hit_selection()
     (its companion figS_decoupling_schematic is SuppFig_decoupling; run
      `python decoupling_supp_figures.py` for the full supplementary set)
  D  screening reliability benchmarks, two parts:
       - database confusion matrix on the evaluable subset
         (8_2batch_blib_pipeline_v3.ipynb sections 7-8)
       - in-vitro peptide-array validation
         (10_comparison_to_invitro_v2.ipynb)

Panel A is the library schematic, drawn by hand; its numbers are the library
statistics printed by this script (`--library-stats`).

Consumes
    config.INFERENCE_SCORES_PKL, config.REDUCED_TP_PKL     stages 06, 07
    config.LFC_TABLE_PKL                                   stage 03
    config.BLIB_KINASES_SRC_CSV
    config.HUMAN_KINOME_PKL, config.INTERACTION_CLASSED_WSOURCE_PKL,
    config.KINBERT_V1_TEST_PKL                             KINBERT corpus
    config.KINEXUS_XLSX                                    peptide-array data

Produces (in config.FIG_DIR)
    mock_boltzmann_to_logfc.{png,pdf}                       panel B
    decoupling/figS_hit_selection_binding_lfsr.{pdf,svg,png}  panel C
    fig05D_confusion_matrix_{y,st,sty}.pdf                  panel D, left
    fig05D_confusion_summary.csv, fig05D_fn_decomposition.csv
    fig05D_invitro_pie.pdf, fig05D_invitro_overlap_summary.csv  panel D, right
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("fig05")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

# Confusion-matrix cells sampled from matplotlib's 'summer': deep green for the
# key result (TP), pale yellow for the large, expected TN block.
_summer = colormaps["summer"]
CELL_VALS = {"TN": 0.95, "FP": 0.65, "FN": 0.40, "TP": 0.10}

FN_CATEGORY_ORDER = ["fp_sub", "inactive_sub", "no_paabd", "not_in_screen", "evaluable"]
FN_CATEGORY_LABELS = {
    "fp_sub": "FP substrate",
    "inactive_sub": "Inactive substrate",
    "no_paabd": "No PAABD motif",
    "not_in_screen": "Not in screen",
    "evaluable": "Evaluable FN",
}
NON_EVALUABLE = FN_CATEGORY_ORDER[:-1]

# Kinexus plate labels -> the construct names used in the screen.
KINEXUS_NAME_MAP = pd.DataFrame({
    "uniprot_acc": [
        "KS6B1_HUMAN_wt", "PLK1_HUMAN", "TBK1_HUMAN", "PDPK1_HUMAN", "IKKB_HUMAN",
        "FYN_HUMAN", "FES_HUMAN", "KSYK_HUMAN", "FRK_HUMAN", "DYRK4_HUMAN",
    ],
    "common_name": [
        "P70S6K", "PLK1", "TBK1", "PDK1", "IKKB",
        "FYN", "FES", "SYK", "FRK", "DYRK4",
    ],
})


# ═══════════════════════════════════════════════════════════════════════════
# Panel D, left: database confusion matrix
# ═══════════════════════════════════════════════════════════════════════════
def assign_paabd(dat: pd.DataFrame, get_sub_15: bool = True) -> pd.DataFrame:
    """Flag which reporter domain could read out each substrate.

    A phosphosite is only visible to a pAABD whose binding motif it matches,
    so these flags decide whether a missed annotation was even evaluable:

        SH2      pY at 0 and a hydrophobic (I/L/V/M) at +3
        14-3-3   pS/pT at 0, R at -3, P at +2
        WW       pS/pT at 0, P at +1
        FHA      pT at 0, D at +3

    Motif definitions as in 8_2batch_blib_pipeline_v3.ipynb. NOTE: these are
    stricter than the ones in 4_filter_unmapped_substrates.ipynb (SH2 there is
    pY alone, FHA allows D/I/S/Y at +3); the confusion-matrix numbers in the
    manuscript use the definitions below.
    """
    dat = dat.copy()
    if get_sub_15:
        dat["substrate_seq_15aa"] = dat.apply(
            lambda s: s["substrate_seq"][s["p_ind"] - 1 - 7: s["p_ind"] - 1 + 8], axis=1
        )
    dat = dat.loc[dat["substrate_seq_15aa"].apply(len) == 15, :].copy()

    seq = dat["substrate_seq_15aa"]
    dat["SH2"] = seq.apply(lambda s: (s[7] == "Y") & (s[10] in {"I", "L", "V", "M"}))
    dat["b14-3-3"] = seq.apply(lambda s: (s[7] in "ST") & (s[4] == "R") & (s[9] == "P"))
    dat["WW"] = seq.apply(lambda s: (s[7] in "ST") & (s[8] == "P"))
    dat["FHA"] = seq.apply(lambda s: (s[7] == "T") & (s[10] == "D"))
    return dat


def load_annotation() -> pd.DataFrame:
    """Curated KSI annotations, restricted to high-confidence sources.

    Two sources are concatenated: the curated interaction database, which
    supplies the positives, and the KINBERT held-out test split, which supplies
    matched negatives (y_true = 0) drawn from the same substrates.
    """
    human_kinome = pd.read_pickle(config.HUMAN_KINOME_PKL)
    log.info("human kinome: %d entries", len(human_kinome))

    interactions = assign_paabd(pd.read_pickle(config.INTERACTION_CLASSED_WSOURCE_PKL))
    interactions = (
        interactions.loc[:, ["kinase", "substrate_seq_15aa", "source_all"]]
        .drop_duplicates()
        .rename(columns={"source_all": "source"})
    )
    interactions["y_true"] = 1
    log.info("interaction DB: %d unique KSIs", len(interactions))

    test_pred = pd.read_pickle(config.KINBERT_V1_TEST_PKL)
    test_pred = assign_paabd(test_pred)
    test_pred = (
        test_pred.loc[:, ["kinase_seq", "substrate_seq_15aa", "y_true", "source"]]
        .merge(human_kinome, on="kinase_seq")
        .loc[:, ["kinase", "substrate_seq_15aa", "source", "y_true"]]
    )

    data_anno = pd.concat([interactions, test_pred], ignore_index=True)
    log.info("combined annotation: %d rows", len(data_anno))

    train_dat = assign_paabd(data_anno, get_sub_15=False)
    train_dat_pos = train_dat.loc[~train_dat["source"].isin(config.DB_EXCLUDE_SOURCES), :]
    log.info("high-confidence DB entries: %d (%d positive)",
             len(train_dat_pos), int((train_dat_pos["y_true"] == 1).sum()))
    return train_dat_pos


def load_screen_for_benchmark() -> pd.DataFrame:
    """Screening hits joined to their kinase accession, benchmark subset."""
    blib = pd.read_csv(config.BLIB_KINASES_SRC_CSV, header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]

    dat = pd.read_pickle(config.REDUCED_TP_PKL)
    dat = pd.merge(blib, dat, on="kinase")
    dat = dat.loc[~dat["kinase"].isin(config.DB_EXCLUDE_KINASES), :]
    log.info("screening pairs retained for the benchmark: %d", len(dat))
    return dat


def load_screened_substrates() -> dict[str, pd.DataFrame]:
    """Every (substrate, kinase) the screen recovered, per library.

    Used only to tell "annotated pair we called negative" from "annotated pair
    whose substrate never worked in the screen at all".
    """
    ksi = pd.read_pickle(config.LFC_TABLE_PKL)
    ksi = ksi.drop_duplicates(["substrate_aa", "kinase_aa"])
    # `substrate` in the stage 03 table is the nucleotide sequence; drop it so
    # that `substrate` means the 15-mer here, as it does in the hit tables.
    ksi = ksi.drop(columns=["substrate"]).rename(columns={"substrate_aa": "substrate"})
    out = {
        "ST": ksi.loc[ksi["kinase_type"] != "Y", :],
        "Y": ksi.loc[ksi["kinase_type"] == "Y", :],
    }
    log.info("screened pairs - ST: %d, Y: %d", len(out["ST"]), len(out["Y"]))
    return out


def inactive_substrates(ksi_lib: pd.DataFrame) -> set:
    """Substrates present in the library that never produced a hit.

    NOTE: hits here are MAGeCK FDR < 0.1 on the raw enrichment, not the
    decoupled call. That is what the notebook does (its `true_positive` branch
    never fires, because the raw screen table has no such column), and it is
    the looser of the two criteria, so a substrate flagged inactive is one that
    failed even by the permissive test.
    """
    hit_mask = ksi_lib["FDR"] < 0.1
    active = set(ksi_lib.loc[hit_mask, "substrate"].dropna())
    screened = set(ksi_lib["substrate"].dropna())
    return screened - active


def classify_false_negatives(curr_df_all: pd.DataFrame,
                             screened: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Give every annotated pair we called negative one evaluability category.

    Priority order, first match wins: a substrate that is itself a false
    positive, then one that is inactive throughout the screen, then one with no
    reader-domain motif, then one absent from the screen. What is left is a
    genuine platform miss.
    """
    fn = curr_df_all.loc[
        (curr_df_all["set"] == "train_pos")
        & (~curr_df_all["true_positive"].astype(bool)), :
    ].copy()
    log.info("annotated pairs called negative: %d", len(fn))

    parts = []
    for lib in ("ST", "Y"):
        part = (
            fn.loc[fn["kinase_type"] == "ST", :] if lib == "ST"
            else fn.loc[fn["kinase_type"] != "ST", :]
        ).copy()
        ksi_lib = screened[lib]
        part["is_fp_sub"] = part["fp_sub"].astype(bool)
        part["in_screen"] = part["substrate_seq_15aa"].isin(
            set(ksi_lib["substrate"].dropna())
        )
        part["is_inactive_sub"] = part["substrate_seq_15aa"].isin(inactive_substrates(ksi_lib))
        if lib == "ST":
            part["has_paabd"] = part["b14-3-3"] | part["WW"] | part["FHA"]
        else:
            part["has_paabd"] = part["SH2"].astype(bool)

        cat = pd.Series("evaluable", index=part.index)
        cat[~part["in_screen"]] = "not_in_screen"
        cat[part["in_screen"] & ~part["has_paabd"]] = "no_paabd"
        cat[part["in_screen"] & part["has_paabd"] & part["is_inactive_sub"]] = "inactive_sub"
        cat[part["is_fp_sub"]] = "fp_sub"
        part["fn_category"] = cat

        log.info("%s false-negative breakdown: %s", lib,
                 part["fn_category"].value_counts().to_dict())
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def evaluable_recall(curr_df_lib: pd.DataFrame, fn_lib: pd.DataFrame, label: str) -> dict:
    """Recall over all annotated positives, and over the evaluable ones only."""
    db_pos = curr_df_lib.loc[curr_df_lib["set"] == "train_pos"]
    tp = int(db_pos["true_positive"].astype(bool).sum())
    fn_total = int((~db_pos["true_positive"].astype(bool)).sum())

    counts = fn_lib["fn_category"].value_counts()
    fn_eval = int(counts.get("evaluable", 0))

    recall_overall = tp / (tp + fn_total) if (tp + fn_total) else float("nan")
    recall_eval = tp / (tp + fn_eval) if (tp + fn_eval) else float("nan")
    coverage = (tp + fn_eval) / (tp + fn_total) if (tp + fn_total) else float("nan")

    result = {"library": label, "TP": tp, "FN_total": fn_total, "FN_evaluable": fn_eval}
    for cat in FN_CATEGORY_ORDER:
        result[f"FN_{cat}"] = int(counts.get(cat, 0))
    result["db_coverage"] = round(coverage, 3)
    result["recall_overall"] = round(recall_overall, 3)
    result["recall_evaluable"] = round(recall_eval, 3)

    log.info("%s: TP %d, FN %d (evaluable %d), coverage %.3f, "
             "recall %.3f -> %.3f on the evaluable subset",
             label, tp, fn_total, fn_eval, coverage, recall_overall, recall_eval)
    return result


def drop_non_evaluable(curr_df_lib: pd.DataFrame, fn_lib: pd.DataFrame) -> pd.DataFrame:
    """Remove the non-evaluable false negatives, matching on the pair key."""
    keys = ["kinase_acc", "substrate_seq_15aa"]
    non_eval = fn_lib.loc[fn_lib["fn_category"] != "evaluable", keys]
    exclude = curr_df_lib.set_index(keys).index.isin(pd.MultiIndex.from_frame(non_eval))
    return curr_df_lib.loc[~exclude]


def _luminance(rgba) -> float:
    r, g, b = rgba[:3]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def plot_confusion(curr_df: pd.DataFrame, title: str, out_path: Path) -> dict:
    """2x2 database-vs-screening confusion matrix with its metrics."""
    s = curr_df["true_positive"].astype(bool)
    is_db_neg = curr_df["set"].eq("train_tn")
    is_db_pos = ~is_db_neg

    tp = int((s & is_db_pos).sum())
    fp = int((s & is_db_neg).sum())
    fn = int((~s & is_db_pos).sum())
    tn = int((~s & is_db_neg).sum())
    cm = np.array([[tn, fp], [fn, tp]])

    key_grid = [["TN", "FP"], ["FN", "TP"]]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.set_facecolor("white")
    for i in range(2):
        for j in range(2):
            rgba = _summer(CELL_VALS[key_grid[i][j]])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=rgba, zorder=0))
            ax.text(j, i, f"{key_grid[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if _luminance(rgba) < 0.45 else "#1a1a1a", zorder=1)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Screening −", "Screening +"], fontsize=11)
    ax.set_yticklabels(["Database −", "Database +"], fontsize=11)
    ax.set_xlabel("Screening", fontsize=12, labelpad=8)
    ax.set_ylabel("Database", fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, pad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    total = cm.sum()
    acc = (tp + tn) / total if total else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    ax.text(1.04, 0.5,
            f"n = {total:,}\nAccuracy  {acc:.3f}\nPrecision {prec:.3f}\nRecall    {rec:.3f}",
            transform=ax.transAxes, va="center", ha="left",
            fontsize=10, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7", edgecolor="#cccccc"))

    fig.savefig(out_path)
    plt.close(fig)
    log.info("wrote %s (TP %d, FP %d, FN %d, TN %d)", out_path.name, tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": acc, "precision": prec, "recall": rec}


def panel_d_confusion(fig_dir: Path) -> None:
    train_dat_pos = load_annotation()
    dat_in_screen = load_screen_for_benchmark()
    screened = load_screened_substrates()

    # Only pairs the database and the screen both cover can be scored.
    curr_df_all = pd.merge(
        dat_in_screen, train_dat_pos,
        left_on=["kinase_acc", "substrate"],
        right_on=["kinase", "substrate_seq_15aa"],
    )
    curr_df_all["set"] = "train_tn"
    curr_df_all.loc[curr_df_all["y_true"] == 1, "set"] = "train_pos"
    log.info("spike-in pairs scored: %d", len(curr_df_all))

    curr_df_st = curr_df_all.loc[curr_df_all["kinase_type"] == "ST", :].copy()
    curr_df_y = curr_df_all.loc[curr_df_all["kinase_type"] != "ST", :].copy()

    fn_all = classify_false_negatives(curr_df_all, screened)
    fn_st = fn_all.loc[fn_all["kinase_type"] == "ST"]
    fn_y = fn_all.loc[fn_all["kinase_type"] != "ST"]

    rec_y = evaluable_recall(curr_df_y, fn_y, "Y library")
    rec_st = evaluable_recall(curr_df_st, fn_st, "ST library")
    rec_all = evaluable_recall(curr_df_all, fn_all, "Combined")

    stats = {}
    for label, curr, fn_lib, fname in (
        ("Y library Database Spike-Ins", curr_df_y, fn_y, "fig05D_confusion_matrix_y.pdf"),
        ("ST library Database Spike-Ins", curr_df_st, fn_st, "fig05D_confusion_matrix_st.pdf"),
        ("ST + Y Database Spike-Ins", curr_df_all, fn_all, "fig05D_confusion_matrix_sty.pdf"),
    ):
        stats[label] = plot_confusion(
            drop_non_evaluable(curr, fn_lib), label, fig_dir / fname
        )

    pd.DataFrame(stats).T.reset_index(names="panel").to_csv(
        fig_dir / "fig05D_confusion_summary.csv", index=False
    )

    # Supplementary table: where the un-recovered annotations went.
    rows = []
    for label, curr, rec in (("Y library", curr_df_y, rec_y),
                             ("ST library", curr_df_st, rec_st),
                             ("Combined", curr_df_all, rec_all)):
        db_pos = curr.loc[curr["set"] == "train_pos"]
        tp = int(db_pos["true_positive"].astype(bool).sum())
        fp = int((curr["true_positive"].astype(bool) & curr["set"].eq("train_tn")).sum())
        tn = int((~curr["true_positive"].astype(bool) & curr["set"].eq("train_tn")).sum())
        rows.append({
            "Library": label, "TP": tp, "FP": fp, "TN": tn,
            "FN (total)": rec["FN_total"],
            "FP sub": rec["FN_fp_sub"],
            "Inactive sub": rec["FN_inactive_sub"],
            "No PAABD": rec["FN_no_paabd"],
            "Not in screen": rec["FN_not_in_screen"],
            "FN (evaluable)": rec["FN_evaluable"],
            "Precision": round(tp / (tp + fp), 3) if (tp + fp) else float("nan"),
            "Recall": rec["recall_evaluable"],
        })
    fn_table = pd.DataFrame(rows)
    fn_table.to_csv(fig_dir / "fig05D_fn_decomposition.csv", index=False)
    log.info("wrote fig05D_fn_decomposition.csv\n%s", fn_table.to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Panel D, right: in-vitro peptide-array comparison
# ═══════════════════════════════════════════════════════════════════════════
def process_kinexus_half(df: pd.DataFrame) -> pd.DataFrame:
    """Background-correct one Kinexus plate half.

    Two corrections: the last column is the non-specific control, subtracted
    row-wise, and the empty-peptide wells give a per-kinase offset, subtracted
    column-wise. The last two rows are controls and are dropped.
    """
    df = df.copy()
    df.columns = ["Sequence", "Pos"] + [c.split(" ")[-1] for c in df.columns[2:]]
    df.iloc[:, 2:-1] = df.iloc[:, 2:-1].sub(df.iloc[:, -1], axis=0)
    mean_empty = df.loc[df["Sequence"] == "empty", :].iloc[:, 2:-1].mean(axis=0)
    df.iloc[:-2, 2:-1] = df.iloc[:-2, 2:-1].sub(mean_empty, axis=1)
    return df.iloc[:-2, 2:-1]


def load_kinexus_sheet(sheet_name: str, n_rows_part1: int) -> pd.DataFrame:
    """One Kinexus sheet (two plate halves = two replicates) in long form."""
    raw = pd.read_excel(config.KINEXUS_XLSX, sheet_name=sheet_name)
    part1_raw = raw.iloc[:n_rows_part1, :]
    part2_raw = raw.iloc[n_rows_part1:, :]

    df1 = process_kinexus_half(part1_raw).melt()
    df2 = process_kinexus_half(part2_raw).melt()

    n_kinases = process_kinexus_half(part1_raw).shape[1]
    substrate_labels = pd.concat([part1_raw["Sequence"].iloc[:-2]] * n_kinases).tolist()

    combined = pd.concat([df1, df2], axis=1)
    combined.columns = ["kinase", "value1", "kinase2", "value2"]
    combined["substrate"] = substrate_labels
    return combined.drop("kinase2", axis=1)


def panel_d_invitro(fig_dir: Path) -> None:
    df_all = pd.concat(
        [load_kinexus_sheet(s, config.KINEXUS_ROWS_PART1) for s in config.KINEXUS_SHEETS],
        ignore_index=True,
    )
    log.info("Kinexus: %d kinase-substrate measurements", len(df_all))

    # Replicate QC: the two plate halves must agree within 30% of the larger
    # absolute signal, otherwise the spot is not trusted either way.
    v1 = df_all["value1"].astype(float)
    v2 = df_all["value2"].astype(float)
    rel_diff = (v1 - v2).abs() / np.maximum(np.maximum(v1.abs(), v2.abs()), 1e-9)
    valid = np.isfinite(v1) & np.isfinite(v2) & (rel_diff <= config.KINEXUS_REL_DIFF_MAX)
    df_valid = df_all.loc[valid, :].copy()
    log.info("replicate QC: %d pass, %d fail (> %g relative difference)",
             int(valid.sum()), int((~valid).sum()), config.KINEXUS_REL_DIFF_MAX)

    # NOTE: the notebook's prose says "signal > 100"; the code it ran, and the
    # figure in the manuscript, use 20. The code is kept.
    df_valid["invitro_pos"] = (
        (df_valid["value1"] > config.KINEXUS_SIGNAL_THRESH)
        & (df_valid["value2"] > config.KINEXUS_SIGNAL_THRESH)
    )
    log.info("in-vitro positives: %d of %d valid spots (%.1f%%)",
             int(df_valid["invitro_pos"].sum()), len(df_valid),
             100 * df_valid["invitro_pos"].mean())

    df_valid = pd.merge(
        df_valid, KINEXUS_NAME_MAP, left_on="kinase", right_on="common_name"
    ).drop("kinase", axis=1)
    df_valid = df_valid.loc[:, ["value1", "value2", "substrate", "invitro_pos",
                                "uniprot_acc", "common_name"]]
    df_valid.columns = ["value1", "value2", "substrate", "invitro_pos",
                        "kinase", "common_name"]

    dat_p_all = pd.read_pickle(config.REDUCED_TP_PKL)
    # NOTE: the notebook redefines the screening call for this comparison as
    # binding_max > 0 (any positive decoupled binding in any reporter), rather
    # than the lfsr/threshold call of stages 06-07. That is what produces the
    # 84.9% quoted on the panel, so it is kept; it is a looser criterion than
    # the canonical `true_positive`.
    dat_p_all = dat_p_all.copy()
    dat_p_all["true_positive"] = dat_p_all["binding_max"] > 0

    both = pd.merge(dat_p_all, df_valid, on=["kinase", "substrate"])
    log.info("pairs measured by both platforms: %d", len(both))

    screen_pos = set(both.index[both["true_positive"]])
    invitro_pos = set(both.index[both["invitro_pos"]])
    n_screen_only = len(screen_pos - invitro_pos)
    n_overlap = len(screen_pos & invitro_pos)
    n_invitro_only = len(invitro_pos - screen_pos)
    pct_of_screen = 100 * n_overlap / len(screen_pos) if screen_pos else 0.0
    pct_of_invitro = 100 * n_overlap / len(invitro_pos) if invitro_pos else 0.0

    log.info("phospho-PCA positives %d, in-vitro positives %d, overlap %d "
             "(%.1f%% of phospho-PCA, %.1f%% of in vitro)",
             len(screen_pos), len(invitro_pos), n_overlap, pct_of_screen, pct_of_invitro)

    summary = pd.DataFrame([
        {"Group": "PhosphoPCA only", "n": n_screen_only,
         "% of PhosphoPCA positives": round(100 * n_screen_only / len(screen_pos), 1)
         if screen_pos else np.nan,
         "% of In vitro positives": np.nan},
        {"Group": "Overlap", "n": n_overlap,
         "% of PhosphoPCA positives": round(pct_of_screen, 1),
         "% of In vitro positives": round(pct_of_invitro, 1)},
        {"Group": "In vitro only", "n": n_invitro_only,
         "% of PhosphoPCA positives": np.nan,
         "% of In vitro positives": round(100 * n_invitro_only / len(invitro_pos), 1)
         if invitro_pos else np.nan},
    ])
    summary.to_csv(fig_dir / "fig05D_invitro_overlap_summary.csv", index=False)
    log.info("wrote fig05D_invitro_overlap_summary.csv")

    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        [n_overlap, n_screen_only],
        labels=[f"Confirmed in vitro\n(n={n_overlap})",
                f"PhosphoPCA only\n(n={n_screen_only})"],
        explode=(0.06, 0),
        colors=[_summer(0.10), _summer(0.80)],
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.65,
        startangle=90,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
    )
    for t in texts:
        t.set_fontsize(11)
    for at in autotexts:
        at.set_fontsize(12)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("In Vitro Kinase-Peptide Validation", fontsize=13, pad=14)
    fig.savefig(fig_dir / "fig05D_invitro_pie.pdf")
    plt.close(fig)
    log.info("wrote fig05D_invitro_pie.pdf")


# ═══════════════════════════════════════════════════════════════════════════
# Panel A: the library statistics quoted on the schematic
# ═══════════════════════════════════════════════════════════════════════════
def library_stats() -> pd.DataFrame:
    """Designed / constructed / screened library content (Table S: libdesign).

    From 8_2batch_blib_pipeline_v3.ipynb section 5. "ecoli constructed" is what
    the PacBio lookup of the plasmid pool recovered, "yeast screened" what the
    Illumina screen recovered.
    """
    blib = pd.read_csv(config.BLIB_KINASES_SRC_CSV, header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]

    y_sub = pd.read_csv(config.Y_SUB_ALL_CSV).drop_duplicates("substrate_seq_15aa")
    st_sub = pd.read_csv(config.ST_SUB_ALL_CSV).drop_duplicates("substrate_seq_15aa")

    y_lookup = pd.read_pickle(config.Y_LOOKUP_PKL)
    st_lookup = pd.read_pickle(config.ST_LOOKUP_PKL)

    ksi = pd.read_pickle(config.LFC_TABLE_PKL).drop_duplicates(["substrate_aa", "kinase_aa"])
    ksi_y = ksi.loc[ksi["kinase_type"] == "Y", :]
    ksi_st = ksi.loc[ksi["kinase_type"] != "Y", :]

    rows = []
    for label, sub, lookup, screened, ktype in (
        ("Y library", y_sub, y_lookup, ksi_y, "Y"),
        ("ST library", st_sub, st_lookup, ksi_st, "ST"),
    ):
        designed_ks = {
            s + "&" + k
            for s in sub["substrate_seq_15aa"].tolist()
            for k in blib.loc[blib["kinase_type"] == ktype, "kinase_seq"].tolist()
        }
        rows.append({
            "library": label,
            "n_kinase": screened["kinase_name"].nunique(),
            "n_substrate_designed": len(sub),
            "n_substrate_ecoli_constructed": int(
                lookup["substrate_aa"].drop_duplicates()
                .isin(sub["substrate_seq_15aa"]).sum()
            ),
            "n_substrate_yeast_screened": int(
                screened["substrate_aa"].drop_duplicates()
                .isin(sub["substrate_seq_15aa"]).sum()
            ),
            "n_ksi_designed": len(designed_ks),
            "n_ksi_ecoli_constructed": len(
                set(lookup["substrate_aa"] + "&" + lookup["kinase_aa"]) & designed_ks
            ),
            "n_ksi_yeast_screened": len(
                set(screened["substrate_aa"] + "&" + screened["kinase_aa"]) & designed_ks
            ),
            # Including substrates that were not part of the design (mutations
            # picked up during library construction).
            "n_ksi_yeast_screened_all": len(
                set(screened["substrate_aa"] + "&" + screened["kinase_aa"])
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", choices=["b", "c", "d", "all"], default="all",
                    help="which panel to build (default: all)")
    ap.add_argument("--library-stats", action="store_true",
                    help="also print and write the panel A library statistics")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # matplotlib's font subsetter is chatty at INFO level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    fig_dir = config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    if args.panel in ("b", "all"):
        log.info("=== panel B: mock Boltzmann -> log2FC ===")
        import mock_boltzmann_to_logfc
        mock_boltzmann_to_logfc.build_panel(fig_dir)

    if args.panel in ("c", "all"):
        log.info("=== panel C: hit selection in the (b, lfsr) plane ===")
        import decoupling_supp_figures as dsf
        dsf.OUT_DIR.mkdir(parents=True, exist_ok=True)
        d = dsf.load_scores()
        dsf.fig_hit_selection(d, "figS_hit_selection_binding_lfsr", yscale="log")

    if args.panel in ("d", "all"):
        log.info("=== panel D, left: database confusion matrix ===")
        panel_d_confusion(fig_dir)
        log.info("=== panel D, right: in-vitro validation ===")
        panel_d_invitro(fig_dir)

    if args.library_stats:
        log.info("=== panel A: library statistics ===")
        stats = library_stats()
        stats.to_csv(fig_dir / "fig05A_library_stats.csv", index=False)
        log.info("\n%s", stats.to_string(index=False))


if __name__ == "__main__":
    main()
