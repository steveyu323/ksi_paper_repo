#!/usr/bin/env python
"""
Supplementary Figure S6 (SuppFig_falsepositive): substrate-only control
libraries define the false-positive threshold.

Panel provenance (code lifted from 7_identifiy_false_positives.ipynb):
  a, b  cell 2   -> supp_plot_outputs/lfc_distribution.pdf
  c     cell 8   -> left panel of supp_plot_outputs/fp_identification_panels.pdf
  d     cell 36  -> right panel of supp_plot_outputs/fp_identification_panels.pdf

Note on panel d: in the notebook, cell 36 coloured the volcano with whatever
`fp_stats` was in memory. The shipped PDF has 2,753 data markers of which 45
are non-zero, which reproduces only with the cell-11 table: promiscuity across
S/T kinases of the control substrates flagged at LFC > 1.5 and FDR < 0.1.
That table is used here explicitly. (The PDF holds 2,754 / 46 marks because
the legend swatch is one more marker.)

Inputs
  ${KSI_DATA_ROOT}/processed_count_table/blib{7,10}_2batch_rra.*_summary.txt
  ksi_analysis_2025/251203_ksi_reduced_df.pkl

Usage:  python sf6_false_positive.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # ksi_analysis_2025
# MAGeCK count/RRA tables. Override with KSI_COUNT_DIR, or set KSI_DATA_ROOT
# and leave the default layout; neither default resolves off the lab machines.
COUNT_DIR = os.environ.get(
    "KSI_COUNT_DIR",
    os.path.join(os.environ.get("KSI_DATA_ROOT", "/path/to/sequencing_data_root"),
                 "processed_count_table"))
OUT_DIR = os.path.join(HERE, "out")
SRC_DIR = os.path.join(OUT_DIR, "source_data")

ST_LIB, Y_LIB = "blib7_2batch_rra", "blib10_2batch_rra"
LFC_THRESH = 1.5
FDR_THRESH = 0.1
THRESH_GRID = np.arange(-4, 4, 0.05)

BLUE, VERMILLION = "#0072B2", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def load_control_library(tag):
    """gene_summary merged onto sgrna_summary, as in notebook 7 cell 2."""
    gene = pd.read_csv(f"{COUNT_DIR}/{tag}.gene_summary.txt", sep="\t")
    sgrna = pd.read_csv(f"{COUNT_DIR}/{tag}.sgrna_summary.txt", sep="\t")
    return pd.merge(gene, sgrna, left_on="id", right_on="Gene")


def promiscuity(df):
    """Fraction of screened kinases that call each substrate positive (cell 8)."""
    total = df.groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    pos = df.loc[df["positive"], :].groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    stats = pd.merge(pos, total, on="substrate")
    stats.columns = ["substrate", "pos_kinase", "tot_kinase"]
    stats["promiscuous_score"] = stats["pos_kinase"] / stats["tot_kinase"]
    return stats


def flagged_substrates(dat_count_fp, thresh):
    return dat_count_fp.loc[
        (dat_count_fp["FDR"] < FDR_THRESH) & (dat_count_fp["LFC"] > thresh), "Gene"
    ]


def main():
    os.makedirs(SRC_DIR, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    lfc_st = load_control_library(ST_LIB)
    lfc_y = load_control_library(Y_LIB)

    dat_all_blib = pd.read_pickle(os.path.join(ROOT, "251203_ksi_reduced_df.pkl"))
    dat_st = dat_all_blib.loc[dat_all_blib["paabd_all"] != "sh2"]

    sgrna_st = pd.read_csv(f"{COUNT_DIR}/{ST_LIB}.sgrna_summary.txt", sep="\t")
    dat_count_fp = sgrna_st.loc[sgrna_st["control_mean"] > 10, :]

    # panel c: fraction of flagged substrates with promiscuity > 0.5
    stat_ls = []
    for thresh in THRESH_GRID:
        subs = flagged_substrates(dat_count_fp, thresh)
        s = promiscuity(dat_st.loc[dat_st["substrate"].isin(subs)])
        stat_ls.append((s["promiscuous_score"] > 0.5).sum() / s.shape[0])
    stat_ls = np.array(stat_ls)
    i_cut = int(np.argmin(np.abs(THRESH_GRID - LFC_THRESH)))

    # panel d: volcano, coloured by promiscuity of control-flagged substrates
    fp_stats = promiscuity(dat_st.loc[dat_st["substrate"].isin(
        flagged_substrates(dat_count_fp, LFC_THRESH))])
    dat_plot = dat_count_fp.copy()
    dat_plot["neg_log10_fdr"] = -np.log10(dat_plot["FDR"].clip(lower=1e-10))
    dat_plot = dat_plot.merge(fp_stats[["substrate", "promiscuous_score"]],
                              left_on="Gene", right_on="substrate", how="left")
    dat_plot["promiscuous_score"] = dat_plot["promiscuous_score"].fillna(0)
    n_scored = int((dat_plot["promiscuous_score"] > 0.1).sum())

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.9),
                             gridspec_kw=dict(hspace=0.65, wspace=0.32))
    xlim_ab = (-3.2, 7.2)

    for ax, df, color, title, yticks in [
        (axes[0, 0], lfc_st, BLUE, "Ser/Thr substrate-only library", [0, 0.15, 0.30, 0.45]),
        (axes[0, 1], lfc_y, VERMILLION, "Tyr substrate-only library", [0, 0.8, 1.6, 2.4]),
    ]:
        sns.histplot(df, x="LFC", bins=50, ax=ax, color=color, alpha=0.35,
                     edgecolor="white", linewidth=0.3, stat="density")
        sns.kdeplot(df["LFC"], ax=ax, color=color, linewidth=1.6, bw_adjust=0.8)
        ax.axvline(LFC_THRESH, color=VERMILLION, ls="--", lw=0.8)
        ax.set_xlim(*xlim_ab)
        ax.set_yticks(yticks)
        ax.set_xlabel("Log fold change")
        ax.set_ylabel("Density")
        ax.set_title(title)
    axes[0, 0].text(LFC_THRESH + 0.1, axes[0, 0].get_ylim()[1] * 0.97,
                    f"LFC = {LFC_THRESH}", color=VERMILLION, fontsize=7, va="top")

    ax = axes[1, 0]
    ax.axvspan(LFC_THRESH, 4, color=VERMILLION, alpha=0.06, lw=0)
    ax.plot(THRESH_GRID, stat_ls * 100, color=BLUE, lw=1.6)
    ax.axvline(LFC_THRESH, color=VERMILLION, ls="--", lw=0.8)
    ax.plot(THRESH_GRID[i_cut], stat_ls[i_cut] * 100, "o", color=VERMILLION, ms=4, zorder=5)
    ax.annotate(f"{stat_ls[i_cut] * 100:.0f} % at the chosen\nLFC = {LFC_THRESH} cut-off",
                xy=(THRESH_GRID[i_cut], stat_ls[i_cut] * 100), xytext=(-0.3, 58),
                ha="right", fontsize=7, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#333333", lw=0.6))
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 105)
    ax.set_xticks(range(-4, 5))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Log fold change threshold")
    ax.set_ylabel("Promiscuous substrates (%)")

    ax = axes[1, 1]
    bg = dat_plot.loc[dat_plot["promiscuous_score"] <= 0.1]
    fg = dat_plot.loc[dat_plot["promiscuous_score"] > 0.1].sort_values("promiscuous_score")
    ax.scatter(bg["LFC"], bg["neg_log10_fdr"], s=2, color="#BBBBBB", lw=0, zorder=1)
    sc = ax.scatter(fg["LFC"], fg["neg_log10_fdr"], c=fg["promiscuous_score"],
                    cmap="viridis", vmin=0, vmax=1, s=4 + 10 * fg["promiscuous_score"],
                    lw=0, zorder=2)
    ax.axvline(LFC_THRESH, color=VERMILLION, ls="--", lw=0.8)
    ax.axhline(-np.log10(FDR_THRESH), color="#444444", ls=":", lw=0.8)
    ax.set_yticks(range(0, 11, 2))
    ax.text(0.02, 0.165, f"FDR = {FDR_THRESH}", transform=ax.transAxes,
            fontsize=7, color="#444444", va="bottom")
    ax.text(0.98, 0.165, f"{n_scored} of {len(dat_plot)} substrates\nwith promiscuity > 0.1",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#444444")
    ax.set_xlabel("Log fold change (substrate-only library)")
    ax.set_ylabel(r"$-\log_{10}$ FDR")
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.05)
    cbar.set_label("Promiscuity score", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    for ax, letter in zip(axes.flat, "abcd"):
        ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"SuppFig_falsepositive.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── source data + summary ─────────────────────────────────────────────────
    pd.concat([lfc_st[["Gene", "LFC"]].assign(library="ST"),
               lfc_y[["Gene", "LFC"]].assign(library="Y")]).to_csv(
        os.path.join(SRC_DIR, "sf6ab_control_library_lfc.csv"), index=False)
    pd.DataFrame({"lfc_threshold": THRESH_GRID.round(2), "frac_promiscuous": stat_ls}).to_csv(
        os.path.join(SRC_DIR, "sf6c_promiscuity_vs_threshold.csv"), index=False)
    dat_plot[["sgrna", "Gene", "LFC", "FDR", "neg_log10_fdr", "promiscuous_score"]].to_csv(
        os.path.join(SRC_DIR, "sf6d_volcano.csv"), index=False)

    print(f"c: min fraction promiscuous = {stat_ls.min():.3f}; "
          f"at LFC {THRESH_GRID[i_cut]:.2f} = {stat_ls[i_cut]:.4f}")
    print(f"d: {n_scored} of {len(dat_plot)} control substrates with promiscuity > 0.1")
    print("wrote", os.path.join(OUT_DIR, "SuppFig_falsepositive.pdf"))


if __name__ == "__main__":
    main()
