#!/usr/bin/env python3
"""SuppFig_falsepositive - the substrate-only control libraries.

From 7_identifiy_false_positives.ipynb (the plotting cells; the thresholds and
the substrate lists themselves are produced by stage 07).

Three panels:
  1. LFC distribution of both substrate-only control libraries. The S/T library
     is bimodal - a peak near 0.5 plus a tail out to LFC 2-4 - while the Y
     library is a thin distribution around zero. Only the S/T tail is
     thresholded.
  2. Fraction of control-library substrates that are promiscuous, as the LFC
     cut is moved. Below LFC 1 fewer than 7% of them are positive for more than
     half of the kinases; the curve turns up at LFC ~ 2.
  3. Volcano of the S/T control library, coloured by promiscuity score, with
     the LFC = 1.5 / FDR = 0.1 cut that stage 07 applies.

Consumes
    config.COUNT_DIR/blib{7,10}_2batch_rra.{gene,sgrna}_summary.txt
    config.REDUCED_PKL                        stage 06
    config.OUT_DIR/fp_promiscuity_sweep.csv   stage 07

Produces (in config.FIG_DIR)
    suppfig_fp_lfc_distribution.pdf
    suppfig_fp_identification_panels.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

log = logging.getLogger("suppfig_fp")

PALETTE = {config.FP_SAMPLE_ST: "#2E86AB", config.FP_SAMPLE_Y: "#E84855"}
KDE_PALETTE = {config.FP_SAMPLE_ST: "#1B5E7B", config.FP_SAMPLE_Y: "#A3000C"}
TITLES = {
    config.FP_SAMPLE_ST: "S/T Substrate-Only Control Library",
    config.FP_SAMPLE_Y: "Y Substrate-Only Control Library",
}


def load_control(sample: str) -> pd.DataFrame:
    """Per-substrate MAGeCK results for one control library."""
    gene = pd.read_csv(config.COUNT_DIR / f"{sample}.gene_summary.txt", sep="\t")
    sgrna = pd.read_csv(config.COUNT_DIR / f"{sample}.sgrna_summary.txt", sep="\t")
    return pd.merge(gene, sgrna, left_on="id", right_on="Gene")


def fig_lfc_distribution(fig_dir: Path) -> None:
    sns.set_style("ticks")
    plt.rcParams.update({
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, sample in zip(axes, (config.FP_SAMPLE_ST, config.FP_SAMPLE_Y)):
        dat = load_control(sample)
        sns.histplot(dat, x="LFC", bins=50, ax=ax, color=PALETTE[sample], alpha=0.45,
                     edgecolor="white", linewidth=0.4, stat="density")
        sns.kdeplot(dat["LFC"], ax=ax, color=KDE_PALETTE[sample], linewidth=2.2,
                    bw_adjust=0.8)
        ax.set_xlim(-6, 8)
        ax.set_xlabel("Log Fold Change (LFC)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(TITLES[sample], fontsize=13, fontweight="bold", pad=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(which="both", direction="out")
        sns.despine(ax=ax, trim=True)
        log.info("%s: %d substrates, median LFC %.2f", sample, len(dat), dat["LFC"].median())

    fig.suptitle("LFC Distribution of Substrate-Only Control Libraries",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = fig_dir / "suppfig_fp_lfc_distribution.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out.name)


def fig_identification_panels(fig_dir: Path) -> None:
    sweep = pd.read_csv(config.OUT_DIR / "fp_promiscuity_sweep.csv")

    reduced = pd.read_pickle(config.REDUCED_PKL)
    pairs_st = reduced.loc[reduced["paabd_all"] != "sh2"]

    # Promiscuity score of every S/T substrate with at least one positive call.
    total = pairs_st.groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    pos = (pairs_st.loc[pairs_st["positive"], :]
           .groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]])
    fp_stats = pd.merge(pos, total, on="substrate")
    fp_stats.columns = ["substrate", "pos_kinase", "tot_kinase"]
    fp_stats["promiscuous_score"] = fp_stats["pos_kinase"] / fp_stats["tot_kinase"]

    control = load_control(config.FP_SAMPLE_ST)
    dat_count_fp = control.loc[
        control["control_mean"] > config.BARCODE_CONTROL_MEAN_MIN, :
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Panel A: promiscuity vs threshold ────────────────────────────────
    ax = axes[0]
    sns.lineplot(x=sweep["lfc_thresh"], y=sweep["frac_promiscuous"], ax=ax,
                 color="steelblue", linewidth=2.5)
    ax.axvline(x=config.FP_LFC_THRESH, color="crimson", linestyle="--", linewidth=1.8,
               label=f"LFC = {config.FP_LFC_THRESH} threshold")
    at_thresh = float(
        sweep.loc[(sweep["lfc_thresh"] - config.FP_LFC_THRESH).abs().idxmin(),
                  "frac_promiscuous"]
    )
    ax.axhline(y=at_thresh, color="crimson", linestyle=":", linewidth=1.4, alpha=0.8)
    ax.annotate(f"{at_thresh:.0%} promiscuous\nat LFC = {config.FP_LFC_THRESH}",
                xy=(config.FP_LFC_THRESH, at_thresh), xytext=(2.2, 0.60),
                fontsize=9, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc="mistyrose", ec="crimson",
                          alpha=0.85))
    ax.axvspan(config.FP_LFC_THRESH, 4, alpha=0.08, color="crimson",
               label=f"FP zone (LFC > {config.FP_LFC_THRESH})")
    ax.set_xlabel("LFC Threshold", fontsize=12)
    ax.set_ylabel("Fraction of substrates with\npromiscuous score > "
                  f"{config.PROMISCUITY_SCORE_CUT}", fontsize=12)
    ax.set_title("Promiscuity vs. LFC Threshold", fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_xlim(-4, 4)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    log.info("at LFC = %g, %.1f%% of control-library substrates are promiscuous",
             config.FP_LFC_THRESH, 100 * at_thresh)

    # ── Panel B: volcano of the control library ──────────────────────────
    ax = axes[1]
    dat_plot = dat_count_fp.copy()
    dat_plot["neg_log10_fdr"] = -np.log10(dat_plot["FDR"].clip(lower=1e-10))
    dat_plot = dat_plot.merge(fp_stats[["substrate", "promiscuous_score"]],
                              left_on="Gene", right_on="substrate", how="left")
    dat_plot["promiscuous_score"] = dat_plot["promiscuous_score"].fillna(0)
    is_fp = (dat_plot["FDR"] < config.FP_FDR_THRESH) & (
        dat_plot["LFC"] > config.FP_LFC_THRESH
    )

    ax.scatter(dat_plot.loc[~is_fp, "LFC"], dat_plot.loc[~is_fp, "neg_log10_fdr"],
               c=dat_plot.loc[~is_fp, "promiscuous_score"], cmap="Reds", vmin=0, vmax=1,
               s=5, alpha=0.5, zorder=1)
    sc = ax.scatter(dat_plot.loc[is_fp, "LFC"], dat_plot.loc[is_fp, "neg_log10_fdr"],
                    c=dat_plot.loc[is_fp, "promiscuous_score"], cmap="Reds",
                    vmin=0, vmax=1, s=5, alpha=0.5, label="FP substrate", zorder=2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Promiscuity score\n(fraction of kinases → positive)", fontsize=9)
    ax.axvline(x=config.FP_LFC_THRESH, color="crimson", linestyle="--", linewidth=1.6,
               label=f"LFC = {config.FP_LFC_THRESH}")
    ax.axhline(y=-np.log10(config.FP_FDR_THRESH), color="navy", linestyle="--",
               linewidth=1.6, label=f"FDR = {config.FP_FDR_THRESH}")
    ax.set_xlabel("LFC (substrate-only library)", fontsize=12)
    ax.set_ylabel("–log₁₀(FDR)", fontsize=12)
    ax.set_title("Volcano Plot — Substrate-Only Library", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    log.info("volcano: %d of %d control substrates in the FP quadrant",
             int(is_fp.sum()), len(dat_plot))

    fig.suptitle(
        f"False Positive Substrate Identification: LFC = {config.FP_LFC_THRESH} "
        f"& FDR < {config.FP_FDR_THRESH} Thresholds",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = fig_dir / "suppfig_fp_identification_panels.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)

    fig_dir = config.FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_lfc_distribution(fig_dir)
    fig_identification_panels(fig_dir)


if __name__ == "__main__":
    main()
