#!/usr/bin/env python
"""
Model-free concordance of the two sequencing replicates.

rep 1 = sequencing batch 1 (blib_fp), rep 2 = batch 2 (blib_fp_b2), merged in
2_generate_lfc_table.ipynb and stored as pre_1/mtx_1 and pre_2/mtx_2 in
decoupling_out/raw_screen_count_mapped.csv (the table the model is fit to).

Per pAABD, at the (kinase, substrate) pair level (barcodes summed, as in the model):
  * pre-selection and post-selection counts, rep 1 vs rep 2 (log10 CPM)
  * log enrichment  l = log(mtx/N_mtx + eps) - log(pre/N_pre + eps)
  * kinase-centred log enrichment  l - mean_kinase(l)  — the naive analogue
    of the model's b_sk, with the per-kinase baseline removed
  * how agreement depends on read depth (pairs binned by min pre count)
  * per-kinase correlation of the kinase-centred enrichment

Natural-log units throughout (as the model), so these are directly
comparable with b_sk.

Outputs (out/):
  raw_concordance_summary.csv       per pAABD x quantity: Pearson, Spearman, n
  raw_concordance_by_depth.csv      per pAABD x depth bin
  raw_concordance_per_kinase.csv    per pAABD x kinase
  fig_raw_replicate_scatter.pdf     rep1 vs rep2 scatter grid
  fig_raw_concordance_by_depth.pdf  correlation vs. depth, per-kinase correlations

Usage (conda env ksi_bayes_qc, or any env with pandas/scipy/matplotlib):
  python 01_raw_replicate_concordance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))      # the pipeline package root
import config  # noqa: E402

OUT = HERE / "out"
SCREEN_CSV = config.SCREEN_COUNT_MAPPED_CSV     # stage 04

PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}
INK = "#222222"
PSEUDO = 0.5
JITTER = np.random.default_rng(0)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Liberation Sans"],
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote out/{name}.pdf/.png")


def load_pairs() -> pd.DataFrame:
    df = pd.read_csv(SCREEN_CSV).iloc[:, 1:]
    p = (df.groupby(["paabd", "kinase_name", "substrate_aa"])[["pre_1", "mtx_1", "pre_2", "mtx_2"]]
         .sum().reset_index().rename(columns={"kinase_name": "kinase", "substrate_aa": "substrate"}))
    out = []
    for pa, g in p.groupby("paabd"):
        g = g.copy()
        for r in (1, 2):
            n_pre, n_mtx = g[f"pre_{r}"].sum(), g[f"mtx_{r}"].sum()
            g[f"cpm_pre_{r}"] = np.log10((g[f"pre_{r}"] + PSEUDO) / n_pre * 1e6)
            g[f"cpm_mtx_{r}"] = np.log10((g[f"mtx_{r}"] + PSEUDO) / n_mtx * 1e6)
            g[f"lfc_{r}"] = (np.log((g[f"mtx_{r}"] + PSEUDO) / n_mtx)
                             - np.log((g[f"pre_{r}"] + PSEUDO) / n_pre))
            g[f"clfc_{r}"] = g[f"lfc_{r}"] - g.groupby("kinase")[f"lfc_{r}"].transform("mean")
        g["min_pre"] = g[["pre_1", "pre_2"]].min(axis=1)
        out.append(g)
    return pd.concat(out, ignore_index=True)


QUANTS = [("cpm_pre", "pre-selection log10 CPM"),
          ("cpm_mtx", "post-selection log10 CPM"),
          ("lfc", "log enrichment"),
          ("clfc", "kinase-centred log enrichment")]
# The scatter figure shows the raw-count quantities only; the kinase-centred
# enrichment stays in the CSV summaries (03_compare_replicate_fits.py reads it).
SCATTER_QUANTS = QUANTS[:3]


def corr(a, b):
    return pearsonr(a, b)[0], spearmanr(a, b)[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_pairs()

    # ---- summary -------------------------------------------------------------
    rows = []
    for pa in PAABD_ORDER:
        g = d[d.paabd == pa]
        for q, lab in QUANTS:
            r, rho = corr(g[f"{q}_1"], g[f"{q}_2"])
            diff = g[f"{q}_2"] - g[f"{q}_1"]
            rows.append({"paabd": pa, "quantity": q, "label": lab, "n_pairs": len(g),
                         "pearson": r, "spearman": rho,
                         "sd_rep_diff": diff.std(), "sd_mean_signal": ((g[f"{q}_1"] + g[f"{q}_2"]) / 2).std()})
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "raw_concordance_summary.csv", index=False)

    # ---- by depth --------------------------------------------------------------
    edges = [0, 50, 100, 200, 500, 1000, 2000, 5000, np.inf]
    dep = []
    for pa in PAABD_ORDER:
        g = d[d.paabd == pa]
        bins = pd.cut(g.min_pre, edges, right=False)
        for iv, gg in g.groupby(bins, observed=True):
            if len(gg) < 30:
                continue
            r, rho = corr(gg.clfc_1, gg.clfc_2)
            dep.append({"paabd": pa, "min_pre_lo": iv.left, "min_pre_hi": iv.right,
                        "n_pairs": len(gg), "pearson_clfc": r, "spearman_clfc": rho,
                        "pearson_lfc": corr(gg.lfc_1, gg.lfc_2)[0]})
    dep = pd.DataFrame(dep)
    dep.to_csv(OUT / "raw_concordance_by_depth.csv", index=False)

    # ---- per kinase ------------------------------------------------------------
    pk = []
    for (pa, k), g in d.groupby(["paabd", "kinase"]):
        r, rho = corr(g.clfc_1, g.clfc_2)
        pk.append({"paabd": pa, "kinase": k, "n_pairs": len(g), "pearson_clfc": r,
                   "spearman_clfc": rho, "median_pre_1": g.pre_1.median(),
                   "median_pre_2": g.pre_2.median()})
    pk = pd.DataFrame(pk)
    pk.to_csv(OUT / "raw_concordance_per_kinase.csv", index=False)

    # ---- scatter grid ---------------------------------------------------------
    fig, axes = plt.subplots(len(PAABD_ORDER), len(SCATTER_QUANTS), figsize=(7.2, 9.3))
    for i, pa in enumerate(PAABD_ORDER):
        g = d[d.paabd == pa]
        for j, (q, lab) in enumerate(SCATTER_QUANTS):
            ax = axes[i, j]
            x, y = g[f"{q}_1"].values, g[f"{q}_2"].values
            lo, hi = np.percentile(np.r_[x, y], [0.1, 99.9])
            ax.hexbin(x, y, gridsize=55, bins="log", cmap="Greys", mincnt=1,
                      extent=(lo, hi, lo, hi), linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=PAABD_COLORS[pa], lw=0.9)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            s = summ[(summ.paabd == pa) & (summ.quantity == q)].iloc[0]
            ax.text(0.04, 0.96, f"r = {s.pearson:.3f}\nρ = {s.spearman:.3f}",
                    transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
            if i == 0:
                ax.set_title(lab)
            if i == len(PAABD_ORDER) - 1:
                ax.set_xlabel("rep 1 (sequencing batch 1)")
            if j == 0:
                ax.set_ylabel(f"{PAABD_LABELS[pa]}\nrep 2 (batch 2)")
    fig.suptitle("Sequencing replicates, raw counts per (kinase, substrate) pair", fontsize=10, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_raw_replicate_scatter")

    # ---- depth + per-kinase -----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.3]})
    for pa in PAABD_ORDER:
        g = dep[dep.paabd == pa]
        x = [edges.index(lo) for lo in g.min_pre_lo]
        ax1.plot(x, g.pearson_clfc, "-o", ms=3.5, color=PAABD_COLORS[pa], label=PAABD_LABELS[pa])
    ax1.set_xticks(range(len(edges) - 1))
    ax1.set_xticklabels([f"{edges[i]:g}–{edges[i + 1]:g}" if np.isfinite(edges[i + 1])
                         else f"≥{edges[i]:g}" for i in range(len(edges) - 1)], rotation=30, ha="right")
    ax1.set_xlabel("input reads per pair, min(pre_1, pre_2)")
    ax1.set_ylabel("Pearson r, kinase-centred\nlog enrichment, rep 1 vs rep 2")
    ax1.set_title("Agreement rises with read depth")
    ax1.legend(frameon=False)
    ax1.set_ylim(0, 1)
    for i, pa in enumerate(PAABD_ORDER):
        g = pk[pk.paabd == pa]
        xs = i + JITTER.uniform(-0.18, 0.18, len(g))
        ax2.scatter(xs, g.pearson_clfc, s=12, color=PAABD_COLORS[pa], zorder=3)
        ax2.hlines(g.pearson_clfc.median(), i - 0.3, i + 0.3, color=INK, lw=1.2)
    ax2.set_xticks(range(len(PAABD_ORDER)))
    ax2.set_xticklabels([PAABD_LABELS[p] for p in PAABD_ORDER])
    ax2.set_ylabel("Pearson r per kinase")
    ax2.set_title("Per-kinase replicate agreement (bar: median)")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    save_fig(fig, "fig_raw_concordance_by_depth")

    print(summ.round(3).to_string())
    print(dep.round(3).to_string())


if __name__ == "__main__":
    main()
