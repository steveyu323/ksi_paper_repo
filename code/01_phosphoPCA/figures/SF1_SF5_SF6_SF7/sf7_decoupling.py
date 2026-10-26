#!/usr/bin/env python
"""
Supplementary Figure S7 (SuppFig_decoupling): binding strength is decoupled
from abundance across phospho-amino-acid-binding domains (PAABDs).

Panel provenance (code lifted from 6_inspect_decoupled_binding.ipynb):
  a, b  cell 11  -> supp_plot_outputs/fig1_activity_abundance.pdf
                    (per kinase x PAABD KDE of binding_strength, gaussian_kde
                    bw=0.4 on [-2, 2]; tick = median kinase_global)
  c     cell 12  -> left heatmap of supp_plot_outputs/fig2_abundance_consistency.pdf
                    (median kinase_global per S/T kinase x PAABD)

The cell-11 grid of histograms is redrawn as overlaid ridgelines (KDE only),
each row scaled to its own tallest trace, matching the manuscript layout.

Inputs (notebook 6 cells 1-4)
  ksi_analysis_2025/decoupling_out/251125_pymc_{sh2,fha,1433,ww}.csv
  ksi_analysis_2025/decoupling_out/251125_pymc_binding_stats_{...}.csv

Usage:  python sf7_decoupling.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # ksi_analysis_2025
DECOUPLING_DIR = os.path.join(ROOT, "decoupling_out")
OUT_DIR = os.path.join(HERE, "out")
SRC_DIR = os.path.join(OUT_DIR, "source_data")

ST_PAABDS = ["fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#0072B2", "fha": "#D55E00", "1433": "#009E73", "ww": "#E69F00"}
PAABD_LS = {"sh2": "-", "fha": "-", "1433": (0, (5, 2)), "ww": (0, (1.5, 1))}
XRANGE = (-2, 2)

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


def load_specificity_df():
    """Notebook 6 cells 1-2: posterior effects merged with binding stats."""
    final_ls = []
    for paabd in ["sh2", "fha", "1433", "ww"]:
        effects = pd.read_csv(f"{DECOUPLING_DIR}/251125_pymc_{paabd}.csv").iloc[:, 1:]
        stats = pd.read_csv(f"{DECOUPLING_DIR}/251125_pymc_binding_stats_{paabd}.csv").iloc[:, 1:]
        effects["paabd"] = paabd
        final = pd.merge(effects, stats, on=["substrate", "kinase"])
        final_ls.append(final.drop(["pair_total_effect"], axis=1))
    return pd.concat(final_ls)


def safe_kde(vals, xrange=XRANGE, bw=0.4, n=400):
    vals = np.asarray(pd.Series(vals).dropna())
    if len(vals) < 3 or np.unique(vals).size < 3:
        return None, None
    try:
        xg = np.linspace(*xrange, n)
        return xg, gaussian_kde(vals, bw_method=bw)(xg)
    except np.linalg.LinAlgError:
        return None, None


def pretty(kinase):
    name = kinase.replace("_HUMAN", "").replace("_", " ").strip()
    return "JAK2" if name == "JAK2 1" else name


def ridgeline(ax, specificity_df, kinases, paabds, title, row_gap=1.1):
    """One row per kinase; KDE per PAABD scaled to the row's tallest trace."""
    records = []
    for r, kinase in enumerate(kinases):
        base = (len(kinases) - 1 - r) * row_gap
        df_k = specificity_df.loc[specificity_df["kinase"] == kinase]
        traces = {}
        for paabd in paabds:
            df_pk = df_k.loc[df_k["paabd"] == paabd]
            if df_pk.empty:
                continue
            xg, yg = safe_kde(df_pk["binding_strength"])
            traces[paabd] = (xg, yg, df_pk["kinase_global"].dropna().median())
        ymax = max((t[1].max() for t in traces.values() if t[1] is not None), default=1)
        ax.axhline(base, color="#CCCCCC", lw=0.5, zorder=0)
        for paabd, (xg, yg, med) in traces.items():
            color = PAABD_COLORS[paabd]
            if xg is not None:
                ax.plot(xg, base + yg / ymax, color=color, ls=PAABD_LS[paabd], lw=1.1)
            ax.plot([med, med], [base - 0.3, base + 0.3], color=color, lw=1.2)
            records.append({"kinase": kinase, "paabd": paabd, "median_abundance": med})
    ax.axvline(0, color="#888888", ls=":", lw=0.6, zorder=0)
    ax.set_yticks([(len(kinases) - 1 - r) * row_gap for r in range(len(kinases))])
    ax.set_yticklabels([pretty(k) for k in kinases])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(*XRANGE)
    ax.set_ylim(-0.4, (len(kinases) - 1) * row_gap + 1.15)
    ax.set_xlabel("Binding strength")
    ax.set_title(title)
    return records


def main():
    os.makedirs(SRC_DIR, exist_ok=True)
    specificity_df = load_specificity_df()

    kinase_paabds = (specificity_df.groupby("kinase")["paabd"]
                     .apply(lambda x: set(x.unique())).to_dict())
    st_kinases = sorted(k for k, ps in kinase_paabds.items() if "fha" in ps)
    y_kinases = sorted(k for k, ps in kinase_paabds.items() if "sh2" in ps)

    # panel c: median kinase_global per S/T kinase x PAABD (cell 12)
    pivot = (specificity_df.loc[specificity_df["kinase"].isin(st_kinases)
                                & specificity_df["paabd"].isin(ST_PAABDS)]
             .groupby(["kinase", "paabd"])["kinase_global"].median()
             .unstack()[ST_PAABDS].reindex(st_kinases))

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 4.6),
                             gridspec_kw=dict(width_ratios=[1.25, 0.75, 1.0], wspace=0.45))

    rec_a = ridgeline(axes[0], specificity_df, st_kinases, ST_PAABDS, "Ser/Thr kinases")
    rec_b = ridgeline(axes[1], specificity_df, y_kinases, ["sh2"], "Tyr kinases")

    leg_a = [Line2D([0], [0], color=PAABD_COLORS[p], ls=PAABD_LS[p], lw=1.1,
                    label=PAABD_LABELS[p]) for p in ST_PAABDS]
    leg_a.append(Line2D([0], [0], color="#222222", lw=1.1, label="Median abundance"))
    axes[0].legend(handles=leg_a, ncol=2, frameon=False, fontsize=7,
                   loc="upper center", bbox_to_anchor=(0.5, -0.13))
    axes[1].legend(handles=[Line2D([0], [0], color=PAABD_COLORS["sh2"], lw=1.1, label="SH2")],
                   frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13))

    ax = axes[2]
    vabs = np.nanmax(np.abs(pivot.values))
    norm = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(val) > vabs * 0.55 else "#222222")
    ax.set_xticks(range(len(ST_PAABDS)))
    ax.set_xticklabels([PAABD_LABELS[p] for p in ST_PAABDS])
    for tick, p in zip(ax.get_xticklabels(), ST_PAABDS):
        tick.set_color(PAABD_COLORS[p])
    ax.set_yticks(range(len(st_kinases)))
    ax.set_yticklabels([pretty(k) for k in st_kinases], fontsize=7)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Median abundance")
    cbar = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.05)
    cbar.set_label("Log fold change", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    for ax, letter, x in zip(axes, "abc", (0.0, 0.0, -0.45)):
        ax.text(x, 1.02, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="right")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"SuppFig_decoupling.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── source data + summary ─────────────────────────────────────────────────
    specificity_df[["kinase", "substrate", "paabd", "binding_strength", "kinase_global"]].to_csv(
        os.path.join(SRC_DIR, "sf7ab_binding_strength.csv"), index=False)
    pd.DataFrame(rec_a + rec_b).to_csv(
        os.path.join(SRC_DIR, "sf7ab_median_abundance.csv"), index=False)
    pivot.rename(columns=PAABD_LABELS).to_csv(os.path.join(SRC_DIR, "sf7c_median_abundance.csv"))

    print(f"S/T kinases: {len(st_kinases)}, Y kinases: {len(y_kinases)}")
    print(pivot.rename(columns=PAABD_LABELS).round(2).to_string())
    print("wrote", os.path.join(OUT_DIR, "SuppFig_decoupling.pdf"))


if __name__ == "__main__":
    main()
