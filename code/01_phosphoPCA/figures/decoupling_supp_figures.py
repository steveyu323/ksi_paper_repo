#!/usr/bin/env python
"""
Supplementary figures for the Bayesian abundance/activity decoupling step.

The decoupling model is fit in 5_enrichment_analysis.ipynb (hierarchical
negative-binomial in PyMC) and inspected in 6_inspect_decoupled_binding.ipynb.
This script regenerates, from the saved posterior summaries:

  1. per-kinase x pAABD pre-decoupled and post-decoupled score histograms
     (grid figures + one-panel-per-pair multipage PDF + statistics table)
  2. a schematic of the motivation and the model
  3. two worked examples where the abundance baseline sets the read-out

Units are the model's natural-log enrichment units, matching
supp_plot_outputs/fig1_activity_abundance.pdf and 251130_ksi_inference_scores.pkl.
Divide by ln 2 for log2.

Figure 05 panel C of the manuscript is figS_hit_selection_binding_lfsr, and
SuppFig_decoupling is figS_decoupling_schematic; fig05_math.py calls into this
module for those panels rather than duplicating them.

Usage:  python decoupling_supp_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

# Paths and thresholds, from config.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

SCORES_PKL = config.INFERENCE_SCORES_PKL     # written by stage 06
OUT_DIR = config.FIG_DIR / "decoupling"

# Hit calls, as in 6_inspect_decoupled_binding.ipynb / stage 06.
LFSR_THRESH = config.LFSR_THRESH
BINDING_THRESH = config.BINDING_THRESH

# ── Style, matching the existing supplementary figures ────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Liberation Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,     # TrueType, so PDF text stays editable
    "svg.fonttype": "none",  # emit <text> elements, not outlined paths
})

PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}

PRE_COLOR = "#9E9E9E"      # pre-decoupled (naive enrichment)
ABUND_COLOR = "#222222"    # fitted abundance / global-activity baseline
HIT_COLOR = "#C1272D"

B_SK = "bₛₖ"          # b_sk
G_K = "gₖ"            # g_k
D_KR = "δₖᵣ"           # delta_kr
BASE_SYM = f"{G_K} + δₖ"   # g_k + mean_r(delta_kr)

XLIM = (-2.6, 3.0)

# ── Worked examples ───────────────────────────────────────────────────────────
# Example 1 / schematic panels D-F use one pAABD and three kinases with
# *moderate* abundance baselines, chosen so that both failure directions are
# visible: KGP1 sits below zero and loses real hits, while PHKG1 and PLK1 sit
# at or above zero and collect naive-positive substrates that are not binders.
EX1_PAABD = "fha"
EX1_KINASES = ["KGP1_HUMAN", "PHKG1_HUMAN", "PLK1_HUMAN"]
EX1_COLORS = {"KGP1_HUMAN": "#C1272D", "PHKG1_HUMAN": "#6A8D3A",
              "PLK1_HUMAN": "#1B5E7B"}

# Example 2: substrates where a moderate baseline flips the *sign* of the score.
EX2_CASES = [("fha", "TFINSQGTGYRQVSD"), ("ww", "RSERGDGSDVKEKTP")]
EX2_ANNOT = {"TFINSQGTGYRQVSD": "UniProt Q12518, Thr395",
             "RSERGDGSDVKEKTP": "UniProt O60829, Ser79"}


# Vector formats first; SVG keeps text as <text> and curves as <path>, so every
# label can be retyped and every curve reshaped in Illustrator.
FORMATS = ("pdf", "svg", "png")


def save_fig(fig, fname, *, tight=False, formats=FORMATS):
    """Write one figure to every output format and report what was written."""
    kw = dict(dpi=300)
    if tight:
        kw["bbox_inches"] = "tight"
    for ext in formats:
        fig.savefig(f"{OUT_DIR}/{fname}.{ext}", **kw)
    plt.close(fig)
    print(f"  wrote {fname}.{'/.'.join(formats)}")


# ═════════════════════════════════════════════════════════════════════════════
# 1. Score table: reconstruct the pre-decoupled read-out
# ═════════════════════════════════════════════════════════════════════════════
def load_scores(path: str = SCORES_PKL) -> pd.DataFrame:
    """Add the pre-decoupled (naive) enrichment score to the posterior summaries.

    The model writes, for observation (substrate s, kinase k, replicate r):

        log mu = log(sf_mtx_r) + log(pre_ksr / sf_pre_r + 1)   <- offset
                 + g_k + delta_kr                              <- abundance / global activity
                 + b_sk                                        <- kinase-centered binding

    so the naive per-replicate log enrichment is log(mtx) - offset, and it
    carries the abundance term. `kinase_global` in the saved table is
    g_k + mean_r(delta_kr); `binding_strength` is the posterior mean of b_sk.
    Averaging the two replicates gives the pre-decoupled score plotted here.
    """
    d = pd.read_pickle(path)

    out = []
    for paabd, g in d.groupby("paabd", sort=False):
        # size factors are computed over all kinases/substrates of the pAABD,
        # exactly as prepare_long_table_for_pymc() does
        tot_pre = np.array([g["pre_1"].sum(), g["pre_2"].sum()], dtype=float)
        tot_mtx = np.array([g["mtx_1"].sum(), g["mtx_2"].sum()], dtype=float)
        sf_pre = tot_pre / np.median(tot_pre)
        sf_mtx = tot_mtx / np.median(tot_mtx)

        g = g.copy()
        for r, (pre_c, mtx_c) in enumerate(zip(["pre_1", "pre_2"], ["mtx_1", "mtx_2"])):
            log_offset = np.log(sf_mtx[r]) + np.log(g[pre_c] / sf_pre[r] + 1.0)
            g[f"_naive_{r}"] = np.log(g[mtx_c] + 0.5) - log_offset
        g["naive_score"] = g[["_naive_0", "_naive_1"]].mean(axis=1)
        out.append(g.drop(columns=["_naive_0", "_naive_1"]))

    d = pd.concat(out, ignore_index=True)
    d["hit"] = (d["lfsr"] <= LFSR_THRESH) & (d["binding_strength"] >= BINDING_THRESH)
    return d


def pair_stats(d: pd.DataFrame) -> pd.DataFrame:
    """One row per kinase x pAABD pair."""
    rows = []
    for (k, p), g in d.groupby(["kinase", "paabd"]):
        n_hit = int(g["hit"].sum())
        rows.append({
            "kinase": k,
            "paabd": p,
            "paabd_label": PAABD_LABELS[p],
            "n_substrates": len(g),
            "abundance_baseline_ln": g["kinase_global"].median(),
            "abundance_baseline_log2": g["kinase_global"].median() / np.log(2),
            "naive_median_ln": g["naive_score"].median(),
            "naive_sd_ln": g["naive_score"].std(),
            "naive_max_ln": g["naive_score"].max(),
            "frac_naive_above_0": (g["naive_score"] > 0).mean(),
            "n_naive_above_0": int((g["naive_score"] > 0).sum()),
            "binding_median_ln": g["binding_strength"].median(),
            "binding_sd_ln": g["binding_strength"].std(),
            "binding_max_ln": g["binding_strength"].max(),
            "n_decoupled_hits": n_hit,
            "n_hits_naive_below_0": int((g["hit"] & (g["naive_score"] < 0)).sum()),
            "frac_hits_naive_below_0": (
                (g["hit"] & (g["naive_score"] < 0)).sum() / n_hit if n_hit else np.nan
            ),
        })
    s = pd.DataFrame(rows)
    return s.sort_values(["paabd", "abundance_baseline_ln"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════════
def safe_kde(vals, xrange=XLIM, bw=0.35, n=400):
    vals = np.asarray(pd.Series(vals).dropna())
    if len(vals) < 3 or np.unique(vals).size < 3:
        return None, None
    try:
        xg = np.linspace(*xrange, n)
        return xg, gaussian_kde(vals, bw_method=bw)(xg)
    except np.linalg.LinAlgError:
        return None, None


def _free_corner(pre, post, base):
    """Pick the top corner of the panel that the drawn curves leave emptiest."""
    lo_edge = XLIM[0] + 0.34 * (XLIM[1] - XLIM[0])
    hi_edge = XLIM[0] + 0.66 * (XLIM[1] - XLIM[0])
    left = right = 0.0
    for vals, w in ((pre, 1.0), (post, 1.0)):
        v = np.asarray(pd.Series(vals).dropna())
        if not len(v):
            continue
        left += w * (v < lo_edge).mean()
        right += w * (v > hi_edge).mean()
    # the abundance baseline marker also occupies space
    left += 0.35 * (base < lo_edge)
    right += 0.35 * (base > hi_edge)
    return "left" if right >= left else "right"


def draw_pair_panel(ax, g, paabd, *, annotate=True, bins=70, fontsize=5.6,
                    headroom=1.34):
    """Overlay the pre- and post-decoupling distributions for one pair."""
    color = PAABD_COLORS[paabd]
    pre = g["naive_score"].dropna()
    post = g["binding_strength"].dropna()
    base = g["kinase_global"].median()

    ax.hist(pre, bins=bins, range=XLIM, color=PRE_COLOR, alpha=0.55,
            density=True, linewidth=0, zorder=1)
    ax.hist(post, bins=bins, range=XLIM, color=color, alpha=0.45,
            density=True, linewidth=0, zorder=2)

    for vals, c in ((pre, "#5A5A5A"), (post, color)):
        xg, yg = safe_kde(vals)
        if xg is not None:
            ax.plot(xg, yg, color=c, linewidth=1.3, zorder=3)

    ax.axvline(0.0, color="#BBBBBB", linewidth=0.8, zorder=0)
    ax.axvline(base, color=ABUND_COLOR, linewidth=1.2, linestyle="--",
               alpha=0.9, zorder=4)

    ax.set_xlim(*XLIM)
    ax.set_ylim(0, ax.get_ylim()[1] * headroom)

    if annotate:
        n_hit = int(g["hit"].sum())
        n_hid = int((g["hit"] & (g["naive_score"] < 0)).sum())
        side = _free_corner(pre, post, base)
        ax.text(0.015 if side == "left" else 0.985, 0.97,
                f"n={len(g)}\nbaseline={base:+.2f}\nhits={n_hit} ({n_hid} <0 naive)",
                transform=ax.transAxes, ha=side, va="top", fontsize=fontsize,
                color="#333333", linespacing=1.25, zorder=7)
    return base


def legend_handles():
    return [
        mpatches.Patch(facecolor=PRE_COLOR, alpha=0.55,
                       label="Pre-decoupling: naive log enrichment"),
        mpatches.Patch(facecolor="#777777", alpha=0.45,
                       label=f"Post-decoupling: binding strength {B_SK}"
                             " (colored by pAABD)"),
        Line2D([0], [0], color=ABUND_COLOR, lw=1.2, ls="--",
               label=f"Fitted abundance baseline, median {BASE_SYM}"),
        Line2D([0], [0], color="#BBBBBB", lw=0.8, label="zero"),
    ]



def draw_error_direction_bars(ax, d, paabd, order, baselines, kcol, title):
    """Stacked bars showing the two error directions of a naive cutoff at zero.

    Above the axis: substrates a naive cutoff keeps, split into those the
    decoupled model also calls and those it does not. Below the axis: confident
    binders whose naive score is negative, which the cutoff would discard.
    """
    d_p = d[d["paabd"] == paabd]
    x = np.arange(len(order))
    both, naive_only, dec_only = [], [], []
    for k in order:
        g = d_p[d_p["kinase"] == k]
        both.append(int((g["hit"] & (g["naive_score"] > 0)).sum()))
        naive_only.append(int(((g["naive_score"] > 0) & ~g["hit"]).sum()))
        dec_only.append(int((g["hit"] & (g["naive_score"] < 0)).sum()))
    both = np.array(both)
    naive_only = np.array(naive_only)
    dec_only = np.array(dec_only)

    ax.bar(x, both, width=0.6, color="#7F7F7F",
           label="agree: decoupled hit and naive > 0")
    ax.bar(x, naive_only, width=0.6, bottom=both, color="#E0C26B",
           label="naive > 0 but not a binder")
    ax.bar(x, -dec_only, width=0.6, color=HIT_COLOR,
           label="binder the naive cutoff misses")
    ax.axhline(0, color="#333333", lw=1.1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k.replace('_HUMAN', '')}\n({b:+.2f})"
                        for k, b in zip(order, baselines)], fontsize=8)
    for t, k in zip(ax.get_xticklabels(), order):
        t.set_color(kcol[k])
        t.set_fontweight("bold")
    ax.set_ylabel("substrates", fontsize=9)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")

    # counts sit inside their own segment, leaving the top band for the legend
    top = max((both + naive_only).max(), 1)
    ax.set_ylim(-max(dec_only.max(), 1) * 2.1, top * 1.62)
    for xi, b, no, dn in zip(x, both, naive_only, dec_only):
        if b > top * 0.08:
            ax.text(xi, b / 2, f"{b}", ha="center", va="center", fontsize=7.4,
                    color="white", fontweight="bold")
        if no > top * 0.08:
            ax.text(xi, b + no / 2, f"+{no}\nspurious", ha="center",
                    va="center", fontsize=7.4, color="#5C4708",
                    fontweight="bold", linespacing=1.3)
        if dn:
            ax.text(xi, -dn - top * 0.055, f"\u2212{dn} missed", ha="center",
                    va="top", fontsize=7.4, color=HIT_COLOR, fontweight="bold")
    ax.legend(fontsize=7.4, frameon=False, loc="upper left", borderpad=0.2,
              labelspacing=0.35)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Grid figures: rows = kinase, cols = pAABD
# ═════════════════════════════════════════════════════════════════════════════
def fig_grid(d: pd.DataFrame, kinases, paabds, title, fname, *, row_h=1.25):
    """Rows = kinase, cols = pAABD (the layout used by fig1_activity_abundance)."""
    n = len(kinases)
    fig_h = row_h * n + 2.4
    fig_w = 3.4 * len(paabds) + 1.2   # extra width for the long kinase labels
    fig, axes = plt.subplots(
        n, len(paabds),
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw=dict(hspace=0.18, wspace=0.10),
    )

    membership = d.groupby("kinase")["paabd"].apply(lambda x: set(x)).to_dict()

    for ri, kinase in enumerate(kinases):
        dk = d[d["kinase"] == kinase]
        for ci, paabd in enumerate(paabds):
            ax = axes[ri][ci]
            if paabd not in membership.get(kinase, set()):
                ax.set_visible(False)
                continue

            g = dk[dk["paabd"] == paabd]
            draw_pair_panel(ax, g, paabd)

            if ri == 0:
                ax.set_title(PAABD_LABELS[paabd], fontsize=12, fontweight="bold",
                             color=PAABD_COLORS[paabd], pad=6)
            if ci == 0:
                ax.set_ylabel(kinase, fontsize=9, rotation=0, ha="right",
                              va="center", labelpad=6)
            else:
                ax.set_yticklabels([])

            if ri == n - 1:
                ax.set_xlabel("log enrichment (natural log)", fontsize=9)
                ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            else:
                ax.set_xticklabels([])

    # margins in figure fractions, so tall figures neither gain a huge top gap
    # nor let the shared legend land on top of the bottom row's axis labels
    fig.subplots_adjust(top=1 - 0.85 / fig_h, bottom=1.45 / fig_h,
                        left=1.55 / fig_w, right=1 - 0.15 / fig_w)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=2,
               fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5 + 0.7 / fig_w, 0.10 / fig_h))
    fig.suptitle(title, fontsize=12.5, fontweight="bold",
                 y=1 - 0.26 / fig_h)

    save_fig(fig, fname)
    print(f"    ({n} kinases x {len(paabds)} pAABDs)")


def fig_pair_panels(d, pairs, title, fname, *, ncol=3, nrow=4, multipage=True,
                    figsize=(11, 9.5)):
    """One panel per (kinase, pAABD) pair, laid out ncol x nrow per page."""
    pairs = list(pairs)
    per_page = ncol * nrow
    n_pages = int(np.ceil(len(pairs) / per_page)) if multipage else 1

    def draw_page(fig, chunk, page):
        axes = fig.subplots(nrow, ncol,
                            gridspec_kw=dict(hspace=0.55, wspace=0.22))
        axf = np.atleast_1d(axes).ravel()
        for ax, (kinase, paabd) in zip(axf, chunk):
            g = d[(d["kinase"] == kinase) & (d["paabd"] == paabd)]
            # per-pair stats go in the title, so the hit rug can own the top
            # band of the axes without colliding with text
            base = draw_pair_panel(ax, g, paabd, annotate=False, bins=60,
                                   headroom=1.20)

            hits = g.loc[g["hit"], "binding_strength"]
            if len(hits):
                y = ax.get_ylim()[1]
                ax.plot(np.clip(hits, *XLIM), np.full(len(hits), y * 0.965),
                        marker="|", ls="none", ms=4, mew=0.7,
                        color=HIT_COLOR, alpha=0.75, zorder=6)

            n_hid = int((g["hit"] & (g["naive_score"] < 0)).sum())
            ax.set_title(f"{kinase}  ·  {PAABD_LABELS[paabd]}",
                         fontsize=9.5, fontweight="bold", pad=14)
            ax.text(0.5, 1.015,
                    f"n={len(g)}   ·   baseline {base:+.2f}   ·   "
                    f"{len(hits)} hits ({n_hid} below 0 naively)",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=6.8, color="#444444")
            ax.set_xlabel("log enrichment (natural log)", fontsize=7.5)
            ax.set_ylabel("density", fontsize=7.5)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
        for ax in axf[len(chunk):]:
            ax.set_visible(False)

        handles = legend_handles() + [
            Line2D([0], [0], color=HIT_COLOR, marker="|", ls="none", ms=6,
                   label=f"decoupled hit (lfsr ≤ {LFSR_THRESH}, "
                         f"{B_SK} ≥ {BINDING_THRESH})")
        ]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, 0.004))
        suffix = f"  ({page + 1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(title + suffix, fontsize=11.5, fontweight="bold", y=0.988)
        fig.subplots_adjust(top=0.925, bottom=0.085, left=0.065, right=0.985)

    if multipage:
        # one combined multipage PDF, plus a per-page SVG/PNG because SVG has
        # no multipage form and the pages are meant to be editable separately
        pagedir = f"{OUT_DIR}/{fname}_pages"
        os.makedirs(pagedir, exist_ok=True)
        with PdfPages(f"{OUT_DIR}/{fname}.pdf") as pdf:
            for page in range(n_pages):
                fig = plt.figure(figsize=figsize)
                draw_page(fig, pairs[page * per_page:(page + 1) * per_page], page)
                pdf.savefig(fig, dpi=300)
                for ext in ("svg", "png"):
                    fig.savefig(f"{pagedir}/{fname}_p{page + 1}.{ext}", dpi=300)
                plt.close(fig)
        print(f"  wrote {fname}.pdf  ({len(pairs)} pairs over {n_pages} pages)")
        print(f"  wrote {fname}_pages/{fname}_p1..{n_pages}.svg/.png")
    else:
        fig = plt.figure(figsize=figsize)
        draw_page(fig, pairs, 0)
        save_fig(fig, fname)
        print(f"    ({len(pairs)} pairs)")


# ═════════════════════════════════════════════════════════════════════════════
# 3. One panel per kinase x pAABD pair, multipage
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# 4. Schematic: motivation + model
# ═════════════════════════════════════════════════════════════════════════════
def fig_schematic(d: pd.DataFrame, stats: pd.DataFrame, fname):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05],
                          hspace=0.42, wspace=0.30,
                          left=0.055, right=0.975, top=0.90, bottom=0.07)

    # ── A. the problem: one read-out, two mixed causes ────────────────────────
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
    axA.set_title("A  One read-out, two causes", loc="left",
                  fontsize=11, fontweight="bold")

    def box(ax, x, y, w, h, text, fc, ec, fs=8, weight="normal", tc="#111111"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.25",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, color=tc, zorder=3,
                linespacing=1.35)

    box(axA, 0.4, 7.4, 4.2, 1.9,
        f"kinase abundance\n& global activity\n{G_K}", "#E8E8E8", "#666666",
        fs=8.5, weight="bold")
    box(axA, 5.4, 7.4, 4.2, 1.9,
        f"sequence-specific\nbinding / phosphorylation\n{B_SK}",
        "#DCEBF3", "#2E86AB", fs=8.5, weight="bold")
    box(axA, 2.4, 3.9, 5.2, 1.6,
        "observed MTX-vs-pre\nenrichment of barcode (s, k)",
        "#F5E6E6", "#C1272D", fs=8.5, weight="bold")
    box(axA, 2.4, 0.9, 5.2, 1.5,
        f"one library-wide threshold\n⇒ hits track {G_K}, not {B_SK}",
        "#FFF6DA", "#C08A00", fs=8.5)

    for x0 in (2.5, 7.5):
        axA.annotate("", xy=(5.0 + 0.35 * np.sign(x0 - 5), 5.6),
                     xytext=(x0, 7.3),
                     arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#555555",
                                     shrinkA=2, shrinkB=2))
    axA.annotate("", xy=(5.0, 2.5), xytext=(5.0, 3.8),
                 arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#C1272D"))
    axA.text(5.0, 6.55, "confounded", ha="center", va="center", fontsize=8,
             style="italic", color="#555555")

    # ── B. how far apart the baselines actually are ───────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    s = stats.sort_values("abundance_baseline_ln")
    lab = [f"{k.replace('_HUMAN', '')} · {PAABD_LABELS[p]}"
           for k, p in zip(s["kinase"], s["paabd"])]
    # the three example kinases are drawn in their own colors so the reader can
    # see that panels D-F use a moderate, not an extreme, slice of this range
    is_ex = [(k in EX1_KINASES) and (p == EX1_PAABD)
             for k, p in zip(s["kinase"], s["paabd"])]
    cols = [EX1_COLORS[k] if e else PAABD_COLORS[p]
            for k, p, e in zip(s["kinase"], s["paabd"], is_ex)]
    axB.barh(np.arange(len(s)), s["abundance_baseline_ln"], color=cols,
             height=0.82, linewidth=0,
             alpha=None)
    axB.axvline(0, color="#333333", lw=0.9)
    ex_pos = [i for i, e in enumerate(is_ex) if e]
    axB.set_yticks(ex_pos)
    axB.set_yticklabels([lab[i].split(" · ")[0] for i in ex_pos], fontsize=6.8)
    for t, i in zip(axB.get_yticklabels(), ex_pos):
        t.set_color(EX1_COLORS[s["kinase"].iloc[i]])
        t.set_fontweight("bold")
    axB.set_ylim(-1, len(s))
    axB.annotate(f"panels D–F\n({PAABD_LABELS[EX1_PAABD]})",
                 xy=(0.62, 0.30), xycoords="axes fraction",
                 fontsize=6.8, color="#555555", style="italic",
                 ha="left", va="center", linespacing=1.4)
    axB.set_xlabel(f"fitted abundance baseline\n{BASE_SYM} (natural log)",
                   fontsize=8.5)
    rng = s["abundance_baseline_ln"].max() - s["abundance_baseline_ln"].min()
    axB.set_title(f"B  Baselines span {rng:.1f} log units\n"
                  f"({rng / np.log(2):.1f} log₂) across "
                  f"{len(s)} kinase × pAABD pairs",
                  loc="left", fontsize=11, fontweight="bold")
    axB.legend(handles=[mpatches.Patch(facecolor=PAABD_COLORS[p],
                                       label=PAABD_LABELS[p])
                        for p in PAABD_ORDER],
               fontsize=7, frameon=False, loc="lower right", ncol=2)

    # ── C. the model ──────────────────────────────────────────────────────────
    axC = fig.add_subplot(gs[0, 2])
    axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
    axC.set_title("C  Hierarchical NB decoupling model", loc="left",
                  fontsize=11, fontweight="bold")

    axC.text(0.0, 9.5, "yₛₖᵣ  ~  NegBinom(μₛₖᵣ, α)",
             fontsize=10.5, ha="left", va="center")
    axC.text(0.0, 8.1, "log μₛₖᵣ  =", fontsize=10.5, ha="left",
             va="center")

    # the four additive terms, each in its own tinted box with a caption below
    terms = [
        ("offsetₛₖᵣ", "input size\n& depth", "#E0E0E0", "#666666"),
        (G_K, "kinase\nabundance", "#F2DFDF", "#C1272D"),
        (D_KR, "replicate\nblock", "#EFE6D0", "#C08A00"),
        (B_SK, "substrate\nbinding", "#DCEBF3", "#2E86AB"),
    ]
    x, w, gap = 2.52, 1.62, 0.34
    for i, (sym, cap, fc, ec) in enumerate(terms):
        axC.add_patch(mpatches.FancyBboxPatch(
            (x, 7.45), w, 1.3,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=fc, edgecolor=ec, linewidth=1.3, zorder=2))
        axC.text(x + w / 2, 8.1, sym, ha="center", va="center",
                 fontsize=8.6 if i == 0 else 10,
                 fontweight="bold", color=ec, zorder=3)
        axC.text(x + w / 2, 6.75, cap, ha="center", va="center", fontsize=7.2,
                 color=ec, linespacing=1.25, zorder=3)
        if i < len(terms) - 1:
            axC.text(x + w + gap / 2, 8.1, "+", ha="center", va="center",
                     fontsize=11, color="#444444")
        x += w + gap

    axC.text(
        0.0, 5.55,
        "Priors — partial pooling shrinks thinly sampled pairs:\n"
        f"   {G_K} ~ N(0, σ_g),   {D_KR} ~ N(0, σ_δ),\n"
        f"   b_raw ~ N(0, σ_b),   α ~ HalfNormal(1)",
        fontsize=8.2, ha="left", va="top", linespacing=1.7)
    axC.text(
        0.0, 3.35,
        "Identifiability — binding is centered within each kinase:\n"
        f"   {B_SK} = b_raw − meanₛ(b_raw),   so Σₛ {B_SK} = 0",
        fontsize=8.2, ha="left", va="top", linespacing=1.7)
    axC.text(
        0.0, 1.85,
        "Anything that moves a kinase's whole substrate profile is\n"
        f"absorbed by {G_K} + {D_KR}; {B_SK} retains only substrate-to-\n"
        f"substrate contrast. Hits are called on the posterior of {B_SK}:\n"
        f"   lfsr ≤ {LFSR_THRESH}  and  {B_SK} ≥ {BINDING_THRESH}.",
        fontsize=8.2, ha="left", va="top", linespacing=1.7)

    # ── D/E. before and after, for the three example kinases only ────────────
    trio = stats[(stats["paabd"] == EX1_PAABD)
                 & stats["kinase"].isin(EX1_KINASES)] \
        .sort_values("abundance_baseline_ln")

    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])

    for ax, col, ttl, xlab in (
        (axD, "naive_score",
         "D  Before decoupling: each distribution sits\n"
         "     where its kinase's baseline puts it",
         "naive log enrichment"),
        (axE, "binding_strength",
         "E  After decoupling: comparable,\n     kinase-centered binding scores",
         f"binding strength {B_SK}"),
    ):
        for _, row in trio.iterrows():
            g = d[(d["kinase"] == row["kinase"]) & (d["paabd"] == row["paabd"])]
            xg, yg = safe_kde(g[col], bw=0.4)
            if xg is None:
                continue
            c = EX1_COLORS[row["kinase"]]
            lab = (f"{row['kinase'].replace('_HUMAN', '')}  "
                   f"(baseline {row['abundance_baseline_ln']:+.2f})")
            ax.fill_between(xg, 0, yg / yg.max(), color=c, alpha=0.22, lw=0)
            ax.plot(xg, yg / yg.max(), color=c, lw=1.9, label=lab)
            if col == "naive_score":
                ax.axvline(row["abundance_baseline_ln"], color=c, lw=1.0,
                           ls="--", alpha=0.8)
        ax.axvline(0, color="#333333", lw=1.1, zorder=2)
        ax.set_xlim(-2.2, 2.4)
        ax.set_ylim(0, 1.30)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"{xlab} (natural log)", fontsize=8.5)
        ax.set_title(ttl, loc="left", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, loc="upper left",
                  handlelength=1.3, borderpad=0.2)

    axD.text(0.985, 0.97,
             f"{PAABD_LABELS[EX1_PAABD]} screen\ndashed: each kinase's baseline",
             transform=axD.transAxes, ha="right", va="top", fontsize=6.8,
             color="#555555", style="italic", linespacing=1.4)

    # ── F. the two error directions a single naive cutoff creates ───────────
    axF = fig.add_subplot(gs[1, 2])
    draw_error_direction_bars(
        axF, d, EX1_PAABD, list(trio["kinase"]),
        list(trio["abundance_baseline_ln"]),
        {k: EX1_COLORS[k] for k in trio["kinase"]},
        f"F  {PAABD_LABELS[EX1_PAABD]} screen: one naive cutoff at zero\n"
        "     errs in opposite directions either side of zero")

    fig.suptitle("Bayesian decoupling of substrate binding from kinase abundance "
                 "and global activity",
                 fontsize=13.5, fontweight="bold", y=0.965)

    save_fig(fig, fname, tight=True)


# ═════════════════════════════════════════════════════════════════════════════
# 5. Example 1: three moderate-baseline kinases, two opposite error directions
# ═════════════════════════════════════════════════════════════════════════════
def fig_example_trio(d, stats, paabd, kinases, fname):
    """Three kinases from one pAABD with moderate baselines either side of zero.

    Only these three distributions are drawn — no full-screen background — so
    the shift between the pre- and post-decoupling scores is readable.
    """
    sub = stats[(stats["paabd"] == paabd) & stats["kinase"].isin(kinases)] \
        .sort_values("abundance_baseline_ln")
    order = list(sub["kinase"])
    lab = PAABD_LABELS[paabd]
    xlim = (-2.2, 2.4)

    fig = plt.figure(figsize=(13.4, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.26,
                          left=0.06, right=0.98, top=0.855, bottom=0.085)

    d_p = d[d["paabd"] == paabd]
    kcol = {k: EX1_COLORS.get(k, PAABD_COLORS[paabd]) for k in order}

    # ── A / B: the three distributions, before and after ────────────────────
    for ci, (col, xlab, tag, ttl) in enumerate((
        ("naive_score", "naive log enrichment", "A",
         "Pre-decoupling — three moderate baselines,\n"
         "     three differently-placed distributions"),
        ("binding_strength", f"binding strength {B_SK}", "B",
         "Post-decoupling — the same three,\n"
         "     now centered on a common scale"),
    )):
        ax = fig.add_subplot(gs[0, ci])
        for k in order:
            g = d_p[d_p["kinase"] == k]
            base = g["kinase_global"].median()
            xg, yg = safe_kde(g[col], xrange=xlim, bw=0.4)
            if xg is None:
                continue
            c = kcol[k]
            ax.fill_between(xg, 0, yg / yg.max(), color=c, alpha=0.22, lw=0)
            ax.plot(xg, yg / yg.max(), color=c, lw=2.0,
                    label=f"{k.replace('_HUMAN', '')}  (baseline {base:+.2f})")
            if col == "naive_score":
                ax.axvline(base, color=c, lw=1.0, ls="--", alpha=0.85)
        ax.axvline(0, color="#333333", lw=1.3, zorder=5)
        if col == "binding_strength":
            ax.axvline(BINDING_THRESH, color="#888888", lw=1.0, ls=":", zorder=5)
            ax.text(BINDING_THRESH + 0.04, 1.02,
                    f"{B_SK} = {BINDING_THRESH}", fontsize=6.8,
                    color="#666666", va="bottom")
        else:
            ax.text(0.06, 1.02, "naive cutoff at 0", fontsize=6.8,
                    color="#333333", va="bottom")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.22)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"{xlab} (natural log)", fontsize=9)
        ax.set_title(f"{tag}  {ttl}", loc="left", fontsize=10.5,
                     fontweight="bold")
        ax.legend(fontsize=7.6, frameon=False, loc="upper left",
                  handlelength=1.3, borderpad=0.2)

    # ── C: the two error directions of a single naive cutoff ────────────────
    axC = fig.add_subplot(gs[0, 2])
    draw_error_direction_bars(
        axC, d, paabd, order, list(sub["abundance_baseline_ln"]), kcol,
        "C  A single naive cutoff errs in opposite\n"
        "     directions either side of zero")

    # ── D / E / F: per-kinase pre vs post, with the hit rug ─────────────────
    for ci, k in enumerate(order):
        ax = fig.add_subplot(gs[1, ci])
        g = d_p[d_p["kinase"] == k]
        base = g["kinase_global"].median()
        c = kcol[k]

        ax.hist(g["naive_score"], bins=70, range=xlim, color=PRE_COLOR,
                alpha=0.6, density=True, lw=0, label="pre-decoupling")
        ax.hist(g["binding_strength"], bins=70, range=xlim, color=c,
                alpha=0.42, density=True, lw=0, label="post-decoupling")
        for vals, cc in ((g["naive_score"], "#5A5A5A"),
                         (g["binding_strength"], c)):
            xg, yg = safe_kde(vals, xrange=xlim)
            if xg is not None:
                ax.plot(xg, yg, color=cc, lw=1.4)
        ax.axvline(0, color="#333333", lw=1.2)
        ax.axvline(base, color=ABUND_COLOR, lw=1.2, ls="--",
                   label=f"baseline {base:+.2f}")
        # headroom for the hit rug, which the legend then sits clear of
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)

        hits = g.loc[g["hit"], "binding_strength"]
        if len(hits):
            y = ax.get_ylim()[1]
            ax.plot(np.clip(hits, *xlim), np.full(len(hits), y * 0.985),
                    marker="|", ls="none", ms=6, mew=0.8, color=HIT_COLOR,
                    label=f"{len(hits)} decoupled hits")
        ax.set_xlim(*xlim)
        n_above = int((g["naive_score"] > 0).sum())
        n_hid = int((g["hit"] & (g["naive_score"] < 0)).sum())
        ax.set_title(f"{'DEF'[ci]}  {k.replace('_HUMAN', '')}  ·  "
                     f"baseline {base:+.2f}",
                     loc="left", fontsize=10.5, fontweight="bold", pad=15)
        ax.text(0.0, 1.015,
                f"naive > 0: {n_above}/{len(g)}   ·   "
                f"decoupled hits: {int(g['hit'].sum())} "
                f"({n_hid} below 0 naively)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=7.6,
                color="#444444")
        ax.set_xlabel("log enrichment (natural log)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper right",
                  bbox_to_anchor=(1.0, 0.93))

    lo_b = sub["abundance_baseline_ln"].min()
    hi_b = sub["abundance_baseline_ln"].max()
    fig.suptitle(
        f"Example 1 — three {lab}-screen kinases with moderate abundance "
        f"baselines ({lo_b:+.2f} to {hi_b:+.2f}): a library-wide cutoff\n"
        "still selects on the baseline, discarding real binders below zero and "
        "admitting non-binders above it",
        fontsize=12.5, fontweight="bold", y=0.975, linespacing=1.5)

    save_fig(fig, fname)




# ═════════════════════════════════════════════════════════════════════════════
# 6. Example 2: cross-kinase comparison for a single substrate
# ═════════════════════════════════════════════════════════════════════════════
def rank_flips(d: pd.DataFrame, min_kinases=5) -> pd.DataFrame:
    """Substrates whose top-scoring kinase changes after decoupling."""
    rows = []
    for (p, s), g in d.groupby(["paabd", "substrate"]):
        if len(g) < min_kinases:
            continue
        nb = g.loc[g["naive_score"].idxmax()]
        db = g.loc[g["binding_strength"].idxmax()]
        if nb["kinase"] == db["kinase"] or not db["hit"] or nb["hit"]:
            continue
        rows.append({
            "paabd": p, "substrate": s, "n_kinases": len(g),
            "naive_top": nb["kinase"], "naive_top_naive": nb["naive_score"],
            "naive_top_binding": nb["binding_strength"], "naive_top_lfsr": nb["lfsr"],
            "decoupled_top": db["kinase"], "decoupled_top_naive": db["naive_score"],
            "decoupled_top_binding": db["binding_strength"],
            "decoupled_top_lfsr": db["lfsr"],
            "naive_rank_of_decoupled_top": int((g["naive_score"] > db["naive_score"]).sum()) + 1,
        })
    return pd.DataFrame(rows)


def fig_example_substrate(d, cases, fname, annot=None):
    """One column per (paabd, substrate): naive ranking above, decoupled below.

    Chosen so that a *moderate* abundance baseline changes the sign of the
    score: kinases below zero naively that turn out to be confident binders,
    and kinases above zero naively whose binding term is negative.
    """
    annot = annot or {}
    ncol = len(cases)
    fig = plt.figure(figsize=(6.9 * ncol, 8.8))
    gs = fig.add_gridspec(2, ncol, hspace=0.32, wspace=0.34,
                          left=0.09, right=0.975, top=0.755, bottom=0.10)

    base_all = d.groupby(["paabd", "kinase"])["kinase_global"].median()

    for ci, (paabd, substrate) in enumerate(cases):
        g = d[(d["paabd"] == paabd) & (d["substrate"] == substrate)].copy()
        g["base"] = [base_all[(paabd, k)] for k in g["kinase"]]
        g["flip"] = np.sign(g["naive_score"]) != np.sign(g["binding_strength"])
        color = PAABD_COLORS[paabd]
        note = annot.get(substrate, "")

        # ── column header ────────────────────────────────────────────────────
        x0 = 0.09 + (0.885 / ncol) * (ci + 0.5)
        fig.text(x0, 0.912,
                 f"{substrate}   ·   {PAABD_LABELS[paabd]} reporter",
                 ha="center", va="center", fontsize=11.5, fontweight="bold",
                 color=color)

        up = g[(g["naive_score"] < 0) & g["hit"]]
        down = g[(g["naive_score"] > 0) & (g["binding_strength"] < 0)]
        lines_ = []
        if note:
            lines_.append(note)
        lines_.append(
            f"baselines {g['base'].min():+.2f} to {g['base'].max():+.2f}   ·   "
            f"{int(g['flip'].sum())} of {len(g)} kinases change sign after "
            "decoupling")
        if len(up):
            r = up.loc[up["binding_strength"].idxmax()]
            lines_.append(
                f"↑  {r['kinase'].replace('_HUMAN', '')} (baseline "
                f"{r['base']:+.2f}) scores {r['naive_score']:+.2f} naively but "
                f"{B_SK} = {r['binding_strength']:+.2f}, lfsr "
                f"{r['lfsr']:.0e} — a called binder")
        if len(down):
            r = down.loc[down["binding_strength"].idxmin()]
            lines_.append(
                f"↓  {r['kinase'].replace('_HUMAN', '')} (baseline "
                f"{r['base']:+.2f}) scores {r['naive_score']:+.2f} naively but "
                f"{B_SK} = {r['binding_strength']:+.2f}, lfsr "
                f"{r['lfsr']:.3f} — not a binder")
        fig.text(x0, 0.838, "\n".join(lines_), ha="center", va="center",
                 fontsize=8.5, color="#333333", linespacing=1.6)

        # ── ranked bars, before and after ───────────────────────────────────
        for ri, (col, xlab, ttl) in enumerate((
            ("naive_score", "naive log enrichment",
             "Pre-decoupling — ranked by the naive read-out"),
            ("binding_strength", f"binding strength {B_SK}",
             "Post-decoupling — ranked by the decoupled score"),
        )):
            ax = fig.add_subplot(gs[ri, ci])
            gs_ = g.sort_values(col, ascending=True)
            y = np.arange(len(gs_))
            cols = [HIT_COLOR if h else PRE_COLOR for h in gs_["hit"]]
            bars = ax.barh(y, gs_[col], color=cols, height=0.72, lw=0)
            # outline the kinases whose score changes sign after decoupling
            for bar, fl in zip(bars, gs_["flip"]):
                if fl:
                    bar.set_edgecolor("#111111")
                    bar.set_linewidth(1.1)
                    bar.set_linestyle((0, (2.5, 1.2)))
            ax.axvline(0, color="#333333", lw=1.2)
            if ri == 1:
                ax.axvline(BINDING_THRESH, color="#888888", lw=1.0, ls=":")

            ax.set_yticks(y)
            ax.set_yticklabels(
                [f"{k.replace('_HUMAN', '')}  ({b:+.2f})"
                 for k, b in zip(gs_["kinase"], gs_["base"])], fontsize=7.8)
            for t, h in zip(ax.get_yticklabels(), gs_["hit"]):
                if h:
                    t.set_color(HIT_COLOR); t.set_fontweight("bold")
            ax.set_ylim(-0.7, len(gs_) - 0.3)
            ax.set_xlabel(f"{xlab} (natural log)", fontsize=9)
            ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold", pad=5)

            span = max(gs_[col].max(), 0) - min(gs_[col].min(), 0)
            pad = 0.035 * span
            for yi, (_, r) in zip(y, gs_.iterrows()):
                v = r[col]
                txt = f"{v:+.2f}"
                if ri == 1 and r["hit"]:
                    txt += f"  (lfsr {r['lfsr']:.0e})"
                ax.text(v + (pad if v >= 0 else -pad), yi, txt, va="center",
                        fontsize=6.8, ha="left" if v >= 0 else "right",
                        color=HIT_COLOR if r["hit"] else "#333333")
            ax.margins(x=0.32 if ri == 1 else 0.22)

    handles = [
        mpatches.Patch(facecolor=HIT_COLOR,
                       label=f"decoupled hit (lfsr ≤ {LFSR_THRESH}, "
                             f"{B_SK} ≥ {BINDING_THRESH})"),
        mpatches.Patch(facecolor=PRE_COLOR, label="not called"),
        mpatches.Patch(facecolor="#FFFFFF", edgecolor="#111111", linewidth=1.1,
                       linestyle=(0, (2.5, 1.2)),
                       label="score changes sign after decoupling"),
        Line2D([0], [0], color="#888888", ls=":", lw=1.0,
               label=f"{B_SK} = {BINDING_THRESH} threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.028))
    fig.text(0.5, 0.012,
             "numbers beside each kinase are that kinase's fitted abundance "
             "baseline", ha="center", va="bottom", fontsize=7.6,
             color="#555555", style="italic")
    fig.suptitle("Example 2 — a moderate abundance baseline is enough to change "
                 "the sign of the score",
                 fontsize=13.5, fontweight="bold", y=0.975)

    save_fig(fig, fname)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Hit selection in (b_sk, lfsr) space
# ═════════════════════════════════════════════════════════════════════════════
N_DRAWS = 4 * 1000                # 4 chains x 1000 draws per pAABD fit
LFSR_ZERO = 1.0 / (2 * N_DRAWS)   # lfsr = 0 (no opposite-sign draw) is drawn here
SEL_XLIM = (-2.8, 3.2)
NONHIT_COLOR = "#9E9E9E"

# Two ways to draw the lfsr axis. Both put confident calls (small lfsr) at the
# top so the selection box sits top-right; ylim is (bottom, top).
#   log:    y = -log10(lfsr), lfsr = 0 floored at LFSR_ZERO
#   linear: y = lfsr on an inverted linear axis
LFSR_SCALES = {
    "log": dict(
        f=lambda v: -np.log10(np.maximum(v, LFSR_ZERO)),
        ylim=(-0.12, -np.log10(LFSR_ZERO) + 0.22),
        ticks=[0.5, 0.1, 0.01, 1e-3, 1.0 / N_DRAWS, LFSR_ZERO],
        ticklabels=["0.5", "0.1", "0.01", "0.001", f"1/{N_DRAWS}", "0"],
        ylabel="local false sign rate, lfsr  (−log₁₀ scale)",
        gridsize=(70, 30),
        label_top=(0.14, "bottom"),     # offset from threshold toward the top
        label_bottom=(0.05, "bottom"),  # offset from ylim bottom toward the top
        note=f"lfsr = 0 (no draw of opposite sign) plotted at 1/{2 * N_DRAWS:,}.",
    ),
    "linear": dict(
        f=lambda v: np.asarray(v, dtype=float),
        ylim=(0.515, -0.012),
        ticks=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ticklabels=["0.5", "0.4", "0.3", "0.2", "0.1", "0"],
        ylabel="local false sign rate, lfsr  (linear, inverted)",
        gridsize=(70, 45),
        label_top=(-0.03, "top"),
        label_bottom=(0.012, "bottom"),
        note="Linear lfsr axis, inverted so that the selected region "
             f"(lfsr ≤ {LFSR_THRESH}) sits at the top.",
    ),
}


def _toward_top(cfg, y, dy):
    """Move y by dy in the visual 'up' direction of the configured axis."""
    return y + dy if cfg["ylim"][1] > cfg["ylim"][0] else y - dy


def _lfsr_axis(ax, cfg, labels=True):
    """y axis labelled in lfsr units."""
    ax.set_yticks(cfg["f"](np.array(cfg["ticks"])))
    ax.set_yticklabels(cfg["ticklabels"] if labels else [])
    ax.set_ylim(*cfg["ylim"])


def _selection_box(ax, cfg, lw=1.0):
    y0 = float(cfg["f"](LFSR_THRESH))
    ytop = cfg["ylim"][1]
    lo, hi = min(y0, ytop), max(y0, ytop)
    ax.add_patch(mpatches.Rectangle(
        (BINDING_THRESH, lo), SEL_XLIM[1] - BINDING_THRESH, hi - lo,
        facecolor=HIT_COLOR, alpha=0.07, edgecolor="none", zorder=0))
    ax.add_patch(mpatches.Rectangle(
        (BINDING_THRESH, lo), SEL_XLIM[1] - BINDING_THRESH, hi - lo,
        facecolor="none", edgecolor=HIT_COLOR, lw=lw, zorder=4, clip_on=False))
    ax.axvline(BINDING_THRESH, color="#555555", lw=0.7, ls="--", zorder=3)
    ax.axhline(y0, color="#555555", lw=0.7, ls="--", zorder=3)


def fig_hit_selection(d: pd.DataFrame, fname, yscale="log"):
    """2-D distribution of posterior mean b_sk vs lfsr, with the hit-calling box.

    yscale: "log" (-log10 lfsr) or "linear" (inverted linear lfsr), see LFSR_SCALES.
    """
    cfg = LFSR_SCALES[yscale]
    ylo, yhi = sorted(cfg["ylim"])
    x_all = d["binding_strength"].clip(*SEL_XLIM)
    y_all = pd.Series(cfg["f"](d["lfsr"]), index=d.index)
    lfsr_ok = d["lfsr"] <= LFSR_THRESH
    b_ok = d["binding_strength"] >= BINDING_THRESH
    n_clip = int(((d["binding_strength"] < SEL_XLIM[0])
                  | (d["binding_strength"] > SEL_XLIM[1])).sum())

    fig = plt.figure(figsize=(11.5, 6.4))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.24,
                             left=0.065, right=0.985, bottom=0.12, top=0.81)
    gA = outer[0].subgridspec(2, 3, width_ratios=[4.2, 1.0, 0.12],
                              height_ratios=[1.0, 4.0], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gA[1, 0])
    ax_top = fig.add_subplot(gA[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gA[1, 1], sharey=ax)
    cax = fig.add_subplot(gA[1, 2])

    # ── A: density of all rows ────────────────────────────────────────────────
    cmap = mcolors.LinearSegmentedColormap.from_list("dens", ["#E6E6E6", "#1A1A1A"])
    hb = ax.hexbin(x_all, y_all, gridsize=cfg["gridsize"], extent=(*SEL_XLIM, ylo, yhi),
                   bins="log", mincnt=1, cmap=cmap, linewidths=0.1,
                   edgecolors="face", zorder=1)
    _selection_box(ax, cfg, lw=1.3)
    ax.set_xlim(*SEL_XLIM)
    _lfsr_axis(ax, cfg)
    ax.set_xlabel(f"posterior mean {B_SK}  (substrate binding, ln units)")
    ax.set_ylabel(cfg["ylabel"])
    cb = fig.colorbar(hb, cax=cax)
    cb.set_label("rows per bin", fontsize=7)
    cb.ax.tick_params(labelsize=6)

    # region counts
    y0 = float(cfg["f"](LFSR_THRESH))
    y_top = _toward_top(cfg, y0, cfg["label_top"][0])
    y_bot = _toward_top(cfg, cfg["ylim"][0], cfg["label_bottom"][0])
    regions = [
        (lfsr_ok & b_ok, (SEL_XLIM[1] - 0.08, y_top, cfg["label_top"][1]), "right",
         HIT_COLOR, "selected\n"),
        (lfsr_ok & ~b_ok, (SEL_XLIM[0] + 0.08, y_top, cfg["label_top"][1]), "left",
         "#333333", f"lfsr ≤ {LFSR_THRESH}, {B_SK} < {BINDING_THRESH}\n"),
        (~lfsr_ok & b_ok, (SEL_XLIM[1] - 0.08, y_bot, cfg["label_bottom"][1]), "right",
         "#333333", f"{B_SK} ≥ {BINDING_THRESH}, lfsr > {LFSR_THRESH}\n"),
        (~lfsr_ok & ~b_ok, (SEL_XLIM[0] + 0.08, y_bot, cfg["label_bottom"][1]), "left",
         "#333333", "neither\n"),
    ]
    for mask, (tx, ty, va), ha, color, label in regions:
        n = int(mask.sum())
        ax.text(tx, ty, f"{label}{n:,} ({100 * n / len(d):.1f}%)", ha=ha,
                va=va, fontsize=7, color=color, zorder=6,
                fontweight="bold" if color == HIT_COLOR else "normal",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.8))

    # marginals: stacked selected / not selected
    hit = d["hit"].to_numpy()
    xb = np.linspace(*SEL_XLIM, 91)
    ax_top.hist([x_all[~hit], x_all[hit]], bins=xb, stacked=True,
                color=[NONHIT_COLOR, HIT_COLOR], lw=0)
    ax_top.axvline(BINDING_THRESH, color="#555555", lw=0.7, ls="--")
    ax_top.tick_params(labelbottom=False)
    ax_top.set_ylabel("rows", fontsize=7)

    yb = np.linspace(ylo, yhi, 61)
    ax_right.hist([y_all[~hit], y_all[hit]], bins=yb, stacked=True,
                  orientation="horizontal", color=[NONHIT_COLOR, HIT_COLOR], lw=0)
    ax_right.axhline(y0, color="#555555", lw=0.7, ls="--")
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel("rows", fontsize=7)
    ax_right.tick_params(axis="x", labelsize=6)

    ax_top.set_title(
        f"A   all substrate × kinase × pAABD rows (n = {len(d):,})",
        loc="left", fontsize=9, fontweight="bold")

    # ── B–E: one panel per pAABD ──────────────────────────────────────────────
    gB = outer[1].subgridspec(2, 2, hspace=0.55, wspace=0.12)
    for i, p in enumerate(PAABD_ORDER):
        axp = fig.add_subplot(gB[i // 2, i % 2])
        g = d[d["paabd"] == p]
        h = g["hit"].to_numpy()
        xp = g["binding_strength"].clip(*SEL_XLIM).to_numpy()
        yp = cfg["f"](g["lfsr"].to_numpy())
        axp.scatter(xp[~h], yp[~h], s=1.2, c=NONHIT_COLOR, alpha=0.25, lw=0,
                    rasterized=True, zorder=1)
        axp.scatter(xp[h], yp[h], s=1.2, c=HIT_COLOR, alpha=0.45, lw=0,
                    rasterized=True, zorder=2)
        _selection_box(axp, cfg, lw=0.9)
        axp.set_xlim(*SEL_XLIM)
        _lfsr_axis(axp, cfg, labels=(i % 2 == 0))
        if i >= 2:
            axp.set_xlabel(f"posterior mean {B_SK}")
        if i % 2 == 0:
            axp.set_ylabel("lfsr")
        axp.set_title(
            f"{'BCDE'[i]}   {PAABD_LABELS[p]}\n"
            f"{int(h.sum()):,} / {len(g):,} selected ({100 * h.mean():.1f}%)",
            loc="left", fontsize=8.5, fontweight="bold", linespacing=1.4)

    handles = [
        mpatches.Patch(facecolor=mcolors.to_rgba(HIT_COLOR, 0.07), edgecolor=HIT_COLOR,
                       label=f"selected: lfsr ≤ {LFSR_THRESH} and {B_SK} ≥ {BINDING_THRESH}"),
        Line2D([], [], color="#555555", lw=0.7, ls="--", label="thresholds"),
        Line2D([], [], marker="o", ls="", ms=4, color=HIT_COLOR, label="selected row"),
        Line2D([], [], marker="o", ls="", ms=4, color=NONHIT_COLOR, label="not selected"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.058, 0.935))
    fig.suptitle("Positive substrate–kinase calls in (binding, lfsr) space",
                 x=0.065, ha="left", y=0.975, fontsize=12, fontweight="bold")
    fig.text(0.065, 0.02,
             f"lfsr from {N_DRAWS:,} posterior draws per fit.  {cfg['note']}  "
             f"{n_clip} rows with "
             f"{B_SK} outside [{SEL_XLIM[0]}, {SEL_XLIM[1]}] drawn at the edge.",
             fontsize=7, color="#555555", ha="left")

    save_fig(fig, fname)


# ═════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading scores and reconstructing the pre-decoupled read-out ...")
    d = load_scores()
    stats = pair_stats(d)
    stats.to_csv(f"{OUT_DIR}/tableS_decoupling_per_pair_stats.csv", index=False)
    d.to_pickle(f"{OUT_DIR}/scores_with_naive.pkl")
    print(f"  {len(d)} substrate x kinase x pAABD rows, {len(stats)} pairs")
    print(f"  wrote tableS_decoupling_per_pair_stats.csv")

    membership = d.groupby("kinase")["paabd"].apply(lambda x: set(x)).to_dict()
    y_kinases = sorted(k for k, p in membership.items() if "sh2" in p)
    st_kinases = sorted(k for k, p in membership.items() if "sh2" not in p)

    print("\nhistogram grids ...")
    fig_grid(d, st_kinases, ["fha", "1433", "ww"],
             "Pre- vs post-decoupling score distributions — S/T kinases "
             "(FHA / 14-3-3 / WW reporters)",
             "figS_pre_post_hist_ST", row_h=1.30)
    # only one pAABD for the Y kinases, so lay them out as a 4 x 2 panel block
    fig_pair_panels(
        d, [(k, "sh2") for k in
            stats[stats["paabd"] == "sh2"].sort_values("abundance_baseline_ln")["kinase"]],
        "Pre- vs post-decoupling score distributions — Y kinases (SH2 reporter)",
        "figS_pre_post_hist_Y", ncol=2, nrow=4, multipage=False, figsize=(9.5, 11.0))

    print("\nper-pair panels ...")
    fig_pair_panels(
        d,
        stats.sort_values(["paabd", "abundance_baseline_ln"])[["kinase", "paabd"]]
             .itertuples(index=False, name=None),
        "Pre- vs post-decoupling score distributions, one panel per "
        "kinase × pAABD pair",
        "figS_pre_post_hist_per_pair", ncol=3, nrow=4)

    print("\nschematic ...")
    fig_schematic(d, stats, "figS_decoupling_schematic")

    print("\nexample 1 ...")
    fig_example_trio(d, stats, EX1_PAABD, EX1_KINASES,
                     "figS_example1_baseline_sets_calls")

    print("\nexample 2 ...")
    flips = rank_flips(d)
    flips.sort_values("decoupled_top_binding", ascending=False).to_csv(
        f"{OUT_DIR}/tableS_decoupling_rank_flips.csv", index=False)
    print(f"  {len(flips)} substrates change top kinase after decoupling"
          f" -> tableS_decoupling_rank_flips.csv")
    fig_example_substrate(d, cases=EX2_CASES,
                          fname="figS_example2_substrate_rank_flip",
                          annot=EX2_ANNOT)

    print("\nhit selection (binding vs lfsr) ...")
    fig_hit_selection(d, "figS_hit_selection_binding_lfsr", yscale="log")
    fig_hit_selection(d, "figS_hit_selection_binding_lfsr_linear", yscale="linear")

    print(f"\ndone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
