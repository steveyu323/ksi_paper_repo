#!/usr/bin/env python
"""
Do the two sequencing replicates give the same answer once each is run
through the decoupling model on its own?

Inputs: out/refits/{paabd}_rep{1,2}_*.csv from 02_refit_per_replicate.py, and
the production (both-replicate) fit summaries
decoupling_out/251125_pymc_binding_stats_{paabd}.csv.

Per pAABD:
  * b_sk from rep 1 vs rep 2 (Pearson, Spearman), beside the same correlation
    for the naive kinase-centred log enrichment (01_raw_replicate_concordance.py)
  * held-out calibration of the posterior uncertainty:
        z = (b_rep1 - b_rep2) / sqrt(sd_rep1^2 + sd_rep2^2)
    The replicates are independent data, so if each posterior SD is honest,
    z ~ N(0, 1): SD(z) ~ 1 and ~5% of |z| > 1.96.
  * hit calls (lfsr <= 0.05 and b >= 0.1) in each replicate: overlap, Jaccard,
    and how many rep-1 hits have the same sign / are at least nominally
    supported (lfsr <= 0.05, b > 0) in rep 2, and vice versa
  * each single-replicate b vs the production b (both replicates)
  * kinase baselines g_k + mean delta_k (centred across kinases, since the
    per-replicate library-size scaling enters them) and hyperparameters
  * convergence of the 8 refits (R-hat, ESS, divergences, E-BFMI)

Outputs (out/):
  replicate_fit_summary.csv       per pAABD
  replicate_fit_pairs_{paabd}.csv per pair: b, sd, lfsr in rep1 / rep2 / production
  replicate_fit_kinase.csv        per pAABD x kinase baselines
  replicate_fit_hyper.csv         sigma_g, sigma_delta, sigma_b, alpha per fit
  replicate_refit_convergence.csv per refit
  fig_replicate_b_scatter.pdf
  fig_replicate_calibration_hits.pdf
  fig_replicate_vs_production.pdf

Usage (conda env ksi_bayes_qc):  python 03_compare_replicate_fits.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))      # the pipeline package root
import config  # noqa: E402

OUT = HERE / "out"
REFIT = OUT / "refits"
PROD = config.DECOUPLING_DIR                    # stage 05

PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}
INK = "#222222"
GREY = "#9E9E9E"
LFSR_THRESH, BINDING_THRESH = config.LFSR_THRESH, config.BINDING_THRESH
HYPER = ["sigma_g", "sigma_delta", "sigma_b", "alpha"]

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


def hit(b, lfsr):
    return (lfsr <= LFSR_THRESH) & (b >= BINDING_THRESH)


def load_pairs(pa: str) -> pd.DataFrame:
    cols = ["substrate", "kinase", "mean", "sd", "lfsr"]
    r1 = pd.read_csv(REFIT / f"{pa}_rep1_b.csv")[cols]
    r2 = pd.read_csv(REFIT / f"{pa}_rep2_b.csv")[cols]
    pr = pd.read_csv(PROD / f"251125_pymc_binding_stats_{pa}.csv").iloc[:, 1:][cols]
    d = (r1.merge(r2, on=["substrate", "kinase"], suffixes=("_r1", "_r2"))
           .merge(pr.rename(columns={"mean": "mean_prod", "sd": "sd_prod", "lfsr": "lfsr_prod"}),
                  on=["substrate", "kinase"], how="left"))
    for s in ("r1", "r2", "prod"):
        d[f"hit_{s}"] = hit(d[f"mean_{s}"], d[f"lfsr_{s}"])
    d["z"] = (d.mean_r1 - d.mean_r2) / np.sqrt(d.sd_r1 ** 2 + d.sd_r2 ** 2)
    return d


def load_global(pa: str, rep: int) -> pd.DataFrame:
    g = pd.read_csv(REFIT / f"{pa}_rep{rep}_global.csv", index_col=0)
    return g


def kinase_baselines(pa: str) -> pd.DataFrame:
    rows = []
    for rep in (1, 2):
        g = load_global(pa, rep)
        kg = g[g.index.str.startswith("g_plus_delta[")]["mean"]
        kg.index = kg.index.str.slice(len("g_plus_delta["), -1)
        rows.append((kg - kg.mean()).rename(f"kinase_global_centred_r{rep}"))
    out = pd.concat(rows, axis=1).reset_index(names="kinase")
    out.insert(0, "paabd", pa)
    return out


def main():
    raw = pd.read_csv(OUT / "raw_concordance_summary.csv")
    summ, kin_rows, hyp_rows, conv_rows, pairs = [], [], [], [], {}
    for pa in PAABD_ORDER:
        d = load_pairs(pa)
        d.to_csv(OUT / f"replicate_fit_pairs_{pa}.csv", index=False)
        pairs[pa] = d
        h1, h2, hp = d.hit_r1, d.hit_r2, d.hit_prod
        sup2 = (d.lfsr_r2 <= 0.05) & (d.mean_r2 > 0)
        sup1 = (d.lfsr_r1 <= 0.05) & (d.mean_r1 > 0)
        z = d.z
        summ.append({
            "paabd": pa, "n_pairs": len(d),
            "pearson_b_r1_r2": pearsonr(d.mean_r1, d.mean_r2)[0],
            "spearman_b_r1_r2": spearmanr(d.mean_r1, d.mean_r2)[0],
            "pearson_raw_centred_lfc": raw[(raw.paabd == pa) & (raw.quantity == "clfc")].pearson.iloc[0],
            "z_sd": z.std(), "z_mean": z.mean(),
            "frac_abs_z_gt_1.96": (z.abs() > 1.96).mean(),
            "hits_r1": int(h1.sum()), "hits_r2": int(h2.sum()), "hits_both": int((h1 & h2).sum()),
            "jaccard_hits": (h1 & h2).sum() / max((h1 | h2).sum(), 1),
            "frac_r1_hits_positive_in_r2": (d.mean_r2[h1] > 0).mean(),
            "frac_r2_hits_positive_in_r1": (d.mean_r1[h2] > 0).mean(),
            "frac_r1_hits_supported_r2": sup2[h1].mean(),
            "frac_r2_hits_supported_r1": sup1[h2].mean(),
            "hits_prod": int(hp.sum()),
            "frac_prod_hits_hit_in_r1": h1[hp].mean(),
            "frac_prod_hits_hit_in_r2": h2[hp].mean(),
            "frac_prod_hits_hit_in_either": (h1 | h2)[hp].mean(),
            "frac_prod_hits_positive_in_both": ((d.mean_r1 > 0) & (d.mean_r2 > 0))[hp].mean(),
            "pearson_b_r1_prod": pearsonr(d.mean_r1, d.mean_prod)[0],
            "pearson_b_r2_prod": pearsonr(d.mean_r2, d.mean_prod)[0],
        })
        kb = kinase_baselines(pa)
        kin_rows.append(kb)
        summ[-1]["pearson_kinase_global_r1_r2"] = pearsonr(kb.kinase_global_centred_r1,
                                                           kb.kinase_global_centred_r2)[0]
        for rep in (1, 2):
            g = load_global(pa, rep)
            hyp_rows.append({"paabd": pa, "fit": f"rep{rep}",
                             **{h: g.loc[h, "mean"] for h in HYPER},
                             **{f"{h}_sd": g.loc[h, "sd"] for h in HYPER}})
            b = pd.read_csv(REFIT / f"{pa}_rep{rep}_b.csv")
            smp = pd.read_csv(REFIT / f"{pa}_rep{rep}_sampler.csv")
            ident = g[g.index.str.startswith("g_plus_delta[")]
            conv_rows.append({
                "paabd": pa, "rep": rep,
                "r_hat_max_b": b.r_hat.max(), "n_b_r_hat_gt_1.01": int((b.r_hat > 1.01).sum()),
                "ess_bulk_min_b": b.ess_bulk.min(), "ess_tail_min_b": b.ess_tail.min(),
                # with one replicate g_k and delta_k1 enter only as their sum, so
                # report the identified kinase baseline, not g / delta separately
                "r_hat_max_kinase_baseline": ident.r_hat.max(),
                "ess_bulk_min_kinase_baseline": ident.ess_bulk.min(),
                "r_hat_max_hyper": g.loc[HYPER].r_hat.max(),
                "ess_bulk_min_hyper": g.loc[HYPER].ess_bulk.min(),
                "r_hat_max_g_delta_unidentified": g[g.index.str.match(r"(g|delta)\[")].r_hat.max(),
                "divergences": int(smp.divergences.sum()),
                "max_treedepth_hits": int(smp.max_treedepth_hits.sum()),
                "e_bfmi_min": smp.bfmi.min(),
            })
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "replicate_fit_summary.csv", index=False)
    kin = pd.concat(kin_rows, ignore_index=True)
    kin.to_csv(OUT / "replicate_fit_kinase.csv", index=False)
    pd.DataFrame(hyp_rows).to_csv(OUT / "replicate_fit_hyper.csv", index=False)
    conv = pd.DataFrame(conv_rows)
    conv.to_csv(OUT / "replicate_refit_convergence.csv", index=False)

    # ── Fig 1: b rep1 vs rep2, + kinase baselines ───────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.6))
    for j, pa in enumerate(PAABD_ORDER):
        d, s = pairs[pa], summ[summ.paabd == pa].iloc[0]
        ax = axes[0, j]
        lo, hi = np.percentile(np.r_[d.mean_r1, d.mean_r2], [0.05, 99.95])
        ax.hexbin(d.mean_r1, d.mean_r2, gridsize=55, bins="log", cmap="Greys", mincnt=1,
                  extent=(lo, hi, lo, hi), linewidths=0, rasterized=True)
        both = d.hit_r1 & d.hit_r2
        ax.scatter(d.mean_r1[both], d.mean_r2[both], s=2.5, color=PAABD_COLORS[pa], lw=0,
                   rasterized=True, label="hit in both reps")
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(PAABD_LABELS[pa])
        ax.text(0.04, 0.96, f"model b: r = {s.pearson_b_r1_r2:.3f}\n"
                f"raw centred LFC: r = {s.pearson_raw_centred_lfc:.3f}",
                transform=ax.transAxes, va="top", fontsize=6.3, color=INK)
        ax.set_xlabel("b̂, fit to rep 1 only")
        if j == 0:
            ax.set_ylabel("b̂, fit to rep 2 only")
            ax.legend(loc="lower right", frameon=False, fontsize=6, markerscale=3)
        ax = axes[1, j]
        k = kin[kin.paabd == pa]
        ax.scatter(k.kinase_global_centred_r1, k.kinase_global_centred_r2, s=14,
                   color=PAABD_COLORS[pa], zorder=3)
        lo, hi = (min(k.kinase_global_centred_r1.min(), k.kinase_global_centred_r2.min()) - 0.2,
                  max(k.kinase_global_centred_r1.max(), k.kinase_global_centred_r2.max()) + 0.2)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.text(0.04, 0.96, f"r = {s.pearson_kinase_global_r1_r2:.3f}\n{len(k)} kinases",
                transform=ax.transAxes, va="top", fontsize=6.3, color=INK)
        ax.set_xlabel("kinase baseline, rep 1 (centred)")
        if j == 0:
            ax.set_ylabel("kinase baseline, rep 2 (centred)")
    fig.suptitle("Independent single-replicate fits of the decoupling model: binding term b_sk (top) "
                 "and per-kinase baseline g_k + δ̄_k (bottom)", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_replicate_b_scatter")

    # ── Fig 2: calibration z + hit concordance ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.1, 1.2, 1]})
    ax = axes[0]
    xs = np.linspace(-5, 5, 300)
    for pa in PAABD_ORDER:
        h, e = np.histogram(pairs[pa].z, bins=np.linspace(-5, 5, 61), density=True)
        ax.plot(0.5 * (e[1:] + e[:-1]), h, color=PAABD_COLORS[pa], lw=1.1,
                label=f"{PAABD_LABELS[pa]}  SD {summ[summ.paabd == pa].z_sd.iloc[0]:.2f}")
    ax.plot(xs, norm.pdf(xs), color=INK, lw=1.0, ls="--", label="N(0,1)")
    ax.set_xlabel("z = (b̂₁ − b̂₂) / √(sd₁² + sd₂²)")
    ax.set_yticks([])
    ax.set_title("Replicate disagreement vs. stated\nposterior uncertainty (held-out calibration)")
    ax.legend(frameon=False, fontsize=6)

    ax = axes[1]
    for i, pa in enumerate(PAABD_ORDER):
        s = summ[summ.paabd == pa].iloc[0]
        only1, both, only2 = s.hits_r1 - s.hits_both, s.hits_both, s.hits_r2 - s.hits_both
        tot = only1 + both + only2
        left = 0
        for w, c, lab in [(only1, "#D0D0D0", "rep 1 only"), (both, PAABD_COLORS[pa], "both"),
                          (only2, "#8C8C8C", "rep 2 only")]:
            ax.barh(i, w / tot, left=left, color=c, edgecolor="white", lw=1.2, height=0.7)
            if w / tot > 0.08:
                ax.text(left + w / tot / 2, i, f"{w}", ha="center", va="center", fontsize=6.5,
                        color="white" if lab == "both" else INK)
            left += w / tot
        ax.text(1.02, i, f"J = {s.jaccard_hits:.2f}", va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(PAABD_ORDER)))
    ax.set_yticklabels([PAABD_LABELS[p] for p in PAABD_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of hits called in either replicate")
    ax.set_title(f"Hit calls per replicate (lfsr ≤ {LFSR_THRESH}, b ≥ {BINDING_THRESH})\n"
                 "light: rep 1 only · color: both · dark: rep 2 only")

    ax = axes[2]
    w = 0.2
    metrics = [("frac_r1_hits_positive_in_r2", "rep-1 hits: b > 0 in rep 2"),
               ("frac_r1_hits_supported_r2", "rep-1 hits: lfsr ≤ .05 in rep 2"),
               ("frac_r2_hits_positive_in_r1", "rep-2 hits: b > 0 in rep 1"),
               ("frac_r2_hits_supported_r1", "rep-2 hits: lfsr ≤ .05 in rep 1")]
    for i, pa in enumerate(PAABD_ORDER):
        s = summ[summ.paabd == pa].iloc[0]
        vals = [s[m] for m, _ in metrics]
        ax.scatter(vals, np.arange(len(metrics)) + (i - 1.5) * w, s=16, color=PAABD_COLORS[pa],
                   label=PAABD_LABELS[pa], zorder=3)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([lab for _, lab in metrics], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.01)
    ax.set_xlabel("fraction")
    ax.set_title("Hits from one replicate are\nsupported by the other")
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    save_fig(fig, "fig_replicate_calibration_hits")

    # ── Fig 3: each replicate vs production ────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.4))
    for j, pa in enumerate(PAABD_ORDER):
        d, s = pairs[pa], summ[summ.paabd == pa].iloc[0]
        for i, rep in enumerate(("r1", "r2")):
            ax = axes[i, j]
            lo, hi = np.percentile(np.r_[d[f"mean_{rep}"], d.mean_prod], [0.05, 99.95])
            ax.hexbin(d.mean_prod, d[f"mean_{rep}"], gridsize=55, bins="log", cmap="Greys",
                      mincnt=1, extent=(lo, hi, lo, hi), linewidths=0, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=PAABD_COLORS[pa], lw=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.text(0.04, 0.96, f"r = {s[f'pearson_b_{rep}_prod']:.3f}", transform=ax.transAxes,
                    va="top", fontsize=6.5, color=INK)
            if i == 0:
                ax.set_title(PAABD_LABELS[pa])
            if i == 1:
                ax.set_xlabel("b̂, production fit (both reps)")
            if j == 0:
                ax.set_ylabel(f"b̂, rep {rep[-1]} only")
    fig.suptitle("Each single-replicate fit vs. the production fit", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_replicate_vs_production")

    print(summ.T.to_string())
    print(conv.to_string())
    print(pd.DataFrame(hyp_rows).round(3).to_string())


if __name__ == "__main__":
    main()
