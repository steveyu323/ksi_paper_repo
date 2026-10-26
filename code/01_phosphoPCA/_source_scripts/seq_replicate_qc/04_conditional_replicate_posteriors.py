#!/usr/bin/env python
"""
Per-replicate binding posteriors with the assay's noise model held fixed,
and a held-out (cross-replicate) predictive check.

Why: 02_refit_per_replicate.py shows that with a single replicate the model
has one observation per (substrate, kinase) and cannot separate binding b_sk
from NB overdispersion. alpha collapses and b is over-shrunk, so that test
mainly shows the second replicate is what identifies the model. The fairer
"do the replicates give the same answer" test fixes the quantities that
describe the assay rather than individual pairs, at their production
posterior means:
    alpha      NB dispersion (one number per pAABD)
    sigma_b    prior scale of b (one number per pAABD)
    g_k + delta_kr  per-kinase, per-replicate baseline (each fit from >1,000 observations)
and estimates b_sk separately from each replicate's count. Given those, the
posterior of each b_sk is one-dimensional,
    p(b | y_r) ∝ N(b; 0, sigma_b) · NB(y_r; exp(log_off_r + g_k + delta_kr + b), alpha),
and is computed exactly on a grid. (The per-kinase centring of b is a shift
of < 0.01 and is ignored.)

Validation: the same grid calculation with *both* replicates' likelihoods
must reproduce the production NUTS posterior of b (reported as
grid_vs_nuts_* in the summary).

Expected concordance: each replicate has half the reads, so even perfectly
consistent replicates will not call identical hit sets. To set that
expectation, b* is drawn from the joint posterior, both replicates' counts are
re-simulated from the model at their real offsets, and the per-replicate
analysis is repeated (N_SIM times). Observed statistics are reported next to
these expected_* values.

Held-out check: the rep-1 posterior of b_sk predicts the rep-2 count
(posterior predictive = grid mixture of NB), and vice versa. Randomised PIT
values should be uniform and the central 50% / 90% predictive intervals
should cover 50% / 90% of held-out counts if the model's uncertainty is
honest.

Outputs (out/):
  conditional_summary.csv            per pAABD
  conditional_pairs_{paabd}.csv      per pair: b mean/sd/lfsr for rep1, rep2, both (grid), production (NUTS)
  fig_conditional_replicates.pdf     rep1 vs rep2 b, validation vs NUTS, held-out PIT
  fig_conditional_hits.pdf           hit-call concordance

Usage (conda env ksi_bayes_qc):  python 04_conditional_replicate_posteriors.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import gammaln
from scipy.stats import nbinom, pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "out"
PROD = ROOT / "decoupling_out"

PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}
INK = "#222222"
LFSR_THRESH, BINDING_THRESH = 0.05, 0.1
GRID = np.linspace(-6, 8, 2801)    # strongest 14-3-3 binders reach b ~ 5
N_SIM = 3
CHUNK = 2000
RNG = np.random.default_rng(0)

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


def nb_logpmf(y, mu, a):
    return (gammaln(y + a) - gammaln(a) - gammaln(y + 1)
            + a * np.log(a / (a + mu)) + y * np.log(mu / (a + mu)))


def summarize(logpost):
    """Normalise a (pairs, grid) log posterior; return weights, mean, sd, lfsr."""
    w = np.exp(logpost - logpost.max(axis=1, keepdims=True))
    w /= w.sum(axis=1, keepdims=True)
    mean = w @ GRID
    sd = np.sqrt(np.clip(w @ GRID ** 2 - mean ** 2, 0, None))
    p_pos = w[:, GRID > 0].sum(1) + 0.5 * w[:, GRID == 0].sum(1)
    return w, mean, sd, np.minimum(p_pos, 1 - p_pos)


def heldout_pit(w, y, base, a):
    """PIT of held-out counts y under the grid-mixture NB predictive.

    w: (pairs, grid) posterior weights from the other replicate;
    base: (pairs,) log_off + g + delta of the held-out replicate.
    """
    mu = np.exp(base[:, None] + GRID[None, :])
    p = a / (a + mu)
    F = (w * nbinom.cdf(y[:, None], a, p)).sum(1)
    Fm = (w * nbinom.cdf(y[:, None] - 1, a, p)).sum(1)
    return Fm + RNG.uniform(size=len(y)) * (F - Fm)


def tls_slope(x, y):
    """Total-least-squares slope of y on x (symmetric in the noise of both)."""
    _, v = np.linalg.eigh(np.cov(x, y))
    return v[1, 1] / v[0, 1]


def hit(b, lfsr):
    return (lfsr <= LFSR_THRESH) & (b >= BINDING_THRESH)


def analyse(pa: str):
    nc = PROD / f"251125_pymc_{pa}.nc"
    cd = xr.open_dataset(nc, group="constant_data", engine="h5netcdf").load()
    y = xr.open_dataset(nc, group="observed_data", engine="h5netcdf").y.values
    post = xr.open_dataset(nc, group="posterior", engine="h5netcdf")
    alpha = float(post.alpha.mean())
    sigma_b = float(post.sigma_b.mean())
    g = post.g.mean(("chain", "draw")).values
    delta = post.delta.mean(("chain", "draw")).values           # (kinase, rep)
    subs, kins = post.substrate.values.astype(str), post.kinase.values.astype(str)

    obs = pd.DataFrame({"s": cd.sub_i.values, "k": cd.kin_i.values, "r": cd.rep_i.values,
                        "y": y, "base": cd.log_off.values + g[cd.kin_i.values]
                        + delta[cd.kin_i.values, cd.rep_i.values]})
    wide = obs.pivot_table(index=["s", "k"], columns="r", values=["y", "base"])
    wide = wide.dropna()                                         # pairs with both reps
    y1, y2 = wide[("y", 0)].values, wide[("y", 1)].values
    b1, b2 = wide[("base", 0)].values, wide[("base", 1)].values
    si = wide.index.get_level_values("s").values
    ki = wide.index.get_level_values("k").values

    logprior = -0.5 * (GRID / sigma_b) ** 2
    res = {k: [] for k in ("m1", "s1", "l1", "m2", "s2", "l2", "mj", "sj", "lj", "pit21", "pit12")}
    sims = [{k: [] for k in ("m1", "l1", "m2", "l2")} for _ in range(N_SIM)]
    for st in range(0, len(y1), CHUNK):
        sl = slice(st, st + CHUNK)
        ll1 = nb_logpmf(y1[sl, None], np.exp(b1[sl, None] + GRID), alpha)
        ll2 = nb_logpmf(y2[sl, None], np.exp(b2[sl, None] + GRID), alpha)
        w1, m1, s1, l1 = summarize(logprior + ll1)
        w2, m2, s2, l2 = summarize(logprior + ll2)
        wj, mj, sj, lj = summarize(logprior + ll1 + ll2)
        # expected concordance: b* ~ joint posterior, both replicates re-simulated
        # at their observed depth/offset, then b re-estimated per replicate
        cdf = np.cumsum(wj, axis=1)
        for sim in sims:
            bs = GRID[np.minimum((cdf < RNG.uniform(size=(len(cdf), 1))).sum(1), len(GRID) - 1)]
            for r, base in (("1", b1[sl]), ("2", b2[sl])):
                mu = np.exp(base + bs)
                ys = RNG.negative_binomial(alpha, alpha / (alpha + mu))
                _, m, _, l = summarize(logprior + nb_logpmf(ys[:, None], np.exp(base[:, None] + GRID), alpha))
                sim[f"m{r}"].append(m); sim[f"l{r}"].append(l)
        for k, v in zip(("m1", "s1", "l1", "m2", "s2", "l2", "mj", "sj", "lj"),
                        (m1, s1, l1, m2, s2, l2, mj, sj, lj)):
            res[k].append(v)
        res["pit21"].append(heldout_pit(w1, y2[sl], b2[sl], alpha))   # predict rep2 from rep1
        res["pit12"].append(heldout_pit(w2, y1[sl], b1[sl], alpha))
    res = {k: np.concatenate(v) for k, v in res.items()}
    sims = [{k: np.concatenate(v) for k, v in sim.items()} for sim in sims]

    d = pd.DataFrame({"substrate": subs[si], "kinase": kins[ki], "y_rep1": y1, "y_rep2": y2,
                      "mean_r1": res["m1"], "sd_r1": res["s1"], "lfsr_r1": res["l1"],
                      "mean_r2": res["m2"], "sd_r2": res["s2"], "lfsr_r2": res["l2"],
                      "mean_both_grid": res["mj"], "sd_both_grid": res["sj"], "lfsr_both_grid": res["lj"],
                      "pit_rep2_from_rep1": res["pit21"], "pit_rep1_from_rep2": res["pit12"]})
    prod = (pd.read_csv(PROD / f"251125_pymc_binding_stats_{pa}.csv").iloc[:, 1:]
            .rename(columns={"mean": "mean_prod", "sd": "sd_prod", "lfsr": "lfsr_prod"}))
    d = d.merge(prod, on=["substrate", "kinase"], how="left")
    for s in ("r1", "r2", "prod"):
        d[f"hit_{s}"] = hit(d[f"mean_{s}"], d[f"lfsr_{s}"])
    d["hit_both_grid"] = hit(d.mean_both_grid, d.lfsr_both_grid)
    d.to_csv(OUT / f"conditional_pairs_{pa}.csv", index=False)

    h1, h2, hp, hg = d.hit_r1, d.hit_r2, d.hit_prod, d.hit_both_grid
    pits = np.r_[d.pit_rep2_from_rep1, d.pit_rep1_from_rep2]
    in50, in90 = (pits >= 0.25) & (pits <= 0.75), (pits >= 0.05) & (pits <= 0.95)
    s = {
        "paabd": pa, "n_pairs": len(d), "alpha": alpha, "sigma_b": sigma_b,
        "grid_vs_nuts_pearson_b": pearsonr(d.mean_both_grid, d.mean_prod)[0],
        "grid_vs_nuts_max_abs_diff_b": float(np.abs(d.mean_both_grid - d.mean_prod).max()),
        "grid_vs_nuts_pearson_sd": pearsonr(d.sd_both_grid, d.sd_prod)[0],
        "grid_vs_nuts_hit_jaccard": (hg & hp).sum() / max((hg | hp).sum(), 1),
        "pearson_b_r1_r2": pearsonr(d.mean_r1, d.mean_r2)[0],
        "spearman_b_r1_r2": spearmanr(d.mean_r1, d.mean_r2)[0],
        # dynamic range: 1 if both sequencing batches put b on the same scale
        "tls_slope_b_r2_on_r1": tls_slope(d.mean_r1, d.mean_r2),
        "sd_b_r1": d.mean_r1.std(), "sd_b_r2": d.mean_r2.std(),
        "median_b_prod_hits_r1": d.mean_r1[d.hit_prod].median(),
        "median_b_prod_hits_r2": d.mean_r2[d.hit_prod].median(),
        "heldout_coverage50": in50.mean(), "heldout_coverage90": in90.mean(),
        "heldout_pit_ks_D": float(np.max(np.abs(np.sort(pits) - (np.arange(1, len(pits) + 1) - 0.5) / len(pits)))),
        "hits_r1": int(h1.sum()), "hits_r2": int(h2.sum()), "hits_both": int((h1 & h2).sum()),
        "jaccard_hits": (h1 & h2).sum() / max((h1 | h2).sum(), 1),
        "frac_r1_hits_positive_in_r2": (d.mean_r2[h1] > 0).mean(),
        "frac_r2_hits_positive_in_r1": (d.mean_r1[h2] > 0).mean(),
        "frac_r1_hits_lfsr05_pos_in_r2": ((d.lfsr_r2 <= 0.05) & (d.mean_r2 > 0))[h1].mean(),
        "frac_r2_hits_lfsr05_pos_in_r1": ((d.lfsr_r1 <= 0.05) & (d.mean_r1 > 0))[h2].mean(),
        "hits_prod": int(hp.sum()),
        "frac_prod_hits_positive_both_reps": ((d.mean_r1 > 0) & (d.mean_r2 > 0))[hp].mean(),
        "frac_prod_hits_hit_in_both_reps": (h1 & h2)[hp].mean(),
        "frac_prod_hits_hit_in_either_rep": (h1 | h2)[hp].mean(),
    }
    # expected values of the same statistics if both replicates followed the model exactly
    ex = []
    for sim in sims:
        e1, e2 = hit(sim["m1"], sim["l1"]), hit(sim["m2"], sim["l2"])
        ex.append({
            "pearson_b_r1_r2": pearsonr(sim["m1"], sim["m2"])[0],
            "hits_r1": e1.sum(), "hits_r2": e2.sum(), "hits_both": (e1 & e2).sum(),
            "jaccard_hits": (e1 & e2).sum() / max((e1 | e2).sum(), 1),
            "frac_r1_hits_positive_in_r2": (sim["m2"][e1] > 0).mean(),
            "frac_r2_hits_positive_in_r1": (sim["m1"][e2] > 0).mean(),
            "frac_r1_hits_lfsr05_pos_in_r2": ((sim["l2"] <= 0.05) & (sim["m2"] > 0))[e1].mean(),
            "frac_r2_hits_lfsr05_pos_in_r1": ((sim["l1"] <= 0.05) & (sim["m1"] > 0))[e2].mean(),
        })
    for k, v in pd.DataFrame(ex).mean().items():
        s[f"expected_{k}"] = v
    return d, s


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, summ = {}, []
    for pa in PAABD_ORDER:
        print(f"[{pa}]")
        pairs[pa], s = analyse(pa)
        summ.append(s)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "conditional_summary.csv", index=False)

    # ── Fig 1: rep1 vs rep2, validation, held-out PIT ─────────────────────
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.2))
    for j, pa in enumerate(PAABD_ORDER):
        d, s, col = pairs[pa], summ.iloc[j], PAABD_COLORS[pa]
        ax = axes[0, j]
        lo, hi = np.percentile(np.r_[d.mean_r1, d.mean_r2], [0.05, 99.95])
        ax.hexbin(d.mean_r1, d.mean_r2, gridsize=55, bins="log", cmap="Greys", mincnt=1,
                  extent=(lo, hi, lo, hi), linewidths=0, rasterized=True)
        hp = d.hit_prod
        ax.scatter(d.mean_r1[hp], d.mean_r2[hp], s=2, color=col, lw=0, rasterized=True,
                   label="production hit")
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.text(0.04, 0.96, f"r = {s.pearson_b_r1_r2:.3f} (expected {s.expected_pearson_b_r1_r2:.3f})\n"
                f"ρ = {s.spearman_b_r1_r2:.3f}\nTLS slope {s.tls_slope_b_r2_on_r1:.2f}",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
        ax.set_title(PAABD_LABELS[pa])
        ax.set_xlabel("b̂ from rep 1")
        if j == 0:
            ax.set_ylabel("b̂ from rep 2")
            ax.legend(loc="lower right", frameon=False, fontsize=6, markerscale=3)

        ax = axes[1, j]
        lo, hi = np.percentile(np.r_[d.mean_prod, d.mean_both_grid], [0.05, 99.95])
        ax.hexbin(d.mean_prod, d.mean_both_grid, gridsize=55, bins="log", cmap="Greys",
                  mincnt=1, extent=(lo, hi, lo, hi), linewidths=0, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=col, lw=0.8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.text(0.04, 0.96, f"r = {s.grid_vs_nuts_pearson_b:.4f}\n"
                f"hit Jaccard {s.grid_vs_nuts_hit_jaccard:.3f}",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
        ax.set_xlabel("b̂, production NUTS fit")
        if j == 0:
            ax.set_ylabel("b̂, grid posterior\n(both reps, fixed hyper.)")

        ax = axes[2, j]
        ax.hist(d.pit_rep2_from_rep1, bins=20, range=(0, 1), density=True, histtype="stepfilled",
                color=col, alpha=0.55, label="rep 2 | rep 1")
        ax.hist(d.pit_rep1_from_rep2, bins=20, range=(0, 1), density=True, histtype="step",
                color=INK, lw=1.0, label="rep 1 | rep 2")
        ax.axhline(1, color=INK, lw=0.6, ls=":")
        ax.set_ylim(0, 1.8)
        ax.text(0.04, 0.96, f"cov50 {s.heldout_coverage50:.3f}\ncov90 {s.heldout_coverage90:.3f}",
                transform=ax.transAxes, va="top", fontsize=6.5, color=INK)
        ax.set_xlabel("held-out PIT")
        if j == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=6)
    axes[0, 0].annotate("per-replicate b", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", fontsize=9, fontweight="bold")
    axes[1, 0].annotate("method check", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", fontsize=9, fontweight="bold")
    axes[2, 0].annotate("held-out calibration", xy=(-0.42, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", fontsize=9, fontweight="bold")
    fig.suptitle("Binding estimates from each sequencing replicate alone (assay noise model fixed "
                 "at production values)", fontsize=9.5, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_conditional_replicates")

    # ── Fig 2: hit concordance ─────────────────────────────────────────────
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
    for i, pa in enumerate(PAABD_ORDER):
        s = summ.iloc[i]
        parts = [(s.hits_r1 - s.hits_both, "#D0D0D0"), (s.hits_both, PAABD_COLORS[pa]),
                 (s.hits_r2 - s.hits_both, "#8C8C8C")]
        tot, left = sum(p for p, _ in parts), 0
        for w, c in parts:
            ax.barh(i, w / tot, left=left, color=c, edgecolor="white", lw=1.2, height=0.7)
            if w / tot > 0.07:
                ax.text(left + w / tot / 2, i, f"{int(w)}", ha="center", va="center", fontsize=6.5,
                        color="white" if c == PAABD_COLORS[pa] else INK)
            left += w / tot
        ax.text(1.02, i, f"J = {s.jaccard_hits:.2f}\n(expected {s.expected_jaccard_hits:.2f})",
                va="center", fontsize=6.2, color=INK)
    ax.set_yticks(range(len(PAABD_ORDER)))
    ax.set_yticklabels([PAABD_LABELS[p] for p in PAABD_ORDER])
    ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of hits called in either replicate")
    ax.set_title("Hits called from one replicate (half the reads)\nlight: rep 1 only · color: both · dark: rep 2 only")
    metrics = [("frac_r1_hits_positive_in_r2", "rep-1 hits: b > 0 in rep 2"),
               ("frac_r1_hits_lfsr05_pos_in_r2", "rep-1 hits: lfsr ≤ .05, b > 0 in rep 2"),
               ("frac_r2_hits_positive_in_r1", "rep-2 hits: b > 0 in rep 1"),
               ("frac_r2_hits_lfsr05_pos_in_r1", "rep-2 hits: lfsr ≤ .05, b > 0 in rep 1"),
               ("frac_prod_hits_positive_both_reps", "production hits: b > 0 in both reps")]
    for i, pa in enumerate(PAABD_ORDER):
        s = summ.iloc[i]
        yy = np.arange(len(metrics)) + (i - 1.5) * 0.17
        ax2.scatter([s[m] for m, _ in metrics], yy, s=15, color=PAABD_COLORS[pa],
                    label=PAABD_LABELS[pa], zorder=3)
        ex = [s.get(f"expected_{m}", np.nan) for m, _ in metrics]
        ax2.scatter(ex, yy, s=22, marker="|", color=INK, lw=1.0, zorder=4,
                    label="expected under model" if i == 0 else None)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels([lab for _, lab in metrics], fontsize=6.5)
    ax2.invert_yaxis(); ax2.set_xlim(0.5, 1.01)
    ax2.set_xlabel("fraction")
    ax2.set_title("Hits from one replicate are supported by the other\n(| = expected if replicates follow the model exactly)")
    ax2.legend(frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save_fig(fig, "fig_conditional_hits")

    print(summ.T.to_string())


if __name__ == "__main__":
    main()
