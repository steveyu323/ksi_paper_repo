#!/usr/bin/env python
"""
MCMC convergence and sampling-quality diagnostics for the decoupling fits.

For each pAABD fit (decoupling_out/251125_pymc_{paabd}.nc):

  * Sampler health: divergences, max-tree-depth hits, acceptance, step size,
    E-BFMI per chain.
  * Rank-normalised split R-hat, bulk/tail ESS and MCSE for every parameter
    that enters the likelihood, grouped as
        hyper      sigma_g, sigma_delta, sigma_b, alpha
        g, delta   as sampled
        kinase_global  g_k + mean_r delta_kr   (the per-kinase baseline used downstream)
        b          binding term, observed (substrate, kinase) pairs only
  * An R-hat reference: R-hat of iid N(0,1) draws with the same (chain, draw,
    n_param) shape as b, i.e. what perfectly mixed chains of this length give.
  * Every parameter with R-hat > 1.01, with its four chain means, the spread
    of those means in posterior-SD units, its lfsr and whether it is a hit.
    (ArviZ R-hat is the max of the bulk and folded/tail versions, so a flag
    can come from tail-width disagreement while the chain means agree.)
  * Chain agreement of the inference itself: b posterior means and hit calls
    (lfsr <= 0.05 and b >= 0.1) recomputed from chains {0,1} and {2,3}
    separately.

Outputs (out/):
  convergence_params_{paabd}.csv   one row per parameter
  convergence_overview.csv         per pAABD x group summary
  flagged_rhat_params.csv          every parameter with R-hat > 1.01
  sampler_health.csv               per pAABD x chain
  chain_split_hits.csv             hit-call agreement between chain halves
  fig_convergence_rhat_ess.pdf     R-hat and ESS distributions vs. iid null
  fig_hyperparameter_traces.pdf    trace + rank plots for the 4 hyperparameters
  fig_sampler_energy.pdf           energy / E-BFMI and tree depth
  fig_chain_split_agreement.pdf    b from chains {0,1} vs {2,3}

Usage (conda env ksi_bayes_qc):  python 01_convergence_diagnostics.py
"""

from __future__ import annotations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from qc_common import (BINDING_THRESH, ESS_THRESH, INK, LFSR_THRESH, NULL_COLOR,
                       OUT, PAABD_COLORS, PAABD_LABELS, PAABD_ORDER, RHAT_THRESH,
                       observed_pairs, open_group, save_fig)

HYPER = ["sigma_g", "sigma_delta", "sigma_b", "alpha"]
HYPER_LABELS = {"sigma_g": "σ_g", "sigma_delta": "σ_δ", "sigma_b": "σ_b", "alpha": "α (NB)"}
GROUP_ORDER = ["hyper", "g", "delta", "kinase_global", "b"]
RNG = np.random.default_rng(0)


def chain_shades(paabd: str, n: int = 4):
    """n shades of the pAABD color, light to dark, one per chain."""
    import matplotlib.colors as mc
    base = np.array(mc.to_rgb(PAABD_COLORS[paabd]))
    dark = base * 0.55
    return [tuple(1 - (1 - dark) * f) for f in np.linspace(0.35, 1.0, n)]


def diag_table(arr: np.ndarray, names, group: str) -> pd.DataFrame:
    """arr: (chain, draw, n). Rank-normalised R-hat, ESS, MCSE per column."""
    ds = az.convert_to_dataset({"x": arr})
    return pd.DataFrame({
        "group": group,
        "param": names,
        "mean": arr.mean(axis=(0, 1)),
        "sd": arr.std(axis=(0, 1)),
        "r_hat": az.rhat(ds).x.values,
        "ess_bulk": az.ess(ds, method="bulk").x.values,
        "ess_tail": az.ess(ds, method="tail").x.values,
        "mcse_mean": az.mcse(ds, method="mean").x.values,
    })


def lfsr_and_mean(b: np.ndarray):
    p = (b > 0).mean(axis=(0, 1))
    return b.mean(axis=(0, 1)), np.minimum(p, 1 - p)


def is_hit(mean, lfsr):
    return (lfsr <= LFSR_THRESH) & (mean >= BINDING_THRESH)


def analyse(paabd: str):
    print(f"[{paabd}] loading")
    post = open_group(paabd, "posterior")
    ss = open_group(paabd, "sample_stats").load()
    kin = post.kinase.values.astype(str)
    sub = post.substrate.values.astype(str)
    reps = post.rep.values

    scal = post[HYPER + ["g", "delta"]].load()
    si, ki = observed_pairs(paabd)
    b_all = post["b"].values                     # (chain, draw, substrate, kinase)
    b = b_all[:, :, si, ki]                      # observed pairs only
    del b_all

    # ---- per-parameter diagnostics -------------------------------------------
    dv = scal.delta.values                       # (c, d, kinase, rep)
    arrays = {                                   # group -> ((chain, draw, n), names)
        "hyper": (np.stack([scal[h].values for h in HYPER], -1), HYPER),
        "g": (scal.g.values, list(kin)),
        "delta": (dv.reshape(*dv.shape[:2], -1), [f"{k}|rep{r}" for k in kin for r in reps]),
        "kinase_global": (scal.g.values + dv.mean(-1), list(kin)),
        "b": (b, [f"{sub[s]}|{kin[k]}" for s, k in zip(si, ki)]),
    }
    params = pd.concat([diag_table(a, n, g) for g, (a, n) in arrays.items()], ignore_index=True)
    params.insert(0, "paabd", paabd)
    params["mcse_over_sd"] = params["mcse_mean"] / params["sd"]
    params.to_csv(OUT / f"convergence_params_{paabd}.csv", index=False)

    # ---- what the R-hat > 1.01 parameters actually look like ----------------
    flagged = []
    for grp, (a, names) in arrays.items():
        d = params[params.group == grp].reset_index(drop=True)
        for i in np.flatnonzero(d.r_hat.values > RHAT_THRESH):
            x = a[:, :, i]
            cm = x.mean(axis=1)
            mean, lfsr = lfsr_and_mean(x[:, :, None])
            flagged.append({
                "paabd": paabd, "group": grp, "param": names[i], "r_hat": d.r_hat[i],
                "ess_bulk": d.ess_bulk[i], "ess_tail": d.ess_tail[i],
                "mean": d["mean"][i], "sd": d.sd[i],
                "chain_means": ";".join(f"{v:.3f}" for v in cm),
                "chain_mean_spread_over_sd": (cm.max() - cm.min()) / d.sd[i],
                "chain_sds": ";".join(f"{v:.3f}" for v in x.std(axis=1)),
                "lfsr": float(lfsr[0]) if grp == "b" else np.nan,
                "is_hit": bool(is_hit(mean, lfsr)[0]) if grp == "b" else np.nan,
            })
    flagged = pd.DataFrame(flagged)

    # ---- iid null for R-hat, same shape as b --------------------------------
    null = az.rhat(az.convert_to_dataset({"x": RNG.standard_normal(b.shape)})).x.values

    # ---- sampler health -------------------------------------------------------
    bfmi = az.bfmi(ss)
    health = pd.DataFrame({
        "paabd": paabd,
        "chain": ss.chain.values,
        "draws": ss.sizes["draw"],
        "divergences": ss.diverging.sum("draw").values,
        "max_treedepth_hits": ss.reached_max_treedepth.sum("draw").values,
        "max_tree_depth": ss.tree_depth.max("draw").values,
        "mean_tree_depth": ss.tree_depth.mean("draw").values,
        "mean_accept": ss.acceptance_rate.mean("draw").values,
        "step_size": ss.step_size.mean("draw").values,
        "e_bfmi": bfmi,
    })

    # ---- chain-half agreement of the inference ------------------------------
    m01, l01 = lfsr_and_mean(b[:2])
    m23, l23 = lfsr_and_mean(b[2:])
    mall, lall = lfsr_and_mean(b)
    h01, h23, hall = is_hit(m01, l01), is_hit(m23, l23), is_hit(mall, lall)
    split = {
        "paabd": paabd,
        "n_pairs": len(mall),
        "hits_all_chains": int(hall.sum()),
        "hits_chains01": int(h01.sum()),
        "hits_chains23": int(h23.sum()),
        "hits_both_halves": int((h01 & h23).sum()),
        "jaccard_halves": (h01 & h23).sum() / max((h01 | h23).sum(), 1),
        "pearson_b_mean_halves": np.corrcoef(m01, m23)[0, 1],
        "max_abs_diff_b_mean": float(np.abs(m01 - m23).max()),
        "median_abs_diff_over_sd": float(np.median(np.abs(m01 - m23) / b.std(axis=(0, 1)))),
    }

    # overview rows
    ov = []
    for grp, d in params.groupby("group", sort=False):
        ov.append({
            "paabd": paabd, "group": grp, "n_params": len(d),
            "r_hat_max": d.r_hat.max(),
            "n_r_hat_gt_1.01": int((d.r_hat > RHAT_THRESH).sum()),
            "frac_r_hat_gt_1.01": (d.r_hat > RHAT_THRESH).mean(),
            "ess_bulk_min": d.ess_bulk.min(), "ess_bulk_median": d.ess_bulk.median(),
            "ess_tail_min": d.ess_tail.min(), "ess_tail_median": d.ess_tail.median(),
            "mcse_over_sd_max": d.mcse_over_sd.max(),
        })
    ov.append({"paabd": paabd, "group": "iid_null_same_shape_as_b", "n_params": len(null),
               "r_hat_max": null.max(), "n_r_hat_gt_1.01": int((null > RHAT_THRESH).sum()),
               "frac_r_hat_gt_1.01": (null > RHAT_THRESH).mean()})

    return dict(params=params, null=null, health=health, split=split, overview=pd.DataFrame(ov),
                flagged=flagged,
                hyper=scal[HYPER], ss=ss, m01=m01, m23=m23, h01=h01, h23=h23)


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════
def fig_rhat_ess(res):
    fig, axes = plt.subplots(len(PAABD_ORDER), 3, figsize=(8.2, 7.2))
    for row, pa in enumerate(PAABD_ORDER):
        r = res[pa]
        p = r["params"]
        col = PAABD_COLORS[pa]
        # R-hat
        ax = axes[row, 0]
        bins = np.linspace(0.996, 1.016, 61)
        ax.hist(np.clip(r["null"], bins[0], bins[-1]), bins=bins, histtype="step",
                color=NULL_COLOR, lw=1.2, ls="--", label="iid null")
        ax.hist(np.clip(p.r_hat, bins[0], bins[-1]), bins=bins, histtype="stepfilled",
                color=col, alpha=0.6, label="fit (all params)")
        ax.axvline(RHAT_THRESH, color=INK, lw=0.8, ls=":")
        ax.set_yscale("log")
        n_hi = int((p.r_hat > RHAT_THRESH).sum())
        n_null = int((r["null"] > RHAT_THRESH).sum())
        ax.text(0.98, 0.95, f"$\\hat{{R}}$ max {p.r_hat.max():.4f}\n>1.01: {n_hi} of {len(p)}\n"
                f"iid null >1.01: {n_null}", transform=ax.transAxes, ha="right", va="top",
                fontsize=6.3, color=INK)
        ax.set_ylabel(f"{PAABD_LABELS[pa]}\n# parameters")
        # ESS bulk / tail
        for j, k in enumerate(["ess_bulk", "ess_tail"], start=1):
            ax = axes[row, j]
            hi = max(p[k].max(), 4000) * 1.05
            ax.hist(p[k], bins=np.linspace(0, hi, 60), color=col, alpha=0.75)
            ax.axvline(ESS_THRESH, color=INK, lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.text(0.98, 0.95, f"min {p[k].min():.0f}\nmedian {p[k].median():.0f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK)
    for j, t in enumerate(["Rank-normalised split $\\hat{R}$", "Bulk ESS (of 4000 draws)",
                           "Tail ESS (of 4000 draws)"]):
        axes[0, j].set_title(t)
        axes[-1, j].set_xlabel(t)
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=6)
    fig.suptitle("Convergence of the decoupling fits: every sampled/derived parameter "
                 "(dotted: $\\hat{R}$ = 1.01, ESS = 400)", fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, "fig_convergence_rhat_ess")


def fig_traces(res):
    fig, axes = plt.subplots(len(PAABD_ORDER), 2 * len(HYPER), figsize=(13, 6.4),
                             gridspec_kw={"width_ratios": [2.2, 1] * len(HYPER)})
    for row, pa in enumerate(PAABD_ORDER):
        hy = res[pa]["hyper"]
        shades = chain_shades(pa)
        for j, h in enumerate(HYPER):
            x = hy[h].values                        # (chain, draw)
            ax_t, ax_r = axes[row, 2 * j], axes[row, 2 * j + 1]
            for c in range(x.shape[0]):
                ax_t.plot(x[c], lw=0.35, color=shades[c], alpha=0.9)
            ranks = pd.Series(x.ravel()).rank().values.reshape(x.shape)
            edges = np.linspace(0, x.size, 21)
            for c in range(x.shape[0]):
                ax_r.hist(ranks[c], bins=edges, histtype="step", color=shades[c], lw=0.9)
            ax_r.axhline(x.shape[1] / 20, color=INK, lw=0.6, ls=":")
            ax_r.set_yticks([])
            ax_r.set_ylim(0, None)
            ax_r.set_xticks([])
            rh = float(az.rhat(x))
            eb = float(az.ess(x))
            ax_t.text(0.02, 0.97, f"$\\hat{{R}}$ {rh:.3f}  ESS {eb:.0f}", transform=ax_t.transAxes,
                      fontsize=6, va="top", color=INK)
            if row == 0:
                ax_t.set_title(f"{HYPER_LABELS[h]} — trace")
                ax_r.set_title("rank")
            if row == len(PAABD_ORDER) - 1:
                ax_t.set_xlabel("draw")
            if j == 0:
                ax_t.set_ylabel(PAABD_LABELS[pa])
    fig.suptitle("Hyperparameter traces (4 chains, shaded light→dark) and rank histograms "
                 "(flat = chains exchangeable; dotted = uniform expectation)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_hyperparameter_traces")


def fig_energy(res):
    fig, axes = plt.subplots(2, len(PAABD_ORDER), figsize=(10, 4.6))
    for j, pa in enumerate(PAABD_ORDER):
        ss = res[pa]["ss"]
        e = ss.energy.values
        e_c = e - e.mean(axis=1, keepdims=True)
        de = np.diff(e, axis=1)
        ax = axes[0, j]
        bins = np.linspace(np.percentile(e_c, 0.5), np.percentile(e_c, 99.5), 50)
        ax.hist(e_c.ravel(), bins=bins, density=True, color=PAABD_COLORS[pa], alpha=0.55,
                label="marginal energy")
        ax.hist(de.ravel(), bins=bins, density=True, histtype="step", color=INK, lw=1.0,
                label="energy transition")
        h = res[pa]["health"]
        ax.set_title(f"{PAABD_LABELS[pa]}   E-BFMI {h.e_bfmi.min():.2f}–{h.e_bfmi.max():.2f}\n"
                     f"divergences {int(h.divergences.sum())} / {int(h.draws.sum())}")
        ax.set_yticks([])
        ax.set_xlabel("energy − mean")
        if j == 0:
            ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax = axes[1, j]
        td = ss.tree_depth.values.ravel()
        vals, cnts = np.unique(td, return_counts=True)
        ax.bar(vals, cnts, color=PAABD_COLORS[pa], width=0.7)
        ax.set_xlabel("tree depth (max allowed 10)")
        ax.set_xlim(0.5, 10.5)
        ax.text(0.02, 0.95, f"max-depth hits {int(h.max_treedepth_hits.sum())}\n"
                f"mean accept {h.mean_accept.mean():.2f}", transform=ax.transAxes,
                fontsize=6.3, va="top", color=INK)
        if j == 0:
            ax.set_ylabel("# draws")
    fig.suptitle("NUTS sampler health. Overlapping energy histograms / E-BFMI > 0.3 "
                 "indicate the momentum resampling explores the posterior efficiently",
                 fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_sampler_energy")


def fig_split(res):
    fig, axes = plt.subplots(1, len(PAABD_ORDER), figsize=(10.5, 2.9))
    for ax, pa in zip(axes, PAABD_ORDER):
        r = res[pa]
        hit = r["h01"] | r["h23"]
        ax.scatter(r["m01"][~hit], r["m23"][~hit], s=2, color=NULL_COLOR, alpha=0.35,
                   rasterized=True, lw=0)
        ax.scatter(r["m01"][hit], r["m23"][hit], s=4, color=PAABD_COLORS[pa], alpha=0.9,
                   rasterized=True, lw=0, label="hit in either half")
        lo = min(r["m01"].min(), r["m23"].min())
        hi = max(r["m01"].max(), r["m23"].max())
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.6, ls=":")
        s = r["split"]
        ax.set_title(PAABD_LABELS[pa])
        ax.text(0.03, 0.97, f"r = {s['pearson_b_mean_halves']:.5f}\n"
                f"hits {s['hits_chains01']} vs {s['hits_chains23']}\n"
                f"shared {s['hits_both_halves']} (Jaccard {s['jaccard_halves']:.3f})",
                transform=ax.transAxes, fontsize=6.3, va="top", color=INK)
        ax.set_xlabel("b̂ from chains 1–2")
        if ax is axes[0]:
            ax.set_ylabel("b̂ from chains 3–4")
            ax.legend(loc="lower right", frameon=False, fontsize=6, markerscale=2)
    fig.suptitle("Independent chain halves give the same binding estimates and hit calls "
                 f"(hit: lfsr ≤ {LFSR_THRESH}, b ≥ {BINDING_THRESH})", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig_chain_split_agreement")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {pa: analyse(pa) for pa in PAABD_ORDER}
    pd.concat([res[p]["overview"] for p in PAABD_ORDER]).to_csv(
        OUT / "convergence_overview.csv", index=False)
    pd.concat([res[p]["flagged"] for p in PAABD_ORDER]).to_csv(
        OUT / "flagged_rhat_params.csv", index=False)
    pd.concat([res[p]["health"] for p in PAABD_ORDER]).to_csv(
        OUT / "sampler_health.csv", index=False)
    pd.DataFrame([res[p]["split"] for p in PAABD_ORDER]).to_csv(
        OUT / "chain_split_hits.csv", index=False)
    fig_rhat_ess(res)
    fig_traces(res)
    fig_energy(res)
    fig_split(res)
    print(pd.concat([res[p]["overview"] for p in PAABD_ORDER]).to_string())


if __name__ == "__main__":
    main()
