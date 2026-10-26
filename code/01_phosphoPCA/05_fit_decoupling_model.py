#!/usr/bin/env python3
"""Stage 05 - hierarchical negative-binomial decoupling fit (PyMC).

From 5_enrichment_analysis.ipynb.

A barcode's methotrexate enrichment mixes two things: how much folded kinase
the cell has at all, and how well that kinase engages this particular
substrate. The first moves every substrate of a kinase together, the second
does not. The model separates them:

    y_skr ~ NegBinomial(mu_skr, alpha)
    log mu_skr = log_offset_skr + g_k + delta_kr + b_sk
    b_sk       = b_raw_sk - mean_s(b_raw_sk)        (centred within kinase)

with g_k the kinase-global term (abundance and global activity), delta_kr its
per-replicate deviation, and b_sk the kinase-centred substrate binding score
that everything downstream uses. The model definition lives in
decoupling_model.py, shared with the QC refits.

One fit per reporter domain, since the four pAABD screens are separate
experiments with their own sequencing depth and their own kinase panel.

This is the expensive stage: PyMC/NUTS, 4 chains x 1000 draws after 1000
tuning steps, roughly 1-1.5 h per pAABD on 4 cores (see the recorded sampler
times in the notebook: 5526 s SH2, 5504 s FHA, 3694 s 14-3-3, 3846 s WW), and
it needs the PyMC container/environment. The posterior netCDFs are ~3 GB each
because the per-observation `mu` is stored, as in the notebook.

Consumes
    config.SCREEN_COUNT_MAPPED_CSV       from stage 04

Produces (in config.DECOUPLING_DIR, one set per pAABD)
    {FIT_PREFIX}_{paabd}.nc                  full posterior
    {FIT_PREFIX}_{paabd}.csv                 posterior-mean effects per pair
    {FIT_PREFIX}_binding_stats_{paabd}.csv   b mean/sd/lfsr per pair
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config
from decoupling_model import fit_decoupled_nb_pymc, prepare_long_table_for_pymc

log = logging.getLogger("stage05")


def summarize_decoupled_effects(idata, long: pd.DataFrame) -> pd.DataFrame:
    """Posterior-mean effects, one row per (substrate, kinase)."""
    g_mean = idata.posterior["g"].mean(dim=("chain", "draw")).to_pandas()
    delta_mean = idata.posterior["delta"].mean(dim=("chain", "draw")).to_pandas()
    b_mean = idata.posterior["b"].mean(dim=("chain", "draw")).to_pandas()

    # One kinase-level adjustment: the replicate deviations averaged over reps.
    delta_bar = delta_mean.mean(axis=1)

    out = (
        b_mean.stack().rename("binding_strength").reset_index()
        .rename(columns={"level_0": "substrate", "level_1": "kinase"})
    )
    out["kinase_global"] = out["kinase"].map(g_mean) + out["kinase"].map(delta_bar)
    out["pair_total_effect"] = out["kinase_global"] + out["binding_strength"]
    return out


def summarize_binding_significance(nc_path, hdi_prob: float = config.HDI_PROB) -> pd.DataFrame:
    """Per-pair summaries of b, including the local false sign rate.

    lfsr = min(P(b > 0), P(b < 0)) over the posterior draws: the probability
    that the sign of the reported effect is wrong. It is used instead of a
    p-value because the question is directional (does this kinase engage this
    substrate more than its average substrate?).
    """
    import arviz as az
    import xarray as xr

    idata = az.from_netcdf(nc_path)
    if not hasattr(idata, "posterior"):
        raise ValueError(f"No posterior group found in {nc_path}")

    post = idata.posterior
    if "b" not in post.data_vars:
        raise KeyError(f"'b' not in posterior of {nc_path}: {list(post.data_vars)}")

    da = post["b"]
    mean_da = da.mean(dim=("chain", "draw"))
    sd_da = da.std(dim=("chain", "draw"))
    prob_pos_da = (da > 0).mean(dim=("chain", "draw"))
    lfsr_da = xr.apply_ufunc(lambda p: np.minimum(p, 1.0 - p), prob_pos_da)
    p2_da = xr.apply_ufunc(lambda x: np.minimum(1.0, 2.0 * x), lfsr_da)

    hdi = az.hdi(da, hdi_prob=hdi_prob)
    if "hdi" not in hdi.dims:
        raise ValueError(f"Unexpected HDI output dims: {hdi.dims}")

    stats = xr.Dataset({
        "mean": mean_da,
        "sd": sd_da,
        "prob_pos": prob_pos_da,
        "lfsr": lfsr_da,
        "p_two_sided": p2_da,
    })
    stats = stats.merge(hdi.sel(hdi="lower").rename({"b": "hdi_low"}), compat="override")
    stats = stats.merge(hdi.sel(hdi="higher").rename({"b": "hdi_high"}), compat="override")

    df = stats.to_dataframe().reset_index()
    # The notebook emits only these columns; the HDI bounds are computed for
    # the merge but not carried downstream.
    return df.loc[:, ["kinase", "substrate", "mean", "sd", "lfsr"]]


def fit_one(paabd: str, screen_df: pd.DataFrame) -> None:
    import arviz as az

    log.info("=== %s ===", paabd)
    long = prepare_long_table_for_pymc(
        screen_df, paabd_value=paabd, pseudocount=config.PSEUDOCOUNT
    )
    log.info("%d observations: %d substrates x %d kinases x %d replicates",
             len(long), long["substrate"].nunique(), long["kinase"].nunique(),
             long["rep"].nunique())

    _, idata = fit_decoupled_nb_pymc(
        long,
        draws=config.NUTS_DRAWS,
        tune=config.NUTS_TUNE,
        chains=config.NUTS_CHAINS,
        target_accept=config.NUTS_TARGET_ACCEPT,
        seed=config.NUTS_SEED,
    )

    nc_path = config.fit_nc(paabd)
    az.to_netcdf(idata, nc_path)
    log.info("wrote %s", nc_path.name)

    summarize_decoupled_effects(idata, long).to_csv(config.fit_effects_csv(paabd))
    log.info("wrote %s", config.fit_effects_csv(paabd).name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paabd", choices=config.PAABDS, action="append",
                    help="fit only this pAABD (repeatable; default: all four)")
    ap.add_argument("--stats-only", action="store_true",
                    help="skip sampling and only re-derive the binding-stats "
                         "CSVs from the saved .nc files")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.DECOUPLING_DIR.mkdir(parents=True, exist_ok=True)
    paabds = args.paabd or config.PAABDS

    if not args.stats_only:
        screen_df = pd.read_csv(config.SCREEN_COUNT_MAPPED_CSV).iloc[:, 1:]
        for paabd in paabds:
            fit_one(paabd, screen_df)

    for paabd in paabds:
        stats = summarize_binding_significance(config.fit_nc(paabd))
        stats.to_csv(config.fit_binding_stats_csv(paabd))
        log.info("wrote %s (%d pairs)",
                 config.fit_binding_stats_csv(paabd).name, len(stats))


if __name__ == "__main__":
    main()
