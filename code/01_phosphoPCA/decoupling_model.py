"""
The hierarchical negative-binomial decoupling model.

Single definition shared by the production fit (05_fit_decoupling_model.py)
and by the QC refits (figures/seq_replicate_qc/),
so that every one of them samples exactly the model, priors, sampler settings
and seed that produced decoupling_out/251125_pymc_{paabd}.nc. Transcribed from
5_enrichment_analysis.ipynb cells 5 and 7.

Two changes from the notebook, neither of which affects sampling:
  * `prepare_long_table_for_pymc` takes `pre_cols` / `mtx_cols` of length 1 to
    build a single-replicate table (the notebook always passes both reps).
  * `fit_decoupled_nb_pymc(store_mu=False)` drops the per-observation `mu`
    Deterministic, which is what made the saved .nc files ~3 GB. The
    production fit keeps it (store_mu=True, the default), as the notebook did.

Model, per pAABD:
    y_skr ~ NB(mu_skr, alpha)
    log mu_skr = log_offset_skr + g_k + delta_kr + b_sk
    b_sk = b_raw_sk - mean_s(b_raw_sk)          (centered per kinase)
    g_k ~ N(0, sigma_g),  delta_kr ~ N(0, sigma_delta),  b_raw_sk ~ N(0, sigma_b)
    sigma_g ~ HN(1), sigma_delta ~ HN(0.5), sigma_b ~ HN(1), alpha ~ HN(1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def prepare_long_table_for_pymc(
    screen_df: pd.DataFrame,
    *,
    paabd_value: str,
    barcode_col: str = "barcode_seq",
    kinase_col: str = "kinase_name",
    substrate_col: str = "substrate_aa",
    paabd_col: str = "paabd",
    pre_cols: tuple[str, ...] = ("pre_1", "pre_2"),
    mtx_cols: tuple[str, ...] = ("mtx_1", "mtx_2"),
    pseudocount: float = 1.0,
):
    """
    Returns a long table with one row per (substrate, kinase, rep):
      columns: substrate, kinase, rep, pre, mtx, sf_pre, sf_mtx, log_offset
    where size factors sf_* are computed on (paabd, condition, rep) across ALL kinases/substrates.
    """
    df = screen_df[screen_df[paabd_col] == paabd_value].copy()

    # numeric + sum barcodes to substrate×kinase level (per replicate column)
    for c in list(pre_cols) + list(mtx_cols):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(pre_cols) + list(mtx_cols))

    grp = [kinase_col, substrate_col]
    agg = {c: "sum" for c in list(pre_cols) + list(mtx_cols)}
    df_sum = df.groupby(grp, as_index=False).agg(agg)

    # build long per rep
    rows = []
    for r, (pre_c, mtx_c) in enumerate(zip(pre_cols, mtx_cols), start=1):
        tmp = df_sum[[kinase_col, substrate_col, pre_c, mtx_c]].copy()
        tmp = tmp.rename(columns={kinase_col: "kinase", substrate_col: "substrate", pre_c: "pre", mtx_c: "mtx"})
        tmp["rep"] = r
        rows.append(tmp)

    long = pd.concat(rows, ignore_index=True)

    # size factors on (paabd, condition, rep): totals across all kinases/substrates
    sf_rows = []
    for r, (pre_c, mtx_c) in enumerate(zip(pre_cols, mtx_cols), start=1):
        total_pre = float(df_sum[pre_c].sum())
        total_mtx = float(df_sum[mtx_c].sum())
        sf_rows.append({"rep": r, "sf_pre": total_pre, "sf_mtx": total_mtx})
    sf = pd.DataFrame(sf_rows)

    # median-scale (optional but usually nice numerically)
    sf["sf_pre"] = sf["sf_pre"] / (np.median(sf["sf_pre"]) if np.median(sf["sf_pre"]) > 0 else 1.0)
    sf["sf_mtx"] = sf["sf_mtx"] / (np.median(sf["sf_mtx"]) if np.median(sf["sf_mtx"]) > 0 else 1.0)

    long = long.merge(sf, on="rep", how="left")

    # offsets for raw-count likelihood:
    # log(mu_raw) = log(sf_mtx) + log(pre/sf_pre + c) + ...
    long["log_offset"] = np.log(long["sf_mtx"].clip(1e-12)) + np.log((long["pre"] / long["sf_pre"]) + pseudocount)

    # keep as integers for NB
    long["pre"] = np.rint(long["pre"]).astype(int)
    long["mtx"] = np.rint(long["mtx"]).astype(int)
    return long


def fit_decoupled_nb_pymc(
    long: pd.DataFrame,
    *,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    seed: int = 0,
    store_mu: bool = True,
):
    """
    long must have columns: substrate, kinase, rep, mtx, log_offset
    Fits hierarchical NB:
      log(mu) = log_offset + g_k + delta_{k,rep} + b_{substrate,kinase_centered}
    """
    import pymc as pm
    import pytensor.tensor as pt

    # index coding
    substrates = pd.Index(sorted(long["substrate"].unique()))
    kinases = pd.Index(sorted(long["kinase"].unique()))
    reps = pd.Index(sorted(long["rep"].unique()))

    sub_idx = substrates.get_indexer(long["substrate"])
    kin_idx = kinases.get_indexer(long["kinase"])
    rep_idx = reps.get_indexer(long["rep"])

    y = long["mtx"].values.astype(int)
    log_offset = long["log_offset"].values.astype(float)

    coords = {
        "obs": np.arange(len(long)),
        "substrate": substrates.astype(str).tolist(),
        "kinase": kinases.astype(str).tolist(),
        "rep": reps.astype(int).tolist(),
    }

    with pm.Model(coords=coords) as model:
        # data containers
        sub_i = pm.Data("sub_i", sub_idx, dims="obs")
        kin_i = pm.Data("kin_i", kin_idx, dims="obs")
        rep_i = pm.Data("rep_i", rep_idx, dims="obs")
        log_off = pm.Data("log_off", log_offset, dims="obs")

        # ---- kinase-global shared across substrates ----
        sigma_g = pm.HalfNormal("sigma_g", 1.0)
        g = pm.Normal("g", 0.0, sigma=sigma_g, dims="kinase")  # kinase-global

        # ---- replicate-pair (block) deviations per kinase ----
        sigma_delta = pm.HalfNormal("sigma_delta", 0.5)
        delta = pm.Normal("delta", 0.0, sigma=sigma_delta, dims=("kinase", "rep"))

        # ---- binding term per substrate×kinase, centered per kinase ----
        sigma_b = pm.HalfNormal("sigma_b", 1.0)
        b_raw = pm.Normal("b_raw", 0.0, sigma=sigma_b, dims=("substrate", "kinase"))

        # center per kinase (mean across substrates = 0 for each kinase)
        b_centered = pm.Deterministic(
            "b",
            b_raw - pt.mean(b_raw, axis=0, keepdims=True),
            dims=("substrate", "kinase"),
        )

        # ---- NB dispersion ----
        alpha = pm.HalfNormal("alpha", 1.0)  # NB2 overdispersion

        # linear predictor for each observation
        eta = (
            log_off
            + g[kin_i]
            + delta[kin_i, rep_i]
            + b_centered[sub_i, kin_i]
        )

        mu = pt.exp(eta)
        if store_mu:
            mu = pm.Deterministic("mu", mu, dims="obs")

        # likelihood
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=y, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
        )

    return model, idata
