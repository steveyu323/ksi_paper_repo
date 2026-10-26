#!/usr/bin/env python
"""
Refit the decoupling model on each sequencing replicate on its own.

The production fit (5_enrichment_analysis.ipynb -> decoupling_out/251125_pymc_*.nc)
pools rep 1 (sequencing batch 1, blib_fp) and rep 2 (batch 2, blib_fp_b2). Here the
same model, priors, sampler settings and seed are run on one replicate at a
time, so each replicate yields its own b_sk, g_k and hit calls with no
information shared between them. 03_compare_replicate_fits.py compares them.

Only summaries are written (the full b draws are ~1 GB per fit):
  out/refits/{paabd}_rep{r}_b.csv        per observed (substrate, kinase) pair:
                                         mean, sd, lfsr, hdi_3%, hdi_97%, r_hat, ess_bulk, ess_tail
  out/refits/{paabd}_rep{r}_global.csv   g, delta, g+delta, sigma_*, alpha summaries + diagnostics
  out/refits/{paabd}_rep{r}_sampler.csv  divergences, tree depth, BFMI per chain
  out/refits/{paabd}_rep{r}_small.nc     posterior of g, delta, sigma_*, alpha + sample_stats

Usage (conda env ksi_bayes_qc):
  python 02_refit_per_replicate.py --paabd sh2 --rep 1
  python 02_refit_per_replicate.py --all      # all 8 fits, in parallel
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))      # the pipeline package root
import config  # noqa: E402

OUT = HERE / "out" / "refits"
SCREEN_CSV = config.SCREEN_COUNT_MAPPED_CSV     # stage 04

PAABDS = ["sh2", "fha", "1433", "ww"]
REPS = [1, 2]

# Same settings as the production fit (5_enrichment_analysis.ipynb, cell 10)
DRAWS, TUNE, CHAINS, TARGET_ACCEPT, SEED = 1000, 1000, 4, 0.9, 1


def run_one(paabd: str, rep: int) -> None:
    import arviz as az
    import numpy as np
    import pandas as pd

    # the production model, shared with 05_fit_decoupling_model.py
    from decoupling_model import fit_decoupled_nb_pymc, prepare_long_table_for_pymc

    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{paabd}_rep{rep}"
    screen_df = pd.read_csv(SCREEN_CSV).iloc[:, 1:]
    long = prepare_long_table_for_pymc(
        screen_df, paabd_value=paabd,
        pre_cols=(f"pre_{rep}",), mtx_cols=(f"mtx_{rep}",), pseudocount=1.0,
    )
    print(f"[{tag}] {len(long)} observations, "
          f"{long.kinase.nunique()} kinases, {long.substrate.nunique()} substrates", flush=True)

    t0 = time.time()
    _, idata = fit_decoupled_nb_pymc(
        long, draws=DRAWS, tune=TUNE, chains=CHAINS,
        target_accept=TARGET_ACCEPT, seed=SEED, store_mu=False,
    )
    print(f"[{tag}] sampled in {time.time() - t0:.0f} s", flush=True)

    post = idata.posterior

    # ---- b for observed pairs only (unobserved grid cells are pure prior) ----
    obs_pairs = long[["substrate", "kinase"]].drop_duplicates()
    b = post["b"].stack(pair=("substrate", "kinase"))
    keep = pd.MultiIndex.from_frame(obs_pairs)
    b = b.sel(pair=keep).transpose("chain", "draw", "pair")
    b_summ = az.summary(az.convert_to_dataset({"b": b.values}), kind="all", round_to="none")
    p_pos = (b > 0).mean(("chain", "draw")).values
    b_df = pd.DataFrame({
        "substrate": obs_pairs["substrate"].values,
        "kinase": obs_pairs["kinase"].values,
        "mean": b_summ["mean"].values,
        "sd": b_summ["sd"].values,
        "lfsr": np.minimum(p_pos, 1 - p_pos),
        "hdi_3%": b_summ["hdi_3%"].values,
        "hdi_97%": b_summ["hdi_97%"].values,
        "r_hat": b_summ["r_hat"].values,
        "ess_bulk": b_summ["ess_bulk"].values,
        "ess_tail": b_summ["ess_tail"].values,
    })
    b_df.to_csv(OUT / f"{tag}_b.csv", index=False)

    # ---- kinase-level and hyperparameters ----
    post["g_plus_delta"] = post["g"] + post["delta"].mean("rep")
    glob = az.summary(idata, var_names=["g", "delta", "g_plus_delta", "sigma_g",
                                        "sigma_delta", "sigma_b", "alpha"],
                      kind="all", round_to="none")
    glob.to_csv(OUT / f"{tag}_global.csv")

    # ---- sampler health ----
    ss = idata.sample_stats
    bfmi = az.bfmi(idata)
    pd.DataFrame({
        "chain": np.arange(post.sizes["chain"]),
        "divergences": ss["diverging"].sum("draw").values,
        "max_treedepth_hits": ss["reached_max_treedepth"].sum("draw").values
            if "reached_max_treedepth" in ss else np.nan,
        "mean_tree_depth": ss["tree_depth"].mean("draw").values,
        "mean_accept": ss["acceptance_rate"].mean("draw").values,
        "bfmi": bfmi,
    }).to_csv(OUT / f"{tag}_sampler.csv", index=False)

    small = az.InferenceData(
        posterior=post[["g", "delta", "sigma_g", "sigma_delta", "sigma_b", "alpha"]],
        sample_stats=ss,
    )
    small.to_netcdf(OUT / f"{tag}_small.nc")
    print(f"[{tag}] done", flush=True)


def run_all() -> None:
    """Launch the 8 fits as separate processes (4 chains each)."""
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    procs = []
    for paabd in PAABDS:
        for rep in REPS:
            log = open(OUT / f"{paabd}_rep{rep}.log", "w")
            p = subprocess.Popen(
                [sys.executable, __file__, "--paabd", paabd, "--rep", str(rep)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            procs.append((paabd, rep, p))
    for paabd, rep, p in procs:
        rc = p.wait()
        print(f"{paabd} rep{rep}: exit {rc}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paabd", choices=PAABDS)
    ap.add_argument("--rep", type=int, choices=REPS)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all:
        run_all()
    elif a.paabd and a.rep:
        run_one(a.paabd, a.rep)
    else:
        ap.error("give --paabd and --rep, or --all")
