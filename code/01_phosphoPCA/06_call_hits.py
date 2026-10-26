#!/usr/bin/env python3
"""Stage 06 - assemble the inference table and call hits.

From 6_inspect_decoupled_binding.ipynb.

Joins the four per-pAABD fits into one table (one row per substrate x kinase x
pAABD, carrying binding_strength, kinase_global, lfsr and the raw counts), then
reduces it to one row per (substrate, kinase): a pair is called positive if
*any* reporter domain that measured it gives

    lfsr <= config.LFSR_THRESH  and  binding_strength >= config.BINDING_THRESH

Any-of rather than all-of, because the reporters are not interchangeable: a
phosphosite is only visible to the domains whose motif it matches, so a pair
measured by three domains may legitimately be positive in one.

Consumes (from stage 05, per pAABD)
    {FIT_PREFIX}_{paabd}.csv, {FIT_PREFIX}_binding_stats_{paabd}.csv
    config.SCREEN_COUNT_MAPPED_PKL       from stage 04

Produces
    config.INFERENCE_SCORES_PKL   per substrate x kinase x pAABD, with counts
    config.REDUCED_PKL            per (substrate, kinase), with the hit call
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("stage06")


def build_specificity_table() -> pd.DataFrame:
    """One row per substrate x kinase x pAABD, scores plus raw counts."""
    parts = []
    for paabd in config.PAABDS:
        effects = pd.read_csv(config.fit_effects_csv(paabd)).iloc[:, 1:]
        stats = pd.read_csv(config.fit_binding_stats_csv(paabd)).iloc[:, 1:]
        effects["paabd"] = paabd
        merged = pd.merge(effects, stats, on=["substrate", "kinase"])
        # pair_total_effect is kinase_global + binding_strength, so it adds
        # nothing to the two columns that are kept.
        merged = merged.drop(["pair_total_effect"], axis=1)
        log.info("%s: %d pairs", paabd, len(merged))
        parts.append(merged)
    specificity_df = pd.concat(parts)

    # Counts summed over the barcodes of each (kinase, substrate, pAABD).
    count_df = pd.read_pickle(config.SCREEN_COUNT_MAPPED_PKL)
    count_df = count_df.drop(["barcode_seq"], axis=1)
    count_df = count_df.groupby(["kinase_name", "substrate_aa", "paabd"]).sum().reset_index()
    count_df.columns = ["kinase", "substrate", "paabd", "pre_1", "mtx_1", "pre_2", "mtx_2"]

    specificity_df = pd.merge(specificity_df, count_df, on=["substrate", "kinase", "paabd"])
    log.info("inference table: %d rows, %d kinases, %d substrates",
             len(specificity_df), specificity_df["kinase"].nunique(),
             specificity_df["substrate"].nunique())
    return specificity_df


def call_true_positive_pairs(
    df: pd.DataFrame,
    *,
    lfsr_thresh: float,
    binding_thresh: float,
    binding_col: str = "binding_strength",
    lfsr_col: str = "lfsr",
    paabd_col: str = "paabd",
) -> pd.DataFrame:
    """Reduce to one row per (substrate, kinase) with the hit call.

    Adds, besides the call itself, which pAABD(s) passed, how many measured the
    pair, and the best binding/lfsr values - all needed downstream to decide
    whether a miss was even evaluable.
    """
    d = df.copy()
    d["_pass"] = (d[lfsr_col] <= lfsr_thresh) & (d[binding_col] >= binding_thresh)

    def agg_one(g: pd.DataFrame) -> pd.Series:
        hit_paabds = sorted(g.loc[g["_pass"], paabd_col].astype(str).unique().tolist())
        all_paabds = sorted(g[paabd_col].astype(str).unique().tolist())
        return pd.Series({
            "true_positive": bool(g["_pass"].any()),
            "paabd_hit": ";".join(hit_paabds),
            "paabd_all": ";".join(all_paabds),
            "n_paabd": int(g[paabd_col].nunique()),
            "hit_count": int(g["_pass"].sum()),
            "binding_max": float(g[binding_col].max()),
            "binding_best_hit": (
                float(g.loc[g["_pass"], binding_col].max()) if g["_pass"].any() else np.nan
            ),
            "lfsr_min": float(g[lfsr_col].min()),
            "lfsr_best_hit": (
                float(g.loc[g["_pass"], lfsr_col].min()) if g["_pass"].any() else np.nan
            ),
        })

    return (
        d.groupby(["substrate", "kinase"], as_index=False, sort=False)
         .apply(agg_one)
         .reset_index(drop=True)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    specificity_df = build_specificity_table()
    specificity_df.to_pickle(config.INFERENCE_SCORES_PKL)
    log.info("wrote %s", config.INFERENCE_SCORES_PKL.name)

    reduced_df = call_true_positive_pairs(
        specificity_df,
        lfsr_thresh=config.LFSR_THRESH,
        binding_thresh=config.BINDING_THRESH,
    )
    # Renamed from true_positive to positive: stage 07 reserves
    # `true_positive` for the call after false-positive substrates are removed.
    reduced_df = reduced_df.rename(columns={"true_positive": "positive"})

    reduced_df.to_pickle(config.REDUCED_PKL)
    log.info("wrote %s: %d pairs, %d called positive (lfsr <= %g, b >= %g)",
             config.REDUCED_PKL.name, len(reduced_df),
             int(reduced_df["positive"].sum()),
             config.LFSR_THRESH, config.BINDING_THRESH)


if __name__ == "__main__":
    main()
