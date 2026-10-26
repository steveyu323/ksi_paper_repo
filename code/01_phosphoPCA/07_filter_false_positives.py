#!/usr/bin/env python3
"""Stage 07 - remove false-positive substrates; write the canonical hit table.

From 7_identifiy_false_positives.ipynb.

Some substrates reconstitute DHFR without any kinase. Two independent lines of
evidence identify them:

  1. The substrate-only control libraries (blib7 for S/T, blib10 for Y) carry
     the substrate with no kinase fused. A construct that is reproducibly
     enriched there (LFC > 1.5 at FDR < 0.1) grows on its own - either because
     the reporter domain binds the unphosphorylated peptide, or because a yeast
     kinase phosphorylates it.
  2. A substrate called positive for nearly every kinase it was measured
     against is not reporting kinase specificity, whatever the control library
     says. The Y library has 8 kinases against the S/T library's 15, so the
     fraction cut is looser there (0.75 vs 0.9).

Their union is excluded, and `true_positive` - the call used from here on - is
`positive and not fp_sub`.

Consumes
    config.REDUCED_PKL                                  from stage 06
    config.COUNT_DIR/blib{7,10}_2batch_rra.sgrna_summary.txt   from stage 00

Produces
    config.REDUCED_TP_PKL        the canonical hit table (adds fp_sub,
                                 true_positive to the stage 06 columns)
    config.OUT_DIR/fp_substrates.csv
                                 the excluded substrates and why
    config.OUT_DIR/fp_promiscuity_sweep.csv
                                 promiscuity vs control-library LFC threshold;
                                 input to figures/suppfig_falsepositive.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

import config

log = logging.getLogger("stage07")


def load_control_library(sample: str) -> pd.DataFrame:
    """Per-substrate MAGeCK results for one substrate-only control library."""
    dat = pd.read_csv(config.COUNT_DIR / f"{sample}.sgrna_summary.txt", sep="\t")
    return dat.loc[dat["control_mean"] > config.BARCODE_CONTROL_MEAN_MIN, :]


def promiscuity_stats(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per substrate: how many of its measured kinases were called positive.

    Substrates with no positive call at all are absent from the result (an
    inner join on the positive counts), which is also what the notebook does.
    """
    total = pairs.groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    pos = (
        pairs.loc[pairs["positive"], :]
        .groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    )
    stats = pd.merge(pos, total, on="substrate")
    stats.columns = ["substrate", "pos_kinase", "tot_kinase"]
    stats["promiscuous_score"] = stats["pos_kinase"] / stats["tot_kinase"]
    return stats


def promiscuity_sweep(pairs: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Fraction of promiscuous substrates as the control-LFC cut is moved.

    For each threshold: take the control-library substrates enriched above it,
    and ask what fraction of them are called positive for more than half of
    their kinases. The curve is flat and low below LFC ~ 1 and turns up sharply
    around LFC ~ 2, which is what motivates the 1.5 cut.
    """
    start, stop, step = config.PROMISCUITY_SWEEP
    rows = []
    for thresh in np.arange(start, stop, step):
        fp_subs = control.loc[
            (control["FDR"] < config.FP_FDR_THRESH) & (control["LFC"] > thresh), "Gene"
        ]
        stats = promiscuity_stats(pairs.loc[pairs["substrate"].isin(fp_subs), :])
        frac = (
            (stats["promiscuous_score"] > config.PROMISCUITY_SCORE_CUT).sum() / len(stats)
            if len(stats)
            else np.nan
        )
        rows.append({"lfc_thresh": thresh, "n_substrates": len(stats), "frac_promiscuous": frac})
    return pd.DataFrame(rows)


def st_false_positives(pairs_st: pd.DataFrame, control_st: pd.DataFrame) -> tuple[list, list]:
    """S/T false-positive substrates: control-library growers, then promiscuous."""
    # NOTE: the notebook builds this mask from the unfiltered sgRNA table and
    # applies it with .loc to the control_mean-filtered one. pandas aligns the
    # boolean Series on the index, so the rows selected are exactly those of
    # the filtered table; using the filtered table directly is equivalent.
    fp_subs = control_st.loc[
        (control_st["FDR"] < config.FP_FDR_THRESH)
        & (control_st["LFC"] > config.FP_LFC_THRESH),
        "Gene",
    ]
    grower_stats = promiscuity_stats(pairs_st.loc[pairs_st["substrate"].isin(fp_subs), :])
    # Every such substrate is excluded, not only the promiscuous ones: at
    # LFC > 1.5, 76% of them are positive for more than half of their kinases.
    growers = grower_stats["substrate"].tolist()
    log.info("S/T: %d control-library growers (LFC > %g, FDR < %g), "
             "%.1f%% of them positive for > %g of their kinases",
             len(growers), config.FP_LFC_THRESH, config.FP_FDR_THRESH,
             100 * (grower_stats["promiscuous_score"] > config.PROMISCUITY_SCORE_CUT).mean(),
             config.PROMISCUITY_SCORE_CUT)

    all_stats = promiscuity_stats(pairs_st)
    promiscuous = all_stats.loc[
        all_stats["promiscuous_score"] > config.PROMISCUITY_THRESH_ST, "substrate"
    ].tolist()
    log.info("S/T: %d substrates positive for > %g of their kinases",
             len(promiscuous), config.PROMISCUITY_THRESH_ST)
    return growers, promiscuous


def y_false_positives(pairs_y: pd.DataFrame) -> list:
    """Y false-positive substrates.

    The Y control library (blib10) has no bimodal enrichment - its LFC is
    centred on zero with no high tail - so there is nothing to threshold, and
    only the promiscuity criterion is applied.
    """
    stats = promiscuity_stats(pairs_y)
    promiscuous = stats.loc[
        stats["promiscuous_score"] > config.PROMISCUITY_THRESH_Y, "substrate"
    ].tolist()
    log.info("Y: %d substrates positive for > %g of their kinases",
             len(promiscuous), config.PROMISCUITY_THRESH_Y)
    return promiscuous


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the promiscuity-vs-threshold sweep "
                         "(figure input only, ~160 passes over the hit table)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    reduced = pd.read_pickle(config.REDUCED_PKL)
    pairs_st = reduced.loc[reduced["paabd_all"] != "sh2"]
    pairs_y = reduced.loc[reduced["paabd_all"] == "sh2"]

    control_st = load_control_library(config.FP_SAMPLE_ST)

    if not args.skip_sweep:
        sweep = promiscuity_sweep(pairs_st, control_st)
        sweep_path = config.OUT_DIR / "fp_promiscuity_sweep.csv"
        sweep.to_csv(sweep_path, index=False)
        log.info("wrote %s (%d thresholds)", sweep_path.name, len(sweep))

    growers, promiscuous_st = st_false_positives(pairs_st, control_st)
    st_fp_sub = list(set(growers + promiscuous_st))
    promiscuous_y = y_false_positives(pairs_y)
    y_fp_sub = list(set(promiscuous_y))
    sty_fp_sub = st_fp_sub + y_fp_sub
    log.info("%d false-positive substrates in total (%d S/T, %d Y)",
             len(sty_fp_sub), len(st_fp_sub), len(y_fp_sub))

    fp_table = pd.DataFrame({"substrate": sty_fp_sub})
    fp_table["library"] = ["ST"] * len(st_fp_sub) + ["Y"] * len(y_fp_sub)
    fp_table["control_library_grower"] = fp_table["substrate"].isin(growers)
    fp_table["promiscuous"] = fp_table["substrate"].isin(promiscuous_st + promiscuous_y)
    fp_path = config.OUT_DIR / "fp_substrates.csv"
    fp_table.to_csv(fp_path, index=False)
    log.info("wrote %s", fp_path.name)

    reduced = reduced.copy()
    reduced["fp_sub"] = reduced["substrate"].isin(sty_fp_sub)
    reduced["true_positive"] = reduced["positive"] & (~reduced["fp_sub"])

    breakdown = reduced.groupby(["positive", "fp_sub"]).size().rename("n_ksi").reset_index()
    for _, row in breakdown.iterrows():
        log.info("positive=%s fp_sub=%s: %d KSIs",
                 row["positive"], row["fp_sub"], row["n_ksi"])

    reduced.to_pickle(config.REDUCED_TP_PKL)
    log.info("wrote %s: %d pairs, %d true positives",
             config.REDUCED_TP_PKL.name, len(reduced),
             int(reduced["true_positive"].sum()))


if __name__ == "__main__":
    main()
