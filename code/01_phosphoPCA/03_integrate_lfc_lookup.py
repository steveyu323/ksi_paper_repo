#!/usr/bin/env python3
"""Stage 03 - join MAGeCK enrichment to the barcode lookup tables.

From 3_integrate_lfc_lookup.ipynb.

Each sublibrary was screened with a different phospho-amino-acid binding
domain (pAABD) fused to one DHFR half, so a sublibrary identifies both the
reporter domain and the substrate chemistry it can read out:

    blib1 -> FHA      blib2 -> 14-3-3     blib3 -> WW     (S/T kinases)
    blib5 -> SH2                                           (Y kinases)

This stage attaches the per-barcode MAGeCK enrichment (LFC, FDR,
control_mean, ...) to the barcode's kinase and substrate identity, adds the
raw two-batch counts, and concatenates the four sublibraries into one table.

Consumes
    config.COUNT_DIR/{sample}.gene_summary.txt, {sample}.sgrna_summary.txt
        for each sample in config.BLIB_SAMPLES (written by stage 00's
        `mageck test` loop over the stage 02 *12* tables)
    config.COUNT_DIR/bliballvariant12{lib}.count.txt      raw counts
    config.Y_LOOKUP_PKL, config.ST_LOOKUP_PKL             from stage 01

Produces
    config.LFC_TABLE_PKL, config.LFC_TABLE_CSV
        one row per (barcode, pAABD): kinase, substrate, enrichment, counts
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import config

log = logging.getLogger("stage03")


def integrate_one(sample: str, paabd: str, kinase_type: str, lookup: pd.DataFrame):
    """Join one sublibrary's MAGeCK results to its barcode lookup table."""
    lib_tag = sample.split("_")[0]

    gene = pd.read_csv(config.COUNT_DIR / f"{sample}.gene_summary.txt", sep="\t")
    sgrna = pd.read_csv(config.COUNT_DIR / f"{sample}.sgrna_summary.txt", sep="\t")
    dat = pd.merge(gene, sgrna, left_on="id", right_on="Gene")

    # MAGeCK's de-novo mode also counts truncated reads; a real barcode is
    # exactly BARCODE_LEN nt.
    dat = dat.loc[dat["id"].str.len() == config.BARCODE_LEN, :]

    dat = pd.merge(dat, lookup, left_on="id", right_on="barcode_seq")
    dat = dat.loc[dat["control_mean"] > config.BARCODE_CONTROL_MEAN_MIN, :].copy()

    # NOTE: the notebook reports this fraction here, after the inner join, so
    # it is 1.0 by construction; kept because it is the figure the notebook
    # output records. The barcodes actually lost are those the merge dropped.
    captured = dat["id"].isin(lookup["barcode_seq"]).sum() / len(dat)
    log.info("%s (%s, %s): %d barcodes, %.3f captured by the lookup table",
             sample, paabd, kinase_type, len(dat), captured)

    dat["paabd"] = paabd
    dat = dat.loc[dat["kinase_type"] == kinase_type, :]
    log.info("  %d barcodes on %s kinases", len(dat), kinase_type)

    raw = pd.read_csv(
        config.COUNT_DIR / f"bliballvariant12{lib_tag}.count.txt", sep="\t"
    ).iloc[:, 1:]
    raw.columns = ["barcode_seq", "pre_1", "mtx_1", "pre_2", "mtx_2"]
    dat = pd.merge(dat, raw)
    log.info("  %d barcodes with two-batch counts", len(dat))
    return dat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    lookups = {
        "Y": pd.read_pickle(config.Y_LOOKUP_PKL),
        "ST": pd.read_pickle(config.ST_LOOKUP_PKL),
    }

    parts = [
        integrate_one(sample, paabd, kinase_type, lookups[kinase_type])
        for sample, paabd, kinase_type in zip(
            config.BLIB_SAMPLES, config.BLIB_PAABDS, config.BLIB_KINASE_TYPES
        )
    ]
    dat_all = pd.concat(parts)

    # A barcode can be sequenced as several distinct nucleotide constructs that
    # translate to the same (kinase, substrate) pair; collapse those. Done per
    # kinase type because the two libraries were screened separately.
    deduped = []
    for kinase_type in ("Y", "ST"):
        part = dat_all.loc[dat_all["kinase_type"] == kinase_type, :]
        part = part.drop_duplicates(["kinase_name", "substrate_aa", "barcode_seq", "paabd"])
        log.info("%s library: %d rows after collapsing", kinase_type, len(part))
        deduped.append(part)
    dat_all = pd.concat(deduped)

    config.LFC_TABLE_PKL.parent.mkdir(parents=True, exist_ok=True)
    dat_all.to_pickle(config.LFC_TABLE_PKL)
    dat_all.to_csv(config.LFC_TABLE_CSV)
    log.info("wrote %s and %s (%d rows, %d kinases)",
             config.LFC_TABLE_PKL.name, config.LFC_TABLE_CSV.name,
             len(dat_all), dat_all["kinase_name"].nunique())


if __name__ == "__main__":
    main()
