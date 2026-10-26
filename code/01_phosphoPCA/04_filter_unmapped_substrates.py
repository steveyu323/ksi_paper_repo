#!/usr/bin/env python3
"""Stage 04 - drop screened substrates with no protein identity.

From 4_filter_unmapped_substrates.ipynb.

The screen recovers more distinct 15-mers than were ordered, because oligo
synthesis and yeast propagation introduce point mutations. A mutated 15-mer
cannot be traced back to a UniProt accession and phosphosite, so it carries no
biological interpretation; it is also, by construction, seen on only a handful
of barcodes. This stage keeps the (kinase, substrate) rows whose substrate is
in the reference 15-mer universe and writes the count table the decoupling
model is fit to.

NOTE on the reference universe (config.SUBSTRATE_BACKGROUND_PKL,
`substrates_background_12345.pkl`): the notebook builds it by concatenating the
15-mer/accession/phosphosite columns of the animal phosphosite table
(config.ANIMAL_PSITES_PKL), the KINBERT interaction table
(config.INTERACTION_CLASSED_PKL), the 2023-06-21 training table
(config.TRAIN_230621_PKL) and two further frames (`sub1234`, `sty_background`)
that were defined in an earlier, unsaved session of the same kernel. The file
therefore cannot be rebuilt from the notebook alone, so it is read here as a
deposited input rather than reconstructed. Its contents are also documented in
`_source_scripts/tableS13_substrate_library/config.py`.

Consumes
    config.LFC_TABLE_CSV                 from stage 03
    config.SUBSTRATE_BACKGROUND_PKL      15-mer -> (accession, site, residue)

Produces (in config.DECOUPLING_DIR)
    raw_screen_count.pkl                 counts for every screened pair
    raw_screen_count_mapped.pkl/.csv     counts for the identifiable subset
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

import config

log = logging.getLogger("stage04")

COUNT_COLS = [
    "barcode_seq", "pre_1", "mtx_1", "pre_2", "mtx_2",
    "kinase_name", "substrate_aa", "paabd",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.DECOUPLING_DIR.mkdir(parents=True, exist_ok=True)

    dat_all = pd.read_csv(config.LFC_TABLE_CSV)
    count_df = dat_all.loc[:, COUNT_COLS]
    count_df.to_pickle(config.SCREEN_COUNT_PKL)
    log.info("wrote %s (%d rows)", config.SCREEN_COUNT_PKL.name, len(count_df))

    background = pd.read_pickle(config.SUBSTRATE_BACKGROUND_PKL)
    mappable = count_df["substrate_aa"].isin(background["substrate_seq_15aa"])

    n_total = count_df["substrate_aa"].nunique()
    n_mapped = count_df.loc[mappable, "substrate_aa"].nunique()
    log.info("%d distinct substrates in the screen", n_total)
    log.info("  %d map to a UniProt accession and residue", n_mapped)
    log.info("  %d do not and are dropped", n_total - n_mapped)

    # Why dropping them is safe: an unmappable substrate is almost always a
    # private mutation on a single barcode, so it is seen with one kinase only.
    kinases_per_unmapped = (
        count_df.loc[~mappable, :]
        .drop_duplicates(["substrate_aa", "kinase_name"])
        .groupby("substrate_aa")["kinase_name"]
        .count()
    )
    log.info("  of those, %d/%d were measured against a single kinase",
             int((kinases_per_unmapped == 1).sum()), len(kinases_per_unmapped))

    count_df_mapped = count_df.loc[mappable, :]
    count_df_mapped.to_pickle(config.SCREEN_COUNT_MAPPED_PKL)
    count_df_mapped.to_csv(config.SCREEN_COUNT_MAPPED_CSV)
    log.info("wrote %s and %s (%d rows, %d dropped)",
             config.SCREEN_COUNT_MAPPED_PKL.name,
             config.SCREEN_COUNT_MAPPED_CSV.name,
             len(count_df_mapped), len(count_df) - len(count_df_mapped))


if __name__ == "__main__":
    main()
