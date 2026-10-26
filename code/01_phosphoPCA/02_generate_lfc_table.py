#!/usr/bin/env python3
"""Stage 02 - raw MAGeCK count tables -> per-sublibrary, two-batch count tables.

From 2_generate_lfc_table.ipynb.

Stage 00 ran MAGeCK in de-novo mode against an empty library file, so each raw
count table holds every distinct sequence the run observed, most of which are
sequencing junk. This stage sanitises those tables: it keeps the reads that
carry the expected amplicon structure, extracts the barcode (variant
libraries) or translates the substrate (substrate-only control libraries),
drops low-coverage members, splits the tables one sublibrary per file, and
merges the two sequencing batches into the `*12*` tables that the per-
sublibrary MAGeCK tests at the end of stage 00 consume.

Consumes
    config.RAW_COUNT_B1_VARIANT / _B1_FP / _B2_VARIANT / _B2_FP
    config.SUBSTRATE_REF_ST_CSV, config.SUBSTRATE_REF_Y_CSV   (coverage QC only)

Produces (in config.COUNT_DIR)
    bliballvariant.count.txt, bliballvariant2.count.txt
    blib_fp.count.txt, blib_fp_b2.count.txt
        sanitised whole-run tables
    bliballvariant{,2}blib{1,2,3,5}.count.txt
    blib_fp{,b2}blib{7,10}.count.txt
        one sublibrary per file, one batch per file
    bliballvariant12blib{1,2,3,5}.count.txt
    blib_fpb12blib{7,10}.count.txt
        the two batches merged, with pre_1/mtx_1/pre_2/mtx_2 columns

Run stage 00's `mageck test` loop after this stage; it reads the *12* tables.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from Bio.Seq import Seq

import config

log = logging.getLogger("stage02")


def sanitise_variant_table(raw_path, out_path=None) -> pd.DataFrame:
    """Keep barcode-bearing reads from a variant-library count table.

    With out_path=None the sanitised whole-run table is not written, matching
    the notebook, which only saved the ones it later re-read.
    """
    dat = pd.read_csv(raw_path, sep="\t")
    n_raw = len(dat)

    dat = dat.loc[dat["sgRNA"].str.contains(config.VARIANT_BARCODE_LEFT, regex=False), :]
    dat = dat.loc[
        dat["sgRNA"].str.contains(config.VARIANT_BARCODE_RIGHT_FILTER, regex=False), :
    ].copy()
    log.info("%s: %d/%d reads keep the barcode flanks (%.1f%%)",
             raw_path.name, len(dat), n_raw, 100 * len(dat) / n_raw)

    dat["Gene"] = dat["sgRNA"].apply(
        lambda s: s.split(config.VARIANT_BARCODE_LEFT)[1]
                   .split(config.VARIANT_BARCODE_RIGHT)[0]
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dat.to_csv(out_path, sep="\t", index=False)
        log.info("  wrote %s", out_path)
    return dat


def sanitise_subonly_table(raw_path, out_path) -> pd.DataFrame:
    """Keep and translate substrate reads from a substrate-only count table.

    The control libraries have no barcode, so the substrate's own nucleotide
    sequence is the identifier and `Gene` is its 15-aa translation.
    """
    dat = pd.read_csv(raw_path, sep="\t")

    dat = dat.loc[dat["sgRNA"].str.contains(config.SUBONLY_LEFT, regex=False), :]
    dat = dat.loc[
        dat["sgRNA"].str.contains(config.SUBONLY_RIGHT_FILTER, regex=False), :
    ].copy()
    dat["sgRNA_trimmed"] = dat["sgRNA"].apply(
        lambda s: s.split(config.SUBONLY_LEFT)[1].split(config.SUBONLY_RIGHT)[0]
    )

    n_flanked = len(dat)
    dat = dat.loc[dat["sgRNA_trimmed"].str.len() == config.SUBONLY_NT_LEN, :].copy()
    log.info("%s: %d/%d flanked reads are full length (%.1f%%)",
             raw_path.name, len(dat), n_flanked, 100 * len(dat) / n_flanked)

    dat["Gene"] = dat["sgRNA_trimmed"].apply(lambda s: str(Seq(s).translate()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dat.to_csv(out_path, sep="\t", index=False)
    log.info("  wrote %s", out_path)
    return dat


def split_sublibrary(dat: pd.DataFrame, lib: str, out_path, drop_na_gene: bool) -> None:
    """Write the (sgRNA, Gene, pre, mtx) table for one sublibrary.

    `drop_na_gene` is applied for the variant libraries only, matching the
    notebook: a barcode read whose flanks were adjacent leaves an empty Gene.
    """
    sub = dat.loc[:, ["sgRNA", "Gene", f"{lib}_pre", f"{lib}_mtx"]]
    sub = sub.loc[sub.iloc[:, 2] + sub.iloc[:, 3] > config.MIN_COUNT_SUM, :]
    if drop_na_gene:
        sub = sub.loc[~sub["Gene"].isna(), :]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, sep="\t", index=False)
    log.info("  %s: %d members -> %s", lib, len(sub), out_path.name)


def merge_batches(lib: str, batch1_path, batch2_path, out_path) -> None:
    """Inner-join the two batches of one sublibrary on (barcode, Gene)."""
    dat1 = pd.read_csv(batch1_path, sep="\t")
    dat2 = pd.read_csv(batch2_path, sep="\t")
    dat1.columns = ["sgRNA", "Gene", f"{lib}_pre_1", f"{lib}_mtx_1"]
    dat2.columns = ["sgRNA", "Gene", f"{lib}_pre_2", f"{lib}_mtx_2"]
    merged = pd.merge(dat1, dat2, on=["Gene", "sgRNA"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)
    log.info("  %s: %d members in both batches -> %s", lib, len(merged), out_path.name)


def report_design_coverage(fp_table: pd.DataFrame) -> None:
    """What fraction of the designed substrate oligos is seen in the library?

    QC only; nothing downstream depends on it. Measured on the substrate-only
    control library, which reads the substrate directly.
    """
    for label, path in (
        ("ST", config.SUBSTRATE_REF_ST_CSV),
        ("Y", config.SUBSTRATE_REF_Y_CSV),
    ):
        ref = pd.read_csv(path, sep=",")
        inserts = (
            ref["Insert Sequence"]
            .apply(lambda s: s.split(config.REF_INSERT_LEFT)[1])
            .apply(lambda s: s.split(config.REF_INSERT_RIGHT)[0])
        )
        frac = inserts.isin(fp_table["sgRNA_trimmed"]).sum() / len(inserts)
        log.info("designed %s substrates recovered: %.2f%%", label, 100 * frac)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-coverage-qc",
        action="store_true",
        help="skip the designed-oligo coverage report (needs the reference CSVs)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cd = config.COUNT_DIR

    # ---- batch 1 -------------------------------------------------------
    log.info("=== batch 1: variant libraries ===")
    var1 = sanitise_variant_table(config.RAW_COUNT_B1_VARIANT, cd / "bliballvariant.count.txt")
    for lib in config.VARIANT_LIBS:
        split_sublibrary(var1, lib, cd / f"bliballvariant{lib}.count.txt", drop_na_gene=True)

    log.info("=== batch 1: substrate-only control libraries ===")
    fp1 = sanitise_subonly_table(config.RAW_COUNT_B1_FP, cd / "blib_fp.count.txt")
    for lib in config.FP_LIBS:
        split_sublibrary(fp1, lib, cd / f"blib_fp{lib}.count.txt", drop_na_gene=False)

    if not args.skip_coverage_qc:
        report_design_coverage(fp1)

    # ---- batch 2 -------------------------------------------------------
    # Batch 2 split the libraries across two runs: blib1/2/3 came back on the
    # 240424 run, blib5 (plus both control libraries) on the 240508 run.
    log.info("=== batch 2: variant libraries blib1-3 ===")
    var2 = sanitise_variant_table(config.RAW_COUNT_B2_VARIANT, cd / "bliballvariant2.count.txt")
    for lib in ["blib1", "blib2", "blib3"]:
        split_sublibrary(var2, lib, cd / f"bliballvariant2{lib}.count.txt", drop_na_gene=True)

    log.info("=== batch 2: blib5 and the control libraries ===")
    # The 240508 run is read twice: once for its barcoded blib5 members, once
    # for its barcode-free substrate-only members.
    # NOTE: the notebook has an earlier, superseded cell that writes
    # bliballvariant2blib5.count.txt from the *batch 1* table; the cell below
    # (notebook cell 60) overwrites it with the real batch-2 blib5 counts, and
    # that is what the deposited table contains. Only the latter is kept here.
    var2b5 = sanitise_variant_table(config.RAW_COUNT_B2_FP)
    split_sublibrary(var2b5, "blib5", cd / "bliballvariant2blib5.count.txt", drop_na_gene=True)

    fp2 = sanitise_subonly_table(config.RAW_COUNT_B2_FP, cd / "blib_fp_b2.count.txt")
    for lib in config.FP_LIBS:
        split_sublibrary(fp2, lib, cd / f"blib_fpb2{lib}.count.txt", drop_na_gene=False)

    # ---- merge the batches ---------------------------------------------
    log.info("=== merging batches ===")
    for lib in config.VARIANT_LIBS:
        merge_batches(
            lib,
            cd / f"bliballvariant{lib}.count.txt",
            cd / f"bliballvariant2{lib}.count.txt",
            cd / f"bliballvariant12{lib}.count.txt",
        )
    for lib in config.FP_LIBS:
        merge_batches(
            lib,
            cd / f"blib_fp{lib}.count.txt",
            cd / f"blib_fpb2{lib}.count.txt",
            cd / f"blib_fpb12{lib}.count.txt",
        )


if __name__ == "__main__":
    main()
