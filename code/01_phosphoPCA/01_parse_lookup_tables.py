#!/usr/bin/env python3
"""Stage 01 - PacBio CCS reads -> barcode -> (kinase, substrate) lookup tables.

From 1_parse_lookup_tables.ipynb.

Each library member is a barcode plus a kinase-substrate fusion on one
plasmid. The Illumina screen (stages 00/02) only ever reads the barcode, so the
assignment of barcode to construct has to come from long reads of the same
plasmid pool. This stage reads the PacBio HiFi FASTA of each library, cuts the
barcode and the kinase-substrate region out of every read by their flanking
sequences, translates them, and reduces the result to one (kinase, substrate)
assignment per barcode.

Consumes
    config.PACBIO_Y_FASTA, config.PACBIO_ST_FASTA   PacBio HiFi reads
    config.BLIB_KINASES_SRC_CSV                     the 23 screened kinases

Produces (in config.LOOKUP_DIR)
    y_pacbio_lookup_raw.pkl / st_pacbio_lookup_raw.pkl
        one row per read that contains the whole construct
    y.lookup.corrected.pkl / st.lookup.corrected.pkl
        one row per barcode, joined to the kinase table
    blib_kinases_with_acc.csv
        the kinase table with the DYRK4 sequence fix applied
"""

from __future__ import annotations

import argparse
import gzip
import logging
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm

import config

log = logging.getLogger("stage01")


def make_pacbio_lookup(fasta_path: Path, out_path: Path) -> pd.DataFrame:
    """Cut barcode / substrate / kinase out of every usable PacBio read."""
    log.info("reading %s", fasta_path)

    identifiers, lengths, seqs = [], [], []
    with gzip.open(fasta_path, "rt") as handle:
        for rec in tqdm(SeqIO.parse(handle, "fasta")):
            identifiers.append(rec.id)
            lengths.append(len(rec.seq))
            seqs.append(str(rec.seq))

    # Reads come off the instrument in either orientation.
    processed = [
        str(Seq(s).reverse_complement()) if config.RC_TRIGGER in s else s
        for s in seqs
    ]
    dat = pd.DataFrame(
        {"identifier": identifiers, "length": lengths, "seq": processed}
    )
    log.info("  %d reads", len(dat))

    # Require every flank, so that both regions can be cut unambiguously.
    for flank in (
        config.BARCODE_LEFT,
        config.BARCODE_RIGHT,
        config.KS_LEFT,
        config.KS_RIGHT,
    ):
        dat = dat.loc[dat["seq"].str.contains(flank, regex=False), :]
    log.info("  %d reads with all four flanks", len(dat))

    dat = dat.copy()
    dat["barcode_seq"] = dat["seq"].apply(
        lambda s: s.split(config.BARCODE_LEFT)[1].split(config.BARCODE_RIGHT)[0]
    )
    dat["ksregion"] = dat["seq"].apply(
        lambda s: s.split(config.KS_LEFT)[1].split(config.KS_RIGHT)[0]
    )

    dat = dat.loc[dat["ksregion"].str.contains(config.LINKER, regex=False), :].copy()
    dat["substrate"] = dat["ksregion"].apply(lambda s: s.split(config.LINKER)[0])
    dat["kinase"] = dat["ksregion"].apply(lambda s: s.split(config.LINKER)[1])
    log.info("  %d reads with the substrate-kinase linker", len(dat))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dat.to_pickle(out_path)
    log.info("  wrote %s", out_path)
    return dat


def load_kinase_table() -> pd.DataFrame:
    """The 23 screened kinase constructs, with the DYRK4 sequence corrected."""
    blib = pd.read_csv(config.BLIB_KINASES_SRC_CSV, header=None)
    blib.columns = ["kinase_type", "kinase_name", "kinase_aa", "kinase_acc"]
    blib.loc[blib["kinase_name"] == "DYRK4_HUMAN", "kinase_aa"] = config.DYRK4_KINASE_AA
    return blib


def build_canonical_lookup(
    df: pd.DataFrame, kinase_table: pd.DataFrame, out_path: Path
) -> pd.DataFrame:
    """Reduce a read-level lookup to one (kinase, substrate) call per barcode."""
    log.info("%d reads in", len(df))

    df = df.loc[df["barcode_seq"].str.len() == config.BARCODE_LEN, :].copy()

    df["substrate_aa"] = df["substrate"].apply(lambda s: str(Seq(s).translate()))
    df = df.loc[df["substrate_aa"].str.len() == config.SUBSTRATE_AA_LEN, :]
    df = df.loc[~df["substrate_aa"].str.contains("*", regex=False), :].copy()

    # The kinase ORF must end in its stop codon and carry no internal one, so
    # that the read really encodes the full-length kinase domain.
    df["kinase_aa"] = df["kinase"].apply(lambda s: str(Seq(s).translate()))
    df = df.loc[df["kinase_aa"].str[-1] == "*", :].copy()
    df["kinase_aa"] = df["kinase_aa"].str[:-1]
    df = df.loc[~df["kinase_aa"].str.contains("*", regex=False), :].copy()

    n_barcodes = df["barcode_seq"].nunique()
    log.info(
        "%d reads over %d barcodes (%.2f reads/barcode) after length and "
        "nonsense filters",
        len(df),
        n_barcodes,
        len(df) / n_barcodes,
    )

    # A barcode can appear on several constructs (recombination during library
    # construction, or PacBio chimeras). Keep it only if one assignment
    # dominates its reads, and use that assignment.
    pair_counts = (
        df.groupby(["barcode_seq", "kinase_aa", "substrate_aa"], dropna=False)
        .size()
        .rename("n_reads")
        .reset_index()
    )
    pair_counts["total_reads"] = pair_counts.groupby("barcode_seq")["n_reads"].transform("sum")
    pair_counts["frac"] = pair_counts["n_reads"] / pair_counts["total_reads"]

    dominant = pair_counts.loc[pair_counts.groupby("barcode_seq")["frac"].idxmax()].copy()
    keep = dominant.loc[
        dominant["frac"] >= config.DOMINANT_FRAC_MIN,
        ["barcode_seq", "kinase_aa", "substrate_aa"],
    ]

    df = df.merge(keep, on=["barcode_seq", "kinase_aa", "substrate_aa"], how="inner")
    df = df.drop_duplicates(["barcode_seq", "kinase_aa", "substrate_aa"])

    # Left join: barcodes whose kinase is not one of the screened constructs
    # keep their sequence but get no name, and are dropped in stage 03.
    df = df.merge(kinase_table, on="kinase_aa", how="left")

    assert df["barcode_seq"].is_unique, "barcode -> construct is not one-to-one"
    log.info(
        "%d barcodes with a dominant (>= %.0f%%) assignment",
        df["barcode_seq"].nunique(),
        100 * config.DOMINANT_FRAC_MIN,
    )
    log.info(
        "%d of them map to a screened kinase",
        int(df["kinase_name"].notna().sum()),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(out_path)
    log.info("wrote %s", out_path)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--skip-parse",
        action="store_true",
        help="reuse the existing *_pacbio_lookup_raw.pkl instead of re-reading "
             "the PacBio FASTA files (the slow part)",
    )
    ap.add_argument(
        "--library",
        choices=["y", "st", "both"],
        default="both",
        help="which library to process (default: both)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.LOOKUP_DIR.mkdir(parents=True, exist_ok=True)

    kinase_table = load_kinase_table()
    kinase_table.to_csv(config.BLIB_KINASES_CSV)
    log.info("wrote %s (%d kinase constructs)", config.BLIB_KINASES_CSV, len(kinase_table))

    libs = ["y", "st"] if args.library == "both" else [args.library]
    specs = {
        "y": (config.PACBIO_Y_FASTA, config.Y_LOOKUP_RAW_PKL, config.Y_LOOKUP_PKL),
        "st": (config.PACBIO_ST_FASTA, config.ST_LOOKUP_RAW_PKL, config.ST_LOOKUP_PKL),
    }

    for lib in libs:
        fasta, raw_pkl, corrected_pkl = specs[lib]
        log.info("--- %s library ---", lib.upper())
        if args.skip_parse:
            raw = pd.read_pickle(raw_pkl)
            log.info("reusing %s (%d reads)", raw_pkl, len(raw))
        else:
            raw = make_pacbio_lookup(fasta, raw_pkl)
        build_canonical_lookup(raw, kinase_table, corrected_pkl)


if __name__ == "__main__":
    main()
