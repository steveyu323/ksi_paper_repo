#!/usr/bin/env bash
# Stage 00 - Illumina reads -> barcode count tables (phospho-PCA screen)
#
# This stage was never committed as a script; it was run interactively in a
# container. It is reconstructed here verbatim from the MAGeCK run logs that
# survive next to each sequencing run (the "Parameters:" line of
# <run>/00_fastq/*.log and <run>/02_fastp_out/*.log), so the commands below are
# the ones that actually produced the deposited count tables.
#
# Provenance of the reconstruction:
#   sequencing_data_231204/00_fastq/bliballvariant.log
#   sequencing_data_231217/02_fastp_out/blib_fp.log
#   sequencing_data_240424/00_fastq/bliballvariant.log
#   sequencing_data_240508/00_fastq/blib_fp_b2.log
#   processed_count_table/*_rra.log            (the mageck test calls)
#
# Tool versions, as recorded in those logs:
#   MAGeCK v0.5.9.5   (run as /opt/conda/bin/mageck, python3.10 conda env)
#   fastp             (merge mode; version string is not retained in the
#                      HTML reports, so it is left unpinned here)
#
# Two things about this stage are easy to get wrong when re-running it:
#   * -l points at an EMPTY library csv (empty_library.csv, 0 bytes). MAGeCK
#     therefore runs in de-novo mode and counts every distinct sequence it
#     observes, rather than matching against a designed barcode whitelist.
#     The barcode -> (kinase, substrate) assignment happens later, in stage 01,
#     against the PacBio lookup table. Do not substitute a real library file.
#   * cutadapt was NOT used for the screen runs. Each run directory has an
#     03_cutadapt_out/, but it is empty for every phospho-PCA run; adapter
#     removal is handled inside fastp's merge. (The only populated
#     03_cutadapt_out belongs to the TBK1 pilot run, which is out of scope.)
#
# Library key, Illumina side. Each blib number is one PAABD reader library:
#
#   blib1   FHA       Ser/Thr arm
#   blib2   14-3-3    Ser/Thr arm
#   blib3   WW        Ser/Thr arm
#   blib5   Src-SH2   Tyr arm          (the Tyr arm was screened against SH2 only)
#   blib7   substrate-only false-positive control
#   blib10  substrate-only false-positive control
#   *_pre   pre-selection
#   *_mtx   methotrexate-selected
#
# Authoritative source: 3_integrate_lfc_lookup.ipynb, which pairs
#   blib_sample_ls = [blib1, blib2, blib3, blib5]
#   paabds         = [fha,   1433,  ww,    sh2 ]
#   kinase_type    = [ST,    ST,    ST,    Y   ]
#
# WARNING - the blib numbers mean something DIFFERENT on the PacBio side.
# The PacBio tube labels were swapped relative to the Illumina numbering, and
# 1_parse_lookup_tables.ipynb says so in its own comments:
#
#   blib1-ksb.CY  ("labeled reversed so Y")   -> y_pacbio_lookup_raw.pkl
#   blib5-ksb.CY  ("labeled reversed so ST")  -> st_pacbio_lookup_raw.pkl
#
# So Illumina blib1 is a Ser/Thr library while PacBio blib1 is the Tyr
# library. A blib number alone is ambiguous across platforms; always say which
# platform you mean. The supplementary table's "ST (blib5) / Y (blib1)" labels
# follow the PacBio convention and are correct for the lookup tables.
#
# Usage:  ./00_reads_to_counts.sh <run_root>
# where <run_root> contains the four sequencing_data_* / 30-* run directories.

set -euo pipefail

RUN_ROOT="${1:?usage: $0 <run_root>}"
EMPTY_LIB="${EMPTY_LIB:-$RUN_ROOT/empty_library.csv}"

# MAGeCK's de-novo mode is driven by an empty library file.
: > "$EMPTY_LIB"

fastp_merge() {
    # fastp_merge <out_dir> <sample> <R1> <R2>
    local out="$1" s="$2" r1="$3" r2="$4"
    mkdir -p "$out"
    fastp -i "$r1" -I "$r2" \
          --merge --merged_out "$out/${s}_merged.fastq.gz" \
          -h "$out/${s}.html" -j "$out/${s}.json"
}

count() {
    # count <work_dir> <name> <sample_labels> <merged fastq...>
    local work="$1" name="$2" labels="$3"; shift 3
    ( cd "$work" && mageck count \
        -l "$EMPTY_LIB" -n "$name" --pdf-report --trim-5 0 \
        --sample-label "$labels" --fastq "$@" )
}

# ---------------------------------------------------------------------------
# Replicate 1 (batch 1)
# ---------------------------------------------------------------------------

# Variant libraries. Samples 26227-26234, interleaved pre/mtx per sublibrary.
R1_DIR="$RUN_ROOT/sequencing_data_231204"
for s in 26227_S5 26228_S6 26229_S7 26230_S8 26231_S9 26232_S10 26233_S11 26234_S12; do
    fastp_merge "$R1_DIR/02_fastp_out" "$s" \
        "$R1_DIR/00_fastq/${s}_R1_001.fastq.gz" "$R1_DIR/00_fastq/${s}_R2_001.fastq.gz"
done
count "$R1_DIR/00_fastq" bliballvariant \
    blib1_pre,blib1_mtx,blib2_pre,blib2_mtx,blib3_pre,blib3_mtx,blib5_pre,blib5_mtx \
    ../02_fastp_out/26227_S5_merged.fastq.gz ../02_fastp_out/26228_S6_merged.fastq.gz \
    ../02_fastp_out/26229_S7_merged.fastq.gz ../02_fastp_out/26230_S8_merged.fastq.gz \
    ../02_fastp_out/26231_S9_merged.fastq.gz ../02_fastp_out/26232_S10_merged.fastq.gz \
    ../02_fastp_out/26233_S11_merged.fastq.gz ../02_fastp_out/26234_S12_merged.fastq.gz

# Substrate-only false-positive control libraries. Note this run was counted
# from inside 02_fastp_out, and the label order is mtx-before-pre.
FP1_DIR="$RUN_ROOT/sequencing_data_231217"
count "$FP1_DIR/02_fastp_out" blib_fp \
    blib10_mtx,blib10_pre,blib7_mtx,blib7_pre \
    ../02_fastp_out/blib10_mtx_amp_S4_merged.fastq.gz \
    ../02_fastp_out/blib10_pre_amp_S3_merged.fastq.gz \
    ../02_fastp_out/blib7_mtx_amp_S2_merged.fastq.gz \
    ../02_fastp_out/blib7_pre_amp_S1_merged.fastq.gz

# ---------------------------------------------------------------------------
# Replicate 2 (batch 2)
# ---------------------------------------------------------------------------

# Variant libraries. All pre samples first, then all mtx samples.
R2_DIR="$RUN_ROOT/sequencing_data_240424"
count "$R2_DIR/00_fastq" bliballvariant \
    blib1_pre,blib2_pre,blib3_pre,blib1_mtx,blib2_mtx,blib3_mtx \
    ../02_fastp_out/blib1_pre_amp_merged.fastq.gz ../02_fastp_out/blib2_pre_amp_merged.fastq.gz \
    ../02_fastp_out/blib3_pre_amp_merged.fastq.gz ../02_fastp_out/blib1_mtx_amp_merged.fastq.gz \
    ../02_fastp_out/blib2_mtx_amp_merged.fastq.gz ../02_fastp_out/blib3_mtx_amp_merged.fastq.gz

# Ser/Thr variant library plus both false-positive controls, replicate 2.
FP2_DIR="$RUN_ROOT/sequencing_data_240508"
count "$FP2_DIR/00_fastq" blib_fp_b2 \
    blib5_pre,blib5_mtx,blib7_pre,blib7_mtx,blib10_pre,blib10_mtx \
    ../02_fastp_out/26528_S104_merged.fastq.gz ../02_fastp_out/26529_S105_merged.fastq.gz \
    ../02_fastp_out/26530_S106_merged.fastq.gz ../02_fastp_out/26531_S107_merged.fastq.gz \
    ../02_fastp_out/26532_S108_merged.fastq.gz ../02_fastp_out/26533_S109_merged.fastq.gz

# ---------------------------------------------------------------------------
# Per-sublibrary enrichment tests
# ---------------------------------------------------------------------------
# Run in the directory holding the per-sublibrary count tables. The *12*
# tables are the two batches merged (stage 02 writes them); the single-batch
# tables give the per-replicate results used in the reproducibility note.

COUNT_DIR="${COUNT_DIR:-$RUN_ROOT/processed_count_table}"

for lib in 1 2 3 5; do
    ( cd "$COUNT_DIR" && mageck test \
        -k "bliballvariant12blib${lib}.count.txt" \
        -t "blib${lib}_mtx_1,blib${lib}_mtx_2" \
        -c "blib${lib}_pre_1,blib${lib}_pre_2" \
        -n "blib${lib}_2batch_rra" --remove-zero both --remove-zero-threshold 0 )

    ( cd "$COUNT_DIR" && mageck test \
        -k "bliballvariantblib${lib}.count.txt" \
        -t "blib${lib}_mtx" -c "blib${lib}_pre" \
        -n "blib${lib}_rra" --remove-zero both --remove-zero-threshold 0 )
done

for lib in 7 10; do
    ( cd "$COUNT_DIR" && mageck test \
        -k "blib_fpb12blib${lib}.count.txt" \
        -t "blib${lib}_mtx_1,blib${lib}_mtx_2" \
        -c "blib${lib}_pre_1,blib${lib}_pre_2" \
        -n "blib${lib}_2batch_rra" --remove-zero both --remove-zero-threshold 0 )

    ( cd "$COUNT_DIR" && mageck test \
        -k "blib_fpblib${lib}.count.txt" \
        -t "blib${lib}_mtx" -c "blib${lib}_pre" \
        -n "blib${lib}_rra" --remove-zero both --remove-zero-threshold 0 )
done
