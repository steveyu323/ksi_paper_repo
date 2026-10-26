"""Paths and analysis constants for the phospho-PCA pipeline.

Every path the pipeline reads or writes is named here; no stage script
hard-codes one. The original notebooks ran inside a container started as

    docker run -v <host data dir>:${DATA_MOUNT} \
               -v <host data dir 2>:${DATA_MOUNT2} ... pymc5-jupyter

with the working directory set to `ksi_analysis_2025/`, so they contain a mix
of container-absolute paths (`${KSI_DATA_ROOT}/...`) and relative ones
(`./decoupling_out/...`, `../kinase_ml/data/...`). Those are replaced by the
four environment variables below.

One wrinkle worth knowing when checking this file against the notebooks:
`2_generate_lfc_table.ipynb` writes some tables to
`${DATA_MOUNT}/<user>/processed_count_table/` and others to
`${DATA_MOUNT}/processed_count_table/`. Those are the same directory seen
through two different bind mounts (the mount root in the earlier sessions, a
per-user subdirectory of it in the later ones); on disk there is only one count
table directory, and both spellings collapse to COUNT_DIR here. The same is
true of the sequencing run directories: the notebook's `sequencing_data_1204`
and `sequencing_data_1217` are today's `sequencing_data_231204` and
`sequencing_data_231217`.

Environment variables
---------------------
KSI_DATA_ROOT
    Sequencing and PacBio data root; the container's `${KSI_DATA_ROOT}`.
    Holds the raw run directories, `processed_count_table/`, and the two
    designed-substrate reference CSVs.
    No usable default: set this. The template value will not resolve.
KSI_REPO_ROOT
    Analysis code/data root; the container's working-tree parent. Holds
    `ksi_analysis_2025/`, `kinase_ml/` and `screening_analysis/`. Read only.
    (Same variable name and default as
    `_source_scripts/tableS13_substrate_library/config.py`.)
    No usable default: set this. The template value will not resolve.
KSI_COUNT_DIR
    MAGeCK count/test output directory written by stage 00 and stage 02.
    Default: $KSI_DATA_ROOT/processed_count_table
KSI_OUT_DIR
    Where this pipeline writes everything it produces. The default is a fresh
    directory; nothing in it is read by any other part of the package.
    Default: <package>/data/results/phosphoPCA/rerun

    Do not point this at <package>/data/results/phosphoPCA itself, nor at
    $KSI_REPO_ROOT/ksi_analysis_2025. The first is the deposited-results view
    and is made of symlinks into the second, which is the original, read-only
    analysis directory: writing there would overwrite the deposited artifacts
    through those symlinks. To re-run against the deposited inputs without
    touching them, copy the inputs a stage needs into a fresh KSI_OUT_DIR
    first (`cp -L`), or set KSI_OUT_DIR to a scratch directory and start from
    the stage whose inputs you have.
KSI_FIG_DIR
    Figure output directory. Default: $KSI_OUT_DIR/figures
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent.parent          # .../ksi_paper_package

DATA_ROOT = Path(os.environ.get("KSI_DATA_ROOT", "/path/to/sequencing_data_root"))

REPO_ROOT = Path(
    os.environ.get(
        "KSI_REPO_ROOT",
        # template only - set KSI_REPO_ROOT to the directory holding
        # ksi_analysis_2025/ and kinase_ml/
        "/path/to/analysis_repo_root",
    )
)

ANALYSIS_DIR = REPO_ROOT / "ksi_analysis_2025"      # notebook working directory
KINASE_ML_DATA = REPO_ROOT / "kinase_ml" / "data"   # KINBERT corpus + predictions
PLANNING_DIR = REPO_ROOT / "kinase_ml" / "code" / "wetlab_planning"
SCREENING_DIR = REPO_ROOT / "screening_analysis"

COUNT_DIR = Path(os.environ.get("KSI_COUNT_DIR", DATA_ROOT / "processed_count_table"))

OUT_DIR = Path(
    os.environ.get(
        "KSI_OUT_DIR", PACKAGE_ROOT / "data" / "results" / "phosphoPCA" / "rerun"
    )
)
FIG_DIR = Path(os.environ.get("KSI_FIG_DIR", OUT_DIR / "figures"))

# Sub-directories of OUT_DIR, named as the notebooks named them.
LOOKUP_DIR = OUT_DIR / "lookup_tables"
DECOUPLING_DIR = OUT_DIR / "decoupling_out"
DARK_KINASE_OUT = OUT_DIR / "dark_kinase_out"

# --------------------------------------------------------------------------
# Stage 01 - PacBio barcode -> (kinase, substrate) lookup tables
# --------------------------------------------------------------------------
# blib1 is the Tyr (Y) variant library, blib5 the Ser/Thr (ST) one. The run
# directory names are reversed relative to the library they carry, which is why
# the notebook labels them "labeled reversed".
PACBIO_Y_FASTA = (
    DATA_ROOT / "20231106_Yu_C" / "blib1-ksb.CY" / "CCS_12462"
    / "m64076_231104_004250.hifi_reads.fasta.gz"
)
PACBIO_ST_FASTA = (
    DATA_ROOT / "20231106_Yu_C" / "blib5-ksb.CY" / "CCS_12461"
    / "m64076_231104_154224.hifi_reads.fasta.gz"
)

# The 23 screened kinase constructs: type, name, kinase-domain aa, UniProt acc.
BLIB_KINASES_SRC_CSV = SCREENING_DIR / "blib_kinases_with_acc.csv"

Y_LOOKUP_RAW_PKL = LOOKUP_DIR / "y_pacbio_lookup_raw.pkl"
ST_LOOKUP_RAW_PKL = LOOKUP_DIR / "st_pacbio_lookup_raw.pkl"
Y_LOOKUP_PKL = LOOKUP_DIR / "y.lookup.corrected.pkl"
ST_LOOKUP_PKL = LOOKUP_DIR / "st.lookup.corrected.pkl"
BLIB_KINASES_CSV = LOOKUP_DIR / "blib_kinases_with_acc.csv"

# Construct anatomy. Reads containing RC_TRIGGER are reverse-complemented
# first; the barcode and the kinase-substrate region are then cut out between
# their flanks, and LINKER splits that region into substrate | kinase.
RC_TRIGGER = "CGTACGCACAATTC"
BARCODE_LEFT = "TATCCGGCTCGTATGTTGTGTGGAATTGTGCGTACG"
BARCODE_RIGHT = "CCTAGGCAGCTATGACCATGATT"
KS_LEFT = "GGCGGTGGCGGGTCTGGTGGAGGAGGGTCAGGTGGTGGCGGCTCGGGAGGTGGAGCTAGC"
KS_RIGHT = "AAGCTTATTAGTTATGTCAC"
LINKER = "GGTGGATCAGGTGGTGGTGGTTCTGGTGGC"

BARCODE_LEN = 19          # designed barcode length
SUBSTRATE_AA_LEN = 15     # designed substrate window length
# A barcode is kept only if one (kinase, substrate) assignment accounts for at
# least this fraction of its reads; that assignment becomes the barcode's.
DOMINANT_FRAC_MIN = 0.90

# 2025-12-10 fix: the DYRK4 kinase-domain sequence in
# blib_kinases_with_acc.csv was wrong, so DYRK4 barcodes failed to join to the
# kinase table. Patched in stage 01 before the join.
DYRK4_KINASE_AA = (
    "YEVLETIGKGSFGQVAKCLDHKNNELVALKIIRNKKRFHQQALMELKILEALRKKDKDNTYNVVHMKDFFYFRN"
    "HFCITFELLGINLYELMKNNNFQGFSLSIVRRFTLSVLKCLQMLSVEKIIHCDLKPENIVLYQKGQASVKVIDF"
    "GSSCYEHQKVYTYIQSRFYRSPEVILGHPYDVAIDMWSLGCITAELYTGYPLFPGENEVEQLACIMEVLGLPPA"
    "GFIQTASRRQTFFDSKGFPKNITNNRGKKRYPDSKDLTMVLKTYDTSFLDFLRRCLVWEPSLRMTPDQALKHAWI"
)

# --------------------------------------------------------------------------
# Stage 02 - MAGeCK count tables -> per-sublibrary, two-batch count tables
# --------------------------------------------------------------------------
# Raw de-novo MAGeCK count tables, as written by stage 00.
RAW_COUNT_B1_VARIANT = DATA_ROOT / "sequencing_data_231204" / "00_fastq" / "bliballvariant.count.txt"
RAW_COUNT_B1_FP = DATA_ROOT / "sequencing_data_231217" / "02_fastp_out" / "blib_fp.count.txt"
RAW_COUNT_B2_VARIANT = DATA_ROOT / "sequencing_data_240424" / "00_fastq" / "bliballvariant.count.txt"
RAW_COUNT_B2_FP = DATA_ROOT / "sequencing_data_240508" / "00_fastq" / "blib_fp_b2.count.txt"

# Designed substrate oligos, used only for the library-coverage QC print.
SUBSTRATE_REF_ST_CSV = DATA_ROOT / "substrates_st_230313.csv"
SUBSTRATE_REF_Y_CSV = DATA_ROOT / "substrates_y_230313.csv"

# Barcode-amplicon flanks in the variant libraries (barcode sits between them).
VARIANT_BARCODE_LEFT = "CGTACG"
VARIANT_BARCODE_RIGHT = "CCTAGG"
# The notebook pre-filters on 'GCCTAGG' (one base more) and then splits on
# 'CCTAGG'; both filters are kept, in that order.
VARIANT_BARCODE_RIGHT_FILTER = "GCCTAGG"

# Substrate-only (false-positive control) libraries carry no barcode, so the
# substrate itself is read out between these flanks and translated.
SUBONLY_LEFT = "GGGAGGTGGAGCTAGC"
SUBONLY_RIGHT_FILTER = "TGGATCAGGTGGTTAGAAGCTTATTAGTT"
SUBONLY_RIGHT = "GGTGGATCAGGTGGTTAGAAGCTTATTAGTT"
SUBONLY_NT_LEN = 45       # 15 aa

# Designed-substrate reference flanks, for the coverage QC only.
REF_INSERT_LEFT = "AGGAGGGTCAGGTGGTGGCGGCTCGGGAGGTGGAGCTAGC"
REF_INSERT_RIGHT = "GGTGGATCAGGTGGTGGTGGTTCTGGTGGC"

# A barcode (or substrate) is kept in a sublibrary only if pre + mtx > this.
MIN_COUNT_SUM = 100

VARIANT_LIBS = ["blib1", "blib2", "blib3", "blib5"]   # 1/2/3 = Y, 5 = ST
FP_LIBS = ["blib7", "blib10"]                         # substrate-only controls

# --------------------------------------------------------------------------
# Stage 03 - join MAGeCK enrichment to the lookup tables
# --------------------------------------------------------------------------
# One MAGeCK test per sublibrary (two batches pooled), mapped to its reporter
# domain and to the library whose lookup table carries its barcodes.
BLIB_SAMPLES = ["blib1_2batch_rra", "blib2_2batch_rra", "blib3_2batch_rra", "blib5_2batch_rra"]
BLIB_PAABDS = ["fha", "1433", "ww", "sh2"]
BLIB_KINASE_TYPES = ["ST", "ST", "ST", "Y"]

BARCODE_CONTROL_MEAN_MIN = 10   # pre-selection mean reads per barcode

LFC_TABLE_PKL = OUT_DIR / "dat_all_blib_2batch.pkl"
LFC_TABLE_CSV = OUT_DIR / "dat_all_blib_2batch.csv"

# --------------------------------------------------------------------------
# Stage 04 - drop substrates with no UniProt identity
# --------------------------------------------------------------------------
# 15-mer -> (accession, phosphosite, residue) union used as the "mappable"
# substrate universe. Built in 4_filter_unmapped_substrates.ipynb from the
# animal phosphosite table and the KINBERT train/test interaction tables; see
# the NOTE in 04_filter_unmapped_substrates.py about its provenance.
SUBSTRATE_BACKGROUND_PKL = ANALYSIS_DIR / "substrates_background_12345.pkl"

# Inputs the notebook used to build that file (kept for reference).
ANIMAL_PSITES_PKL = KINASE_ML_DATA / "230301_all_animal_psites.pkl"
INTERACTION_CLASSED_PKL = KINASE_ML_DATA / "interactions" / "interaction_classed_221017.pkl"
TRAIN_230621_PKL = KINASE_ML_DATA / "training_data" / (
    "interations_e2e_1n1p_classed_random_withacc_230621.pkl"
)

SCREEN_COUNT_PKL = DECOUPLING_DIR / "raw_screen_count.pkl"
SCREEN_COUNT_MAPPED_PKL = DECOUPLING_DIR / "raw_screen_count_mapped.pkl"
SCREEN_COUNT_MAPPED_CSV = DECOUPLING_DIR / "raw_screen_count_mapped.csv"

# --------------------------------------------------------------------------
# Stage 05 - hierarchical NB decoupling fit (PyMC)
# --------------------------------------------------------------------------
PAABDS = ["sh2", "fha", "1433", "ww"]

# The fit identifier the notebook used; the deposited posteriors are
# decoupling_out/251125_pymc_{paabd}.nc. Keep it so file names match the
# figure scripts and the QC folders.
FIT_PREFIX = "251125_pymc"

# NUTS settings, verbatim from 5_enrichment_analysis.ipynb cell 10.
NUTS_DRAWS = 1000
NUTS_TUNE = 1000
NUTS_CHAINS = 4
NUTS_TARGET_ACCEPT = 0.9
NUTS_SEED = 1
PSEUDOCOUNT = 1.0          # added to normalised pre counts in the log offset
HDI_PROB = 0.94            # arviz default, used for the b HDI columns


def fit_nc(paabd: str) -> Path:
    """Posterior (netCDF) for one pAABD."""
    return DECOUPLING_DIR / f"{FIT_PREFIX}_{paabd}.nc"


def fit_effects_csv(paabd: str) -> Path:
    """Posterior-mean effects table (binding_strength, kinase_global, ...)."""
    return DECOUPLING_DIR / f"{FIT_PREFIX}_{paabd}.csv"


def fit_binding_stats_csv(paabd: str) -> Path:
    """Per-pair b summaries (mean, sd, lfsr)."""
    return DECOUPLING_DIR / f"{FIT_PREFIX}_binding_stats_{paabd}.csv"


# --------------------------------------------------------------------------
# Stage 06 - hit calling
# --------------------------------------------------------------------------
# A (substrate, kinase) pair is a hit if ANY reporter domain measures it with
# lfsr <= LFSR_THRESH and posterior-mean b >= BINDING_THRESH.
LFSR_THRESH = 0.05   # study-wide since 2026-10-21; see CHANGELOG_lfsr0.05
BINDING_THRESH = 0.1

INFERENCE_SCORES_PKL = OUT_DIR / "251130_ksi_inference_scores.pkl"
REDUCED_PKL = OUT_DIR / "251203_ksi_reduced_df.pkl"

# --------------------------------------------------------------------------
# Stage 07 - false-positive substrates -> canonical hit table
# --------------------------------------------------------------------------
# MAGeCK tests on the substrate-only control libraries (two batches pooled):
# blib7 = S/T controls, blib10 = Y controls.
FP_SAMPLE_ST = "blib7_2batch_rra"
FP_SAMPLE_Y = "blib10_2batch_rra"

# A substrate-only construct that grows reproducibly on its own is treated as a
# false-positive substrate.
FP_LFC_THRESH = 1.5
FP_FDR_THRESH = 0.1
# Independently of the control library, substrates called positive for almost
# every kinase they were measured against are also dropped. The Y library has
# fewer kinases (8 vs 15), hence the looser cut.
PROMISCUITY_THRESH_ST = 0.9
PROMISCUITY_THRESH_Y = 0.75
# Threshold used for the promiscuity-vs-LFC curve in the supplementary figure.
PROMISCUITY_SCORE_CUT = 0.5
PROMISCUITY_SWEEP = (-4.0, 4.0, 0.05)    # start, stop, step

REDUCED_TP_PKL = OUT_DIR / "251203_ksi_reduced_df_tp.pkl"

# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
PAABD_ORDER = ["sh2", "fha", "1433", "ww"]
PAABD_LABELS = {"sh2": "SH2", "fha": "FHA", "1433": "14-3-3", "ww": "WW"}
PAABD_COLORS = {"sh2": "#2E86AB", "fha": "#E84855", "1433": "#3BB273", "ww": "#F4A261"}

# --- Fig 05 panel D: database confusion matrix ---------------------------
HUMAN_KINOME_PKL = KINASE_ML_DATA / "human_kinome_seq_230301.pkl"
INTERACTION_CLASSED_WSOURCE_PKL = (
    KINASE_ML_DATA / "interactions" / "interaction_classed_221017_wsource.pkl"
)
KINBERT_V1_TEST_PKL = KINASE_ML_DATA / "kinbertv1_pred" / "kinbertv1_DB_Test.pkl"

ST_SUB_ALL_CSV = PLANNING_DIR / "st_sub_all.csv"
Y_SUB_ALL_CSV = PLANNING_DIR / "y_sub_all.csv"
ST_SPIKE_INS_CSV = PLANNING_DIR / "st_spike_ins.csv"
Y_SPIKE_INS_CSV = PLANNING_DIR / "y_spike_ins.csv"

# Annotation sources excluded from the "high confidence" database: mass spec,
# yeast two-hybrid and purely computational predictions.
DB_EXCLUDE_SOURCES = {"ms", "y2h", "phosphonetwork", "deepkinzero", "phosphoelm", "kinasephos"}
# Phosphomimetic/engineered constructs are dropped from the benchmark because
# the database annotates the wild-type kinase, not the variant.
DB_EXCLUDE_KINASES = {"KS6B1_HUMAN_active", "KS6B1_HUMAN_wt", "PLK1_HUMAN_active"}

# --- Fig 05 panel D: in-vitro (Kinexus peptide array) comparison ---------
KINEXUS_XLSX = ANALYSIS_DIR / "kinexus_results.xlsx"
KINEXUS_SHEETS = ("raw_data_1", "raw_data_2")
KINEXUS_ROWS_PART1 = 112            # rows in the first plate half of each sheet
KINEXUS_REL_DIFF_MAX = 0.3          # replicate QC: |v1-v2| / max(|v1|,|v2|)
KINEXUS_SIGNAL_THRESH = 20          # both replicates must exceed this

# --- Fig 06 panel A: tree comparison ------------------------------------
# Manning (2002) human protein-kinase tree, with bracketed bootstrap values.
KINOME_NEWICK = ANALYSIS_DIR / "Human-PK-tree_newick.txt"
# Substrate-set tree after leaf-order alignment and manual renaming to the
# kinome tree's labels (written by 11_ksi_tree_comparison_v2.ipynb, then
# renamed by hand in the figure draft).
SCREEN_NEWICK_RENAMED = ANALYSIS_DIR / "screening_tree_reordered_renamed_newick.nwk"

# Screening-tree construction (11_ksi_tree_comparison_v2.ipynb cell 13).
TREE_LFSR_MAX = 0.001        # stricter than the hit-calling threshold
TREE_MIN_KINASES_Y = 3       # substrate must be measured against > this many
TREE_MIN_KINASES_ST = 6
TREE_METRIC = "jaccard"
TREE_LINKAGE = "average"

# Screening-tree leaf -> kinome-tree leaf. Engineered variants (T252D, T210D),
# the ambiguous JAK2 entry, and ADCK1/PIM1 (no reliable kinome leaf) are left
# out of the paired distance analysis.
SCREEN_TO_KINOME = {
    "FES": "TYR_FES",
    "KSYK": "TYR_SYK",
    "TNK1": "TYR_TNK1",
    "FRK": "TYR_FRK",
    "FYN": "TYR_FYN",
    "ABL2": "TYR_ABL2",
    "STYK1": "TYR_STYK1",
    "M3K7": "TKL_MAP3K7",
    "KGP1": "AGC_PRKG1",
    "PHKG1": "CAMK_PHKG1",
    "KKCC1": "CAMK_CAMKK1",
    "KS6B1": "AGC_RPS6KB1",
    "PLK1": "CAMK_PLK1",
    "TBK1": "OTHER_TBK1",
    "PDPK1": "AGC_PDPK1",
    "IKKB": "OTHER_IKBKB",
    "MAST1": "AGC_MAST1",
    "TITIN": "CAMK_TTN",
}
TREE_VARIANT_LEAVES = ("KS6B1 T252D", "PLK1 T210D", "JAK2 - 1")

# --- Fig 06 panel B: dark-kinase motifs and GO --------------------------
DARK_KINASES = ["PHKG1_HUMAN", "KKCC1_HUMAN", "ADCK1_HUMAN", "DYRK4_HUMAN"]
# Kinase set used for the ST-library motif panels in
# 11_ksi_motif_analysis-stlib.ipynb (see NOTES.md).
STLIB_MOTIF_KINASES = [
    "IKKB_HUMAN", "M3K7_HUMAN", "PHKG1_HUMAN", "KGP1_HUMAN", "PDPK1_HUMAN",
    "TBK1_HUMAN", "PLK1_HUMAN", "KS6B1_HUMAN_wt", "KKCC1_HUMAN",
]

# Promiscuity pre-filter applied before the motif/GO analyses, so that a
# handful of substrates that light up for most kinases do not dominate the
# positional statistics. 12_dark_kinase_analysis.ipynb / 12_2 use
# (score < 0.3, measured against > 5 kinases); the ST-library motif notebook
# uses score < 0.8 instead.
MOTIF_PROMISCUITY_MAX = 0.3
MOTIF_MIN_KINASES = 5
STLIB_PROMISCUITY_MAX = 0.8

PHOSPHO_POSITION = 7         # centre of the 15-mer

# UniProt ID-mapping exports for the dark-kinase GO analysis.
DARK_KINASE_IN = ANALYSIS_DIR / "dark_kinase_in"
DARK_SUB_IDMAPPING_TSV = DARK_KINASE_IN / "substrate_dark_idmapping.tsv"
ST_BACKGROUND_IDMAPPING_TSV = DARK_KINASE_IN / "st_background_uniprot_mapping.tsv"

GO_COLS = [
    "Gene Ontology (biological process)",
    "Gene Ontology (cellular component)",
    "Gene Ontology (molecular function)",
]
GO_ORGANISMS = ("Homo sapiens (Human)", "Mus musculus (Mouse)")
GO_MAX_BG = 300              # GO terms on more than this many background
GO_MIN_BG = 0                # proteins are too generic to test
# Display filters for the per-kinase GO panels (12_2 cells 29-32).
GO_PANEL_BG_MIN = 11
GO_PANEL_BG_MAX = 150
GO_PANEL_HIT_MIN = {"ADCK1_HUMAN": 3, "PHKG1_HUMAN": 2, "KKCC1_HUMAN": 2}
GO_PANEL_P_MAX = 0.2
GO_PANEL_TOP_N = 7
