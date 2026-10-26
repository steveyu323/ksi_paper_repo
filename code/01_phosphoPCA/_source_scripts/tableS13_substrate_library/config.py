"""Input paths for the Table S13 substrate-library manifest.

All paths are resolved relative to REPO_ROOT so the folder can be moved as a
unit. Override REPO_ROOT with the KSI_REPO_ROOT environment variable.
"""
import os
from pathlib import Path

REPO_ROOT = Path(
    os.environ.get(
        "KSI_REPO_ROOT",
        # template only - set KSI_REPO_ROOT to the directory holding
        # ksi_analysis_2025/ and kinase_ml/
        "/path/to/analysis_repo_root",
    )
)

ANALYSIS_DIR = REPO_ROOT / "ksi_analysis_2025"
PLANNING_DIR = REPO_ROOT / "kinase_ml" / "code" / "wetlab_planning"

# --- Designed library (what was ordered as oligos) -------------------------
# Written by 230310_select_substrate_oligo.ipynb, cells 32 and 47.
ST_SUB_ALL = PLANNING_DIR / "st_sub_all.csv"
Y_SUB_ALL = PLANNING_DIR / "y_sub_all.csv"

# --- Class provenance ------------------------------------------------------
# Documented kinase-substrate pairs used as spike-in controls (cells 10, 39).
# Carry the source kinase accession, the PAABD reporter, and the DB split
# (test_pos / train_pos / train_tn).
ST_SPIKE_INS = PLANNING_DIR / "st_spike_ins.csv"
Y_SPIKE_INS = PLANNING_DIR / "y_spike_ins.csv"

# Redundancy-filtered PhosphoSitePlus-derived candidate pools (cells 14, 43).
ST_SUB_PSP = PLANNING_DIR / "st_sub_psp.csv"
Y_SUB_PSP = PLANNING_DIR / "y_sub_psp.csv"

# --- Protein identity mapping ----------------------------------------------
# 15-mer -> (UniProt accession, phosphosite index, phosphoacceptor residue).
# Built by 4_filter_unmapped_substrates.ipynb as the union of the animal
# phosphosite table and the KinBERT train/test interaction tables.
SUBSTRATE_BACKGROUND = ANALYSIS_DIR / "substrates_background_12345.pkl"

# --- Observed screen (optional provenance column) --------------------------
# Unique (kinase, substrate) rows actually recovered from the barcode screen.
SCREEN_COUNTS = ANALYSIS_DIR / "decoupling_out" / "raw_screen_count.pkl"

# --- Design constants, copied from 230310_select_substrate_oligo.ipynb -----
SPIKE_IN_DIST_THRESH = 7   # cells 6, 34   -> controls de-duplicated at < 7
PSP_DIST_THRESH = 9        # cells 13, 41  -> candidates kept at distance > 8
ANIMAL_DIST_THRESH = 11    # cell 18       -> Y-library animal pool only

OUT_DIR = Path(__file__).resolve().parent / "outputs"
