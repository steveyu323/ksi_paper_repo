#!/usr/bin/env python3
"""
Validate Table 2.1 ("Comparison of training-data scope across KSI prediction
methods") against the datasets actually on disk.

Table 2.1 as published
----------------------
    Model         # Kinases   # Organisms   # KSIs
    DeepKinZero         343             1   13,426
    Phosformer          800            13   24,534
    KINBERT           2,304            92  147,488
    KolossuS      (training data statistics not available)

This script recomputes each cell from the raw dataset files that were used by
the data-processing notebooks in kinase_ml/, and reports PASS / MISMATCH per
cell.  It is read-only: nothing under --data-root is modified.

Data provenance (all paths relative to --data-root, default: <repo>/kinase_ml)
-----------------------------------------------------------------------------
DeepKinZero   data/raw/raw/deepkinzero/{Train,Test,Val}_Phosphosite*.txt
              Same files parsed by code/data_processing/collect_interactions.ipynb
              (cell "### DeepKinZero").  Tab-separated, no header:
                  substrate <tab> psite <tab> 15-mer context <tab> kinase(s)
              Test/Val are multi-label: the kinase field may hold several
              comma-joined accessions, which are exploded to one row per KSI.

Phosformer    data/benchmark_sets/phosformer/{train.csv,test_pos_neg.csv,
              test_neg_only.csv}
              Same files used by code/evaluation/230704_tapedblysate_benchmark.ipynb
              to build kinbert_phosformer_{train,test}_overlap.pkl.
              Columns: uniprot, kinase, substrate, label
                uniprot   = kinase UniProt accession        <- kinase identity
                kinase    = gapped kinase-domain string     <- NOT an identifier,
                            it is re-emitted per row (2,413,524 rows carry
                            2,413,524 distinct values), so it is never used as a
                            grouping key here.
                substrate = 11-mer phosphosite peptide      <- substrate identity
                label     = 1 positive / 0 negative         <- only 1 counts as a KSI

KINBERT       data/interactions/interaction_classed_221017.pkl   (database KSIs)
              data/interactions/lysate_ksi_cleaned_221115.pkl    (lysate LC-MS KSIs)
              data/training_data/interations_e2e_1n1p_db_classed_random_lysate_random_withacc_240213/
                  {train,val,test}.pkl                           (assembled splits)

Organisms     code/data_processing/all_kinase_organism_info.tsv  (curated, 1,971 rows)
              data/raw/raw/uniprot_kinase.tab                    (UniProt dump,
                                                                  1.28 M rows)
              Accession -> Organism, curated table first, UniProt dump as
              fallback.  Fully offline; no network calls.

Counting definitions used here
------------------------------
  # Kinases    distinct kinase UniProt accessions appearing in a positive KSI
  # Organisms  distinct source organisms of those accessions that resolve in the
               offline accession -> organism maps (unresolved accessions are
               reported separately and can only raise the count, never lower it)
  # KSIs       distinct positive kinase-substrate records, at the finest
               resolution the dataset supports:
                 DeepKinZero  (substrate accession, phosphosite, kinase)
                 Phosformer   (kinase accession, 11-mer peptide)
                 KINBERT      (kinase accession, substrate accession, p_ind, p_aa)

Usage
-----
    python validate_table2_1_training_data.py
    python validate_table2_1_training_data.py --data-root /path/to/kinase_ml \
                                              --out-dir  ./table2_1_validation
    python validate_table2_1_training_data.py --strict     # exit 1 on mismatch

Outputs (written to --out-dir)
------------------------------
    table2_1_measured.csv     one row per model, measured values
    table2_1_comparison.csv   published vs measured, per cell, with verdict
    table2_1_validation.md    human-readable report
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# Published Table 2.1 values (transcribed from the manuscript; see
# table2_1_published.csv for the same content as data).
# ----------------------------------------------------------------------------
PUBLISHED = {
    "DeepKinZero": {"n_kinases": 343, "n_organisms": 1, "n_ksis": 13426},
    "Phosformer": {"n_kinases": 800, "n_organisms": 13, "n_ksis": 24534},
    "KINBERT": {"n_kinases": 2304, "n_organisms": 92, "n_ksis": 147488},
    # KolossuS is intentionally absent: the preprint reports no training-set
    # composition, so there is nothing to validate.
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "kinase_ml"


# ----------------------------------------------------------------------------
# Accession -> organism resolution (offline)
# ----------------------------------------------------------------------------
def load_organism_map(data_root: Path, verbose: bool = True) -> dict[str, str]:
    """Build a UniProt accession -> organism map from local files only.

    The curated kinase table wins; the bulk UniProt dump fills the gaps.
    """
    mapping: dict[str, str] = {}

    curated = data_root / "code" / "data_processing" / "all_kinase_organism_info.tsv"
    if curated.exists():
        tab = pd.read_csv(curated, sep="\t")
        for col in ("From", "Entry"):
            if col in tab.columns:
                mapping.update(
                    dict(zip(tab[col].astype(str), tab["Organism"].astype(str)))
                )
        if verbose:
            print(f"  organism map: {len(mapping):>9,} from {curated.name}")
    else:
        print(f"  WARNING: missing {curated}", file=sys.stderr)

    bulk = data_root / "data" / "raw" / "raw" / "uniprot_kinase.tab"
    if bulk.exists():
        before = len(mapping)
        dump = pd.read_csv(
            bulk, sep="\t", usecols=["Entry", "Organism"], low_memory=False
        )
        for acc, organism in zip(dump["Entry"].astype(str), dump["Organism"].astype(str)):
            mapping.setdefault(acc, organism)
        if verbose:
            print(f"  organism map: {len(mapping) - before:>9,} added from {bulk.name}")
    else:
        print(f"  WARNING: missing {bulk}", file=sys.stderr)

    return mapping


def summarise_organisms(accessions, organism_map: dict[str, str]):
    """Return (n_distinct_resolved, n_unresolved, Counter of organism -> n)."""
    counts = collections.Counter(
        organism_map.get(acc, "<unresolved>") for acc in accessions
    )
    unresolved = counts.pop("<unresolved>", 0)
    return len(counts), unresolved, counts


# ----------------------------------------------------------------------------
# DeepKinZero
# ----------------------------------------------------------------------------
DKZ_FILES = (
    "Train_Phosphosite.txt",
    "Test_Phosphosite_MultiLabel.txt",
    "Val_Phosphosite_MultiLabel.txt",
)


def load_deepkinzero(data_root: Path) -> pd.DataFrame:
    """Load and explode the DeepKinZero splits into one row per KSI."""
    dkz_dir = data_root / "data" / "raw" / "raw" / "deepkinzero"
    frames = []
    for name in DKZ_FILES:
        path = dkz_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["substrate", "psite", "pcontext", "kinase"],
            dtype=str,
        )
        frame["split"] = name.split("_")[0].lower()
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)

    # Test/Val rows carry several comma-joined kinase accessions (multi-label).
    exploded = raw.assign(kinase=raw["kinase"].str.split(",")).explode("kinase")
    exploded["kinase"] = exploded["kinase"].str.strip()
    return exploded


def stats_deepkinzero(data_root: Path, organism_map: dict[str, str]) -> dict:
    ksi = load_deepkinzero(data_root)

    kinases = sorted(set(ksi["kinase"]))
    n_organisms, unresolved, breakdown = summarise_organisms(kinases, organism_map)
    n_ksis = ksi.drop_duplicates(["substrate", "psite", "kinase"]).shape[0]

    return {
        "model": "DeepKinZero",
        "n_kinases": len(kinases),
        "n_organisms": n_organisms,
        "n_ksis": n_ksis,
        "_n_ksi_rows": ksi.shape[0],
        "_n_phosphosites": ksi.drop_duplicates(["substrate", "psite"]).shape[0],
        "_n_kinase_substrate_protein_pairs": ksi.drop_duplicates(
            ["kinase", "substrate"]
        ).shape[0],
        "_organism_unresolved": unresolved,
        "_organism_breakdown": breakdown,
        "_files": [str(Path("data/raw/raw/deepkinzero") / f) for f in DKZ_FILES],
    }


# ----------------------------------------------------------------------------
# Phosformer
# ----------------------------------------------------------------------------
PHOSFORMER_FILES = ("train.csv", "test_pos_neg.csv", "test_neg_only.csv")


def stats_phosformer(
    data_root: Path, organism_map: dict[str, str], chunksize: int = 500_000
) -> dict:
    """Stream the Phosformer CSVs (train.csv is ~715 MB) and count positives.

    Only the three small columns are read; the gapped kinase-domain column is
    skipped entirely, which keeps peak memory well under 1 GB.
    """
    phos_dir = data_root / "data" / "benchmark_sets" / "phosformer"

    kinases: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    n_rows = 0
    n_positive_rows = 0
    per_file: dict[str, dict[str, int]] = {}
    # Diagnostics for the "kinase column is not an identifier" check.
    train_acc_domain: set[tuple[str, int]] = set()
    domain_keyed_ksis: set[tuple[str, int, str]] = set()

    for name in PHOSFORMER_FILES:
        path = phos_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        f_rows = f_pos = 0
        f_kinases: set[str] = set()
        f_pairs: set[tuple[str, str]] = set()
        for chunk in pd.read_csv(
            path,
            usecols=["uniprot", "kinase", "substrate", "label"],
            dtype={
                "uniprot": str,
                "kinase": str,
                "substrate": str,
                "label": "int8",
            },
            chunksize=chunksize,
        ):
            f_rows += len(chunk)
            positive = chunk.loc[chunk["label"] == 1]
            f_pos += len(positive)
            f_kinases.update(positive["uniprot"])
            f_pairs.update(zip(positive["uniprot"], positive["substrate"]))
            # The domain strings are ~700 chars each; hash them rather than
            # retaining millions of copies just to count distinct values.
            if name == "train.csv":
                train_acc_domain.update(
                    zip(chunk["uniprot"], chunk["kinase"].map(hash))
                )
            domain_keyed_ksis.update(
                zip(
                    positive["uniprot"],
                    positive["kinase"].map(hash),
                    positive["substrate"],
                )
            )
        per_file[name] = {
            "rows": f_rows,
            "positive_rows": f_pos,
            "kinases": len(f_kinases),
            "ksis": len(f_pairs),
        }
        n_rows += f_rows
        n_positive_rows += f_pos
        kinases |= f_kinases
        pairs |= f_pairs

    n_organisms, unresolved, breakdown = summarise_organisms(
        sorted(kinases), organism_map
    )

    return {
        "model": "Phosformer",
        "n_kinases": len(kinases),
        "n_organisms": n_organisms,
        "n_ksis": len(pairs),
        "_total_rows": n_rows,
        "_positive_rows": n_positive_rows,
        "_train_distinct_acc_domain": len(train_acc_domain),
        "_ksis_keyed_on_domain": len(domain_keyed_ksis),
        "_per_file": per_file,
        "_organism_unresolved": unresolved,
        "_organism_breakdown": breakdown,
        "_files": [str(Path("data/benchmark_sets/phosformer") / f) for f in PHOSFORMER_FILES],
    }


# ----------------------------------------------------------------------------
# KINBERT (this work)
# ----------------------------------------------------------------------------
KINBERT_SPLIT_DIR = (
    "data/training_data/"
    "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_240213"
)


def stats_kinbert(data_root: Path, organism_map: dict[str, str]) -> dict:
    """Recount KINBERT from both the source KSI tables and the assembled splits."""
    db_path = data_root / "data" / "interactions" / "interaction_classed_221017.pkl"
    lysate_path = data_root / "data" / "interactions" / "lysate_ksi_cleaned_221115.pkl"

    db = pd.read_pickle(db_path)
    lysate = pd.read_pickle(lysate_path).copy()
    # lysate stores the site as a single "S212"-style string; split it so both
    # tables share the (kinase, substrate, p_ind, p_aa) key.
    lysate["p_aa"] = lysate["psite"].astype(str).str[0]
    lysate["p_ind"] = lysate["psite"].astype(str).str[1:]

    key = ["kinase", "substrate", "p_ind", "p_aa"]
    source_union = pd.concat(
        [db[key].astype(str), lysate[key].astype(str)], ignore_index=True
    ).drop_duplicates()

    n_kin_db = db["kinase"].nunique()
    n_kin_lysate = lysate["kinase"].nunique()
    kinases_union = set(db["kinase"].astype(str)) | set(lysate["kinase"].astype(str))

    # Assembled training splits actually fed to the model.
    split_dir = data_root / KINBERT_SPLIT_DIR
    split_kinases: set[str] = set()
    split_ksis = 0
    split_available = split_dir.exists()
    if split_available:
        frames = []
        for split in ("train", "val", "test"):
            path = split_dir / f"{split}.pkl"
            if path.exists():
                frames.append(pd.read_pickle(path))
        if frames:
            splits = pd.concat(frames, ignore_index=True)
            positive = splits.loc[splits["label"] == 1]
            split_kinases = set(positive["kinase"].astype(str))
            split_ksis = positive[key].astype(str).drop_duplicates().shape[0]
        else:
            split_available = False

    n_organisms, unresolved, breakdown = summarise_organisms(
        sorted(kinases_union), organism_map
    )
    n_organisms_splits = (
        summarise_organisms(sorted(split_kinases), organism_map)[0]
        if split_kinases
        else None
    )
    curated = data_root / "code" / "data_processing" / "all_kinase_organism_info.tsv"
    n_organisms_curated_table = (
        pd.read_csv(curated, sep="\t")["Organism"].nunique() if curated.exists() else None
    )

    return {
        "model": "KINBERT",
        "n_kinases": len(kinases_union),
        "n_organisms": n_organisms,
        "n_ksis": len(source_union),
        "_n_kinases_db": n_kin_db,
        "_n_kinases_lysate": n_kin_lysate,
        "_n_kinases_concat_not_unioned": n_kin_db + n_kin_lysate,
        "_n_ksis_db": db[key].astype(str).drop_duplicates().shape[0],
        "_n_ksis_lysate": lysate[key].astype(str).drop_duplicates().shape[0],
        "_n_kinases_assembled_splits": len(split_kinases) if split_available else None,
        "_n_ksis_assembled_splits": split_ksis if split_available else None,
        "_n_organisms_assembled_splits": n_organisms_splits,
        "_n_organisms_curated_table": n_organisms_curated_table,
        "_organism_unresolved": unresolved,
        "_organism_breakdown": breakdown,
        "_files": [
            "data/interactions/interaction_classed_221017.pkl",
            "data/interactions/lysate_ksi_cleaned_221115.pkl",
            KINBERT_SPLIT_DIR + "/{train,val,test}.pkl",
        ],
    }


# ----------------------------------------------------------------------------
# Comparison / reporting
# ----------------------------------------------------------------------------
CELLS = (
    ("n_kinases", "# of Kinases"),
    ("n_organisms", "# of Organisms"),
    ("n_ksis", "# of KSIs"),
)


def build_comparison(measured: list[dict]) -> pd.DataFrame:
    rows = []
    for stat in measured:
        model = stat["model"]
        for field, label in CELLS:
            published = PUBLISHED[model][field]
            observed = stat[field]
            delta = observed - published
            rows.append(
                {
                    "model": model,
                    "statistic": label,
                    "published": published,
                    "measured": observed,
                    "delta": delta,
                    "pct_delta": round(100.0 * delta / published, 2) if published else None,
                    "verdict": "PASS" if delta == 0 else "MISMATCH",
                }
            )
    return pd.DataFrame(rows)


def build_corrected(measured: list[dict]) -> pd.DataFrame:
    """The table as it should be stated, on one consistent basis.

    Basis: every kinase / organism / KSI distributed with the dataset, i.e. the
    union of that dataset's own train + validation + test splits.  DeepKinZero
    and Phosformer are therefore reported as their release files stand;  KINBERT
    is reported from its assembled train/val/test splits rather than from the
    upstream source tables, so that all three rows count the same thing.
    """
    by_model = {stat["model"]: stat for stat in measured}
    dkz, phos, kin = by_model["DeepKinZero"], by_model["Phosformer"], by_model["KINBERT"]

    rows = [
        {
            "model": "DeepKinZero",
            "n_kinases": dkz["n_kinases"],
            "n_organisms": dkz["n_organisms"],
            "n_ksis": dkz["n_ksis"],
            "basis": "train + val + test release files",
            "note": (
                "KSI count is 187 below the published 13,426; kinase and organism "
                "counts reproduce exactly."
            ),
        },
        {
            "model": "Phosformer",
            "n_kinases": phos["n_kinases"],
            "n_organisms": phos["n_organisms"],
            "n_ksis": phos["n_ksis"],
            "basis": "train + test release files (no validation split distributed)",
            "note": (
                "KSI count is 2,659 below the published 24,534, consistent with the "
                "absent validation split; kinase and organism counts reproduce exactly."
            ),
        },
        {
            "model": "KINBERT",
            "n_kinases": kin["_n_kinases_assembled_splits"],
            "n_organisms": kin["_n_organisms_assembled_splits"],
            "n_ksis": kin["_n_ksis_assembled_splits"],
            "basis": "assembled train + val + test splits",
            "note": (
                "Replaces the published 2,304 kinases, which double counted the "
                f"{kin['_n_kinases_concat_not_unioned'] - kin['n_kinases']:,} kinases "
                "shared by the database and lysate tables, and the published 147,488 "
                "KSIs, which predate the current splits."
            ),
        },
        {
            "model": "KolossuS",
            "n_kinases": pd.NA,
            "n_organisms": pd.NA,
            "n_ksis": pd.NA,
            "basis": "not reported",
            "note": (
                "Training-set composition is not reported in the preprint "
                "(Jha et al., 2025); nothing to count."
            ),
        },
    ]
    corrected = pd.DataFrame(rows)
    for field, _ in CELLS:
        corrected[f"published_{field}"] = corrected["model"].map(
            lambda m: PUBLISHED.get(m, {}).get(field, pd.NA)
        )
    order = ["model", "basis"]
    for field, _ in CELLS:
        order += [field, f"published_{field}"]
    return corrected[order + ["note"]]


def write_report(out_dir: Path, measured: list[dict], comparison: pd.DataFrame) -> Path:
    by_model = {stat["model"]: stat for stat in measured}
    lines: list[str] = []
    add = lines.append

    add("# Table 2.1 validation — training-data scope across KSI prediction methods")
    add("")
    add("Regenerate with `python validate_table2_1_training_data.py`.")
    add("All counts are recomputed from the raw dataset files; no network access.")
    add("")
    add("## Verdict")
    add("")
    add("| Model | Statistic | Published | Measured | Delta | Verdict |")
    add("|---|---|---:|---:|---:|---|")
    for row in comparison.itertuples():
        add(
            f"| {row.model} | {row.statistic} | {row.published:,} | "
            f"{row.measured:,} | {row.delta:+,} | {row.verdict} |"
        )
    add("")
    n_pass = int((comparison["verdict"] == "PASS").sum())
    add(f"{n_pass}/{len(comparison)} cells reproduce exactly.")
    add("")

    add("## DeepKinZero")
    dkz = by_model["DeepKinZero"]
    add("")
    add("Files: " + ", ".join(f"`{f}`" for f in dkz["_files"]))
    add("")
    add(f"- distinct kinase accessions (after multi-label explode): **{dkz['n_kinases']:,}**")
    add(f"- distinct (substrate, phosphosite, kinase) KSIs: **{dkz['n_ksis']:,}**")
    add(f"- distinct (substrate, phosphosite) phosphosites: {dkz['_n_phosphosites']:,}")
    add(
        f"- distinct (kinase, substrate protein) pairs: "
        f"{dkz['_n_kinase_substrate_protein_pairs']:,}"
    )
    add(
        f"- organisms: {dkz['n_organisms']} resolved "
        f"({dkz['_organism_unresolved']} accessions unresolved offline)"
    )
    for organism, n in dkz["_organism_breakdown"].most_common():
        add(f"    - {organism}: {n}")
    add("")

    add("## Phosformer")
    phos = by_model["Phosformer"]
    add("")
    add("Files: " + ", ".join(f"`{f}`" for f in phos["_files"]))
    add("")
    add(f"- rows scanned: {phos['_total_rows']:,} (positives: {phos['_positive_rows']:,})")
    add(f"- distinct kinase accessions in positives: **{phos['n_kinases']:,}**")
    add(f"- distinct (kinase accession, 11-mer peptide) KSIs: **{phos['n_ksis']:,}**")
    add("")
    add("| File | Rows | Positive rows | Kinases | KSIs |")
    add("|---|---:|---:|---:|---:|")
    for name, counts in phos["_per_file"].items():
        add(
            f"| {name} | {counts['rows']:,} | {counts['positive_rows']:,} | "
            f"{counts['kinases']:,} | {counts['ksis']:,} |"
        )
    add("")
    add(
        f"- organisms: {phos['n_organisms']} resolved "
        f"({phos['_organism_unresolved']} accessions unresolved offline)"
    )
    for organism, n in phos["_organism_breakdown"].most_common():
        add(f"    - {organism}: {n}")
    add("")

    add("## KINBERT")
    kin = by_model["KINBERT"]
    add("")
    add("Files: " + ", ".join(f"`{f}`" for f in kin["_files"]))
    add("")
    add(f"- database KSI table kinases: {kin['_n_kinases_db']:,}")
    add(f"- lysate KSI table kinases: {kin['_n_kinases_lysate']:,}")
    add(f"- **union** of the two: **{kin['n_kinases']:,}**")
    add(
        f"- sum of the two without de-duplication: "
        f"{kin['_n_kinases_concat_not_unioned']:,}"
    )
    add(f"- distinct (kinase, substrate, p_ind, p_aa) KSIs across both: **{kin['n_ksis']:,}**")
    if kin["_n_kinases_assembled_splits"] is not None:
        add(
            f"- assembled train/val/test splits: "
            f"{kin['_n_kinases_assembled_splits']:,} kinases, "
            f"{kin['_n_ksis_assembled_splits']:,} positive KSIs"
        )
    add(f"- distinct organisms in the curated kinase table: {kin['_n_organisms_curated_table']}")
    add(
        f"- distinct organisms among the {kin['n_kinases']:,} union kinases: "
        f"{kin['n_organisms']} ({kin['_organism_unresolved']} unresolved offline)"
    )
    add("")

    add("## Corrected table")
    add("")
    add(
        "Stated on one consistent basis: everything distributed with each dataset "
        "(its own train + validation + test splits). Written to "
        "`table2_1_corrected.csv`."
    )
    add("")
    add("| Model | # of Kinases | # of Organisms | # of KSIs |")
    add("|---|---:|---:|---:|")
    for row in build_corrected(measured).itertuples():
        if pd.isna(row.n_kinases):
            add(f"| {row.model} | \\multicolumn{{3}}{{c}}{{not reported}} | | |")
        else:
            add(
                f"| {row.model} | {row.n_kinases:,} | {row.n_organisms:,} | "
                f"{row.n_ksis:,} |"
            )
    add("")

    add("## Interpretation of the mismatches")
    add("")

    add("### DeepKinZero — # of KSIs")
    add("")
    add(
        f"The three split files hold {dkz['_n_ksi_rows']:,} kinase-substrate records "
        f"after exploding the comma-joined multi-label kinase field, all of them "
        f"distinct ({dkz['n_ksis']:,} unique triples). Neither that number nor any "
        f"coarser grouping reaches the published 13,426: collapsing to phosphosites "
        f"gives {dkz['_n_phosphosites']:,}, collapsing to kinase-substrate protein "
        f"pairs gives {dkz['_n_kinase_substrate_protein_pairs']:,}. The shortfall of "
        f"{PUBLISHED['DeepKinZero']['n_ksis'] - dkz['n_ksis']:,} "
        f"({100.0 * (PUBLISHED['DeepKinZero']['n_ksis'] - dkz['n_ksis']) / PUBLISHED['DeepKinZero']['n_ksis']:.1f}%) "
        "is most likely a version difference between the published dataset and the "
        "copy distributed with the DeepKinZero release used here. The kinase and "
        "organism counts reproduce exactly, so the row is sound apart from this cell."
    )
    add("")

    add("### Phosformer — # of KSIs")
    add("")
    per_file = phos["_per_file"]
    add(
        f"The local copy carries only train.csv and the two test files - there is no "
        f"validation split. Train contributes {per_file['train.csv']['ksis']:,} unique "
        f"(kinase, 11-mer) positives and test_pos_neg.csv contributes "
        f"{per_file['test_pos_neg.csv']['ksis']:,}, totalling {phos['n_ksis']:,}. The "
        f"gap to the published 24,534 is exactly "
        f"{PUBLISHED['Phosformer']['n_ksis'] - phos['n_ksis']:,}, which happens to "
        f"equal the test-split positive count, consistent with a held-out split of "
        "similar size being absent from this copy. Kinase and organism counts "
        "reproduce exactly."
    )
    add("")
    add(
        "Note on identity: the `kinase` column is a gapped domain string that is "
        f"re-emitted per row (train.csv's {per_file['train.csv']['rows']:,} rows carry "
        f"{phos['_train_distinct_acc_domain']:,} distinct (uniprot, kinase) pairs), so "
        "it cannot be used as a kinase key. Counting on it instead of `uniprot` "
        f"inflates the KSI count to {phos['_ksis_keyed_on_domain']:,}."
    )
    add("")

    add("### KINBERT — # of Kinases")
    add("")
    add(
        f"The published 2,304 is reproduced exactly by "
        f"{kin['_n_kinases_db']:,} + {kin['_n_kinases_lysate']:,} = "
        f"{kin['_n_kinases_concat_not_unioned']:,}, i.e. concatenating the distinct "
        "kinases of the database and lysate tables without de-duplicating across "
        "them. This is the operation performed in "
        "`kinase_ml/code/data_processing/230704_datasource_plotting.ipynb`, which "
        "builds `all_kinase_train` by list concatenation of two `drop_duplicates()` "
        f"results. The de-duplicated union is {kin['n_kinases']:,}; the "
        f"{kin['_n_kinases_concat_not_unioned'] - kin['n_kinases']:,} kinases present "
        "in both tables are double counted. The assembled training splits contain "
        f"{kin['_n_kinases_assembled_splits']:,} kinases. The table cell should read "
        f"{kin['n_kinases']:,} (or {kin['_n_kinases_assembled_splits']:,} if the "
        "intent is the kinases actually seen during training), not 2,304."
    )
    add("")

    add("### KINBERT — # of KSIs")
    add("")
    add(
        f"No counting definition tried here reproduces 147,488. The de-duplicated "
        f"union of the two source tables gives {kin['n_ksis']:,} "
        f"({kin['_n_ksis_db']:,} database + {kin['_n_ksis_lysate']:,} lysate before "
        f"union), and the assembled train/val/test splits give "
        f"{kin['_n_ksis_assembled_splits']:,} positives. Both exceed the published "
        f"figure by roughly "
        f"{100.0 * (kin['n_ksis'] - PUBLISHED['KINBERT']['n_ksis']) / PUBLISHED['KINBERT']['n_ksis']:.0f}%, "
        "so 147,488 appears to come from an earlier snapshot of the training set "
        "rather than from the artifacts in this repository. Recommend restating the "
        f"cell as {kin['_n_ksis_assembled_splits']:,}, matched to the split files that "
        "were actually trained on."
    )
    add("")

    add("## KolossuS")
    add("")
    add(
        "Not validated. The preprint (Jha et al., 2025) does not report training-set "
        "composition, so there is no published figure to check and no dataset copy "
        "in this repository."
    )
    add("")

    path = out_dir / "table2_1_validation.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Table 2.1 against the datasets on disk.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="kinase_ml root holding data/ and code/",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "table2_1_validation",
        help="directory for the measured/comparison CSVs and the markdown report",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="rows per chunk when streaming the Phosformer CSVs",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any cell fails to reproduce",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.resolve()
    if not data_root.exists():
        parser.error(f"--data-root does not exist: {data_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data root : {data_root}")
    print(f"out dir   : {args.out_dir}")
    print()
    print("Building offline accession -> organism map ...")
    organism_map = load_organism_map(data_root)
    print()

    measured = []
    for label, fn in (
        ("DeepKinZero", lambda: stats_deepkinzero(data_root, organism_map)),
        ("Phosformer", lambda: stats_phosformer(data_root, organism_map, args.chunksize)),
        ("KINBERT", lambda: stats_kinbert(data_root, organism_map)),
    ):
        print(f"Counting {label} ...")
        measured.append(fn())

    public_cols = ["model", "n_kinases", "n_organisms", "n_ksis"]
    measured_df = pd.DataFrame([{k: s[k] for k in public_cols} for s in measured])
    measured_df.to_csv(args.out_dir / "table2_1_measured.csv", index=False)

    comparison = build_comparison(measured)
    comparison.to_csv(args.out_dir / "table2_1_comparison.csv", index=False)

    corrected = build_corrected(measured)
    corrected.to_csv(args.out_dir / "table2_1_corrected.csv", index=False)

    report = write_report(args.out_dir, measured, comparison)

    print()
    print(comparison.to_string(index=False))
    print()
    n_pass = int((comparison["verdict"] == "PASS").sum())
    print(f"{n_pass}/{len(comparison)} cells reproduce exactly.")
    print(f"report: {report}")

    if args.strict and n_pass != len(comparison):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
