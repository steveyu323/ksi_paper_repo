#!/usr/bin/env python3
"""Build the Table S13 candidate substrate library manifest.

Reconstructs the designed screening library from the wet-lab planning files,
labels each member with its design provenance, and maps it back to a UniProt
accession and phosphosite.

Nothing here is imputed. A 15-mer that does not resolve to an accession is
emitted with empty identity fields and counted in the summary, rather than
dropped or filled in.

Usage
-----
    python3 build_table_s13.py
    python3 build_table_s13.py --no-screen-column   # skip the screen join
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

# Keep `python3 -I build_table_s13.py` working: isolated mode drops the
# script's own directory from sys.path.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import config as cfg  # noqa: E402


# --------------------------------------------------------------------------
# Load the designed library
# --------------------------------------------------------------------------
def load_designed_library() -> pd.DataFrame:
    """Return one row per designed 15-mer, tagged with its library."""
    frames = []
    for lib, path in (("ST", cfg.ST_SUB_ALL), ("Y", cfg.Y_SUB_ALL)):
        df = pd.read_csv(path)
        df = df.rename(columns={df.columns[0]: "window_15mer"})
        df = df[["window_15mer"]].copy()
        df["library"] = lib
        frames.append(df)

    lib_df = pd.concat(frames, ignore_index=True)
    lib_df["window_15mer"] = lib_df["window_15mer"].str.strip().str.upper()

    n_rows = len(lib_df)
    lib_df = lib_df.drop_duplicates(subset=["library", "window_15mer"])
    if len(lib_df) != n_rows:
        print(
            f"  note: dropped {n_rows - len(lib_df)} exact duplicate rows "
            f"within a library",
            file=sys.stderr,
        )
    return lib_df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Design provenance / class
# --------------------------------------------------------------------------
def load_spike_ins() -> pd.DataFrame:
    """Documented kinase-substrate pairs, one row per (15-mer, library).

    A spike-in 15-mer can appear under several target kinases, reporter
    domains and DB splits. Those are collapsed into sorted, semicolon-joined
    fields so the manifest keeps one row per substrate without discarding
    which kinases it was documented against.
    """
    frames = []
    for lib, path in (("ST", cfg.ST_SPIKE_INS), ("Y", cfg.Y_SPIKE_INS)):
        df = pd.read_csv(path)
        df = df[["substrate_seq_15aa", "kinase", "PAABD", "p_aa", "set"]].copy()
        df["library"] = lib
        frames.append(df)
    spikes = pd.concat(frames, ignore_index=True)
    spikes["substrate_seq_15aa"] = (
        spikes["substrate_seq_15aa"].str.strip().str.upper()
    )

    def join_unique(s: pd.Series) -> str:
        return ";".join(sorted({str(x) for x in s.dropna()}))

    agg = (
        spikes.groupby(["library", "substrate_seq_15aa"])
        .agg(
            spikein_kinase_acc=("kinase", join_unique),
            spikein_reporter_domain=("PAABD", join_unique),
            spikein_db_split=("set", join_unique),
        )
        .reset_index()
        .rename(columns={"substrate_seq_15aa": "window_15mer"})
    )
    return agg


def load_pool(path, library: str) -> set[str]:
    df = pd.read_csv(path)
    return set(df[df.columns[0]].astype(str).str.strip().str.upper())


def assign_class(lib_df: pd.DataFrame, spikes: pd.DataFrame) -> pd.DataFrame:
    """Attach design_set and the human-readable Class column.

    Precedence follows the design notebook (cells 27 and 44), where the
    spike_in label is applied last and therefore wins over psp/animal.
    """
    psp = {
        "ST": load_pool(cfg.ST_SUB_PSP, "ST"),
        "Y": load_pool(cfg.Y_SUB_PSP, "Y"),
    }

    df = lib_df.merge(spikes, on=["library", "window_15mer"], how="left")

    is_spike = df["spikein_kinase_acc"].notna()
    in_psp = df.apply(
        lambda r: r["window_15mer"] in psp[r["library"]], axis=1
    )

    df["design_set"] = "animal_random"
    df.loc[in_psp, "design_set"] = "psp_random"
    df.loc[is_spike, "design_set"] = "spike_in"

    class_map = {
        "spike_in": "Control (documented)",
        "psp_random": "Candidate (PhosphoSitePlus)",
        "animal_random": "Candidate (animal phosphoproteome)",
    }
    df["substrate_class"] = df["design_set"].map(class_map)

    # Spike-ins drawn from the train_tn split are documented NON-substrates.
    # Surfacing this prevents the control block being read as all-positive.
    df["control_polarity"] = ""
    split = df["spikein_db_split"].fillna("")
    df.loc[split.str.contains("pos", na=False), "control_polarity"] = "positive"
    df.loc[split.str.contains("tn", na=False), "control_polarity"] = "negative"
    both = split.str.contains("pos", na=False) & split.str.contains("tn", na=False)
    df.loc[both, "control_polarity"] = "mixed"

    return df


# --------------------------------------------------------------------------
# Protein identity mapping
# --------------------------------------------------------------------------
def map_protein_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Join each 15-mer to UniProt accession + phosphosite.

    A 15-mer window is not unique to one protein: identical windows occur in
    paralogues, isoforms and orthologues across the animal phosphosite set.
    We therefore report a deterministic primary assignment (lowest accession,
    then lowest site index) alongside the full match count and the complete
    accession list, so ambiguity stays visible in the deposited manifest.
    """
    bg = pd.read_pickle(cfg.SUBSTRATE_BACKGROUND)
    bg = bg[["substrate_seq_15aa", "substrate", "p_ind", "p_aa"]].copy()
    bg["substrate_seq_15aa"] = bg["substrate_seq_15aa"].str.strip().str.upper()
    bg = bg.rename(columns={"substrate_seq_15aa": "window_15mer"})
    bg = bg.drop_duplicates(subset=["window_15mer", "substrate", "p_ind"])

    wanted = set(df["window_15mer"])
    bg = bg[bg["window_15mer"].isin(wanted)]

    bg["p_ind"] = pd.to_numeric(bg["p_ind"], errors="coerce").astype("Int64")
    bg = bg.sort_values(["window_15mer", "substrate", "p_ind"])

    primary = (
        bg.groupby("window_15mer")
        .head(1)
        .rename(
            columns={
                "substrate": "uniprot_acc",
                "p_ind": "phosphosite_position",
                "p_aa": "phosphoacceptor",
            }
        )
    )

    stats = (
        bg.groupby("window_15mer")
        .agg(
            n_protein_matches=("substrate", "nunique"),
            all_uniprot_acc=(
                "substrate",
                lambda s: ";".join(sorted(set(s))),
            ),
        )
        .reset_index()
    )

    out = df.merge(primary, on="window_15mer", how="left")
    out = out.merge(stats, on="window_15mer", how="left")
    out["n_protein_matches"] = out["n_protein_matches"].fillna(0).astype(int)
    out["mapping_status"] = out["uniprot_acc"].notna().map(
        {True: "mapped", False: "unmapped"}
    )

    # Fall back to the central residue when no DB record supplies p_aa.
    central = out["window_15mer"].str[7]
    out["phosphoacceptor"] = out["phosphoacceptor"].fillna(central)
    return out


def add_screen_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Flag which designed substrates were actually recovered in the screen."""
    counts = pd.read_pickle(cfg.SCREEN_COUNTS)
    seen = (
        counts.drop_duplicates(["substrate_aa", "kinase_name"])
        .groupby("substrate_aa")["kinase_name"]
        .nunique()
    )
    df["n_kinases_recovered"] = (
        df["window_15mer"].map(seen).fillna(0).astype(int)
    )
    df["recovered_in_screen"] = df["n_kinases_recovered"] > 0
    return df


# --------------------------------------------------------------------------
# Assemble
# --------------------------------------------------------------------------
def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Stable SUB-<LIB>-#### IDs, ordered controls-first then alphabetically.

    Sorting is fully determined by the data, so re-running reproduces the
    same ID for the same substrate.
    """
    df = df.sort_values(
        ["library", "design_set", "window_15mer"],
        key=lambda c: c.map({"spike_in": 0, "psp_random": 1, "animal_random": 2})
        if c.name == "design_set"
        else c,
        kind="mergesort",
    ).reset_index(drop=True)

    df["substrate_id"] = [
        f"SUB-{row.library}-{i:04d}"
        for i, row in enumerate(df.itertuples(), start=1)
    ]
    # Restart numbering within each library.
    df["substrate_id"] = (
        df.groupby("library").cumcount().add(1).astype(str).str.zfill(4)
    )
    df["substrate_id"] = "SUB-" + df["library"] + "-" + df["substrate_id"]
    return df


COLUMNS = [
    "substrate_id",
    "library",
    "uniprot_acc",
    "all_uniprot_acc",
    "n_protein_matches",
    "phosphosite_position",
    "phosphoacceptor",
    "window_15mer",
    "substrate_class",
    "design_set",
    "control_polarity",
    "spikein_kinase_acc",
    "spikein_reporter_domain",
    "spikein_db_split",
    "mapping_status",
    "recovered_in_screen",
    "n_kinases_recovered",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-screen-column", action="store_true")
    args = ap.parse_args()

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1. loading designed library ...")
    df = load_designed_library()
    print(f"   {len(df)} designed substrates "
          f"({(df.library == 'ST').sum()} ST, {(df.library == 'Y').sum()} Y)")

    print("2. assigning design provenance ...")
    df = assign_class(df, load_spike_ins())

    print("3. mapping protein identity (large join, ~30 s) ...")
    df = map_protein_identity(df)

    if args.no_screen_column:
        df["recovered_in_screen"] = pd.NA
        df["n_kinases_recovered"] = pd.NA
    else:
        print("4. joining screen recovery ...")
        df = add_screen_recovery(df)

    df = assign_ids(df)
    out = df[COLUMNS].copy()

    main_csv = cfg.OUT_DIR / "tableS13_substrate_library.csv"
    out.to_csv(main_csv, index=False)

    repr_csv = cfg.OUT_DIR / "tableS13_representative_rows.csv"
    rep = (
        out.groupby(["library", "substrate_class"], dropna=False)
        .head(2)
        .sort_values(["library", "substrate_class"])
    )
    rep.to_csv(repr_csv, index=False)

    # ---- summary ----------------------------------------------------------
    lines = []
    lines.append(f"total designed substrates: {len(out)}")
    lines.append("")
    lines.append("by library and class:")
    tab = (
        out.groupby(["library", "substrate_class"]).size().rename("n").reset_index()
    )
    lines.append(tab.to_string(index=False))
    lines.append("")
    lines.append("identity mapping:")
    lines.append(out["mapping_status"].value_counts().to_string())
    n_ambig = (out["n_protein_matches"] > 1).sum()
    lines.append(f"ambiguous (>1 source protein): {n_ambig}")
    lines.append("")
    lines.append("control polarity (spike-ins only):")
    lines.append(
        out.loc[out.design_set == "spike_in", "control_polarity"]
        .value_counts()
        .to_string()
    )
    if not args.no_screen_column:
        lines.append("")
        lines.append("recovered in screen:")
        lines.append(out["recovered_in_screen"].value_counts().to_string())

    summary = "\n".join(lines)
    (cfg.OUT_DIR / "summary_stats.txt").write_text(summary + "\n")
    print()
    print(summary)
    print()
    print(f"wrote {main_csv}")
    print(f"wrote {repr_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
