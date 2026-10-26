"""
Tables for the curated ("demo") virus figures.

Source notebook: ``eval_6_prediction_against_kinome_virus_demo.ipynb``.

Figures covered
---------------
  eval_figs/eval_virus_dms_kinase_psite_demo.{png,pdf,svg}   (p1)
  eval_figs/eval_virus_dms_drug_kinase_demo.{png,pdf,svg}    (p2)
  eval_figs/eval_virus_dms_drug_psite_demo.{png,pdf,svg}     (p_drug)

All three are heatmaps over a hand-curated subset of viral phospho-sites:
p1 is kinase by site coloured by logit, p2 is kinase by drug coloured by
whether the drug targets that kinase, p_drug is drug by site.

Where the numbers come from
---------------------------
The pipeline enters at ``250118_virus_against_kinome_hits.pkl`` rather than at
the raw model output.  That cached file is the exact object the figures were
built from; re-deriving it from the model pickles changes 28 of its 96 rows,
because the top-ten-kinase selection sorts on logits with ties and pandas'
default sort is not stable.

The three external drug databases the notebook merges are not on this machine,
but their only product is ``drug_kinase_uniprot.pkl``, which is cached here and
which reproduces the notebook's recorded output exactly.

p2 caveat: its drug axis is chosen by a second non-stable sort, so re-running
picks a different two drugs per kinase.  The true 44 drugs are recovered from
the figure's own vector artifact plus the exported pair list, and the axis is
pinned to them; the script reports how far a free re-run drifts.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

from eval_table_utils import Checker, require, table_path, write_verification
from virus_pipeline import (
    build_drug_long, build_logit_long, build_top_df, complete_with_zeros,
    complete_with_zeros_drug, fill_missing_hits,
)

HITS_PKL = "./250118_virus_against_kinome_hits.pkl"
DRUG_PKL = "./drug_kinase_uniprot.pkl"
EVIDENCE_CSV = "./claude_evidence_kinase_drug_virus.csv"
DEMO_PAIRS_CSV = "./260119_demo_kinase_virus_drug_pairs.csv"

# The p1 rows the notebook printed at cell 48, for the site with the most
# predicted kinases.  These are logits straight off the plotted frame.
P1_S202_LOGITS = {
    "NEK10": 3.4531, "ICK": 3.1406, "PCTAIRE1": 3.0625, "MOK": 3.0000,
    "PCTAIRE2": 2.7812, "GSK3B": 2.7031, "GSK3A": 2.5469, "CDC7": 2.4531,
    "CDKL2": 2.3750, "PCTAIRE3": 2.3594,
}

# Axis contents read out of eval_virus_dms_kinase_psite_demo.pdf.
P1_KINASE_AXIS = [
    "AKT1", "ALK", "CDC2", "CDK11", "CDK2", "CDK4", "CDK5", "CDK7", "CK1e",
    "Erk1", "Erk2", "Erk7", "FLT3", "FLT4", "FMS", "GSK3B", "JAK2", "KDR",
    "PFTAIRE2", "PKCa", "PKCb", "PKCe", "SRPK2",
]

# Axis contents read out of eval_virus_dms_drug_kinase_demo.pdf.
P2_KINASE_AXIS = [
    "AKT1", "ALK", "CDC2", "CDK2", "CDK4", "CDK5", "CDK7", "CK1e", "Erk1",
    "Erk2", "Erk7", "FLT3", "FLT4", "FMS", "GSK3B", "JAK2", "KDR",
    "PFTAIRE2", "PKCa", "PKCb", "PKCe", "SRPK2",
]

# The two drugs on the p2 axis that are absent from the exported pair list.
P2_EXTRA_DRUGS = ["Ensartinib", "alpha-Tocopherol succinate"]

# The full XL820 column of the plotted p2 grid, printed at cell 70.
P2_XL820_HITS = {k: (1 if k == "KDR" else 0) for k in P2_KINASE_AXIS}


def build_p1(top_df):
    """Kinase-by-site heatmap, restricted to the curated examples."""
    logit_long = build_logit_long(top_df)

    # The curated subset: keep only kinase/virus pairs with literature support,
    # at most two sites each.
    searched_links = pd.read_csv(require(EVIDENCE_CSV))
    searched_links_examples = (
        searched_links.loc[~searched_links["virus_drug_evidence"].isna(), :]
        .groupby(["virus", "kinase", "drug", "sequence"]).head(1))

    curated = pd.merge(logit_long, searched_links_examples,
                       left_on=["substrate", "kinase"],
                       right_on=["virus", "kinase"])
    curated = curated.iloc[:, [0, 1, 2, 3]]
    curated = curated.drop_duplicates().groupby(["substrate", "kinase"]).head(2)
    curated.columns = ["substrate", "psite", "kinase", "logit"]

    # Three influenza rows are added back by position.
    flu_top_df = top_df.iloc[[1, 6, 9], :]
    logit_long_flu = build_logit_long(flu_top_df).groupby(["substrate", "psite"]).head(1)
    curated = pd.concat([curated, logit_long_flu])

    kinase_order = curated["kinase"].dropna().astype(str).unique().tolist()
    filled = complete_with_zeros(curated, kinase_order)
    return logit_long, curated, filled


def build_p2(top_df, curated_kinases):
    """Kinase-by-drug hit grid, with the drug axis pinned to the figure."""
    drug_long, drug_long_pre = build_drug_long(top_df, curated_kinases)
    free_rerun = fill_missing_hits(drug_long)

    # Pin the drug axis to the 44 drugs the figure actually shows.
    demo_pairs = pd.read_csv(require(DEMO_PAIRS_CSV), index_col=0)
    figure_drugs = list(dict.fromkeys(
        demo_pairs["drug"].astype(str).tolist() + P2_EXTRA_DRUGS))

    observed = (drug_long_pre.loc[drug_long_pre["kinase"].isin(curated_kinases), :]
                [["kinase", "drug", "hit"]]
                .groupby(["kinase", "drug"], as_index=False).agg(hit=("hit", "max")))

    grid = (pd.DataFrame({"kinase": [k for k in P2_KINASE_AXIS]})
            .merge(pd.DataFrame({"drug": figure_drugs}), how="cross"))
    pinned = grid.merge(observed, on=["kinase", "drug"], how="left")
    pinned["hit"] = pinned["hit"].fillna(0).astype(int)
    pinned["drug_lbl"] = pinned["drug"].apply(
        lambda s: s[:10] + "..." if len(s) > 10 else s)
    return pinned, free_rerun, figure_drugs


def build_p_drug(curated, drug_long_pinned, p1_max_logit):
    """Drug-by-site heatmap.

    The exported pair list is the plotted frame just before the grid is
    completed, so it is used directly: re-deriving it permutes the drug order
    within each group, which would permute the figure's x axis.

    That file has no generator.  Nothing in the repository writes or reads
    ``260119_demo_kinase_virus_drug_pairs.csv``, so the copy on disk is the only
    one, and this figure cannot be rebuilt without it.
    """
    virus_drug_long = pd.read_csv(require(DEMO_PAIRS_CSV), index_col=0)
    filled, drug_order = complete_with_zeros_drug(virus_drug_long)
    # The fill scale of this figure is capped at half of p1's maximum.
    fill_limit = p1_max_logit / 2
    return virus_drug_long, filled, drug_order, fill_limit


def main():
    checks = []

    top_df = build_top_df(require(HITS_PKL), require(DRUG_PKL), logit_min=3)
    print(f"\ntop_df: {top_df.shape[0]} sites")

    # ---------------- p1: kinase by phospho-site ----------------
    print("\n=== eval_virus_dms_kinase_psite_demo ===")
    logit_long, curated, p1_filled = build_p1(top_df)

    p = table_path("eval_virus_dms_kinase_psite_demo.csv")
    p1_filled.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}  ({len(p1_filled)} cells)")

    c = Checker("eval_virus_dms_kinase_psite_demo")
    # The nucleocapsid label carries a leading space from the name map, which
    # a pandas repr hides, so compare on the stripped value.
    s202 = logit_long[(logit_long["substrate"].str.strip() == "SARS2:Nucleocapsid") &
                      (logit_long["psite"] == "S202")]
    s202_map = dict(zip(s202["kinase"].astype(str), s202["logit"]))
    for kinase, want in P1_S202_LOGITS.items():
        c.check(f"SARS2:Nucleocapsid S202 / {kinase} logit",
                s202_map.get(kinase), want, 1e-9)
    got_kinases = sorted(p1_filled["kinase"].astype(str).unique())
    c.check("number of kinases on the x axis",
            float(len(got_kinases)), float(len(P1_KINASE_AXIS)), 0)
    c.check("kinase axis matches the figure",
            0.0 if got_kinases == sorted(P1_KINASE_AXIS) else 1.0, 0.0, 0)
    c.check("number of phospho-sites on the y axis",
            float(p1_filled[["substrate", "psite"]].drop_duplicates().shape[0]), 20.0, 0)
    p1_max = float(p1_filled["logit"].max())
    c.check("colour-scale maximum", p1_max, 8.0625, 1e-9)
    checks.append(c.report())

    # ---------------- p2: kinase by drug ----------------
    print("\n=== eval_virus_dms_drug_kinase_demo ===")
    p2_pinned, p2_free, figure_drugs = build_p2(top_df, curated["kinase"])

    p = table_path("eval_virus_dms_drug_kinase_demo.csv")
    p2_pinned.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}  ({len(p2_pinned)} cells)")

    c = Checker("eval_virus_dms_drug_kinase_demo")
    c.check("number of kinases on the x axis",
            float(p2_pinned["kinase"].nunique()), 22.0, 0)
    c.check("number of drugs on the y axis",
            float(p2_pinned["drug"].nunique()), 44.0, 0)
    c.check("grid cells", float(len(p2_pinned)), 968.0, 0)
    xl820 = p2_pinned[p2_pinned["drug"] == "XL820"].set_index("kinase")["hit"].to_dict()
    for kinase, want in P2_XL820_HITS.items():
        c.check(f"XL820 targets {kinase}", float(xl820.get(kinase, -1)), float(want), 0)
    checks.append(c.report())

    # How far an unpinned re-run drifts, recorded rather than hidden.
    overlap = len(set(p2_free["drug"].astype(str)) & set(figure_drugs))
    print(f"  free re-run recovers {overlap} of the figure's "
          f"{len(figure_drugs)} drugs (tie-break in sort_values('nsub'))")
    p2_free.to_csv(table_path("eval_virus_dms_drug_kinase_demo_free_rerun.csv"), index=False)

    # ---------------- p_drug: drug by phospho-site ----------------
    print("\n=== eval_virus_dms_drug_psite_demo ===")
    pairs, pdrug_filled, drug_order, fill_limit = build_p_drug(
        curated, p2_pinned, p1_max)

    p = table_path("eval_virus_dms_drug_psite_demo.csv")
    pdrug_filled.to_csv(p, index=False)
    print(f"  wrote {os.path.basename(p)}  ({len(pdrug_filled)} cells, "
          f"colour scale capped at {fill_limit})")

    c = Checker("eval_virus_dms_drug_psite_demo")
    c.check("number of drugs on the x axis", float(len(drug_order)), 42.0, 0)
    c.check("number of phospho-sites on the y axis",
            float(pdrug_filled[["substrate", "psite"]].drop_duplicates().shape[0]), 19.0, 0)
    c.check("grid cells", float(len(pdrug_filled)), 798.0, 0)
    c.check("colour-scale maximum", fill_limit, 4.03125, 1e-9)
    # Cell 78 of the notebook checked that no drug is named after a kinase.
    c.check("no rows for a drug named Erk1",
            float(len(pdrug_filled[pdrug_filled["drug"] == "Erk1"])), 0.0, 0)
    checks.append(c.report())

    write_verification(checks, "VERIFICATION_05_virus_demo.csv")


if __name__ == "__main__":
    main()
