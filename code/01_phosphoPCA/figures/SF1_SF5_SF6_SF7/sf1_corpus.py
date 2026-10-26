#!/usr/bin/env python
"""
Supplementary Figure S1 (SuppFig_corpus): scope and composition of the KINBERT
training corpus.

  a  kinases / organisms / kinase-substrate pairs for three sequence-based KSI
     predictors (Table S1, tab:scope). The KINBERT row is recomputed from the
     corpus; DeepKinZero and Phosformer are the values reported by their authors.
  b  kinases binned by known-substrate count, DB only vs DB + lysate
     (Table S3, tab:ipk).

KINBERT counts follow eval_0_datasource_stats.ipynb cells 18-23, with the fix
from audit_261022/scripts/audit_01_corpus_stats.py: kinases are the distinct
accessions in the corpus (2,021), not len(lysate_list + db_list) (2,304, which
double counts the 304 kinases shared by the two tables).
  kinases    distinct `kinase` among positive rows
  organisms  distinct organisms of corpus kinases (UniProt annotation table)
  pairs      distinct (substrate_seq_15aa, kinase_domain_seq) among positives
  bins       dedup on (substrate_seq_15aa, kinase_domain_seq), first occurrence
             kept, then count per kinase; pd.cut right-closed at 5/20/50/100/200

Inputs
  ksi_analysis_2025/pairs_aug_v2.csv  (DB + lysate, the KINBERT corpus)
  ksi_analysis_2025/pairs_aug_v1.csv  (DB only)
  kinase_ml/code/data_processing/all_kinase_organism_info.tsv

Usage:  python sf1_corpus.py      (reads ~1.2 GB of CSV in chunks, ~2 min)
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # ksi_analysis_2025
PARENT = os.path.dirname(ROOT)
OUT_DIR = os.path.join(HERE, "out")
SRC_DIR = os.path.join(OUT_DIR, "source_data")

CORPUS_V2 = os.path.join(ROOT, "pairs_aug_v2.csv")
CORPUS_V1 = os.path.join(ROOT, "pairs_aug_v1.csv")
ORGANISM_TSV = os.path.join(PARENT, "kinase_ml/code/data_processing/all_kinase_organism_info.tsv")

# as reported by the methods' authors (Table S1)
PUBLISHED = {
    "DeepKinZero": {"Kinases": 343, "Organisms": 1, "Kinase-substrate pairs": 13426},
    "Phosformer": {"Kinases": 800, "Organisms": 13, "Kinase-substrate pairs": 24534},
}
MODELS = ["DeepKinZero", "Phosformer", "KINBERT"]
STATS = ["Kinases", "Organisms", "Kinase-substrate pairs"]

BINS = [0, 5, 20, 50, 100, 200, float("inf")]
BIN_LABELS = ["1–5", "5–20", "20–50", "50–100", "100–200", ">200"]
CHUNK = 500_000

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#C8C8C8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})


def read_positives(path):
    cols = ["kinase", "label", "kinase_domain_seq", "substrate_seq_15aa"]
    chunks = [c.loc[c["label"] == 1, ["kinase", "kinase_domain_seq", "substrate_seq_15aa"]]
              for c in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False)]
    return pd.concat(chunks, ignore_index=True).astype(str)


def substrate_bins(pos):
    ks_df = pos.drop_duplicates(["substrate_seq_15aa", "kinase_domain_seq"])
    per_kinase = ks_df.groupby("kinase").size()
    binned = pd.cut(per_kinase, bins=BINS, labels=BIN_LABELS, right=True, include_lowest=True)
    return binned.value_counts().reindex(BIN_LABELS).fillna(0).astype(int)


def main():
    os.makedirs(SRC_DIR, exist_ok=True)

    v2 = read_positives(CORPUS_V2)
    org = pd.read_csv(ORGANISM_TSV, sep="\t", usecols=["From", "Organism"])
    corpus_kinases = set(v2["kinase"])
    kinbert = {
        "Kinases": len(corpus_kinases),
        "Organisms": org.loc[org["From"].isin(corpus_kinases), "Organism"].nunique(),
        "Kinase-substrate pairs": len(v2.drop_duplicates(["substrate_seq_15aa", "kinase_domain_seq"])),
    }
    scope = pd.DataFrame({**PUBLISHED, "KINBERT": kinbert}).T.loc[MODELS, STATS]

    bins = pd.DataFrame({"DB + lysate": substrate_bins(v2)})
    del v2
    bins["DB only"] = substrate_bins(read_positives(CORPUS_V1))

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7.1, 2.0))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.3], wspace=0.3)
    gs_a = outer[0, 0].subgridspec(1, 3, wspace=0.45)
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(3)]
    ax_b = fig.add_subplot(outer[0, 1])

    y = np.arange(len(MODELS))[::-1]
    xmax = {"Kinases": 3000, "Organisms": 130, "Kinase-substrate pairs": 230_000}
    ticks = {"Kinases": ([0, 1000, 2000], ["0", "1k", "2k"]),
             "Organisms": ([0, 50, 100], ["0", "50", "100"]),
             "Kinase-substrate pairs": ([0, 75_000, 150_000], ["0", "75k", "150k"])}
    for i, (ax, stat) in enumerate(zip(axes_a, STATS)):
        vals = scope[stat].values
        ax.barh(y, vals, height=0.55, color=[GREY, GREY, BLUE])
        for yy, v in zip(y, vals):
            ax.text(v + xmax[stat] * 0.03, yy, f"{v:,}", va="center", fontsize=7, color="#333333")
        ax.set_xlim(0, xmax[stat])
        ax.set_xticks(*ticks[stat])
        ax.set_xlabel(stat)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS if i == 0 else [])
        ax.tick_params(axis="y", length=0)

    x = np.arange(len(BIN_LABELS))
    ax_b.plot(x, bins["DB + lysate"], "-o", color=BLUE, ms=3.5, lw=1.2, label="DB + lysate")
    ax_b.plot(x, bins["DB only"], "--s", color=ORANGE, ms=3.5, lw=1.2, label="DB only")
    ax_b.set_yscale("log")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(BIN_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    ax_b.set_xlabel("Known substrates per kinase")
    ax_b.set_ylabel("Number of kinases")
    ax_b.legend(frameon=False, fontsize=7.5, loc="upper right")

    axes_a[0].text(-0.25, 1.04, "a", transform=axes_a[0].transAxes, fontsize=9,
                   fontweight="bold", va="bottom")
    ax_b.text(-0.25, 1.04, "b", transform=ax_b.transAxes, fontsize=9,
              fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"SuppFig_corpus.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── source data + summary ─────────────────────────────────────────────────
    scope.to_csv(os.path.join(SRC_DIR, "sf1a_training_scope.csv"))
    bins.to_csv(os.path.join(SRC_DIR, "sf1b_substrate_bins.csv"))
    print("a:\n" + scope.to_string())
    print("b:\n" + bins.T.to_string())
    print("   totals:", bins.sum().to_dict())
    print("wrote", os.path.join(OUT_DIR, "SuppFig_corpus.pdf"))


if __name__ == "__main__":
    main()
