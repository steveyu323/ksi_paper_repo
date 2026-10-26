#!/usr/bin/env python
"""
Supplementary Figure S5 (SuppFig_library): library construction, screening
and database benchmark.

Panel provenance
  a  1_parse_lookup_tables.ipynb cells 2, 6 (make_pacbio_lookup /
     build_canonical_lookup read counts; Table S8 / tab:libstats)
       Initial CCS  -> all HiFi reads
       Barcode QC   -> both barcode flanks present
       Flank QC     -> + ks_left, ks_right and linker present
       ORF QC       -> + 19-nt barcode, 15-aa substrate without stop,
                         kinase ORF ending in a single stop codon
  b  8_2batch_blib_pipeline_v2.ipynb cells 13, 15, 18 (stats_df;
     Table S9 / tab:libdesign)
  c  8_2batch_blib_pipeline_v2.ipynb cells 9-11, 21-26, 32
     (confusion_matrix_summary.csv; Table S10 / tab:spike)

Panel a streams ~5.4 M PacBio reads (a few minutes); the counts are cached in
out/source_data/sf5a_ccs_qc_counts.csv. Pass --recount to rebuild the cache.

Usage:  python sf5_library.py [--recount]
"""

import argparse
import gzip
import os
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import BiopythonWarning
from Bio.Seq import Seq, translate

warnings.simplefilter("ignore", BiopythonWarning)
warnings.simplefilter("ignore", pd.errors.SettingWithCopyWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # ksi_analysis_2025
PARENT = os.path.dirname(ROOT)
OUT_DIR = os.path.join(HERE, "out")
SRC_DIR = os.path.join(OUT_DIR, "source_data")

# Sequencing data root. Set KSI_DATA_ROOT to wherever the PacBio run lives;
# the default is only a template and will not resolve anywhere.
DATA_ROOT = os.environ.get("KSI_DATA_ROOT", "/path/to/sequencing_data_root")

CCS = {
    "ST (blib5)": os.path.join(
        DATA_ROOT, "20231106_Yu_C/blib5-ksb.CY/CCS_12461/"
        "m64076_231104_154224.hifi_reads.fasta.gz"),
    "Y (blib1)": os.path.join(
        DATA_ROOT, "20231106_Yu_C/blib1-ksb.CY/CCS_12462/"
        "m64076_231104_004250.hifi_reads.fasta.gz"),
}
# make_pacbio_lookup defaults (1_parse_lookup_tables.ipynb cell 2)
RC_TRIGGER = "CGTACGCACAATTC"
BARCODE_LEFT = "TATCCGGCTCGTATGTTGTGTGGAATTGTGCGTACG"
BARCODE_RIGHT = "CCTAGGCAGCTATGACCATGATT"
KS_LEFT = "GGCGGTGGCGGGTCTGGTGGAGGAGGGTCAGGTGGTGGCGGCTCGGGAGGTGGAGCTAGC"
KS_RIGHT = "AAGCTTATTAGTTATGTCAC"
LINKER = "GGTGGATCAGGTGGTGGTGGTTCTGGTGGC"

STAGES = ["Initial\nCCS", "Barcode\nQC", "Flank\nQC", "ORF\nQC"]
EXCLUDE_SOURCES = {"ms", "y2h", "phosphonetwork", "deepkinzero", "phosphoelm", "kinasephos"}
EXCLUDE_KINASES = {"KS6B1_HUMAN_active", "KS6B1_HUMAN_wt", "PLK1_HUMAN_active"}

BLUE, ORANGE, GREEN = "#0072B2", "#E69F00", "#009E73"

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


# ── panel a ───────────────────────────────────────────────────────────────────
def read_fasta_seqs(path):
    with gzip.open(path, "rt") as handle:
        chunks = []
        for line in handle:
            if line.startswith(">"):
                if chunks:
                    yield "".join(chunks)
                chunks = []
            else:
                chunks.append(line.strip())
        if chunks:
            yield "".join(chunks)


def orf_pass(barcode, substrate, kinase):
    """build_canonical_lookup filters before the barcode-assignment step."""
    if len(barcode) != 19:
        return False
    sub_aa = translate(substrate)
    if len(sub_aa) != 15 or "*" in sub_aa:
        return False
    kin_aa = translate(kinase)
    return kin_aa.endswith("*") and "*" not in kin_aa[:-1]


def count_ccs_stages(path):
    n = [0, 0, 0, 0]
    for s in read_fasta_seqs(path):
        if RC_TRIGGER in s:
            s = str(Seq(s).reverse_complement())
        n[0] += 1
        if BARCODE_LEFT not in s or BARCODE_RIGHT not in s:
            continue
        n[1] += 1
        if KS_LEFT not in s or KS_RIGHT not in s:
            continue
        ksregion = s.split(KS_LEFT)[1].split(KS_RIGHT)[0]
        if LINKER not in ksregion:
            continue
        n[2] += 1
        barcode = s.split(BARCODE_LEFT)[1].split(BARCODE_RIGHT)[0]
        substrate, kinase = ksregion.split(LINKER)[0], ksregion.split(LINKER)[1]
        if orf_pass(barcode, substrate, kinase):
            n[3] += 1
    return n


def panel_a_counts(recount):
    cache = os.path.join(SRC_DIR, "sf5a_ccs_qc_counts.csv")
    if os.path.exists(cache) and not recount:
        return pd.read_csv(cache, index_col=0)
    rows = {}
    for lib, path in CCS.items():
        print(f"counting {lib}: {path}")
        rows[lib] = count_ccs_stages(path)
    df = pd.DataFrame(rows, index=[s.replace("\n", " ") for s in STAGES]).T
    df.to_csv(cache)
    return df


# ── panel b ───────────────────────────────────────────────────────────────────
def load_screen():
    """8_2batch_blib_pipeline_v2.ipynb cell 15."""
    blib = pd.read_csv(os.path.join(PARENT, "screening_analysis/blib_kinases_with_acc.csv"), header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]
    ksi_df = pd.read_pickle(os.path.join(ROOT, "dat_all_blib_2batch.pkl")).drop_duplicates(
        ["substrate_aa", "kinase_aa"])
    ksi_df = ksi_df.drop(["substrate"], axis=1)
    ksi_df.columns = [
        "id", "num", "neg|score", "neg|p-value", "neg|fdr", "neg|rank",
        "neg|goodsgrna", "neg|lfc", "pos|score", "pos|p-value", "pos|fdr",
        "pos|rank", "pos|goodsgrna", "pos|lfc", "sgrna", "Gene",
        "control_count", "treatment_count", "control_mean", "treat_mean", "LFC",
        "control_var", "adj_var", "score", "p.low", "p.high", "p.twosided",
        "FDR", "high_in_treatment", "identifier", "length", "seq",
        "barcode_seq", "ksregion", "kinase", "substrate",
        "kinase_seq", "kinase_type", "kinase_name", "kinase_acc", "paabd",
        "pre_1", "mtx_1", "pre_2", "mtx_2",
    ]
    return blib, ksi_df


def panel_b_counts(blib, ksi_df):
    """8_2batch_blib_pipeline_v2.ipynb cells 13, 18."""
    wl = os.path.join(PARENT, "kinase_ml/code/wetlab_planning")
    y_sub_all = pd.read_csv(f"{wl}/y_sub_all.csv").drop_duplicates("substrate_seq_15aa")
    st_sub_all = pd.read_csv(f"{wl}/st_sub_all.csv").drop_duplicates("substrate_seq_15aa")
    y_lookup = pd.read_pickle(os.path.join(ROOT, "lookup_tables/y.lookup.corrected.pkl"))
    st_lookup = pd.read_pickle(os.path.join(ROOT, "lookup_tables/st.lookup.corrected.pkl"))
    ksi_df_y = ksi_df.loc[ksi_df["kinase_type"] == "Y", :]
    ksi_df_st = ksi_df.loc[ksi_df["kinase_type"] != "Y", :]

    out = []
    for lib, sub_all, lookup, ksi, kinase_seqs in [
        ("Y library", y_sub_all, y_lookup, ksi_df_y,
         blib.loc[blib["kinase_type"] == "Y", "kinase_seq"].tolist()),
        ("ST library", st_sub_all, st_lookup, ksi_df_st,
         ksi_df_st.loc[ksi_df_st["kinase_type"] == "ST", "kinase_seq"].drop_duplicates().tolist()),
    ]:
        subs = sub_all["substrate_seq_15aa"]
        designed = {s + "&" + k for s in subs.tolist() for k in kinase_seqs}
        ecoli = set(lookup["substrate_aa"] + "&" + lookup["kinase_aa"]) & designed
        yeast = set(ksi["substrate"] + "&" + ksi["kinase_seq"]) & designed
        out.append({
            "library": lib,
            "n_kinase": ksi["kinase_name"].nunique(),
            "n_substrate_designed": len(sub_all),
            "n_substrate_ecoli_constructed": int(lookup["substrate_aa"].drop_duplicates().isin(subs).sum()),
            "n_substrate_yeast_screened": int(ksi["substrate"].drop_duplicates().isin(subs).sum()),
            "n_ksi_designed": len(designed),
            "n_ksi_ecoli_constructed": len(ecoli),
            "n_ksi_yeast_screened": len(yeast),
        })
    return pd.DataFrame(out).set_index("library")


# ── panel c ───────────────────────────────────────────────────────────────────
def assign_paabd(dat, get_sub_15=True):
    """8_2batch_blib_pipeline_v2.ipynb cell 5 (motif flags used by later cells)."""
    if get_sub_15:
        dat["substrate_seq_15aa"] = dat.apply(
            lambda s: s["substrate_seq"][s["p_ind"] - 1 - 7: s["p_ind"] - 1 + 8], axis=1)
    return dat.loc[dat["substrate_seq_15aa"].apply(len) == 15, :]


def panel_c_stats(blib):
    """8_2batch_blib_pipeline_v2.ipynb cells 9-11, 21-26, 32."""
    data = os.path.join(PARENT, "kinase_ml/data")
    human_kinome = pd.read_pickle(f"{data}/human_kinome_seq_230301.pkl")

    interaction_classed = assign_paabd(
        pd.read_pickle(f"{data}/interactions/interaction_classed_221017_wsource.pkl"))
    interaction_classed = (interaction_classed
                           .loc[:, ["kinase", "substrate_seq_15aa", "source_all"]]
                           .drop_duplicates().rename(columns={"source_all": "source"}))
    interaction_classed["y_true"] = 1

    test_pred = assign_paabd(pd.read_pickle(f"{data}/kinbertv1_pred/kinbertv1_DB_Test.pkl"))
    test_pred = (test_pred.loc[:, ["kinase_seq", "substrate_seq_15aa", "y_true", "source"]]
                 .merge(human_kinome, on="kinase_seq")
                 .loc[:, ["kinase", "substrate_seq_15aa", "source", "y_true"]])

    data_anno = pd.concat([interaction_classed, test_pred], ignore_index=True)
    train_dat_pos = data_anno.loc[~data_anno["source"].isin(EXCLUDE_SOURCES), :]

    dat_in_screen = pd.read_pickle(os.path.join(ROOT, "251203_ksi_reduced_df_tp.pkl"))
    dat_in_screen = pd.merge(blib, dat_in_screen, on="kinase")
    dat_in_screen = dat_in_screen.loc[~dat_in_screen["kinase"].isin(EXCLUDE_KINASES), :]

    curr_all = pd.merge(dat_in_screen, train_dat_pos,
                        left_on=["kinase_acc", "substrate"],
                        right_on=["kinase", "substrate_seq_15aa"])
    curr_all["set"] = "train_tn"
    curr_all.loc[curr_all["y_true"] == 1, "set"] = "train_pos"

    rows = []
    for label, df in [("Y library", curr_all.loc[curr_all["kinase_type"] != "ST"]),
                      ("ST library", curr_all.loc[curr_all["kinase_type"] == "ST"]),
                      ("Combined", curr_all)]:
        s = df["true_positive"].astype(bool)
        db_pos = ~df["set"].eq("train_tn")
        TP, FP = int((s & db_pos).sum()), int((s & ~db_pos).sum())
        FN, TN = int((~s & db_pos).sum()), int((~s & ~db_pos).sum())
        total = TP + FP + FN + TN
        rows.append({"Library": label, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "Total": total,
                     "Accuracy": (TP + TN) / total, "Precision": TP / (TP + FP),
                     "Recall": TP / (TP + FN)})
    return pd.DataFrame(rows)


# ── figure ────────────────────────────────────────────────────────────────────
def plot(ccs, design, conf):
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.3),
                             gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.1, 0.95]))

    ax = axes[0]
    x = np.arange(len(STAGES))
    ax.plot(x, ccs.loc["ST (blib5)"] / 1e6, "-o", color=BLUE, ms=3.5, lw=1.2, label="ST (blib5)")
    ax.plot(x, ccs.loc["Y (blib1)"] / 1e6, "--s", color=ORANGE, ms=3.5, lw=1.2, label="Y (blib1)")
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylim(0, 3.4)
    ax.set_xlabel("Quality-control stage")
    ax.set_ylabel("Long reads (millions)")
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)

    ax = axes[1]
    cols = ["n_ksi_designed", "n_ksi_ecoli_constructed", "n_ksi_yeast_screened"]
    w = 0.36
    for off, lib, color in [(-w / 2, "ST library", BLUE), (w / 2, "Y library", ORANGE)]:
        vals = design.loc[lib, cols].values / 1e3
        bars = ax.bar(np.arange(3) + off, vals, w, color=color, label=lib)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + w / 2, v + 0.3, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=6.5, color="#333333")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Designed", "Constructed\n($E.\\ coli$)", "Screened\n(yeast)"], fontsize=7.5)
    ax.set_ylim(0, 36)
    ax.set_ylabel("Kinase-substrate pairs (thousands)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2, handlelength=1.2,
              bbox_to_anchor=(1.02, 1.04))

    ax = axes[2]
    metrics = ["Accuracy", "Recall", "Precision"]
    ax.axhspan(0.5, 1.5, color="#EEEEEE", lw=0, zorder=0)
    for k, (lib, label, color, marker) in enumerate([
        ("Y library", "Y", BLUE, "o"), ("ST library", "ST", ORANGE, "s"),
        ("Combined", "Combined", GREEN, "^"),
    ]):
        row = conf.set_index("Library").loc[lib]
        ax.scatter([row[m] for m in metrics], np.arange(3) + 0.22 - 0.22 * k, color=color,
                   marker=marker, s=14, zorder=3,
                   label=f"{label} ($n$ = {int(row['Total'])})")
    ax.set_yticks(range(3))
    ax.set_yticklabels(metrics)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Value")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", handletextpad=0.2,
              borderaxespad=0.2)

    for ax, letter in zip(axes, "abc"):
        ax.text(-0.15, 1.04, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"SuppFig_library.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recount", action="store_true", help="re-stream the PacBio FASTAs for panel a")
    args = ap.parse_args()
    os.makedirs(SRC_DIR, exist_ok=True)

    ccs = panel_a_counts(args.recount)
    blib, ksi_df = load_screen()
    design = panel_b_counts(blib, ksi_df)
    del ksi_df
    conf = panel_c_stats(blib)

    design.to_csv(os.path.join(SRC_DIR, "sf5b_library_design.csv"))
    conf.round(3).to_csv(os.path.join(SRC_DIR, "sf5c_confusion_matrix_summary.csv"), index=False)
    plot(ccs, design, conf)

    print("a:\n" + ccs.to_string())
    print("b:\n" + design.to_string())
    att = 1 - design["n_ksi_yeast_screened"] / design["n_ksi_designed"]
    print("   overall pair attrition:", att.round(3).to_dict())
    print("c:\n" + conf.round(3).to_string(index=False))
    print("wrote", os.path.join(OUT_DIR, "SuppFig_library.pdf"))


if __name__ == "__main__":
    main()
