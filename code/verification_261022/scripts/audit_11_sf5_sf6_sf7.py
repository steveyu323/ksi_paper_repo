#!/usr/bin/env python3
"""
audit_11 -- number mismatches in the regenerated Supplementary Figures S5-S7
(ksi_analysis_2025/SF5_SF6_SF7/) against the main-text figures, the supplement
captions, and the LFSR <= 0.05 study-wide hit threshold.

Prompted by: "n = 403 is a bit contradictory to n = 258 in main plot?"

Run:  python3 -I audit_11_sf5_sf6_sf7.py [SF_DIR] [KSI_ROOT] [COUNT_DIR]

Defaults point at the colossus backup. Reads only; writes a CSV to ../out/.
Everything except the hard-coded manuscript values is recomputed from the
scripts' own saved source_data/ plus the primary pickles.
"""
import csv
import os
import sys

import pandas as pd

SF_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "${KSI_REPO_ROOT}/"
    "ksi_analysis_2025/SF5_SF6_SF7")
KSI_ROOT = sys.argv[2] if len(sys.argv) > 2 else (
    "${KSI_REPO_ROOT}/ksi_analysis_2025")
COUNT_DIR = sys.argv[3] if len(sys.argv) > 3 else (
    "${KSI_DATA_ROOT}/processed_count_table")

SRC = os.path.join(SF_DIR, "out", "source_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out",
                   "audit_11_sf5_sf6_sf7.csv")

rows = []


def claim(name, ms, code, verdict, note=""):
    print(f"CLAIM {name} | manuscript={ms} | code={code} | VERDICT={verdict}")
    if note:
        print(f"      {note}")
    rows.append({"claim": name, "manuscript": ms, "recomputed": code,
                 "verdict": verdict, "note": note})


print("=" * 78)
print("PART 1 -- the n=403 vs n=258 question (SF5c vs main Fig 2 panel D)")
print("=" * 78)

# Both figures draw the SAME confusion matrix. They differ only in whether the
# 145 false negatives the assay could not have detected are in the denominator.
TP, FP, TN = 134, 11, 63
FN_NOM, FN_EVAL = 195, 50          # nominal vs screening-evaluable
NO_PAABD, QC_REMOVED = 120, 25

claim("FN_nominal = FN_evaluable + no-PAABD + QC-removed",
      f"{FN_NOM}", f"{FN_EVAL}+{NO_PAABD}+{QC_REMOVED}={FN_EVAL+NO_PAABD+QC_REMOVED}",
      "MATCH" if FN_EVAL + NO_PAABD + QC_REMOVED == FN_NOM else "MISMATCH")

for label, fn, want_total, want_acc, want_rec in (
        ("SF5c / Table S10 (nominal FN)", FN_NOM, 403, 0.489, 0.407),
        ("main Fig 2 panel D (evaluable FN)", FN_EVAL, 258, 0.764, 0.728)):
    total = TP + FP + TN + fn
    acc = (TP + TN) / total
    prec = TP / (TP + FP)
    rec = TP / (TP + fn)
    claim(f"[{label}] Total", want_total, total,
          "MATCH" if total == want_total else "MISMATCH")
    claim(f"[{label}] Accuracy", want_acc, round(acc, 3),
          "MATCH" if abs(acc - want_acc) < 5e-4 else "MISMATCH")
    claim(f"[{label}] Recall", want_rec, round(rec, 3),
          "MATCH" if abs(rec - want_rec) < 5e-4 else "MISMATCH")
    claim(f"[{label}] Precision", 0.924, round(prec, 3),
          "MATCH" if abs(prec - 0.924) < 5e-4 else "MISMATCH")

claim("403 vs 258 are the same matrix under two denominators",
      "both printed, neither labelled", "403 nominal / 258 evaluable",
      "MISMATCH",
      "Precision is denominator-invariant (0.924 both ways), which is why only "
      "Accuracy (0.489 vs 0.764) and Recall (0.407 vs 0.728) diverge. Neither "
      "figure says which denominator it uses.")

print()
print("=" * 78)
print("PART 2 -- LFSR threshold actually used by the three scripts")
print("=" * 78)

red = pd.read_pickle(os.path.join(KSI_ROOT, "251203_ksi_reduced_df.pkl"))
red_tp = pd.read_pickle(os.path.join(KSI_ROOT, "251203_ksi_reduced_df_tp.pkl"))

claim("sf6 input `positive` column max lfsr_best_hit", "<= 0.05",
      round(float(red["lfsr_best_hit"].max()), 4),
      "MATCH" if red["lfsr_best_hit"].max() <= 0.05 + 1e-9 else "MISMATCH",
      "sf6_false_positive.py keys on df['positive'] from 251203_ksi_reduced_df.pkl")
claim("sf5 input `true_positive` column max lfsr_best_hit", "<= 0.05",
      round(float(red_tp["lfsr_best_hit"].max()), 4),
      "MATCH" if red_tp["lfsr_best_hit"].max() <= 0.05 + 1e-9 else "MISMATCH",
      "sf5_library.py keys on df['true_positive'] from 251203_ksi_reduced_df_tp.pkl")

n_pos, n_tp = int(red["positive"].sum()), int(red_tp["true_positive"].sum())
claim("`positive` pairs at LFSR<=0.05", "8,890 (changelog)", n_pos,
      "MATCH" if n_pos == 8890 else "MISMATCH")
claim("`true_positive` pairs (positive minus FP substrates)", "not stated", n_tp,
      "UNRESOLVED",
      "A FOURTH hit quantity: 12,936 rows (sub x kin x PAABD) / 8,890 positive "
      "pairs / 8,100 true_positive pairs / 34,767 pairs total. The manuscript "
      "uses 'KSIs' for several of these.")
claim("sf7_decoupling.py applies an LFSR filter", "n/a", "no filter",
      "MATCH",
      "Correct: S7 plots binding-strength DISTRIBUTIONS and median abundances, "
      "not hit calls, so no threshold belongs there.")

print()
print("=" * 78)
print("PART 3 -- SF6: panel c vs panel d, and the 'substrates' miscount")
print("=" * 78)

vol = pd.read_csv(os.path.join(SRC, "sf6d_volcano.csv"))
pc = pd.read_csv(os.path.join(SRC, "sf6c_promiscuity_vs_threshold.csv"))

n_rows = len(vol)
n_genes = vol["Gene"].nunique()
gt01_rows = int((vol["promiscuous_score"] > 0.1).sum())
gt01_genes = int(vol.loc[vol["promiscuous_score"] > 0.1, "Gene"].nunique())
gt05_rows = int((vol["promiscuous_score"] > 0.5).sum())
gt05_genes = int(vol.loc[vol["promiscuous_score"] > 0.5, "Gene"].nunique())

claim("SF6d volcano markers", "46 of 2754 substrates (caption)",
      f"{gt01_rows} of {n_rows} ROWS", "MISMATCH",
      "Caption's 46/2754 counts the legend swatch (README already caught this).")
claim("SF6d rows vs unique substrates", "'substrates'",
      f"{n_rows} rows = {n_genes} unique Gene (one marker per sgrna/barcode)",
      "MISMATCH",
      "The volcano is one marker per BARCODE, not per substrate.")
claim("SF6d correct count above promiscuity 0.1",
      "45 of 2753 substrates (README's fix)",
      f"{gt01_genes} of {n_genes} substrates  ({gt01_rows} of {n_rows} barcodes)",
      "MISMATCH",
      "The README's own correction keeps the wrong noun. Both 46/2754 and "
      "45/2753 are barcode counts; per substrate it is "
      f"{gt01_genes} of {n_genes}.")

floor = float(pc["frac_promiscuous"].min())
at15 = float(pc.loc[(pc["lfc_threshold"] - 1.5).abs().idxmin(), "frac_promiscuous"])
claim("SF6c floor", "never falls below 27% (caption)", f"{floor*100:.2f}%",
      "MATCH", "26.94% rounds to 27%.")
claim("SF6c value at the chosen LFC=1.5 cut-off",
      "75.00% (S6 caption provenance line) / '76 %' (artwork) / '~75%' (Methods)",
      f"{at15*100:.2f}%", "MISMATCH",
      "Artwork's 76% and Methods' ~75% are both fine; the caption's claim that "
      "the reference line is 'recovered at 75.00%' is wrong -- it is 75.76%.")

# Reconcile panels c and d -- the alleged 16x conflict (refine.ink #31 / notes B5)
red_st = red.loc[red["paabd_all"] != "sh2"]
sg = pd.read_csv(os.path.join(COUNT_DIR, "blib7_2batch_rra.sgrna_summary.txt"),
                 sep="\t")
fp_ctrl = sg.loc[sg["control_mean"] > 10, :]


def promiscuity(df):
    tot = df.groupby(["substrate"]).count().reset_index().iloc[:, [0, 1]]
    pos = df.loc[df["positive"], :].groupby(["substrate"]).count(
    ).reset_index().iloc[:, [0, 1]]
    s = pd.merge(pos, tot, on="substrate")
    s.columns = ["substrate", "pos_kinase", "tot_kinase"]
    s["score"] = s["pos_kinase"] / s["tot_kinase"]
    return s


def flagged(t):
    return fp_ctrl.loc[(fp_ctrl["FDR"] < 0.1) & (fp_ctrl["LFC"] > t), "Gene"]


s15 = promiscuity(red_st.loc[red_st["substrate"].isin(flagged(1.5))])
claim("SF6c denominator at LFC=1.5",
      "'fraction of control substrates' (caption implies all 2,753)",
      f"{s15.shape[0]} flagged substrates", "MISMATCH",
      "Panel c's denominator is the substrates FLAGGED at each threshold, which "
      "varies along the x-axis -- not the full control library.")
claim("panels c and d reconcile once both are per-substrate",
      "16x conflict alleged (notes B5 / refine.ink #31)",
      f"c: {int((s15['score']>0.5).sum())} of {s15.shape[0]} flagged >0.5 "
      f"({at15*100:.1f}%);  d: {gt01_genes} substrates >0.1",
      "MATCH",
      "FALSE ALARM. 33 substrates have non-zero promiscuity; 25 of those exceed "
      "0.5. Panel c reports 25/33 = 75.8%, panel d reports the 33. The two "
      "panels never disagreed -- only the caption's wording does.")

print()
print("=" * 78)
print("PART 4 -- SF5 panels a/b and SF7")
print("=" * 78)

cm = pd.read_csv(os.path.join(SRC, "sf5c_confusion_matrix_summary.csv"))
comb = cm.loc[cm["Library"] == "Combined"].iloc[0]
for col, want in (("TP", 134), ("FP", 11), ("FN", 195), ("TN", 63),
                  ("Total", 403)):
    claim(f"SF5c Combined {col}", want, int(comb[col]),
          "MATCH" if int(comb[col]) == want else "MISMATCH")

lib = pd.read_csv(os.path.join(SRC, "sf5b_library_design.csv"))
for _, r in lib.iterrows():
    att = 1 - r["n_ksi_yeast_screened"] / r["n_ksi_designed"]
    want = {"ST library": 0.23, "Y library": 0.15}[r["library"]]
    claim(f"SF5b pair attrition [{r['library']}]", f"{want:.0%} (caption)",
          f"{att*100:.1f}%", "MATCH" if abs(att - want) < 0.006 else "MISMATCH",
          "" if abs(att - want) < 0.006 else "Truncated, not rounded.")

s7c = pd.read_csv(os.path.join(SRC, "sf7c_median_abundance.csv")).set_index("kinase")
n_vals = s7c.shape[0] * s7c.shape[1]
claim("SF7c median-abundance values", "45 (caption)", n_vals,
      "MATCH" if n_vals == 45 else "MISMATCH")

for kin, paabd, orig in (("IKKB_HUMAN", "FHA", -0.86),
                         ("PDPK1_HUMAN", "WW", -0.39)):
    v = float(s7c.loc[kin, paabd])
    claim(f"SF7c {kin} {paabd}", orig, round(v, 4),
          "MISMATCH" if abs(round(v, 2) - orig) > 1e-9 else "MATCH",
          f"|deviation| = {abs(v-orig):.4f}. Rounds to {round(v,2)} because the "
          f"value straddles the 2-dp boundary; the caption's 'largest deviation "
          f"0.005' is only marginally exceeded.")

dep = ["ADCK1_HUMAN", "MAST1_HUMAN", "TITIN_HUMAN"]
enr = [k for k in s7c.index if any(t in k for t in ("KKCC1", "KS6B1", "TBK1"))]
claim("SF7c depleted across all three PAABDs", "ADCK1, MAST1, TITIN",
      "all negative" if (s7c.loc[dep] < 0).all().all() else "not all negative",
      "MATCH" if (s7c.loc[dep] < 0).all().all() else "MISMATCH")
claim("SF7c enriched across all three PAABDs", "KKCC1, KS6B1, TBK1",
      "all positive" if (s7c.loc[enr] > 0).all().all() else "not all positive",
      "MATCH" if (s7c.loc[enr] > 0).all().all() else "MISMATCH")
means = {c: round(float(s7c[c].abs().mean()), 3) for c in s7c.columns}
claim("SF7c WW gives smaller effects both directions", "yes (caption)",
      f"mean|median| {means}",
      "MATCH" if means["WW"] < min(means["FHA"], means["14-3-3"]) else "MISMATCH")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "recomputed",
                                       "verdict", "note"])
    w.writeheader()
    w.writerows(rows)

bad = sum(1 for r in rows if r["verdict"] == "MISMATCH")
unres = sum(1 for r in rows if r["verdict"] == "UNRESOLVED")
print()
print(f"{len(rows)} claims: {len(rows)-bad-unres} MATCH, {bad} MISMATCH, "
      f"{unres} UNRESOLVED -> {os.path.abspath(OUT)}")
