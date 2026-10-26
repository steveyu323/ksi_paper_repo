#!/usr/bin/env python3
"""audit_10 — internal arithmetic of Supp Tables S8/S9 and the counts quoted in
Results/Fig 1/Fig 2 captions. Pure arithmetic on the printed values: catches
self-inconsistency, not provenance. No data roots needed."""
import csv, sys, os

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "audit_10_library_tables.csv")
rows = []
def claim(name, ms, code, ok):
    v = "MATCH" if ok else "MISMATCH"
    print(f"CLAIM {name} | manuscript={ms} | code={code} | VERDICT={v}")
    rows.append({"claim": name, "manuscript": ms,
                                                "recomputed": code, "verdict": v})

# ---- Table S8: mean reads per barcode = after-ORF-QC reads / final unique barcodes
S8 = {"ST (blib5)": dict(initial=3039052, bc=2445887, flank=2133880, orf=1631649,
                         final=121070, mean=11.27),
      "Y (blib1)":  dict(initial=2381935, bc=1935748, flank=1691475, orf=1286517,
                         final=205272, mean=5.31)}
for lib, d in S8.items():
    calc = d["orf"] / d["final"]
    claim(f"S8 mean_reads_per_barcode[{lib}]", d["mean"], round(calc, 4),
          abs(calc - d["mean"]) < 0.005)
    mono = d["initial"] >= d["bc"] >= d["flank"] >= d["orf"]
    claim(f"S8 QC_stages_monotonic[{lib}]", "decreasing", mono, mono)

# Results text rounds these to 11.3 and 5.3
for lib, want in (("ST (blib5)", 11.3), ("Y (blib1)", 5.3)):
    got = round(S8[lib]["mean"], 1)
    claim(f"Results rounding[{lib}]", want, got, got == want)

# ---- Table S9: designed pairs must equal kinases x designed substrates
S9 = {"Y":  dict(kin=8,  sub_des=1950, sub_con=1939, sub_scr=1932,
                 p_des=15600, p_con=14873, p_scr=13220),
      "ST": dict(kin=15, sub_des=1978, sub_con=1968, sub_scr=1964,
                 p_des=29670, p_con=25214, p_scr=22695)}
for lib, d in S9.items():
    exp = d["kin"] * d["sub_des"]
    claim(f"S9 designed_pairs[{lib}] = kinases x designed_substrates",
          d["p_des"], exp, exp == d["p_des"])
    # constructed/screened pairs should NOT exceed kinases x corresponding substrates
    for stage, pk, sk in (("constructed", "p_con", "sub_con"),
                          ("screened", "p_scr", "sub_scr")):
        ceil = d["kin"] * d[sk]
        ok = d[pk] <= ceil
        claim(f"S9 {stage}_pairs[{lib}] <= kinases x {stage}_substrates",
              d[pk], f"<= {ceil}", ok)
    mono = d["sub_des"] >= d["sub_con"] >= d["sub_scr"]
    claim(f"S9 substrate_attrition_monotonic[{lib}]", "decreasing", mono, mono)
    mono_p = d["p_des"] >= d["p_con"] >= d["p_scr"]
    claim(f"S9 pair_attrition_monotonic[{lib}]", "decreasing", mono_p, mono_p)
    # SuppFig_library panel b quotes overall attrition
    att = 1 - d["p_scr"] / d["p_des"]
    claim(f"S9 overall_pair_attrition[{lib}]", {"ST": 0.23, "Y": 0.15}[lib],
          round(att, 4), abs(att - {"ST": 0.23, "Y": 0.15}[lib]) < 0.005)

# ---- Totals quoted in Fig 1 / Fig 2 panel A vs Results
con_tot = S9["Y"]["p_con"] + S9["ST"]["p_con"]
scr_tot = S9["Y"]["p_scr"] + S9["ST"]["p_scr"]
claim("Fig1/Fig2A total_KSIs (constructed)", 40087, con_tot, con_tot == 40087)
claim("Results total_KSIs (screened)", 35915, scr_tot, scr_tot == 35915)
claim("constructed != screened (labels must differ)", "both shown as 'KSIs'",
      f"{con_tot} vs {scr_tot}", False)

# ---- Table S9 kinase counts vs the 23-construct panel
claim("S9 kinase_total", 23, S9["Y"]["kin"] + S9["ST"]["kin"],
      S9["Y"]["kin"] + S9["ST"]["kin"] == 23)

# ---- Substrate pool size quoted in Results ("~2,000")
for lib, d in S9.items():
    claim(f"Results substrate_pool ~2000 [{lib}]", "~2000", d["sub_des"],
          1900 <= d["sub_des"] <= 2100)

# ---- Table S8 follow-up: which denominator reproduces the printed mean?
# "After ORF QC reads / Final unique barcodes" does NOT give the printed mean.
# The printed means imply a LARGER barcode denominator, consistent across both
# libraries at ~84% of barcodes being confidently assigned -- i.e. the mean
# appears to be taken over ALL observed barcodes, before the >=90% unambiguous
# assignment filter that produces "Final unique barcodes".
for lib, d in S8.items():
    implied = d["orf"] / d["mean"]
    frac = d["final"] / implied
    claim(f"S8 implied_barcode_denominator[{lib}]",
          f"final unique = {d['final']}", f"implied = {implied:,.0f}", False)
    claim(f"S8 implied_assignment_rate[{lib}]", "not stated",
          f"{frac*100:.1f}% confidently assigned", 0.80 < frac < 0.90)

with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "recomputed", "verdict"])
    w.writeheader(); w.writerows(rows)
n_bad = sum(1 for r in rows if r["verdict"] != "MATCH")
print(f"\n{len(rows)} claims checked, {n_bad} MISMATCH -> {os.path.abspath(OUT)}")
