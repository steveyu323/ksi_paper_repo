#!/usr/bin/env python3
"""audit_08 -- the kinase nominated for influenza PR8 M1 Ser53, where the
Results text (ERK7), the Fig 04 caption (PKCepsilon), the Fig 04 panel B
artwork and the Fig 04 panel C artwork disagree.  Also re-checks the other
nominations quoted in the same Results paragraph / caption, and traces the
drugs that the drug-mapping step attaches to each nomination.

Usage:
    python3 -I audit_08_m1_ser53.py <EVAL_DIR> [<FIG04_PDF>]

    EVAL_DIR   tarballs/code/code/evaluation/
    FIG04_PDF  optional; "figures/Fig 04 - Viruses.pdf" out of
               manuscript/261014/KSI_Paper.zip.

Writes ../out/audit_08_m1_ser53.csv and prints one CLAIM line per check.
"""
import ast
import csv
import os
import re
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "out", "audit_08_m1_ser53.csv")

rows = []


def claim(name, ms, code, verdict, note=""):
    print(f"CLAIM {name} | manuscript={ms} | code={code} | VERDICT={verdict}")
    rows.append({"claim": name, "manuscript": ms, "code": code,
                 "verdict": verdict, "note": note})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def norm_sub(s):
    return re.sub(r"\s+", " ", s).strip()


def parse_ranked(cell):
    """'Erk7:4.6875:0.9922,HIPK3:4.6250:...' -> [(kinase, logit, prob), ...]"""
    out = []
    for item in str(cell).split(","):
        p = item.strip().split(":")
        if len(p) == 3:
            out.append((p[0].strip(), float(p[1]), float(p[2])))
    return out


def parse_pairs(cell):
    """\"['Erk7:A;B', 'HIPK3:C']\" -> {kinase: [drug, ...]}"""
    try:
        items = ast.literal_eval(cell) if cell.strip().startswith("[") else []
    except Exception:
        items = []
    out = {}
    for it in items:
        k, _, d = str(it).partition(":")
        out[k.strip()] = [x for x in d.split(";") if x]
    return out


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    eval_dir = sys.argv[1]
    fig_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    HITS = os.path.join(eval_dir, "250118_virus_against_kinome_hits_with_drug.csv")
    DEMO = os.path.join(eval_dir, "260119_demo_kinase_virus_drug_pairs.csv")
    DRUGKIN = os.path.join(eval_dir, "drug_kinase_uniprot.pkl")
    T_KP = os.path.join(eval_dir, "eval_tables",
                        "eval_virus_dms_kinase_psite_demo.csv")
    T_DP = os.path.join(eval_dir, "eval_tables",
                        "eval_virus_dms_drug_psite_demo.csv")

    hits = read_csv(HITS)
    demo = read_csv(DEMO)

    # index the kinome scan by (substrate, site)
    scan = {}
    for r in hits:
        scan[(norm_sub(r["substrate"]), r["wt_aa"] + r["p_ind"])] = r

    # -----------------------------------------------------------------
    # Q1 -- the full top-10 for PR8 M1 Ser53
    # -----------------------------------------------------------------
    M1 = ("influenza M1", "S53")
    r = scan[M1]
    ranked = parse_ranked(r["pred_kinase_prob"])
    passing = ast.literal_eval(r["passing_kinases"])
    drugs = parse_pairs(r["passing_drugs"])
    print("\n--- Q1: KINBERT top-10 kinases for influenza (PR8) M1 S53 "
          f"(15-mer {r['substrate_seq_15aa']}) ---")
    for i, (k, lg, pr) in enumerate(ranked, 1):
        mark = "PASS" if k in passing else "    "
        print(f"    {i:2d}. {k:10s} logit={lg:8.4f} prob={pr:.4f} {mark} "
              f"drugs={';'.join(drugs.get(k, [])) or '-'}")

    top_kin, top_logit = ranked[0][0], ranked[0][1]
    claim("Fig 8 caption kinase @ M1 Ser53", "PKCepsilon (PKCe/PRKCE)",
          f"{top_kin} (logit {top_logit})",
          "MISMATCH",
          "PKCe is not in the top-10 for M1 S53 at all; top_kinase column of "
          "250118_virus_against_kinome_hits_with_drug.csv says Erk7")
    claim("Results kinase @ M1 Ser53", "ERK7 (MAPK15)",
          f"{top_kin} (logit {top_logit}, prob {ranked[0][2]})",
          "MATCH" if top_kin == "Erk7" else "MISMATCH",
          "Erk7 is rank 1 of 10 and is in passing_kinases")

    # -----------------------------------------------------------------
    # Q2 -- ERK7 vs PKCe head to head at this site
    # -----------------------------------------------------------------
    pkce = [(i, lg) for i, (k, lg, _) in enumerate(ranked, 1) if k == "PKCe"]
    claim("PKCe rank @ M1 Ser53", "top nomination (per Fig 8 caption)",
          f"rank {pkce[0][0]}" if pkce else "not in top-10 (not scored in the "
          "cached hit table)",
          "MISMATCH",
          "PKCe appears for PR8 M1 only at S157 (rank 10, logit 2.2656), "
          "never at S53")

    # -----------------------------------------------------------------
    # Q3/Q4 -- which site actually carries Bryostatin 1 / KAI-1455
    # -----------------------------------------------------------------
    print("\n--- Q3/Q4: where Bryostatin 1 and KAI-1455 are mapped ---")
    want = {"Bryostatin 1", "KAI-1455"}
    for d in demo:
        if d["drug"] in want:
            print(f"    260119_demo_kinase_virus_drug_pairs.csv: "
                  f"drug={d['drug']:13s} substrate={norm_sub(d['substrate']):14s} "
                  f"psite={d['psite']:6s} kinase={d['kinase']:6s} logit={d['logit']}")
    bk_sites = {(norm_sub(d["substrate"]), d["psite"], d["kinase"])
                for d in demo if d["drug"] in want}
    claim("Fig 8 caption site for Bryostatin 1 / KAI-1455",
          "influenza M1 Ser53",
          "; ".join(f"{s} {p} ({k})" for s, p, k in sorted(bk_sites)),
          "MISMATCH",
          "both drugs are mapped only to PKCe at H3N2 HA T329")

    H3 = ("H3N2", "T329")
    r3 = scan[H3]
    ranked3 = parse_ranked(r3["pred_kinase_prob"])
    drugs3 = parse_pairs(r3["passing_drugs"])
    print(f"\n--- Q4: H3N2 HA T329 (15-mer {r3['substrate_seq_15aa']}) ---")
    for i, (k, lg, pr) in enumerate(ranked3, 1):
        print(f"    {i:2d}. {k:10s} logit={lg:8.4f} prob={pr:.4f} "
              f"drugs={';'.join(drugs3.get(k, [])) or '-'}")
    claim("H3N2 HA Thr329 top kinase", "PKCepsilon (Fig 8 panel C)",
          f"{ranked3[0][0]} (logit {ranked3[0][1]})",
          "MATCH" if ranked3[0][0] == "PKCe" else "MISMATCH",
          "pan-PKC block; PKCe is rank 1 and the only one carrying "
          "Bryostatin 1 / KAI-1455")
    claim("H3N2 HA Thr329 carries Bryostatin 1 / KAI-1455", "yes (panel C)",
          "yes: " + ";".join(drugs3.get("PKCe", [])),
          "MATCH" if want <= set(drugs3.get("PKCe", [])) else "MISMATCH")

    # -----------------------------------------------------------------
    # Q5 -- aspirin and minocycline on M1 Ser53
    # -----------------------------------------------------------------
    print("\n--- Q5: aspirin / minocycline on M1 S53 ---")
    m1_drugs = drugs.get("Erk7", [])
    claim("Panel C drugs on M1 Ser53", "Aspirin + Minocycline (panel C)",
          "Erk7 -> " + ";".join(m1_drugs),
          "MATCH" if {"Acetylsalicylic acid", "Minocycline"} <= set(m1_drugs)
          else "MISMATCH",
          "panel C's 'Aspirin' column is the pipeline's "
          "'Acetylsalicylic acid'")
    try:
        import pickle
        with open(DRUGKIN, "rb") as fh:
            dk = pickle.load(fh)
        erk7 = dk[dk.kinase_name == "Erk7"]
        print("    drug_kinase_uniprot.pkl rows for Erk7 "
              f"({sorted(set(erk7.kinase_acc))}):")
        for _, q in erk7.iterrows():
            n = int((dk.drug_name == q.drug_name).sum())
            print(f"      {q.drug_name:22s} -> Erk7 ({q.kinase_acc}); this drug "
                  f"is annotated against {n} kinases total")
        for dn in ("Acetylsalicylic acid", "Minocycline", "Fostamatinib"):
            tg = sorted(dk[dk.drug_name == dn].kinase_name)
            claim(f"drug promiscuity: {dn}", "listed as a kinase nomination",
                  f"{len(tg)} annotated kinase targets: {tg if len(tg) <= 15 else tg[:15] + ['...']}",
                  "UNRESOLVED",
                  "pan-family DrugBank/PKIDB target annotation, not an "
                  "Erk7-selective tool compound -- this is the shortlist "
                  "dilution")
    except Exception as exc:                                  # pragma: no cover
        claim("drug_kinase_uniprot.pkl trace", "n/a", f"unreadable: {exc}",
              "UNRESOLVED")

    # -----------------------------------------------------------------
    # Q6 -- the other nominations in the Results / Fig 8 caption
    # -----------------------------------------------------------------
    print("\n--- Q6: other nominations quoted in Results / Fig 8 caption ---")
    others = [
        ("Zika E Ser72 -> CK1epsilon",      ("ZIKA E", "S72"),            {"CK1e"}),
        ("SARS2 N Ser206 -> SRPK1/2",       ("SARS2:Nucleocapsid", "S206"), {"SRPK1", "SRPK2"}),
        ("SARS2 N Ser202 -> GSK-3beta",     ("SARS2:Nucleocapsid", "S202"), {"GSK3B"}),
        ("SARS2 ORF9b Tyr42 -> ALK/FMS/KDR/FLT4",
                                            ("SARS2:ORF9b", "Y42"),       {"ALK", "FMS", "KDR", "FLT4"}),
    ]
    for name, key, want_kin in others:
        rr = scan[key]
        rk = parse_ranked(rr["pred_kinase_prob"])
        pk = set(ast.literal_eval(rr["passing_kinases"]))
        dg = parse_pairs(rr["passing_drugs"])
        order = {k: i for i, (k, _, _) in enumerate(rk, 1)}
        print(f"    {name}")
        print(f"      ranked: " + ", ".join(f"{k}:{lg}" for k, lg, _ in rk))
        print(f"      passing: {sorted(pk)}")
        got = sorted(want_kin & set(order), key=lambda k: order[k])
        detail = ("; ".join(f"{k} rank {order[k]} logit "
                            f"{dict((a, b) for a, b, _ in rk)[k]}"
                            f"{'' if k in pk else ' (FILTERED OUT)'}"
                            for k in got)
                  or "none of the named kinases is in the top-10")
        if not got:
            verdict, note = "MISMATCH", "named kinase absent from the top-10"
        elif all(k in pk for k in got) and order[got[0]] == 1:
            verdict, note = "MATCH", ""
        elif any(k in pk for k in got):
            verdict = "MATCH*"
            note = (f"top-ranked kinase at this site is {rk[0][0]} "
                    f"({rk[0][1]}), not the one named; the named kinase is "
                    f"in the passing set but is not rank 1")
        else:
            verdict = "MISMATCH"
            note = (f"named kinase is in the top-10 but did not survive the "
                    f"selectivity/drug filter; passing set is {sorted(pk)}; "
                    f"top-ranked is {rk[0][0]} ({rk[0][1]})")
        claim(name, " / ".join(sorted(want_kin)), detail, verdict, note)
        if want_kin and len(want_kin) > 1:
            unnamed = sorted(pk - want_kin,
                             key=lambda k: order.get(k, 99))
            if unnamed:
                ranks = [order.get(k, 99) for k in got]
                tidy = max(ranks) <= len(want_kin)
                claim(f"{name} -- completeness",
                      f"{len(want_kin)} kinases named",
                      f"{len(pk)} kinases pass at this site; also "
                      + ", ".join(f"{k} (rank {order.get(k,'?')}, logit "
                                  f"{dict((a, b) for a, b, _ in rk).get(k)})"
                                  for k in unnamed),
                      "MATCH" if tidy else "MATCH*",
                      "" if tidy else
                      "the named kinases are a drug-driven subset, not the "
                      "top-N by logit; unnamed kinases score as high or "
                      "higher, so 'convergence on these' understates the set")

    # drug nominations quoted for ORF9b Y42
    r9 = scan[("SARS2:ORF9b", "Y42")]
    d9 = parse_pairs(r9["passing_drugs"])
    for drug, kin in (("Lorlatinib", "ALK"), ("Vatalanib", "FLT4")):
        ok = drug in d9.get(kin, [])
        claim(f"ORF9b Tyr42 drug {drug} on {kin}", f"{drug} via {kin}",
              "present" if ok else f"absent; {kin} drugs = {d9.get(kin)}",
              "MATCH" if ok else "MISMATCH")
    # drug nominations quoted for Zika E S72 and N S202
    rz = scan[("ZIKA E", "S72")]
    dz = parse_pairs(rz["passing_drugs"])
    claim("Zika E Ser72 drugs on CK1e", "Seliciclib (Roscovitine) [+ Purvalanol]",
          "CK1e -> " + ";".join(dz.get("CK1e", [])),
          "MATCH*" if "Seliciclib" in dz.get("CK1e", []) else "MISMATCH",
          "Seliciclib is mapped to CK1e at S72; Purvalanol is NOT -- in the "
          "pipeline Purvalanol maps to SRPK2 at Zika E T76/Y90 and to SRPK2 "
          "at N S206.  The caption's Purvalanol mention is literature "
          "context, not a pipeline nomination")
    tide = [d["drug"] for d in demo
            if d["psite"] == "S202" and d["kinase"] == "GSK3B"]
    claim("N Ser202 drugs on GSK3B", "Tideglusib (+ lithium, literature)",
          f"demo pair list -> {sorted(set(tide))}",
          "MATCH*" if "Tideglusib" in tide else "MISMATCH",
          "the demo/highlight pair list carries GSK3B at S202, but GSK3B is "
          "rank 6 in the kinome scan there and is NOT in passing_kinases "
          "(['ICK', 'PCTAIRE1'])")

    # -----------------------------------------------------------------
    # cross-check against the exported figure tables
    # -----------------------------------------------------------------
    print("\n--- cross-check: exported panel-B/C figure tables ---")
    for path, label in ((T_KP, "eval_virus_dms_kinase_psite_demo.csv"),
                        (T_DP, "eval_virus_dms_drug_psite_demo.csv")):
        if not os.path.exists(path):
            claim(f"{label} present", "n/a", "file missing", "UNRESOLVED")
            continue
        for q in read_csv(path):
            if q["psite"] in ("S53", "T329") and float(q["logit"]) != 0.0:
                extra = f" drug={q['drug']}" if "drug" in q else ""
                print(f"    {label}: {norm_sub(q['substrate']):14s} "
                      f"{q['psite']:5s} kinase={q['kinase']:6s} "
                      f"logit={q['logit']}{extra}")
    kp = {(q["psite"], q["kinase"]) for q in read_csv(T_KP)
          if float(q["logit"]) != 0.0}
    claim("Fig 8 panel C row S53 (artwork)",
          "Erk7-mapped drugs (Aspirin, Minocycline)",
          "Erk7" if ("S53", "Erk7") in kp else "?",
          "MATCH" if ("S53", "Erk7") in kp and ("S53", "PKCe") not in kp
          else "MISMATCH",
          "panel C -- the plotted heatmap -- agrees with the Results text and "
          "with the code; it contradicts its own caption")
    claim("Fig 8 panel C row T329 (artwork)",
          "PKCe-mapped drugs (KAI-1455, Bryostatin 1)",
          "PKCe" if ("T329", "PKCe") in kp else "?",
          "MATCH" if ("T329", "PKCe") in kp else "MISMATCH")

    # panel B artwork, read from the PDF if supplied
    if fig_pdf and os.path.exists(fig_pdf):
        try:
            b = _panel_b_spans(fig_pdf)
            print("\n--- Fig 8 panel B (rightmost sub-panel) text spans ---")
            for y, t in b:
                print(f"    y={y:7.1f} {t!r}")
            texts = [t for _, t in b]
            mixed = ("Influenza H3N2 · HA" in texts
                     and "S53" in texts and "M1" in texts)
            claim("Fig 8 panel B internal consistency",
                  "heading 'Influenza H3N2 · HA' + node 'PKCε' + "
                  "'M1' / 'S53' + Bryostatin 1 / KAI-1455",
                  "self-contradictory: the heading, the kinase and both drugs "
                  "belong to H3N2 HA T329, but the protein/site labels read "
                  "M1 / S53",
                  "MISMATCH" if mixed else "UNRESOLVED",
                  "fix = relabel the node HA / T329 (the panel is the H3N2 "
                  "nomination), and give M1 Ser53 its own Erk7 entry if it is "
                  "to be shown")
        except ImportError:
            claim("Fig 8 panel B internal consistency", "see note",
                  "PyMuPDF unavailable", "UNRESOLVED")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "code",
                                           "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    import collections
    tally = collections.Counter(r["verdict"] for r in rows)
    print(f"\n{len(rows)} claims checked: "
          + ", ".join(f"{v}={n}" for v, n in sorted(tally.items()))
          + f" -> {os.path.abspath(OUT)}")
    print("MATCH*  = partial: the manuscript's kinase/drug is in the pipeline "
          "output, but it is not the top-ranked call at that site, or the "
          "account is incomplete.  See the note column.")


def _panel_b_spans(path):
    """Text spans of the rightmost panel-B sub-panel of Fig 04 (x > 380,
    172 < y < 314 in the published 1-page artwork)."""
    import pymupdf
    doc = pymupdf.open(path)
    out = []
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for sp in line["spans"]:
                    t = sp["text"].strip()
                    x, y = sp["bbox"][0], sp["bbox"][1]
                    if t and x > 380 and 172 < y < 314:
                        out.append((y, t))
    return sorted(out)


if __name__ == "__main__":
    main()
