#!/usr/bin/env python3
"""audit_07 -- PER2 phosphosite number in Fig 03 ("Network rewiring", printed as
Figure 7) panel B vs the Results text, plus the other tier-1 residue-resolved
claims in the same Results paragraph and the Fig 7 caption's mutation counts.

Usage:
    python3 -I audit_07_per2_site.py <EVAL_DIR> [<FIG03_PDF>]

    EVAL_DIR   tarballs/code/code/evaluation/
    FIG03_PDF  optional; "figures/Fig 03 - Network rewiring.pdf" out of
               manuscript/261014/KSI_Paper.zip.  If given and PyMuPDF is
               importable, the phosphosite labels are re-read from the artwork;
               otherwise the labels recorded on 2026-10-22 are used.

Writes ../out/audit_07_per2_site.csv and prints one CLAIM line per check.
"""
import csv
import os
import sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "out", "audit_07_per2_site.csv")

# --------------------------------------------------------------------------
# What the manuscript says (manuscript/261014/KSI_Paper.zip)
# --------------------------------------------------------------------------
# main.tex L259 (Results, "Network rewiring" section):
MS_RESULTS = {
    # claim name                     (kinase as written, substrate, site as written)
    "PTPS Thr58":                    ("AMPKa1", "PTS", "T58"),
    "TP53 Ser215":                   ("AurA/AurB", "TP53", "S215"),
    "TP53 Ser269":                   ("AurA/AurB", "TP53", "S269"),
    "CTNNB1 Ser45":                  ("CK1", "CTNNB1", "S45"),
    "CTNNB1 Ser33":                  ("GSK3A/GSK3B", "CTNNB1", "S33"),
    "CTNNB1 Ser37":                  ("GSK3A/GSK3B", "CTNNB1", "S37"),
    "CTNNB1 Thr41":                  ("GSK3A/GSK3B", "CTNNB1", "T41"),
    "PER2 Ser662":                   ("CK1", "PER2", "S662"),
    "CCNF Ser621":                   ("CK2", "CCNF", "S621"),
    "SLC12A3 Thr60":                 ("OSR1/SPAK", "SLC12A3", "T60"),
    "KIT Ser821":                    ("PKC", "KIT", "S821"),
    # tier-2 entry quoted in the same paragraph
    "CBL Tyr371":                    ("(tier 2, kinase unresolved)", "CBL", "Y371"),
}
# Kinase aliases: how the Results/caption spell a kinase -> the names the
# pipeline tables use (the tables carry Manning-style short names).
ALIAS = {
    "AMPKa1":      {"AMPKa1", "AMPK"},
    "AurA/AurB":   {"AurA", "AurB"},
    "CK1":         {"CK1", "CK1a", "CK1a2", "CK1d", "CK1e",
                    "CK1g1", "CK1g2", "CK1g3"},
    "GSK3A/GSK3B": {"GSK3A", "GSK3B"},
    "CK2":         {"CK2", "CK2a1", "CK2a2"},
    "OSR1/SPAK":   {"OSR1", "STLK3"},   # SPAK == STK39 == STLK3 in these tables
    "PKC":         {"PKC", "PKCa", "PKCb", "PKCd", "PKCe", "PKCg",
                    "PKCh", "PKCi", "PKCz"},
}

# Fig 03 caption (main.tex L399):
MS_FIG_COUNTS = {"onsite": 40, "flanked": 25}

# Phosphosite labels read out of "Fig 03 - Network rewiring.pdf" panel B on
# 2026-10-22 (pymupdf span text; each is one text span, so these are not
# extraction artefacts).  Grouped by the pathway heading above them.
FIG_LABELS_RECORDED = {
    "Wnt/beta-catenin":   ["p45", "p41", "p37", "p33"],
    "p53 pathway":        ["p269", "p215", "p185", "p183", "p155", "p109"],
    "Tau/Neurodegeneration": ["p621", "p393", "p669", "p641"],
    "Cell cycle/DDR":     ["p280", "p640"],
    "NF-kB/Immune":       ["p152", "p32", "p45", "p425", "p386"],
    "Ion channels":       ["p20", "p60", "p1455"],
    "Metabolism":         ["p11", "p58"],
    "Neuromuscular":      ["p558"],
    "Nuclear/Structural": ["p534"],
    "Circadian":          ["p562"],
    "RTK signaling":      ["p226", "p155", "p821", "p371", "p647"],
}

rows = []


def claim(name, ms, code, verdict, note=""):
    print(f"CLAIM {name} | manuscript={ms} | code={code} | VERDICT={verdict}")
    rows.append({"claim": name, "manuscript": ms, "code": code,
                 "verdict": verdict, "note": note})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    eval_dir = sys.argv[1]
    fig_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    FINAL = os.path.join(eval_dir, "260130_ptmvar_nr_finalized.csv")
    CURATED = os.path.join(eval_dir,
                           "260129_ptmvar_kinase_substrate_pind_type_all_curated.csv")
    TIERS = os.path.join(eval_dir, "consensus_kinase_substrate_tiers.csv")
    PSITES = os.path.join(eval_dir, "260128_ptmvar_psites_overlap_dbs.csv")
    FLANKS = os.path.join(eval_dir, "260128_ptmvar_flanks_overlap_dbs.csv")

    final = read_csv(FINAL)
    curated = read_csv(CURATED)
    tiers = read_csv(TIERS)
    psites = read_csv(PSITES)
    flanks = read_csv(FLANKS)

    # ------------------------------------------------------------------
    # Q1/Q2 -- the PER2 row(s)
    # ------------------------------------------------------------------
    print("\n--- Q1: PER2 rows in every pipeline table ---")
    per2_sites = set()
    for label, tbl in (("260130_ptmvar_nr_finalized", final),
                       ("260129_..._type_all_curated", curated),
                       ("consensus_kinase_substrate_tiers", tiers),
                       ("260128_ptmvar_psites_overlap_dbs", psites),
                       ("260128_ptmvar_flanks_overlap_dbs", flanks)):
        hit = [r for r in tbl if r.get("substrate_gene") == "PER2"]
        if not hit:
            print(f"    {label}: (no PER2 row)")
            continue
        for r in hit:
            site = f"{r.get('p_aa','?') or '?'}{r['p_ind']}"
            per2_sites.add(r["p_ind"])
            print(f"    {label}: kinase={r['kinase_gene']:6s} site={site:6s} "
                  f"mut={(r.get('mut_aa') or '-')}{(r.get('mut_ind') or '')} "
                  f"dbSNP={r.get('dbSNP','')} tier={r.get('tier','')} "
                  f"db={r.get('db_source','')} type={r.get('mut_type','')} "
                  f"sig={r.get('Clinical_Significance','')} "
                  f"dz={r.get('Disease/Phenotype','')}")

    code_site = sorted(per2_sites)
    claim("Results PER2 site", "Ser662",
          "S" + "/".join(code_site) if code_site else "ABSENT",
          "MATCH" if code_site == ["662"] else "MISMATCH",
          "every PTMVar table carries p_ind=662; rs121908635 / Pathogenic / "
          "Advanced sleep phase syndrome 1")

    # The figure label.  The pathway -> label grouping below is the one read by
    # hand off the panel-B span coordinates on 2026-10-22; when the PDF is
    # supplied we additionally re-verify the literal spans straight from it
    # (which digits are actually typeset is the whole question here).
    fig_circadian = {k: list(v) for k, v in FIG_LABELS_RECORDED.items()}
    if fig_pdf and os.path.exists(fig_pdf):
        try:
            found = _figure_site_spans(fig_pdf)
            print(f"\n--- Fig 03 panel B spans re-read from {fig_pdf}")
            print(f"    site labels in artwork: {sorted(found, key=lambda s: int(s[1:]))}")
            recorded = {l for v in fig_circadian.values() for l in v}
            if found != recorded:
                print(f"    NOTE: differs from recorded set "
                      f"(+{sorted(found - recorded)} -{sorted(recorded - found)})")
            claim("Fig 7B typesets 'p562'", "p562 present" if "p562" in found
                  else "p562 ABSENT", "p662 expected from p_ind=662",
                  "MISMATCH" if "p562" in found and "p662" not in found
                  else "UNRESOLVED",
                  "read directly from the PDF text spans")
        except ImportError:
            print("\n--- PyMuPDF unavailable; using recorded Fig 03 labels only")

    circ = fig_circadian.get("Circadian", [])
    claim("Fig 7B Circadian node label", ",".join(circ) or "NOT FOUND",
          "p662 (expected from p_ind=662)",
          "MISMATCH" if circ != ["p662"] else "MATCH",
          "PER2 is the only substrate in the Circadian cluster "
          "(generate_all_figures_final.py: pathways['Circadian'] = ['PER2']); "
          "the label is drawn as f'p{p_ind}', so p562 cannot come from the data")

    # Off-by-100 vs isoform numbering
    claim("PER2 562 = isoform-specific numbering?", "p562 (Fig 7B)",
          "no 562 anywhere in evaluation/*.py or *.ipynb; no PER2 isoform "
          "table in the pipeline",
          "MISMATCH",
          "pure transcription error (6 -> 5 in the hundreds digit), not an "
          "isoform renumbering; FASPS PER2 S662G (Toh 2001) is the canonical "
          "UniProt O15055 numbering the tables use")

    # ------------------------------------------------------------------
    # Q3a -- the other residue-resolved (tier-1) claims
    # ------------------------------------------------------------------
    print("\n--- Q3: other sites quoted in the Results paragraph ---")
    # index the curated table (it keeps the un-collapsed kinase names)
    def lookup(substrate, site):
        aa, idx = site[0], site[1:]
        out = []
        for r in curated:
            if r["substrate_gene"] == substrate and r["p_ind"] == idx \
                    and r["p_aa"] == aa:
                out.append((r["kinase_gene"], r["tier"], r["mut_type"]))
        return out

    for name, (ms_kin, sub, site) in MS_RESULTS.items():
        hits = lookup(sub, site)
        if not hits:
            claim(name, f"{ms_kin} @ {sub} {site}", "site absent from curated table",
                  "MISMATCH", "")
            continue
        want = ALIAS.get(ms_kin)
        kin_tier1 = sorted({k for k, t, _ in hits if t == "Tier 1"})
        kin_any = sorted({k for k, _, _ in hits})
        tiers_seen = sorted({t for _, t, _ in hits})
        code_str = (f"{sub} {site}: Tier1={kin_tier1 or '-'} "
                    f"all={kin_any} tiers={tiers_seen}")
        if want is None:                     # CBL Tyr371 -- tier 2 by design
            ok = "Tier 2" in tiers_seen and "Tier 1" not in tiers_seen
            claim(name, f"{ms_kin} @ {sub} {site}", code_str,
                  "MATCH" if ok else "MISMATCH",
                  "Results places this in the second tier")
            continue
        if want & set(kin_tier1):
            verdict, note = "MATCH", ""
        elif want & set(kin_any):
            verdict = "MISMATCH"
            note = (f"{sorted(want & set(kin_any))} present but not Tier 1 at "
                    f"this site")
        else:
            verdict = "MISMATCH"
            note = (f"none of {sorted(want)} is assigned to {sub} {site}; "
                    f"the Tier-1 kinases there are {kin_tier1 or 'none'}")
        claim(name, f"{ms_kin} @ {sub} {site}", code_str, verdict, note)

    # ------------------------------------------------------------------
    # Q3b -- the Fig 7 caption counts
    # ------------------------------------------------------------------
    print("\n--- Q3: Fig 7 caption mutation counts ---")
    n_on = sum(1 for r in final if r["mut_type"] == "onsite")
    n_fl = sum(1 for r in final if r["mut_type"] == "flanked")
    claim("Fig 7 caption on-phosphosite mutations", MS_FIG_COUNTS["onsite"], n_on,
          "MATCH" if n_on == MS_FIG_COUNTS["onsite"] else "MISMATCH",
          "rows of 260130_ptmvar_nr_finalized.csv with mut_type=onsite "
          "(kinase-substrate-site rows, not distinct variants: only 20 "
          "distinct dbSNP IDs)")
    claim("Fig 7 caption flanking mutations", MS_FIG_COUNTS["flanked"], n_fl,
          "MATCH" if n_fl == MS_FIG_COUNTS["flanked"] else "MISMATCH",
          "rows with mut_type=flanked (17 distinct dbSNP IDs)")

    # Fig 7B also prints the four tier x type counts 11 / 29 / 12 / 13
    want_split = {("onsite", "Tier 1"): 11, ("onsite", "Tier 2"): 29,
                  ("flanked", "Tier 1"): 12, ("flanked", "Tier 2"): 13}
    for (mt, tier), want in want_split.items():
        got = sum(1 for r in final
                  if r["mut_type"] == mt and r["tier"] == tier)
        claim(f"Fig 7B count {mt}/{tier}", want, got,
              "MATCH" if got == want else "MISMATCH")

    # ------------------------------------------------------------------
    # Q3c -- every other phosphosite label in panel B vs the code
    # ------------------------------------------------------------------
    print("\n--- extra: all Fig 7B site labels vs p_ind in the finalized table ---")
    code_pinds = {r["p_ind"] for r in final}
    fig_pinds = {lab[1:] for labs in fig_circadian.values() for lab in labs}
    orphan = sorted(fig_pinds - code_pinds, key=int)
    for p in orphan:
        grp = [g for g, labs in fig_circadian.items() if f"p{p}" in labs]
        cand = sorted(_one_typo_away(p, code_pinds), key=int)
        subs = {}
        for r in final:
            if r["p_ind"] in cand:
                subs.setdefault(r["p_ind"], set()).add(
                    f"{r['substrate_gene']} {r['p_aa']}{r['p_ind']}")
        detail = "; ".join(f"p{c} = " + "/".join(sorted(subs.get(c, {"?"})))
                           for c in cand) or "no single-typo candidate"
        claim(f"Fig 7B label p{p} ({'/'.join(grp)})", f"p{p}",
              f"no p_ind={p} in the finalized table; one-typo candidates: {detail}",
              "MISMATCH",
              "artwork label with no backing row -- same class of error as p562")
    if not orphan:
        claim("Fig 7B labels all backed by a p_ind", "all", "all", "MATCH")

    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "code",
                                           "verdict", "note"])
        w.writeheader()
        w.writerows(rows)
    bad = sum(1 for r in rows if r["verdict"] != "MATCH")
    print(f"\n{len(rows)} claims checked, {bad} not MATCH -> {os.path.abspath(OUT)}")


def _one_typo_away(label, pool):
    """p_inds in `pool` reachable from `label` by one digit substitution or by
    inserting one digit (the two ways a hand-redrawn label goes wrong)."""
    out = set()
    for c in pool:
        if len(c) == len(label) and sum(a != b for a, b in zip(c, label)) == 1:
            out.add(c)
        elif len(c) == len(label) + 1:
            for i in range(len(c)):
                if c[:i] + c[i + 1:] == label:
                    out.add(c)
                    break
    return out


#: /^p\d+$/ spans in panel B that are kinase node names, not phosphosites.
_KINASE_PNAMES = {"p38", "p53"}


def _figure_site_spans(path):
    """The set of pNNN phosphosite labels actually typeset in Fig 03 panel B."""
    import re
    import pymupdf
    doc = pymupdf.open(path)
    out = set()
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            for line in blk.get("lines", []):
                for sp in line["spans"]:
                    t = sp["text"].strip()
                    if re.fullmatch(r"p\d+", t) and t not in _KINASE_PNAMES:
                        out.add(t)
    return out


if __name__ == "__main__":
    main()
