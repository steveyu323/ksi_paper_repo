#!/usr/bin/env python3
"""audit_03 / audit_04 — phospho-PCA kinase panel composition.

Audits:
  03  "24 kinases" (Summary + Introduction) vs "23 kinase constructs"
      (Table S12 title / Methods arithmetic), and whether DYRK4 was designed
      but absent from the screened library.
  04  Which arm Titin was screened in: Methods puts it in the tyrosine arm,
      Table S12 in the serine/threonine arm.

Provenance of the "arm" call: the screening table carries both `kinase_type`
(ST|Y) and `paabd` (the PAABD reader actually co-expressed). The two arms used
disjoint reader sets -- Src-SH2 only for Y, and WW/FHA/14-3-3beta for ST -- so
the observed PAABDs are an independent check on kinase_type.

Usage:
    python3 -I audit_03_kinase_panel.py KSI_DIR SCREEN_DIR WET_DIR [TEX_DIR]
                                        [--lookup-dir DIR] [--no-big-csv]

    KSI_DIR     .../${KSI_REPO_ROOT}/ksi_analysis_2025
    SCREEN_DIR  .../${KSI_REPO_ROOT}/screening_analysis
    WET_DIR     .../${KSI_REPO_ROOT}/kinase_ml/code/wetlab_planning
    TEX_DIR     dir holding main.tex + supplementary.tex (optional; if absent
                the manuscript-side values recorded below are used verbatim)
    --lookup-dir  KSI_DIR/lookup_tables, to test whether DYRK4 reads exist in
                the E. coli-constructed (PacBio) library.  Loads ~1.2 GB of
                pickles; skipped unless given.

No data root is ever added to sys.path; every path arrives through argv.
"""
import csv
import os
import re
import sys

csv.field_size_limit(10 ** 9)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "out"))
OUT_CSV = os.path.join(OUT_DIR, "audit_03_kinase_panel.csv")
OUT_TXT = os.path.join(OUT_DIR, "audit_03_methods_replacement.txt")

# ---------------------------------------------------------------- manuscript
# main.tex (261014/KSI_Paper.zip, main.tex of 2026-10-21 16:46)
#   L138 SUMMARY      "... high-confidence substrates for 24 kinases ..."
#   L149 INTRODUCTION "... against 24 representative kinases in a pooled format"
#   L1012 METHODS     "8 tyrosine kinases (FRK, FYN, ABL2, STYK1, and the Titin
#                      pseudokinase, plus [3 additional tyrosine kinases - to
#                      confirm]) and 15 serine/threonine kinases (...)"
# supplementary.tex
#   L730 Table S12    "The 23 kinase constructs screened by phospho-PCA."
MS = {
    "summary_count": 24,
    "intro_count": 24,
    "tableS12_title_count": 23,
    "tableS12_st": 15,
    "tableS12_y": 8,
    "methods_st": 15,
    "methods_y": 8,
    "methods_y_named": ["FRK", "FYN", "ABL2", "STYK1", "TITIN"],
    "methods_y_placeholder_n": 3,
    "methods_st_named": ["KKCC1", "KS6B1_HUMAN_wt", "KGP1", "PHKG1", "PLK1",
                         "M3K7", "TBK1", "IKKB", "ADCK1",
                         "KS6B1_HUMAN_active", "PLK1_HUMAN_active"],
    "titin_arm_methods": "Y",
    "titin_arm_tableS12": "ST",
}

# Table S12 rows, in table order.  Parsed from supplementary.tex when TEX_DIR is
# supplied; this literal copy is the fallback and the regression baseline.
TABLE_S12 = {
    "ST": ["TITIN_HUMAN", "PHKG1_HUMAN", "PIM1_HUMAN", "KKCC1_HUMAN",
           "PLK1_HUMAN", "PLK1_HUMAN_active", "MAST1_HUMAN", "PDPK1_HUMAN",
           "KGP1_HUMAN", "KS6B1_HUMAN_wt", "KS6B1_HUMAN_active", "TBK1_HUMAN",
           "IKKB_HUMAN", "ADCK1_HUMAN", "M3K7_HUMAN"],
    "Y": ["JAK2_HUMAN_1", "FES_HUMAN", "ABL2_HUMAN", "FRK_HUMAN", "FYN_HUMAN",
          "KSYK_HUMAN", "TNK1_HUMAN", "STYK1_HUMAN"],
}

# entry name -> HGNC / common symbol used in Methods prose
SYMBOL = {
    "TITIN_HUMAN": "TTN (Titin pseudokinase)",
    "PHKG1_HUMAN": "PHKG1",
    "PIM1_HUMAN": "PIM1",
    "KKCC1_HUMAN": "CAMKK1 (CaMKK1)",
    "PLK1_HUMAN": "PLK1",
    "PLK1_HUMAN_active": "PLK1-T210D",
    "MAST1_HUMAN": "MAST1",
    "PDPK1_HUMAN": "PDPK1/PDK1",
    "KGP1_HUMAN": "PRKG1",
    "KS6B1_HUMAN_wt": "RPS6KB1/S6K1",
    "KS6B1_HUMAN_active": "RPS6KB1-T252D",
    "TBK1_HUMAN": "TBK1",
    "IKKB_HUMAN": "IKBKB/IKKbeta",
    "ADCK1_HUMAN": "ADCK1",
    "M3K7_HUMAN": "MAP3K7/TAK1",
    "JAK2_HUMAN_1": "JAK2 (first kinase domain)",
    "FES_HUMAN": "FES",
    "ABL2_HUMAN": "ABL2",
    "FRK_HUMAN": "FRK",
    "FYN_HUMAN": "FYN",
    "KSYK_HUMAN": "SYK",
    "TNK1_HUMAN": "TNK1",
    "STYK1_HUMAN": "STYK1",
    "DYRK4_HUMAN": "DYRK4",
}

VARIANTS = {"PLK1_HUMAN_active", "KS6B1_HUMAN_active"}
# variant -> wild-type construct it shares an identity with
VARIANT_PARENT = {"PLK1_HUMAN_active": "PLK1_HUMAN",
                  "KS6B1_HUMAN_active": "KS6B1_HUMAN_wt"}

ST_READERS = {"ww", "fha", "1433"}
Y_READERS = {"sh2"}

ARM_LONG = {"ST": "serine/threonine", "Y": "tyrosine"}

# ------------------------------------------------------------------ plumbing
_rows = []
_counts = {"MATCH": 0, "MISMATCH": 0, "UNRESOLVED": 0}


def claim(name, ms, code, verdict):
    if verdict is True:
        verdict = "MATCH"
    elif verdict is False:
        verdict = "MISMATCH"
    _counts[verdict] = _counts.get(verdict, 0) + 1
    print(f"CLAIM {name} | manuscript={ms} | code={code} | VERDICT={verdict}")
    _rows.append((name, ms, code, verdict))


def die(msg):
    sys.stderr.write(f"audit_03: {msg}\n")
    raise SystemExit(2)


# -------------------------------------------------------------- tex parsing
def parse_tex(tex_dir):
    """Re-read the manuscript-side numbers so the audit tracks edits."""
    got = {}
    main = os.path.join(tex_dir, "main.tex")
    supp = os.path.join(tex_dir, "supplementary.tex")
    if os.path.exists(main):
        txt = open(main, encoding="utf-8", errors="replace").read()
        m = re.search(r"substrates for (\d+) kinases", txt)
        if m:
            got["summary_count"] = int(m.group(1))
        m = re.search(r"against (\d+) representative kinases", txt)
        if m:
            got["intro_count"] = int(m.group(1))
        m = re.search(r"The panel comprised (\d+) tyrosine kinases.*?and (\d+) "
                      r"serine/threonine kinases", txt, re.S)
        if m:
            got["methods_y"] = int(m.group(1))
            got["methods_st"] = int(m.group(2))
        m = re.search(r"\{?\[?\}?(\d+) additional tyrosine kinases", txt)
        if m:
            got["methods_y_placeholder_n"] = int(m.group(1))
        got["methods_titin_in_y_sentence"] = bool(
            re.search(r"tyrosine kinases \([^)]*Titin pseudokinase", txt))
    if os.path.exists(supp):
        txt = open(supp, encoding="utf-8", errors="replace").read()
        m = re.search(r"The (\d+) kinase constructs screened by phospho-PCA", txt)
        if m:
            got["tableS12_title_count"] = int(m.group(1))
        blk = re.search(r"Serine/threonine arm\}\}(.*?)\\bottomrule", txt, re.S)
        if blk:
            body = blk.group(1)
            st_part, _, y_part = body.partition("Tyrosine arm}}")
            def entries(chunk):
                out = []
                for line in chunk.split("\\\\"):
                    m2 = re.search(r"\\texttt\{([A-Z0-9_]+\\?_HUMAN)\}", line)
                    if not m2:
                        continue
                    name = m2.group(1).replace("\\_", "_")
                    first = line.split("&")[0]
                    if "active" in first:
                        name += "_active"
                    elif re.search(r"_\{\\text\{wt\}\}|\bwt\b", first):
                        name += "_wt"
                    elif re.search(r"JAK2\$?_", first):
                        name += "_1"
                    out.append(name)
                return out
            got["tableS12_ST"] = entries(st_part)
            got["tableS12_Y"] = entries(y_part)
    return got


# ------------------------------------------------- screening-data extraction
def census_from_big_csv(path):
    """(entry_name -> arm, entry_name -> set(paabd)) streamed from the raw
    merged screening table.  This is the authoritative screening record."""
    arms, readers, nrow = {}, {}, 0
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        # the file has a leading unnamed index column
        try:
            i_name = hdr.index("kinase_name")
            i_type = hdr.index("kinase_type")
            i_pa = hdr.index("paabd")
        except ValueError:
            die(f"{path}: expected kinase_name/kinase_type/paabd in header")
        for row in rd:
            nrow += 1
            name, typ, pa = row[i_name], row[i_type], row[i_pa]
            prev = arms.setdefault(name, typ)
            if prev != typ:
                die(f"{name}: inconsistent kinase_type {prev} vs {typ}")
            readers.setdefault(name, set()).add(pa)
    return arms, readers, nrow


def census_from_pickles(ksi_dir, design_arm):
    """Fallback / corroboration: reduced hit table (kinase set) + per-PAABD
    inference table (readers).  Arm then comes from the design reference."""
    import pandas as pd
    red = os.path.join(ksi_dir, "251203_ksi_reduced_df_tp.pkl")
    inf = os.path.join(ksi_dir, "251130_ksi_inference_scores.pkl")
    if not (os.path.exists(red) and os.path.exists(inf)):
        return None, None, 0
    kin = set(pd.read_pickle(red)["kinase"].unique())
    d = pd.read_pickle(inf)
    readers = {k: set(map(str, g)) for k, g in d.groupby("kinase")["paabd"]}
    arms = {k: design_arm.get(k, "?") for k in kin}
    return arms, readers, len(d)


# ------------------------------------------------------------------ DYRK4
def dyrk4_in_library(lookup_dir, ref_seq_orig, ref_seq_fixed, design_seq):
    """Does any PacBio read of the constructed library carry DYRK4?

    kinase_aa is translated from the read, not taken from the reference, so a
    substring probe against the designed sequence measures real abundance and
    is insensitive to the N-terminal mis-annotation that stopped DYRK4 from
    joining to the screening table.  The same probe is applied to every ST
    construct so DYRK4's representation can be read against its peers.
    """
    import pandas as pd
    res = {}
    for tag, fn in (("ST", "st.lookup.corrected.pkl"),
                    ("Y", "y.lookup.corrected.pkl")):
        p = os.path.join(lookup_dir, fn)
        if not os.path.exists(p):
            res[tag] = None
            continue
        ka = pd.read_pickle(p)["kinase_aa"]
        core = ref_seq_fixed[30:90]
        exact = int((ka == ref_seq_fixed).sum() + (ka == ref_seq_orig).sum())
        contains = ka.str.contains(core, regex=False)
        # error-tolerant abundance of every designed construct of this arm
        abundance = {}
        for name, seq in design_seq.items():
            if len(seq) < 160:
                continue
            abundance[name] = int(ka.str.contains(seq[120:160], regex=False).sum())
        res[tag] = {
            "n_reads_total": int(len(ka)),
            "n_exact_full_length": exact,
            "n_reads_with_kinase_domain_core": int(contains.sum()),
            "n_distinct_variants": int(ka[contains].nunique()),
            "variant_prefixes": sorted({s[:12] for s in ka[contains].unique()}),
            "abundance_by_probe": abundance,
        }
    return res


# ---------------------------------------------------------------------- main
def main(argv):
    pos = [a for a in argv if not a.startswith("--")]
    flags = [a for a in argv if a.startswith("--")]
    if len(pos) < 3:
        die("usage: audit_03_kinase_panel.py KSI_DIR SCREEN_DIR WET_DIR "
            "[TEX_DIR] [--lookup-dir DIR] [--no-big-csv]")
    ksi_dir, screen_dir, wet_dir = (os.path.abspath(p) for p in pos[:3])
    tex_dir = os.path.abspath(pos[3]) if len(pos) > 3 else None
    lookup_dir = None
    for f in flags:
        if f.startswith("--lookup-dir="):
            lookup_dir = os.path.abspath(f.split("=", 1)[1])
    if lookup_dir is None and "--lookup-dir" in argv:
        i = argv.index("--lookup-dir")
        if i + 1 < len(argv):
            lookup_dir = os.path.abspath(argv[i + 1])
    use_big = "--no-big-csv" not in flags

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- manuscript side
    ms = dict(MS)
    s12 = {k: list(v) for k, v in TABLE_S12.items()}
    if tex_dir:
        got = parse_tex(tex_dir)
        for k in ("summary_count", "intro_count", "tableS12_title_count",
                  "methods_st", "methods_y", "methods_y_placeholder_n"):
            if k in got:
                ms[k] = got[k]
        if got.get("tableS12_ST"):
            s12["ST"], s12["Y"] = got["tableS12_ST"], got["tableS12_Y"]
        print(f"# manuscript source: {tex_dir}")
    else:
        print("# manuscript source: literals recorded in this script")
    ms["tableS12_st"], ms["tableS12_y"] = len(s12["ST"]), len(s12["Y"])
    s12_arm = {k: "ST" for k in s12["ST"]}
    s12_arm.update({k: "Y" for k in s12["Y"]})

    # ---- design reference (the 24-construct design list)
    ref_path = os.path.join(screen_dir, "blib_kinases_with_acc.csv")
    if not os.path.exists(ref_path):
        die(f"missing design reference {ref_path}")
    design_arm, design_seq = {}, {}
    with open(ref_path, newline="") as fh:
        for r in csv.reader(fh):
            if len(r) < 4:
                continue
            design_arm[r[1]] = r[0]
            design_seq[r[1]] = r[2]
    print(f"# design reference: {ref_path} ({len(design_arm)} constructs)")

    # the sequence correction applied in 1_parse_lookup_tables.ipynb (251210)
    fixed_path = os.path.join(ksi_dir, "lookup_tables", "blib_kinases_with_acc.csv")
    dyrk4_fixed = design_seq.get("DYRK4_HUMAN", "")
    if os.path.exists(fixed_path):
        with open(fixed_path, newline="") as fh:
            rd = csv.reader(fh)
            h = next(rd)
            ix = {c: i for i, c in enumerate(h)}
            for r in rd:
                if r[ix["kinase_name"]] == "DYRK4_HUMAN":
                    dyrk4_fixed = r[ix["kinase_aa"]]

    # ---- screening side
    big = os.path.join(ksi_dir, "dat_all_blib_2batch.csv")
    src = None
    if use_big and os.path.exists(big):
        print(f"# screening source: {big} (streaming)")
        scr_arm, scr_readers, n = census_from_big_csv(big)
        src = f"dat_all_blib_2batch.csv ({n} rows)"
    else:
        print("# screening source: reduced/inference pickles")
        scr_arm, scr_readers, n = census_from_pickles(ksi_dir, design_arm)
        if scr_arm is None:
            die("no screening table available")
        src = f"251203_ksi_reduced_df_tp.pkl + 251130_ksi_inference_scores.pkl ({n} rows)"

    screened = sorted(scr_arm)
    st_scr = sorted(k for k in screened if scr_arm[k] == "ST")
    y_scr = sorted(k for k in screened if scr_arm[k] == "Y")

    # ---- corroborate arm from PAABD readers alone
    reader_arm = {}
    for k, rs in scr_readers.items():
        rs = {str(x).lower() for x in rs}
        if rs and rs <= Y_READERS:
            reader_arm[k] = "Y"
        elif rs and rs <= ST_READERS:
            reader_arm[k] = "ST"
        else:
            reader_arm[k] = "MIXED:" + ",".join(sorted(rs))

    print()
    print(f"# screening census ({src})")
    print(f"#   ST arm ({len(st_scr)}): {', '.join(st_scr)}")
    print(f"#   Y  arm ({len(y_scr)}): {', '.join(y_scr)}")
    print()

    # =============================================== CLAIMS
    # --- item 03: panel size
    claim("Summary kinase count", ms["summary_count"], len(screened),
          ms["summary_count"] == len(screened))
    claim("Introduction kinase count", ms["intro_count"], len(screened),
          ms["intro_count"] == len(screened))
    claim("Table S12 title kinase count", ms["tableS12_title_count"],
          len(screened), ms["tableS12_title_count"] == len(screened))
    claim("Table S12 Ser/Thr arm size", ms["tableS12_st"], len(st_scr),
          ms["tableS12_st"] == len(st_scr))
    claim("Table S12 tyrosine arm size", ms["tableS12_y"], len(y_scr),
          ms["tableS12_y"] == len(y_scr))
    claim("Methods Ser/Thr arm size", ms["methods_st"], len(st_scr),
          ms["methods_st"] == len(st_scr))
    claim("Methods tyrosine arm size", ms["methods_y"], len(y_scr),
          ms["methods_y"] == len(y_scr))
    claim("Panel total screened constructs", ms["tableS12_title_count"],
          len(screened), len(screened) == ms["tableS12_title_count"])

    distinct = len({VARIANT_PARENT.get(k, k) for k in screened})
    claim("Distinct kinase identities (variants merged)",
          "not stated in manuscript", distinct, "UNRESOLVED")

    # --- item 04: Titin arm
    titin_code = scr_arm.get("TITIN_HUMAN", "ABSENT")
    claim("Methods places Titin in arm", ARM_LONG.get(ms["titin_arm_methods"]),
          ARM_LONG.get(titin_code, titin_code),
          ms["titin_arm_methods"] == titin_code)
    claim("Table S12 places Titin in arm",
          ARM_LONG.get(ms["titin_arm_tableS12"]),
          ARM_LONG.get(titin_code, titin_code),
          ms["titin_arm_tableS12"] == titin_code)
    t_readers = ",".join(sorted(str(x).lower() for x in
                                scr_readers.get("TITIN_HUMAN", [])))
    claim("Titin PAABD readers (Methods implies Src-SH2 only)", "sh2",
          t_readers or "none", t_readers == "sh2")
    claim("Titin arm implied by PAABD readers alone",
          ARM_LONG.get(ms["titin_arm_tableS12"]),
          ARM_LONG.get(reader_arm.get("TITIN_HUMAN"), reader_arm.get("TITIN_HUMAN")),
          reader_arm.get("TITIN_HUMAN") == ms["titin_arm_tableS12"])

    # --- every kinase: Table S12 arm vs screening arm
    bad = [k for k in screened if s12_arm.get(k) != scr_arm[k]]
    claim("Table S12 arm assignment, all 23 kinases",
          f"{len(screened) - len(bad)}/{len(screened)} agree",
          "all agree" if not bad else "disagree: " + ",".join(bad), not bad)
    bad_rd = [k for k in screened if reader_arm.get(k) != scr_arm[k]]
    claim("kinase_type vs PAABD-reader arm, all 23 kinases",
          "consistent", "all consistent" if not bad_rd
          else "inconsistent: " + ",".join(bad_rd), not bad_rd)
    missing = [k for k in s12_arm if k not in scr_arm]
    extra = [k for k in scr_arm if k not in s12_arm]
    claim("Table S12 roster == screened roster", "identical",
          "identical" if not (missing or extra)
          else f"S12-only={missing} screen-only={extra}",
          not (missing or extra))

    # --- Methods placeholder arithmetic
    named_y = [k for k in ms["methods_y_named"]
               if k != "TITIN"]  # Titin is the disputed one
    n_add_y = len(y_scr) - len(named_y)
    claim("Methods '[N additional tyrosine kinases]'",
          ms["methods_y_placeholder_n"], n_add_y,
          ms["methods_y_placeholder_n"] == n_add_y)
    n_add_st = len(st_scr) - len(ms["methods_st_named"])
    claim("Methods '[remaining serine/threonine kinases]'",
          "unspecified", f"{n_add_st} remaining", "UNRESOLVED")

    # --- DYRK4
    in_design = "DYRK4_HUMAN" in design_arm
    claim("DYRK4 present in design reference", "designed, then removed",
          f"present in {os.path.basename(ref_path)} "
          f"(arm={design_arm.get('DYRK4_HUMAN')})", in_design)
    prim = os.path.join(wet_dir, "230504_primers.csv")
    prim_hit = False
    if os.path.exists(prim):
        prim_hit = any(r and r[0] == "DYRK4_HUMAN"
                       for r in csv.reader(open(prim, newline="")))
    claim("DYRK4 cloning primer designed", "designed",
          f"present in {os.path.basename(prim)}" if prim_hit else "absent",
          prim_hit)
    claim("DYRK4 present in screening data", "absent from final library",
          "absent (0 rows)" if "DYRK4_HUMAN" not in scr_arm
          else "PRESENT", "DYRK4_HUMAN" not in scr_arm)
    claim("Design roster size vs screened roster size",
          f"{len(design_arm)} designed -> {ms['tableS12_title_count']} screened",
          f"{len(design_arm)} designed -> {len(screened)} screened",
          len(design_arm) - len(screened) == 1 and
          len(screened) == ms["tableS12_title_count"])

    dy_lib = None
    if lookup_dir:
        dy_lib = dyrk4_in_library(lookup_dir,
                                  design_seq.get("DYRK4_HUMAN", ""), dyrk4_fixed,
                                  {k: v for k, v in design_seq.items()
                                   if design_arm.get(k) == "ST"})
        st = dy_lib.get("ST") or {}
        claim("DYRK4 full-length construct in E. coli library (PacBio)",
              "not represented", f"0 exact reads of "
              f"{st.get('n_reads_total', '?')} ST reads",
              st.get("n_exact_full_length", 0) == 0)
        claim("DYRK4 kinase-domain reads in E. coli library",
              "not represented",
              f"{st.get('n_reads_with_kinase_domain_core', 0)} reads "
              f"({st.get('n_distinct_variants', 0)} variants, N-term "
              f"{','.join(st.get('variant_prefixes', [])[:1])}... != designed "
              f"{dyrk4_fixed[:12]}...)",
              "UNRESOLVED")
        ab = st.get("abundance_by_probe", {})
        if ab.get("DYRK4_HUMAN") is not None and len(ab) > 1:
            others = {k: v for k, v in ab.items() if k != "DYRK4_HUMAN"}
            lo = min(others, key=others.get)
            claim("DYRK4 library representation vs weakest screened ST kinase",
                  "not represented",
                  f"DYRK4={ab['DYRK4_HUMAN']} reads vs {lo}={others[lo]} "
                  f"({others[lo] / max(ab['DYRK4_HUMAN'], 1):.0f}x more)",
                  ab["DYRK4_HUMAN"] * 10 < others[lo])
        print()
        print("# DYRK4 library-presence detail")
        for tag, d in dy_lib.items():
            if d is None:
                print(f"#   {tag}: (lookup table absent)")
                continue
            ab = d.pop("abundance_by_probe", {})
            print(f"#   {tag}: {d}")
            if ab:
                print(f"#   {tag} reads per designed construct (40aa probe):")
                for k, v in sorted(ab.items(), key=lambda kv: kv[1]):
                    print(f"#       {k:22s} {v:7d}")

    # =============================================== panel CSV
    all_names = sorted(set(scr_arm) | set(s12_arm) | set(design_arm))
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kinase", "arm_in_screening_data", "arm_in_tableS12",
                    "paabds_observed", "is_variant", "in_design_only"])
        for k in all_names:
            w.writerow([
                k,
                scr_arm.get(k, ""),
                s12_arm.get(k, ""),
                ";".join(sorted(str(x).lower() for x in scr_readers.get(k, []))),
                "yes" if k in VARIANTS else "no",
                "yes" if (k in design_arm and k not in scr_arm) else "no",
            ])
    print(f"\n# wrote {OUT_CSV}")

    # =============================================== Methods replacement text
    st_order = [k for k in s12["ST"] if k in scr_arm] + \
               [k for k in st_scr if k not in s12["ST"]]
    y_order = [k for k in s12["Y"] if k in scr_arm] + \
              [k for k in y_scr if k not in s12["Y"]]
    txt = (
        "Replacement for main.tex METHODS (the sentence beginning "
        "\"The panel comprised ...\"):\n\n"
        "The panel comprised {ny} tyrosine kinases ({ylist}) and {nst} "
        "serine/threonine kinases ({stlist}); the complete panel with resolved "
        "HGNC symbols is listed in Table S12.\n".format(
            ny=len(y_order),
            nst=len(st_order),
            ylist=", ".join(SYMBOL.get(k, k) for k in y_order),
            stlist=", ".join(SYMBOL.get(k, k) for k in st_order),
        )
    )
    with open(OUT_TXT, "w") as fh:
        fh.write(txt)
    print(f"# wrote {OUT_TXT}\n")
    print(txt)

    print(f"# verdicts: {_counts}")
    return 1 if _counts.get("MISMATCH") else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
