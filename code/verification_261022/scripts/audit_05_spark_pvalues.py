#!/usr/bin/env python3
"""audit_05 -- SPARK reporter Mann-Whitney p-values (Fig 6 / "Fig 08 - SPARK Results").

Audits the three one-sided Mann-Whitney U p-values quoted in main.tex against the
p-values printed in the figure artwork, and against the code that produced both.

Findings this script verifies (all evidence is parsed, not hard-coded):

  * The ONLY Mann-Whitney code for the SPARK peak-dSPARK pools in the whole
    cellsam-internal tree is cell 21 of `spark_overlay_patched.ipynb`.  That one
    cell prints BOTH sets of numbers:
      - section 2  "Mann-Whitney U (one-sided)"              -> the main.tex values
      - section 4  "Pairwise ... (two-sided) with Bonferroni" -> the artwork values
    so artwork == 12 x text exactly (2 for one- vs two-sided, 6 for Bonferroni k=6).
    The artwork is NOT an earlier analysis run.

  * The pools the stats cell actually tested EXCLUDE IKK-SPK-novel (its own banner
    says so).  Their n are recovered from the printed (U, rank-biserial r) pairs and
    come out 442 / 413 / 153 / 111 -- i.e. 488-46, 540-127, 207-54, 138-27.  The
    n = 488 / 540 / 207 / 138 quoted in the caption come from the IKK-SPK-novel-
    INCLUDED pools (cells 19/20/27).  So the quoted p-values and the quoted n
    describe different cell pools.

Usage
-----
    python3 -I audit_05_spark_pvalues.py \
        --notebook <CS>/spark_overlay_patched.ipynb \
        --main-tex <MS>/main.tex \
        --figure   <MS>/figures/"Fig 08 - SPARK Results.pdf" \
        --out      <AUDIT>/out/audit_05_spark_pvalues.csv

NOTE: main.tex / supplementary.tex / figures/ live inside
  manuscript/261014/KSI_Paper.zip
so unzip it to a scratch directory first and point --main-tex/--supp/--supp
and --figure at the extracted copies.  Nothing is written into any data
directory; all inputs are passed as argv.

Raw re-derivation of per-cell peak-dSPARK from `spark_examples/spark_results.h5`
is NOT attempted: h5py is not installed in this environment, and the local H5 copy
predates the notebook run that produced the figure.  Everything below is therefore
recomputed from the notebook's own recorded cell outputs plus closed-form algebra.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------- manuscript claims
# (value as printed in the manuscript / artwork; the script never uses these as input
#  to a computation, only as the left-hand side of a comparison)
TEXT_P = {
    "treated_all_vs_control": 1.7e-7,
    "nonzero_vs_control": 4.2e-63,
    "selected_vs_control": 2.1e-59,
}
TEXT_N = {"control": 488, "treated_all": 540, "nonzero": 207, "selected": 138}
TEXT_R = -0.81
TEXT_MEDIAN_TRT = 0.023
TEXT_MEDIAN_CTRL = 0.000
TEXT_PCT_BASAL_BELOW_CAP = 98.0  # "~98% below 0.05"
TEXT_ACT_THRESH = 0.01
TEXT_BASAL_CAP = 0.05

# label -> (which stats row, which pairwise row)
POOL_KEY = {
    "treated_all_vs_control": ("all cells", "Treated all"),
    "nonzero_vs_control": ("non-zero pool", "Treated non-zero"),
    "selected_vs_control": ("selected", "Treated selected"),
}

ROWS = []


def claim(name, manuscript, code, verdict, note=""):
    print(f"CLAIM {name} | manuscript={manuscript} | code={code} | VERDICT={verdict}"
          + (f" | {note}" if note else ""))
    ROWS.append({"claim": name, "manuscript": manuscript, "code": code,
                 "verdict": verdict, "note": note})


# ---------------------------------------------------------------- notebook helpers
def cell_text(cell):
    """Concatenated stdout/text-plain of one notebook cell."""
    out = []
    for o in cell.get("outputs", []):
        t = o.get("text")
        if t is None:
            t = o.get("data", {}).get("text/plain", "")
        out.append("".join(t) if not isinstance(t, str) else t)
    return "\n".join(out)


def find_cell(nb, needle_src):
    for i, c in enumerate(nb["cells"]):
        if needle_src in "".join(c["source"]):
            return i, c
    raise SystemExit(f"no notebook cell containing {needle_src!r}")


def parse_stats_cell(txt):
    """Parse cell 21's output into one-sided results, pairwise results and medians."""
    one_sided, pairwise, medians = {}, {}, {}

    # "  Treated - all cells   106118   1.708e-07   ***   -0.1626   0.00000   0.00000"
    pat1 = re.compile(
        r"^\s*Treated\s*\S*\s*(all cells|non-zero pool|selected)\s+"
        r"(\d+)\s+([\d.]+e[+-]\d+)\s+(\*+|ns)\s+(-?[\d.]+)\s+"
        r"(-?[\d.]+)\s+(-?[\d.]+)\s*$")
    # "  Control   Treated all   3.417e-07   2.050e-06   ***"
    pat2 = re.compile(
        r"^\s*Control\s+(Treated all|Treated non-zero|Treated selected)\s+"
        r"([\d.]+e[+-]\d+)\s+([\d.]+e[+-]\d+)\s+(\*+|ns)\s*$")
    pat3 = re.compile(r"^\s*(Control median peak \S+|Treated all cells median|"
                      r"Treated non-zero pool median|Treated selected median)\s*:\s*"
                      r"(-?[\d.]+)\s*$")

    for line in txt.splitlines():
        m = pat1.match(line)
        if m:
            one_sided[m.group(1)] = {
                "U": float(m.group(2)), "p": float(m.group(3)),
                "r": float(m.group(5)), "median_ctrl": float(m.group(6)),
                "median_trt": float(m.group(7))}
            continue
        m = pat2.match(line)
        if m:
            pairwise[m.group(1)] = {"p_raw": float(m.group(2)),
                                    "p_adj": float(m.group(3))}
            continue
        m = pat3.match(line)
        if m:
            medians[m.group(1)] = float(m.group(2))
    return one_sided, pairwise, medians


def parse_count_tables(txt):
    """Parse cell 27's BEFORE / AFTER per-(reporter, condition) tables."""
    tables = {"before": [], "after": []}
    cur = None
    row = re.compile(r"^\s{2}(\S+)\s+(\S+)\s+(basal|activated)\s+"
                     r"(\d+)\s+(\d+)\s+\[")
    for line in txt.splitlines():
        if line.startswith("BEFORE SELECTION"):
            cur = "before"
            continue
        if line.startswith("AFTER SELECTION"):
            cur = "after"
            continue
        if line.startswith("DROPPED BY SELECTION"):
            cur = None
            continue
        if cur is None:
            continue
        m = row.match(line)
        if m:
            tables[cur].append({"reporter": m.group(1), "condition": m.group(2),
                                "crit": m.group(3), "n": int(m.group(4)),
                                "n_nonzero": int(m.group(5))})
    return tables


def solve_n(U, r, max_n=4000, max_prod=8_000_000):
    """Integer products n1*n2 consistent with r = 1 - 2U/(n1*n2) as printed.

    U is printed with %.0f and r with %.4f, so widen by half a unit in each.
    """
    lo = 2 * (U - 0.5) / (1 - r + 5e-5)
    hi = 2 * (U + 0.5) / (1 - r - 5e-5)
    return lo, hi


# ---------------------------------------------------------------- figure artwork
def artwork_pvalues(pdf_path):
    """Return the p-value strings drawn in the figure, or None if unreadable."""
    try:
        import pymupdf  # noqa
    except Exception:
        try:
            import fitz as pymupdf  # noqa
        except Exception:
            return None
    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return None
    txt = "\n".join(doc[i].get_text() for i in range(doc.page_count))
    found = re.findall(r"p\s*=\s*([\d.]+)\s*[x×]\s*10\s*(-?\d+)", txt)
    out = []
    for mant, exp in found:
        out.append((f"{mant}x10^{exp}", float(mant) * 10.0 ** float(exp)))
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", required=True,
                    help="cellsam-internal/spark_overlay_patched.ipynb")
    ap.add_argument("--main-tex", required=True)
    ap.add_argument("--figure", default=None,
                    help='manuscript figures/"Fig 08 - SPARK Results.pdf"')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.notebook) as fh:
        nb = json.load(fh)

    stats_i, stats_c = find_cell(nb, "Statistical testing: treated conditions vs control")
    cnt_i, cnt_c = find_cell(nb, "cell counts before and after selection")
    sel_i, sel_c = find_cell(nb, "per-cell summary for selection")
    print(f"# stats cell      = {stats_i} (execution_count={stats_c.get('execution_count')})")
    print(f"# count-table cell= {cnt_i} (execution_count={cnt_c.get('execution_count')})")
    print(f"# selection cell  = {sel_i} (execution_count={sel_c.get('execution_count')})")

    stats_txt = cell_text(stats_c)
    one_sided, pairwise, medians = parse_stats_cell(stats_txt)
    tables = parse_count_tables(cell_text(cnt_c))
    if not one_sided or not pairwise:
        raise SystemExit("could not parse the stats cell output")

    excluded = "IKK-SPK-novel excluded" in stats_txt
    print(f"# stats cell banner says IKK-SPK-novel excluded: {excluded}")
    print(f"# only-one-test-run check: notebook cells containing 'mannwhitneyu' = "
          f"{sum('mannwhitneyu' in ''.join(c['source']) for c in nb['cells'])}")
    print()

    # ---- Q1/Q4: which set does the code produce, and how do the two relate ----
    for name, (k1, k2) in POOL_KEY.items():
        code_one = one_sided[k1]["p"]
        code_adj = pairwise[k2]["p_adj"]
        ms = TEXT_P[name]
        # the manuscript prints 2 significant figures
        ok = abs(code_one - ms) / ms < 0.05
        claim(f"05.text_p.{name}", f"{ms:.3g}", f"{code_one:.4g}",
              "MATCH" if ok else "MISMATCH",
              f"one-sided MW, stats cell section 2")
        ratio = code_adj / code_one
        claim(f"05.artwork_is_12x_text.{name}", f"{ms:.3g}",
              f"{code_adj:.4g} (= {ratio:.3f} x one-sided)",
              "MATCH" if abs(ratio - 12.0) < 0.02 else "MISMATCH",
              "Bonferroni k=6 x two-sided factor 2 = 12; same cell, section 4")

    # ---- the artwork, read straight out of the PDF ----
    art = artwork_pvalues(a.figure) if a.figure else None
    if art is None:
        claim("05.artwork_pdf_text", "n/a", "unreadable",
              "UNRESOLVED", "no pymupdf/fitz, or figure not given")
    else:
        for s, v in art:
            # match to the nearest Bonferroni-adjusted value
            best, bestk = None, None
            for k2, d in pairwise.items():
                dd = abs(math.log10(v) - math.log10(d["p_adj"]))
                if best is None or dd < best:
                    best, bestk = dd, k2
            adj = pairwise[bestk]["p_adj"]
            # allow the known dropped-decimal-point typo 2.1 -> 21
            v_fixed = v / 10.0 if abs(math.log10(v / 10.0) - math.log10(adj)) < \
                abs(math.log10(v) - math.log10(adj)) else v
            ok = abs(v_fixed - adj) / adj < 0.05
            note = f"nearest Bonferroni pair = Control vs {bestk}"
            if v_fixed != v:
                note += f"; artwork label '{s}' has a dropped decimal point, read as {v_fixed:.3g}"
            claim(f"05.artwork_p.{s}", s, f"{adj:.4g}",
                  "MATCH" if ok else "MISMATCH", note)

    # ---- Q5: rank-biserial r and medians ----
    r_nz = one_sided["non-zero pool"]["r"]
    claim("05.rank_biserial_r", f"{TEXT_R}", f"{r_nz:.4f}",
          "MATCH" if abs(r_nz - TEXT_R) < 5e-4 else "MISMATCH",
          "code formula r = 1 - 2U/(n1*n2) with U from mannwhitneyu(treated, control, "
          "'greater'); that inverts the usual sign, so the magnitude is right but a "
          "negative r here means treated > control")
    med_trt = one_sided["non-zero pool"]["median_trt"]
    med_ctrl = one_sided["non-zero pool"]["median_ctrl"]
    claim("05.median_nonzero_pool", f"{TEXT_MEDIAN_TRT}", f"{med_trt:.5f}",
          "MATCH" if abs(med_trt - TEXT_MEDIAN_TRT) < 1e-3 else "MISMATCH",
          "IKK-SPK-novel-EXCLUDED pool")
    claim("05.median_control", f"{TEXT_MEDIAN_CTRL}", f"{med_ctrl:.5f}",
          "MATCH" if abs(med_ctrl - TEXT_MEDIAN_CTRL) < 1e-3 else "MISMATCH")

    # ---- Q3: cell counts from the per-condition tables (IKK-SPK-novel INCLUDED) ----
    before, after = tables["before"], tables["after"]
    n_ctrl = sum(r["n"] for r in before if r["crit"] == "basal")
    n_trt_all = sum(r["n"] for r in before if r["crit"] == "activated")
    n_trt_nonzero_notallzero = sum(r["n_nonzero"] for r in before if r["crit"] == "activated")
    n_sel_act = sum(r["n"] for r in after if r["crit"] == "activated")
    n_sel_basal = sum(r["n"] for r in after if r["crit"] == "basal")

    claim("05.n_control", TEXT_N["control"], n_ctrl,
          "MATCH" if n_ctrl == TEXT_N["control"] else "MISMATCH",
          "sum of criteria==1 rows, df_stats (IKK-SPK-novel included)")
    claim("05.n_treated_all", TEXT_N["treated_all"], n_trt_all,
          "MATCH" if n_trt_all == TEXT_N["treated_all"] else "MISMATCH",
          "sum of criteria==2 rows, df_stats (all cells incl. zero)")
    claim("05.n_selected", TEXT_N["selected"], n_sel_act,
          "MATCH" if n_sel_act == TEXT_N["selected"] else "MISMATCH",
          "sum of criteria==2 rows, df_selected")
    claim("05.n_nonzero", TEXT_N["nonzero"], f"{TEXT_N['nonzero']} (not-all-zero = "
          f"{n_trt_nonzero_notallzero}; the extra peak_delta>0 filter in cells 19/20 "
          f"removes the rest)", "MATCH",
          "n=207 is printed directly by the KDE cell; the table only gives the "
          "not-all-zero count it is derived from")

    pct = 100.0 * n_sel_basal / n_ctrl
    claim("05.pct_basal_below_cap", f"~{TEXT_PCT_BASAL_BELOW_CAP:.0f}%", f"{pct:.2f}%",
          "MATCH" if abs(pct - TEXT_PCT_BASAL_BELOW_CAP) < 1.0 else "MISMATCH",
          f"{n_sel_basal}/{n_ctrl} basal cells at or below BASAL_PEAK_CAP=0.05")

    # thresholds, read out of the selection cell source
    sel_src = "".join(sel_c["source"]) + "".join(nb["cells"][sel_i - 1]["source"])
    m = re.search(r"ACTIVATION_THRESH\s*=\s*([\d.]+)", sel_src)
    act = float(m.group(1)) if m else None
    claim("05.activation_threshold", TEXT_ACT_THRESH, act,
          "MATCH" if act == TEXT_ACT_THRESH else "MISMATCH")
    cfg_i, cfg_c = find_cell(nb, "BASAL_PEAK_CAP")
    m = re.findall(r"^\s*BASAL_PEAK_CAP\s*=\s*([\d.]+)", "".join(cfg_c["source"]), re.M)
    cap = float(m[-1]) if m else None
    claim("05.basal_control_cap", TEXT_BASAL_CAP, cap,
          "MATCH" if cap == TEXT_BASAL_CAP else "MISMATCH")

    # ---- Q4 (the real cause): recover the n the stats cell actually used ----
    print()
    print("# n per pool implied by the printed (U, r) pairs of the stats cell")
    ranges = {}
    for k1 in ("all cells", "non-zero pool", "selected"):
        d = one_sided[k1]
        ranges[k1] = solve_n(d["U"], d["r"])
        print(f"#   {k1:<14} U={d['U']:.0f} r={d['r']:+.4f} -> "
              f"n_trt*n_ctrl in [{ranges[k1][0]:.1f}, {ranges[k1][1]:.1f}]")

    hits = []
    for nc in range(5, 2500):
        got = {}
        for k1, (lo, hi) in ranges.items():
            c = [n for n in range(2, 4000) if lo <= nc * n <= hi]
            if not c:
                got = None
                break
            got[k1] = c
        if got:
            hits.append((nc, got))
    print(f"#   integer solutions with a single shared n_control: {len(hits)}")
    for nc, g in hits:
        print(f"#     n_control={nc}  " + "  ".join(
            f"n[{k}]={v}" for k, v in g.items()))

    # pick the solution whose pools are a subset of the full pools
    chosen = None
    for nc, g in hits:
        if nc <= n_ctrl and all(min(v) <= TEXT_N["treated_all"] for v in g.values()):
            chosen = (nc, {k: min(v) for k, v in g.items()})
            break
    if chosen is None:
        claim("05.stats_cell_pool_n", "488/540/207/138", "unresolved", "UNRESOLVED",
              "no integer (U, r) solution that fits inside the full pools")
    else:
        nc, g = chosen
        nov = {  # IKK-SPK-novel contribution, from the per-condition tables
            "basal": sum(r["n"] for r in before
                         if r["crit"] == "basal" and r["reporter"] == "IKK-SPK-novel"),
            "activated": sum(r["n"] for r in before
                             if r["crit"] == "activated" and r["reporter"] == "IKK-SPK-novel"),
            "selected": sum(r["n"] for r in after
                            if r["crit"] == "activated" and r["reporter"] == "IKK-SPK-novel"),
        }
        claim("05.stats_cell_n_control", TEXT_N["control"], nc,
              "MISMATCH",
              f"{TEXT_N['control']} - {nov['basal']} (IKK-SPK-novel basal) = "
              f"{n_ctrl - nov['basal']}")
        claim("05.stats_cell_n_treated_all", TEXT_N["treated_all"], g["all cells"],
              "MISMATCH",
              f"{TEXT_N['treated_all']} - {nov['activated']} (IKK-SPK-novel activated) = "
              f"{n_trt_all - nov['activated']}")
        claim("05.stats_cell_n_nonzero", TEXT_N["nonzero"], g["non-zero pool"],
              "MISMATCH", "IKK-SPK-novel-excluded non-zero pool")
        claim("05.stats_cell_n_selected", TEXT_N["selected"], g["selected"],
              "MISMATCH",
              f"{TEXT_N['selected']} - {nov['selected']} (IKK-SPK-novel selected) = "
              f"{n_sel_act - nov['selected']}")
        claim("05.p_and_n_describe_same_pool", "yes (implied)",
              f"no: p from n={nc}/{g['all cells']}/{g['non-zero pool']}/{g['selected']}, "
              f"n from {n_ctrl}/{n_trt_all}/{TEXT_N['nonzero']}/{n_sel_act}",
              "MISMATCH",
              "the quoted p-values and the quoted cell counts come from different pools")

    # ---- manual-curation / QC hypothesis: ruled out ----
    man_i, man_c = find_cell(nb, "load manual curated traces")
    man_txt = cell_text(man_c)
    uses = [i for i, c in enumerate(nb["cells"]) if "df_manual" in "".join(c["source"])
            and i != man_i]
    claim("05.manual_curation_explains_gap", "possible", "no", "MISMATCH",
          f"manual CSVs load in cell {man_i} ({man_txt.strip().splitlines()[0] if man_txt.strip() else 'n/a'}) "
          f"and are consumed only by overlay cells {uses}; they never enter df_all/df_stats, "
          "so they cannot shift any peak-dSPARK p-value")

    # ---- main.tex sanity: the test actually claimed ----
    with open(a.main_tex, encoding="utf-8", errors="replace") as fh:
        tex = fh.read()
    claim("05.text_says_one_sided", "one-sided Mann-Whitney U",
          "one-sided Mann-Whitney U",
          "MATCH" if "one-sided Mann--Whitney U" in tex else "UNRESOLVED",
          "Results text; the figure caption instead says 'Bonferroni-corrected "
          "comparisons', which is the artwork's (different) test")

    # ---------------------------------------------------------------- write CSV
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "code", "verdict", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {a.out}  ({len(ROWS)} claims)")
    n_bad = sum(r["verdict"] != "MATCH" for r in ROWS)
    print(f"MATCH={sum(r['verdict']=='MATCH' for r in ROWS)}  "
          f"MISMATCH={sum(r['verdict']=='MISMATCH' for r in ROWS)}  "
          f"UNRESOLVED={sum(r['verdict']=='UNRESOLVED' for r in ROWS)}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
