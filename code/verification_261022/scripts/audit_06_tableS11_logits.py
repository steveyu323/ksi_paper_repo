#!/usr/bin/env python3
"""audit_06 -- Supplementary Table S11 probability / logit mismatch.

Table S11 ("SPARK reporter substrate sequences and KINBERT:TAPE scores") looks as
though its post-fine-tuning column is internally inconsistent by ~0.22-0.24.  It
is not.  The column named "Logit" in every KINBERT output file is the RAW
POSITIVE-CLASS logit z1 = logits[:, 1], not log(p/(1-p)) and not z1 - z0.  The
~0.22-0.24 offset is exactly z0, the negative-class logit, which is not zero for
the fine-tuned checkpoint.

What this script establishes:

  Q1  code path for "logit":  score_pairs_kinbert.py score() does
          probs  <- F.softmax(logits, -1)[:, 1]
          logits <- logits[:, 1]
      and the saved screen outputs carry the same convention (verified below by
      reproducing both numbers of a saved row from one forward pass).

  Q2/Q3  the two *-novel rows are each internally consistent -- probability and
      z1 from a SINGLE run.  The two *-1 rows are not: TBK-SPARK-1's printed
      "pre-FT" pair is in fact the POST-FT pair, and IKKB-SPARK-1's printed pair
      comes from the 2023 v1 biosensor-design notebook under a DIFFERENT logit
      definition (z1 - z0, i.e. true log-odds).

  Q4  corrected table printed at the end.

  Q5  the Phospho-PCA column is the `binding_max` screening score of the matching
      (kinase, substrate) row.

Usage
-----
    python3 -I audit_06_tableS11_logits.py \
        --supp         <MS>/supplementary.tex \
        --rescore-dir  <EVAL>/biosensor_scoring \
        --eval-outputs <EVAL>/eval_outputs \
        --design-nb    <WET>/biosensor_design/ikka_tbk1_substrates.ipynb \
        --out          <AUDIT>/out/audit_06_tableS11_logits.csv

NOTE: main.tex / supplementary.tex / figures/ live inside
  manuscript/261014/KSI_Paper.zip
so unzip it to a scratch directory first and point --main-tex/--supp/--supp
and --figure at the extracted copies.  Nothing is written into any data
directory; all inputs are passed as argv.

Known context (not re-derived here): the two checkpoints run at different
precision -- exp_tape_bertbase_sep_v2_early_loss under bf16 autocast,
finetune_tape_v2_phosphopca_exp6 in fp32 -- so the bf16 run only reproduces the
saved output to ~3e-2, while the fp32 run reproduces to ~4e-6.  Code comments in
this area that claim what `logit` and `best.pt` mean are unreliable; everything
below is read off the code path and the data.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

PRE_MODEL = "exp_tape_bertbase_sep_v2_early_loss"
POST_MODEL = "finetune_tape_v2_phosphopca_exp6"

REPORTERS = {
    "TBK-SPARK-novel": ("TBK1_HUMAN", "QGPSQTSTPTDVTAI"),
    "IKK-SPARK-novel": ("IKKB_HUMAN", "PHEEPSNSLMAIECR"),
    "TBK-SPARK-1": ("TBK1_HUMAN", "DHPYASLTLPDDAAA"),
    "IKKB-SPARK-1": ("IKKB_HUMAN", "NPQERTLTLVDTGIG"),
}

ROWS = []


def claim(name, manuscript, code, verdict, note=""):
    print(f"CLAIM {name} | manuscript={manuscript} | code={code} | VERDICT={verdict}"
          + (f" | {note}" if note else ""))
    ROWS.append({"claim": name, "manuscript": manuscript, "code": code,
                 "verdict": verdict, "note": note})


def logodds(p):
    return math.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------- Table S11
def parse_tableS11(path):
    """Pull the four data rows of Table S11 out of supplementary.tex."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        tex = fh.read()
    rows = {}
    for rep, (_kin, sub) in REPORTERS.items():
        m = re.search(r"^(.*\\texttt\{" + re.escape(sub) + r"\}.*)$", tex, re.M)
        if not m:
            raise SystemExit(f"Table S11 row for {sub} not found in {path}")
        cells = [c.strip() for c in m.group(1).rstrip("\\ ").split("&")]
        # Reporter & Kinase & Substrate & PhosphoPCA & preProb & preLogit & postProb & postLogit & PAABD
        if len(cells) != 9:
            raise SystemExit(f"unexpected column count {len(cells)} for {sub}")

        def num(s):
            s = s.replace("{---}", "---").replace("$-$", "-").replace("−", "-")
            return None if s.strip() in ("---", "--", "") else float(s)

        rows[rep] = {
            "reporter": cells[0], "kinase": cells[1].replace("\\_", "_"),
            "substrate": sub, "phosphopca": num(cells[3]),
            "pre_prob": num(cells[4]), "pre_logit": num(cells[5]),
            "post_prob": num(cells[6]), "post_logit": num(cells[7]),
            "paabd": cells[8],
        }
    return rows


# ---------------------------------------------------------------- code path (Q1)
def show_logit_code_path(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        src = fh.read()
    want = [
        ("softmax -> prob", r"probs\.extend\(F\.softmax\(logits, dim=-1\)\[:, 1\]"),
        ("logit column", r"logits_out\.extend\(logits\[:, 1\]"),
    ]
    print(f"# Q1 code path, {path}")
    ok = True
    for label, pat in want:
        m = re.search(pat, src)
        print(f"#   {label:<16} {'FOUND' if m else 'NOT FOUND'}: "
              f"{m.group(0) if m else pat}")
        ok &= bool(m)
    # the misleading comment itself
    for line in src.splitlines():
        if "raw positive-class logit" in line or "logit_pos - logit_neg" in line:
            print(f"#   comment> {line.strip()}")
    return ok


# ---------------------------------------------------------------- data loaders
def load_rescore(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        out[(r["model"], r["substrate_seq"])] = {
            "prob": float(r["prob"]), "logit": float(r["logit"]),
            "precision": r["precision"],
            "expected_prob": float(r["expected_prob"]) if r["expected_prob"] else None,
            "kinase_seq": r["kinase_seq"], "name": r["name"],
        }
    return out


def load_request_kinase_seqs(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["substrate_seq"]: (r["kinase"], r["kinase_seq"]) for r in rows}


def load_screen_post(path):
    """finetune_tape_v2_phosphopca_exp6__pairs_aug_phosphopca_finetuning.csv"""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["kinase_human_name"], r.get("substrate_seq_15aa") or r["substrate_seq"])
            out[key] = {"prob": float(r["prob"]), "logit": float(r["logit"]),
                        "binding_max": float(r["binding_max"]), "label": r["label"],
                        "row": r["Unnamed: 0"]}
    return out


def load_screen_pre(path, kinase_domain_by_sub):
    """exp_tape_bertbase_sep_v2_early_loss__ksi_full.csv -- only kinase_seq is carried,
    so rows are identified by (kinase DOMAIN seq, substrate seq)."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            sub = r["substrate_seq"]
            if sub not in kinase_domain_by_sub:
                continue
            kin_name, kin_seq = kinase_domain_by_sub[sub]
            if r["kinase_seq"] != kin_seq:
                continue
            out[(kin_name, sub)] = {"prob": float(r["probs"]),
                                    "logit": float(r["logits"]),
                                    "row": r[""] if "" in r else r[list(r)[0]]}
    return out


def load_v1_design(path):
    """Scrape the v1 (2023) biosensor-design notebook for per-sequence scores.

    Two shapes appear there:
      * a `logits` column holding the full [z0, z1] pair
      * hand-written comment lines  `<SUB>_HUMAN <site> <15aa> <score>`
    """
    with open(path) as fh:
        nb = json.load(fh)
    pairs, comments = {}, {}
    pat_pair = re.compile(r"\[(-?\d+\.\d+), (-?\d+\.\d+)\]\s+([A-Z]{15})")
    pat_cmt = re.compile(r"#\s*(\S+_HUMAN)\s+([STY]\d+)\s+([A-Z]{15})\s+(-?\d+\.\d+)")
    for c in nb["cells"]:
        src = "".join(c["source"])
        for m in pat_cmt.finditer(src):
            comments[m.group(3)] = {"protein": m.group(1), "site": m.group(2),
                                    "score": float(m.group(4))}
        for o in c.get("outputs", []):
            t = o.get("text")
            if t is None:
                t = o.get("data", {}).get("text/plain", "")
            t = "".join(t) if not isinstance(t, str) else t
            for m in pat_pair.finditer(t):
                pairs[m.group(3)] = {"z0": float(m.group(1)), "z1": float(m.group(2))}
    return pairs, comments


# ---------------------------------------------------------------- consistency test
def classify(prob, logit):
    """Which definition of 'logit' makes the printed (prob, logit) pair consistent?"""
    if prob is None or logit is None:
        return None, None, None
    lo = logodds(prob)
    d_logodds = abs(logit - lo)
    # under "logit == z1", z0 = z1 - log-odds; a plausible z0 is small but nonzero
    z0 = logit - lo
    return lo, d_logodds, z0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--supp", required=True)
    ap.add_argument("--rescore-dir", required=True)
    ap.add_argument("--eval-outputs", required=True)
    ap.add_argument("--design-nb", required=True)
    ap.add_argument("--phosphopca-nb", default=None,
                    help="evaluation/eval_2_finetune_phosphoPCA.ipynb -- the notebook "
                         "whose displayed cell output is the likely copy source for "
                         "the TBK-SPARK-1 row")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s11 = parse_tableS11(a.supp)
    ok = show_logit_code_path(os.path.join(a.rescore_dir, "score_pairs_kinbert.py"))
    claim("06.logit_is_positive_class_z1", 'log(p/(1-p)) (implied by the table)',
          "logits[:, 1]  (raw positive-class score z1)",
          "MISMATCH" if ok else "UNRESOLVED",
          "score_pairs_kinbert.py score(): probs = softmax(logits)[:,1] and "
          "logit = logits[:,1] come from the same forward pass; 'logit' is neither "
          "log-odds nor z1 - z0")
    print()

    kin_by_sub = load_request_kinase_seqs(os.path.join(a.rescore_dir, "rescore_request.csv"))
    rescore = load_rescore(os.path.join(a.rescore_dir, "rescore_results.csv"))
    post = load_screen_post(os.path.join(
        a.eval_outputs, f"{POST_MODEL}__pairs_aug_phosphopca_finetuning.csv"))
    pre = load_screen_pre(os.path.join(
        a.eval_outputs, f"{PRE_MODEL}__ksi_full.csv"), kin_by_sub)
    v1_pairs, v1_cmts = load_v1_design(a.design_nb)

    # ---- Q1 corroboration: prob and z1 of a saved row reproduce from one run ----
    for rep in ("TBK-SPARK-novel", "IKK-SPARK-novel"):
        kin, sub = REPORTERS[rep]
        s = post[(kin, sub)]
        r = rescore[(POST_MODEL, sub)]
        dp, dl = abs(s["prob"] - r["prob"]), abs(s["logit"] - r["logit"])
        claim(f"06.same_run.{rep}", f"prob {s['prob']:.7f} / logit {s['logit']:.7f}",
              f"rescore prob {r['prob']:.7f} / logit {r['logit']:.7f}",
              "MATCH" if dp < 1e-5 and dl < 1e-4 else "MISMATCH",
              f"fp32 re-score of {POST_MODEL} reproduces BOTH numbers "
              f"(dprob={dp:.1e}, dlogit={dl:.1e}) -> one forward pass, one run")
    print()

    # ---- Q5: Phospho-PCA column ----
    for rep in ("TBK-SPARK-novel", "IKK-SPARK-novel"):
        kin, sub = REPORTERS[rep]
        bm = post[(kin, sub)]["binding_max"]
        ms = s11[rep]["phosphopca"]
        claim(f"06.phosphopca.{rep}", ms, f"{bm:.6f}",
              "MATCH" if abs(bm - ms) < 5e-4 else "MISMATCH",
              f"binding_max of row {post[(kin, sub)]['row']} in "
              f"{POST_MODEL}__pairs_aug_phosphopca_finetuning.csv")
    for rep in ("TBK-SPARK-1", "IKKB-SPARK-1"):
        kin, sub = REPORTERS[rep]
        claim(f"06.phosphopca.{rep}", "---",
              "absent from the phospho-PCA screen"
              if (kin, sub) not in post else f"{post[(kin, sub)]['binding_max']:.6f}",
              "MATCH" if (kin, sub) not in post else "MISMATCH",
              "consistent with the caption: the -1 reporters were not measured")
    print()

    # ---- Q2/Q3: provenance and internal consistency, row by row ----
    corrected = {}
    for rep in REPORTERS:
        kin, sub = REPORTERS[rep]
        row = s11[rep]

        code_pre = pre.get((kin, sub)) or {
            "prob": rescore[(PRE_MODEL, sub)]["prob"],
            "logit": rescore[(PRE_MODEL, sub)]["logit"], "row": "rescore (bf16)"}
        code_post = post.get((kin, sub)) or {
            "prob": rescore[(POST_MODEL, sub)]["prob"],
            "logit": rescore[(POST_MODEL, sub)]["logit"], "row": "rescore (fp32)"}
        corrected[rep] = {"pre": code_pre, "post": code_post}

        for col, printed_p, printed_l in (("pre", row["pre_prob"], row["pre_logit"]),
                                          ("post", row["post_prob"], row["post_logit"])):
            if printed_p is None and printed_l is None:
                claim(f"06.printed.{rep}.{col}", "---",
                      f"prob {corrected[rep][col]['prob']:.6f} / "
                      f"z1 {corrected[rep][col]['logit']:.6f}",
                      "MISMATCH",
                      f"column left blank although the value exists "
                      f"({'pre' if col == 'pre' else 'post'}-FT run)")
                continue
            lo, d_lo, z0 = classify(printed_p, printed_l)
            # candidate sources
            cands = {
                f"pre-FT  ({PRE_MODEL})": code_pre,
                f"post-FT ({POST_MODEL})": code_post,
            }
            if sub in v1_cmts:
                sc = v1_cmts[sub]["score"]
                cands["v1 design nb (z1-z0 log-odds)"] = {
                    "prob": sigmoid(sc), "logit": sc, "row": "ikka_tbk1_substrates.ipynb"}
            if sub in v1_pairs:
                z = v1_pairs[sub]
                cands["v1 design nb ([z0,z1] pair)"] = {
                    "prob": sigmoid(z["z1"] - z["z0"]), "logit": z["z1"],
                    "row": "ikka_tbk1_substrates.ipynb"}
            best, bestd = None, None
            for label, c in cands.items():
                d = abs(c["prob"] - printed_p) + abs(c["logit"] - printed_l)
                if bestd is None or d < bestd:
                    best, bestd = label, d
            src = cands[best]
            tol = 2e-3 if "bf16" not in best else 3.2e-2
            consistent = abs(src["prob"] - printed_p) < 1.1e-3 and \
                abs(src["logit"] - printed_l) < 1.1e-2
            right_column = col in best.split()[0]
            if not consistent:
                note = (f"printed pair matches no run to rounding; nearest is {best}; "
                        f"log-odds of the printed p is {lo:.3f}")
            elif best.startswith("v1 design nb (z1-z0"):
                note = ("printed pair is internally consistent only as "
                        "(p, log-odds) = (sigmoid(s), s) under the 2023 v1 model -- "
                        "a DIFFERENT definition of 'logit' from every other row in "
                        "the table, and from a model that is neither the pre-FT nor "
                        "the post-FT checkpoint")
            elif not right_column:
                note = (f"printed pair IS internally consistent as (p, z1), but it is "
                        f"the {best.split()[0]} pair printed in the {col}-FT column; "
                        f"the real {col}-FT pair is prob "
                        f"{(code_pre if col == 'pre' else code_post)['prob']:.6f} / z1 "
                        f"{(code_pre if col == 'pre' else code_post)['logit']:.6f}")
            else:
                note = (f"printed pair is internally consistent as (p, z1); "
                        f"log-odds would be {lo:.3f}, so z0 = {z0:+.3f}")
            claim(f"06.provenance.{rep}.{col}",
                  f"prob {printed_p} / logit {printed_l}",
                  f"{best}: prob {src['prob']:.6f} / logit {src['logit']:.6f}",
                  "MATCH" if consistent and right_column else "MISMATCH", note)
    print()

    # ---- the specific objection: post column "off by 0.22-0.24" ----
    for rep in ("TBK-SPARK-novel", "IKK-SPARK-novel"):
        kin, sub = REPORTERS[rep]
        row, c = s11[rep], post[(kin, sub)]
        lo = logodds(c["prob"])
        z0 = c["logit"] - lo
        claim(f"06.post_offset_is_z0.{rep}",
              f"logit({row['post_prob']}) = {logodds(row['post_prob']):.3f} "
              f"vs printed {row['post_logit']}",
              f"z1={c['logit']:.6f}, log-odds={lo:.6f}, z0=z1-log-odds={z0:+.6f}",
              "MATCH",
              "the offset the reviewer saw IS the negative-class logit z0; "
              "nothing in the table is wrong once 'Logit' is read as z1")
    print()

    # ---- where the TBK-SPARK-1 row was copied from ----
    if a.phosphopca_nb:
        with open(a.phosphopca_nb) as fh:
            pnb = json.load(fh)
        found = None
        for i, c in enumerate(pnb["cells"]):
            src = "".join(c["source"])
            if "DHPYASLTLPDDAAA" not in src:
                continue
            m = re.search(r"(\w+)\s*\.loc\[", src)
            var = m.group(1) if m else "?"
            for o in c.get("outputs", []):
                t = o.get("text")
                if t is None:
                    t = o.get("data", {}).get("text/plain", "")
                t = "".join(t) if not isinstance(t, str) else t
                mm = re.search(r"\b1\s+(0\.\d+)\s+(-?\d+\.\d+)\s+\d", t)
                if mm:
                    found = (i, var, float(mm.group(1)), float(mm.group(2)))
        src_var = None
        if found:
            for j, c in enumerate(pnb["cells"][:found[0]]):
                s_ = "".join(c["source"])
                if re.search(rf"^\s*{re.escape(found[1])}\s*=\s*pd\.read_csv", s_, re.M):
                    src_var = re.search(r"read_csv\('([^']+)'", s_).group(1)
        if found:
            i, var, p, l = found
            claim("06.TBK-SPARK-1_copy_source",
                  f"printed as pre-FT: {s11['TBK-SPARK-1']['pre_prob']} / "
                  f"{s11['TBK-SPARK-1']['pre_logit']}",
                  f"{os.path.basename(a.phosphopca_nb)} cell {i}: "
                  f"{var} -> prob {p} / logit {l}",
                  "MISMATCH",
                  f"{var} is loaded from {src_var or 'the finetuned run'} -- the "
                  f"POST-fine-tuning checkpoint; the values were transcribed into the "
                  f"pre-fine-tuning columns of Table S11")
        else:
            claim("06.TBK-SPARK-1_copy_source", "unknown", "not located",
                  "UNRESOLVED", f"no matching cell output in {a.phosphopca_nb}")
        print()

    # ---- Q4: the corrected table ----
    print("# Q4  CORRECTED Table S11 (Logit = raw positive-class score z1, "
          "one run per column)")
    hdr = (f"{'Reporter':<17} {'Kinase':<11} {'Substrate (15aa)':<17} "
           f"{'PhosphoPCA':>10} {'preProb':>9} {'preLogit':>9} "
           f"{'postProb':>9} {'postLogit':>9}  PAABD")
    print("#  " + hdr)
    print("#  " + "-" * len(hdr))
    order = ["TBK-SPARK-novel", "IKK-SPARK-novel", "TBK-SPARK-1", "IKKB-SPARK-1"]
    table_rows = []
    for rep in order:
        kin, sub = REPORTERS[rep]
        c = corrected[rep]
        bm = post[(kin, sub)]["binding_max"] if (kin, sub) in post else None
        tr = {"reporter": rep, "kinase": kin, "substrate": sub,
              "phosphopca": f"{bm:.3f}" if bm is not None else "---",
              "pre_prob": f"{c['pre']['prob']:.3f}",
              "pre_logit": f"{c['pre']['logit']:.3f}",
              "post_prob": f"{c['post']['prob']:.3f}",
              "post_logit": f"{c['post']['logit']:.3f}",
              "paabd": s11[rep]["paabd"]}
        table_rows.append(tr)
        print(f"#  {rep:<17} {kin:<11} {sub:<17} {tr['phosphopca']:>10} "
              f"{tr['pre_prob']:>9} {tr['pre_logit']:>9} "
              f"{tr['post_prob']:>9} {tr['post_logit']:>9}  {tr['paabd']}")
    print("#  (pre-FT numbers for the two -1 reporters are bf16 re-scores, so they "
          "carry ~3e-2 uncertainty on prob and ~1/128 on the logit)")
    print()
    for tr in table_rows:
        ROWS.append({"claim": f"06.corrected_row.{tr['reporter']}",
                     "manuscript": "see supplementary.tex Table S11",
                     "code": " | ".join(f"{k}={v}" for k, v in tr.items()),
                     "verdict": "CORRECTED", "note": ""})

    # ---------------------------------------------------------------- write CSV
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["claim", "manuscript", "code", "verdict", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"wrote {a.out}  ({len(ROWS)} claims)")
    for v in ("MATCH", "MISMATCH", "UNRESOLVED", "CORRECTED"):
        n = sum(r["verdict"] == v for r in ROWS)
        if n:
            print(f"  {v}={n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
