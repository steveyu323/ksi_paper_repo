#!/usr/bin/env python3
"""
Audit items 01 + 02 -- KINBERT training-corpus statistics.

Claims under audit
------------------
01  Results text / Supp Table S1 / SuppFig_corpus.pdf panel a:
        KINBERT corpus = 2,304 kinases, 92 organisms, 147,488 kinase-substrate pairs
        Phosformer = 800 / 13 / 24,534 ; DeepKinZero = 343 / 1 / 13,426
    Figure 2 artwork ("Fig 02 - KINBERT.pdf", called Figure 4 in the review notes)
    instead prints 2,021 kinases / 92 organisms / 162,447 interactions, and its
    per-source panel prints kinase counts 331/336/818/2,277 and organism
    counts 1/10/11/92.

02  Supp Table S3 ("kinases binned by known substrate count"):
        DB+lysate: 1129, 290, 79, 63, 66, 189   -> 1816
        DB only:   1211, 375, 113, 44, 24, 23   -> 1790
    Neither total reaches 2,304.

Where the manuscript numbers were computed
------------------------------------------
``<EVAL>/eval_0_datasource_stats.ipynb``
  cell 18  v1 ("DB only") stats   -> n_kinase=1973, n_org=92,
                                     n_ksi_entry=29077, n_ksi_seq=26267
  cell 20  v2 ("DB + lysate")     -> n_kinase=2304, n_org=92,
                                     n_ksi_entry=170606, n_ksi_seq=147488
  cell 22  v2 substrate-count bins -> 1129/290/79/63/66/189
  cell 23  v1 substrate-count bins -> 1211/375/113/44/24/23
  cell 25  the wide table printed as Supp Table S3
  cells 15/27/28 -> the three UpSet PDFs whose numbers the Fig 2 per-source
                    panel reuses (decoded into eval_tables/ks_pair_upset*.csv
                    by eval_table_scripts/table_00_datasource_upset.py)

This script recomputes everything from the surviving corpus CSVs and prints one
``CLAIM ... | VERDICT=...`` line per claim.

Usage
-----
    python3 -I audit_01_corpus_stats.py [DATA_ROOT] [EVAL_ROOT] [OUT_DIR]

defaults:
    DATA_ROOT = <restore>/data/data
    EVAL_ROOT = <restore>/code/code/evaluation
    OUT_DIR   = <this script>/../out

The corpus CSVs are ~1 GB, so they are read in chunks (only the needed
columns); nothing is sampled and no number below is an estimate.
"""

import csv
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESTORE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

DATA_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESTORE, "data", "data")
EVAL_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RESTORE, "code", "code", "evaluation")
OUT_DIR = sys.argv[3] if len(sys.argv) > 3 else os.path.abspath(os.path.join(HERE, "..", "out"))

INTERACTIONS = os.path.join(DATA_ROOT, "interactions", "interactions")
EVAL_TABLES = os.path.join(EVAL_ROOT, "eval_tables")

CHUNK = 200_000

# The notebook's own bin edges / labels (cells 22-23), kept verbatim so the
# recomputation reproduces Supp Table S3 exactly, label typos and all.
BINS = [0, 5, 20, 50, 100, 200, float("inf")]
BIN_LABELS = ["1-5", "5-20", "20-50", "50-100", "100-200", ">200"]

rows = []          # for the output CSV
claims = []        # (name, manuscript, code, verdict, note)


def record(key, value, note=""):
    rows.append({"quantity": key, "recomputed_value": value, "note": note})
    return value


def claim(name, manuscript, code, note=""):
    if code is None:
        verdict = "UNRESOLVED"
    elif str(manuscript) == str(code):
        verdict = "MATCH"
    else:
        verdict = "MISMATCH"
    claims.append((name, manuscript, code, verdict, note))
    print(f"CLAIM {name} | manuscript={manuscript} | code={code} | VERDICT={verdict}"
          + (f" | {note}" if note else ""))


# ---------------------------------------------------------------------------
# 1. the corpus itself
# ---------------------------------------------------------------------------

def corpus_stats(path, label):
    """Recompute every corpus-level count from one pairs_aug_*.csv.

    Mirrors ``eval_0_datasource_stats.ipynb``:
      * positives only (``label == 1``), as the notebook does
      * "KSI entry"    = unique (substrate, kinase, p_aa, p_ind)  -> site-level pair
      * "KSI sequence" = unique (substrate_seq_15aa, kinase_domain_seq)
    """
    cols = ["substrate", "kinase", "p_ind", "p_aa", "label",
            "kinase_domain_seq", "substrate_seq_15aa"]
    n_rows = n_pos = 0
    kinases = set()
    kinase_domains = set()
    site_pairs = set()
    seq_pairs = set()
    # the notebook's own "n_ksi_entry" key (cells 18/20):
    # drop_duplicates(['substrate_seq_15aa','kinase_domain_seq','substrate','p_ind'])
    nb_entry_pairs = set()
    # counts per kinase, on the sequence-deduplicated frame (the notebook's ks_df)
    seq_pairs_by_kinase = {}
    # counts per kinase, on the site-level frame (no sequence collapse)

    for chunk in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False):
        n_rows += len(chunk)
        pos = chunk[chunk["label"] == 1]
        n_pos += len(pos)
        sub = pos["substrate"].astype(str)
        kin = pos["kinase"].astype(str)
        pid = pos["p_ind"].astype(str)
        paa = pos["p_aa"].astype(str)
        kdom = pos["kinase_domain_seq"].astype(str)
        s15 = pos["substrate_seq_15aa"].astype(str)

        kinases.update(kin)
        kinase_domains.update(kdom)
        for t in zip(sub, kin, paa, pid):
            site_pairs.add(t)
        for t in zip(s15, kdom, sub, pid):
            nb_entry_pairs.add(t)
        for t in zip(s15, kdom, kin):
            # kinase is carried along so the per-kinase histogram can be built
            # from the same deduplicated key the notebook used
            seq_pairs.add(t[:2])
            seq_pairs_by_kinase.setdefault(t, None)

    # collapse the (s15, kdom, kinase) keys -> per-kinase counts.  NOTE: the
    # notebook deduplicates on (substrate_seq_15aa, kinase_domain_seq) only and
    # *then* groups by `kinase`, so a (s15, kdom) seen under two different
    # accessions survives once, under whichever accession pandas kept.  That is
    # reproduced below by deduplicating on (s15, kdom) first.
    # dicts preserve insertion order, so iterating `seq_pairs_by_kinase` walks the
    # rows in file order and `kept` therefore keeps the first occurrence of each
    # (s15, kdom) -- exactly what pandas drop_duplicates() does in the notebook.
    kept = {}
    for s15, kdom, kin in seq_pairs_by_kinase:
        kept.setdefault((s15, kdom), kin)
    per_kinase_seq = {}
    for kin in kept.values():
        per_kinase_seq[kin] = per_kinase_seq.get(kin, 0) + 1

    per_kinase_site = {}
    for sub, kin, paa, pid in site_pairs:
        per_kinase_site[kin] = per_kinase_site.get(kin, 0) + 1

    return {
        "label": label,
        "n_rows": n_rows,
        "n_pos_rows": n_pos,
        "n_kinase_acc": len(kinases),
        "n_kinase_domain_seq": len(kinase_domains),
        "n_site_pairs": len(site_pairs),
        "n_seq_pairs": len(seq_pairs),
        "n_nb_ksi_entry": len(nb_entry_pairs),
        "kinases": kinases,
        "per_kinase_seq": per_kinase_seq,
        "per_kinase_site": per_kinase_site,
    }


def histogram(per_kinase):
    counts = pd.Series(per_kinase, dtype="int64")
    binned = pd.cut(counts, bins=BINS, labels=BIN_LABELS,
                    right=True, include_lowest=True)
    out = binned.value_counts().reindex(BIN_LABELS).fillna(0).astype(int)
    return list(out.values), int(out.sum()), int(counts.isna().sum()), int(binned.isna().sum())


print("=" * 78)
print("AUDIT 01/02 -- KINBERT corpus statistics")
print(f"DATA_ROOT = {DATA_ROOT}")
print(f"EVAL_ROOT = {EVAL_ROOT}")
print("=" * 78)

v2_path = os.path.join(DATA_ROOT, "pairs_aug_v2.csv")
v1_path = os.path.join(DATA_ROOT, "pairs_aug_v1.csv")

v2 = corpus_stats(v2_path, "v2 / DB+lysate") if os.path.exists(v2_path) else None
v1 = corpus_stats(v1_path, "v1 / DB only") if os.path.exists(v1_path) else None

if v2:
    print(f"\n-- pairs_aug_v2.csv (DB + lysate, the KINBERT corpus) --")
    for k in ("n_rows", "n_pos_rows", "n_kinase_acc", "n_kinase_domain_seq",
              "n_site_pairs", "n_seq_pairs", "n_nb_ksi_entry"):
        print(f"   {k:24s} {v2[k]}")
        record(f"v2.{k}", v2[k])
if v1:
    print(f"\n-- pairs_aug_v1.csv (DB only) --")
    for k in ("n_rows", "n_pos_rows", "n_kinase_acc", "n_kinase_domain_seq",
              "n_site_pairs", "n_seq_pairs", "n_nb_ksi_entry"):
        print(f"   {k:24s} {v1[k]}")
        record(f"v1.{k}", v1[k])

# ---------------------------------------------------------------------------
# 2. organisms
# ---------------------------------------------------------------------------
org_path = os.path.join(INTERACTIONS, "all_kinase_organism_info_old.tsv")
n_org_table = n_org_corpus = n_org_mapped = None
if os.path.exists(org_path):
    org = pd.read_csv(org_path, sep="\t").loc[:, ["From", "Organism"]]
    org.columns = ["kinase", "organism"]
    # what the notebook did (cell 20): nunique over the WHOLE annotation table,
    # with no restriction to kinases that are actually in the corpus
    n_org_table = record("organism.nunique_whole_annotation_table",
                         int(org["organism"].nunique()),
                         "cell 20 n_org: not restricted to the corpus")
    if v2:
        m = org[org["kinase"].isin(v2["kinases"])]
        n_org_mapped = record("organism.corpus_kinases_mapped_to_annotation",
                              int(m["kinase"].nunique()))
        n_org_corpus = record("organism.nunique_over_corpus_kinases",
                              int(m["organism"].nunique()))
        print(f"\n-- organisms --")
        print(f"   annotation table rows                {len(org)}")
        print(f"   organisms in whole annotation table  {n_org_table}")
        print(f"   corpus kinases found in annotation   {n_org_mapped} / {v2['n_kinase_acc']}")
        print(f"   organisms spanned by corpus kinases  {n_org_corpus}")

# ---------------------------------------------------------------------------
# 3. where 2,304 came from
# ---------------------------------------------------------------------------
db_pkl = os.path.join(INTERACTIONS, "interaction_classed_221017.pkl")
ly_pkl = os.path.join(INTERACTIONS, "lysate_ksi_cleaned_221115.pkl")
concat_len = union_len = None
if os.path.exists(db_pkl) and os.path.exists(ly_pkl):
    dbk = pd.read_pickle(db_pkl)["kinase"].drop_duplicates().tolist()
    lyk = pd.read_pickle(ly_pkl)["kinase"].drop_duplicates().tolist()
    concat_len = record("n_kinase.cell20_recipe_len_lysate_plus_db", len(lyk + dbk),
                        "len() of two concatenated lists -- NOT deduplicated")
    union_len = record("n_kinase.union_lysate_db", len(set(lyk) | set(dbk)))
    record("n_kinase.db_only_nunique", len(dbk))
    record("n_kinase.lysate_only_nunique", len(lyk))
    record("n_kinase.lysate_db_overlap", len(set(lyk) & set(dbk)))
    print(f"\n-- provenance of 2,304 --")
    print(f"   interaction_classed_221017.pkl  kinase nunique  {len(dbk)}")
    print(f"   lysate_ksi_cleaned_221115.pkl   kinase nunique  {len(lyk)}")
    print(f"   len(lysate_list + db_list)  [cell 20 n_kinase]  {concat_len}")
    print(f"   true union                                      {union_len}")
    print(f"   kinases double-counted                           {len(set(lyk) & set(dbk))}")

# ---------------------------------------------------------------------------
# 4. the Supp Table S3 histogram
# ---------------------------------------------------------------------------
print(f"\n-- Supp Table S3 histogram, recomputed --")
hist = {}
for tag, d in (("v2", v2), ("v1", v1)):
    if not d:
        continue
    seq_h, seq_tot, _, seq_nan = histogram(d["per_kinase_seq"])
    site_h, site_tot, _, site_nan = histogram(d["per_kinase_site"])
    hist[tag] = {"seq": (seq_h, seq_tot), "site": (site_h, site_tot)}
    print(f"   {tag} notebook recipe (dedup on (s15,kdom) then groupby kinase):")
    print(f"       {dict(zip(BIN_LABELS, seq_h))}  total={seq_tot}  (NaN bins: {seq_nan})")
    print(f"   {tag} site-level (every distinct (sub,kin,p_aa,p_ind) counted):")
    print(f"       {dict(zip(BIN_LABELS, site_h))}  total={site_tot}  (NaN bins: {site_nan})")
    for lbl, n in zip(BIN_LABELS, seq_h):
        record(f"{tag}.S3_bin_seqdedup[{lbl}]", n)
    record(f"{tag}.S3_bin_seqdedup_total", seq_tot)
    for lbl, n in zip(BIN_LABELS, site_h):
        record(f"{tag}.S3_bin_sitelevel[{lbl}]", n)
    record(f"{tag}.S3_bin_sitelevel_total", site_tot)

# ---------------------------------------------------------------------------
# 5. per-source counts (from the decoded UpSet figures)
# ---------------------------------------------------------------------------
print(f"\n-- per-source counts (eval_tables/ks_pair_upset*_set_sizes.csv) --")
upset = {}
for kind, fname in (("pair", "ks_pair_upset_set_sizes.csv"),
                    ("kinase", "ks_pair_upset_nkinase_set_sizes.csv"),
                    ("organism", "ks_pair_upset_norganism_set_sizes.csv")):
    p = os.path.join(EVAL_TABLES, fname)
    if not os.path.exists(p):
        print(f"   {fname}: MISSING")
        continue
    df = pd.read_csv(p)
    upset[kind] = dict(zip(df["category"], df["set_size"]))
    print(f"   {kind:9s} {upset[kind]}")
    for cat, n in upset[kind].items():
        record(f"upset.{kind}[{cat}]", int(n))
    # union, from the disjoint intersection table
    ip = os.path.join(EVAL_TABLES, fname.replace("_set_sizes", "_intersections"))
    if os.path.exists(ip):
        u = int(pd.read_csv(ip)["intersection_size"].sum())
        upset[kind + "_union"] = u
        record(f"upset.{kind}.union_of_four_sources", u)
        print(f"   {kind:9s} union over the 4 sources = {u}")

# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("VERDICTS")
print("=" * 78)

if v2:
    claim("01.kinases.text_SuppTableS1_SuppFigS1a", 2304, v2["n_kinase_acc"],
          "2,304 = len(lysate+db) un-deduplicated; corpus has 2,021 distinct accessions")
    claim("01.kinases.Fig02_artwork", 2021, v2["n_kinase_acc"])
    claim("01.pairs.text_SuppTableS1_SuppFigS1a", 147488, v2["n_seq_pairs"],
          "unique (substrate_seq_15aa, kinase_domain_seq)")
    claim("01.pairs.Fig02_artwork", 162447, v2["n_site_pairs"],
          "unique (substrate, kinase, p_aa, p_ind) -- both are real, different units")
if v1:
    # main.tex l.199 introduces the four curated databases, says "The corpus
    # reaches 147,488 interactions across 2,304 kinases and 92 organisms", and
    # only then says "On top of this database-only set, we added a Titania
    # LC-MS/MS dataset ... to produce a more diverse ``DB+Lysate'' set".  Both
    # quoted numbers are DB+lysate (v2) numbers, not database-only (v1) ones.
    claim("01.text.pairs_attributed_to_DB_only_corpus", 147488, v1["n_seq_pairs"],
          "main.tex l.199 attributes 147,488 to the database-only corpus; "
          "DB-only is 26,267 sequence pairs (29,894 site pairs / 29,077 by the "
          "notebook's n_ksi_entry key). 147,488 is the DB+lysate number.")
    claim("01.text.kinases_attributed_to_DB_only_corpus", 2304, v1["n_kinase_acc"],
          "same sentence; DB-only corpus has 1,995 distinct kinase accessions")

if n_org_corpus is not None:
    claim("01.organisms.text_and_Fig02", 92, n_org_corpus,
          f"cell 20 used the whole annotation table ({n_org_table} organisms); "
          f"restricted to corpus kinases it is {n_org_corpus}")
if concat_len is not None:
    claim("01.provenance.2304_is_len_of_concat_lists", 2304, concat_len,
          f"true union = {union_len}")

if "v2" in hist:
    ms_v2 = [1129, 290, 79, 63, 66, 189]
    got, tot = hist["v2"]["seq"]
    for lbl, m, g in zip(BIN_LABELS, ms_v2, got):
        claim(f"02.S3.DBplusLysate[{lbl}]", m, g)
    claim("02.S3.DBplusLysate.total", sum(ms_v2), tot,
          "the bins are exhaustive and non-overlapping; the total IS the kinase count "
          "of the sequence-deduplicated frame")
if "v1" in hist:
    ms_v1 = [1211, 375, 113, 44, 24, 23]
    got, tot = hist["v1"]["seq"]
    for lbl, m, g in zip(BIN_LABELS, ms_v1, got):
        claim(f"02.S3.DBonly[{lbl}]", m, g)
    claim("02.S3.DBonly.total", sum(ms_v1), tot)

if v2 and "v2" in hist:
    claim("02.S3.total_vs_2304_gap", 2304 - sum([1129, 290, 79, 63, 66, 189]),
          2304 - hist["v2"]["seq"][1],
          f"explained as 2304-2021={2304 - v2['n_kinase_acc']} double-counted + "
          f"2021-{hist['v2']['seq'][1]}={v2['n_kinase_acc'] - hist['v2']['seq'][1]} "
          f"accessions dropped by the (s15,kdom) dedup "
          f"({v2['n_kinase_acc']} accessions share only {v2['n_kinase_domain_seq']} "
          f"distinct domain sequences)")

if "kinase" in upset:
    for cat, ms in (("Cell Lysate LC-MS", 331), ("Phospho.ELM", 336),
                    ("PhosphoSitePlus", 818), ("Kinasephos3.0", 2277)):
        claim(f"01.Fig02_panel.kinases[{cat}]", ms, int(upset["kinase"].get(cat, 0)))
if "organism" in upset:
    for cat, ms in (("Cell Lysate LC-MS", 1), ("PhosphoSitePlus", 10),
                    ("Phospho.ELM", 11), ("Kinasephos3.0", 92)):
        claim(f"01.Fig02_panel.organisms[{cat}]", ms, int(upset["organism"].get(cat, 0)))

    # Row alignment inside the Fig 02 panel.  The panel prints one shared label
    # column and two number columns.  Text extracted from
    # "figures/Fig 02 - KINBERT.pdf" puts each number on a label's baseline:
    #   y=44.5  "Cell ..."        kinases 331    organisms 1
    #   y=53.2  "Phospho.ELM"     kinases 336    organisms 10
    #   y=61.9  "PhosphoSitePlus" kinases 818    organisms 11
    #   y=70.4  "Kinasephos3.0"   kinases 2,277  organisms 92
    # The UpSet PDFs sort their rows by set size independently per panel, so the
    # organism column was transcribed in the organism UpSet's row order rather
    # than re-keyed to the kinase panel's label order.
    FIG02_ORG_ROWS = {"Cell Lysate LC-MS": 1, "Phospho.ELM": 10,
                      "PhosphoSitePlus": 11, "Kinasephos3.0": 92}
    for cat, printed in FIG02_ORG_ROWS.items():
        claim(f"01.Fig02_rowalign.organisms[{cat}]", printed,
              int(upset["organism"].get(cat, 0)),
              "Fig 02 panel C row label vs ks_pair_upset_norganism.pdf")

# comparators: no code on this tree recomputes them
for name, val in (("01.comparator.phosformer_kinases", 800),
                  ("01.comparator.phosformer_organisms", 13),
                  ("01.comparator.phosformer_pairs", 24534),
                  ("01.comparator.deepkinzero_kinases", 343),
                  ("01.comparator.deepkinzero_organisms", 1),
                  ("01.comparator.deepkinzero_pairs", 13426)):
    claim(name, val, None,
          "quoted from the source publications in eval_0_datasource_stats.ipynb "
          "cell 14 (markdown); no code on this tree recomputes it")

# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, "audit_01_corpus_stats.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["section", "quantity", "manuscript_value", "recomputed_value",
                "verdict", "note"])
    for q in rows:
        w.writerow(["recomputed", q["quantity"], "", q["recomputed_value"], "",
                    q["note"]])
    for name, ms, code, verdict, note in claims:
        w.writerow(["claim", name, ms, "" if code is None else code, verdict, note])
print(f"\nwrote {csv_path}  ({len(rows)} recomputed quantities, {len(claims)} claims)")
