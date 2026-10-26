#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUDIT ITEM 09 - substrate library composition and spike-in provenance (Table S13),
plus the provenance of the Table S10 database-vs-screening benchmark (403 pairs)
and its overlap with KINBERT's training corpus.

Re-runnable.  All data paths are passed as argv so that nothing is written into,
and no interpreter is started from inside, a data directory.

Usage
-----
python3 -I audit_09_spikein_provenance.py \
    --planning   <.../kinase_ml/code/wetlab_planning> \
    --kinase-ml  <.../kinase_ml> \
    --analysis   <.../ksi_analysis_2025> \
    --screening  <.../screening_analysis> \
    --corpus-old <.../tarballs/training_data/training_data/training_data> \
    --splits     <.../tarballs/data/data> \
    --out        <.../audit_261022/out>

Defaults point at the locations used for the 2026-10-22 audit.

Outputs
-------
out/audit_09_spikein_provenance.csv      one row per claim
out/audit_09_training_overlap_pairs.csv  every screened (kinase, 15-mer) pair
                                         found in a KINBERT corpus split
"""

import argparse
import os
import sys

import pandas as pd

# --------------------------------------------------------------------------- #
# defaults
# --------------------------------------------------------------------------- #
COLOSSUS = os.environ.get("KSI_REPO_ROOT", "/path/to/analysis_repo_root")
TARBALLS = os.environ.get("KSI_RESTORE_ROOT", "/path/to/restore_root")

DEF = {
    "planning":   f"{COLOSSUS}/kinase_ml/code/wetlab_planning",
    "kinase_ml":  f"{COLOSSUS}/kinase_ml",
    "analysis":   f"{COLOSSUS}/ksi_analysis_2025",
    "screening":  f"{COLOSSUS}/screening_analysis",
    "corpus_old": f"{TARBALLS}/training_data/training_data/training_data",
    "splits":     f"{TARBALLS}/data/data",
    "out":        f"{TARBALLS}/manuscript/audit_261022/out",
}

# Filters hard-coded in 8_2batch_blib_pipeline_v2.ipynb (cells 21, 22)
EXCLUDE_SOURCES = {"ms", "y2h", "phosphonetwork", "deepkinzero",
                   "phosphoelm", "kinasephos"}
EXCLUDE_KINASES = {"KS6B1_HUMAN_active", "KS6B1_HUMAN_wt", "PLK1_HUMAN_active"}

CLAIMS = []


def claim(name, manuscript, code, verdict=None, note=""):
    """Record and print one CLAIM line."""
    if verdict is None:
        verdict = "MATCH" if str(manuscript) == str(code) else "MISMATCH"
    print(f"CLAIM {name} | manuscript={manuscript} | code={code} "
          f"| VERDICT={verdict}")
    if note:
        print(f"      note: {note}")
    CLAIMS.append({"claim": name, "manuscript": manuscript, "code": code,
                   "verdict": verdict, "note": note})


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# --------------------------------------------------------------------------- #
# shared helpers (semantics copied verbatim from the authors' notebooks)
# --------------------------------------------------------------------------- #
def add_window_15(df):
    """Vectorised equivalent of assign_paabd()'s 15-mer extraction.

    Notebook code:  substrate_seq[p_ind-1-7 : p_ind-1+8], then len==15.
    p_ind < 8 makes the start index negative, which in the original apply()
    produced a short/empty string that the len==15 filter removed; the guard
    below reproduces that outcome without relying on negative-index wrap.
    """
    seqs = df["substrate_seq"].tolist()
    inds = df["p_ind"].tolist()
    win = [s[p - 8:p + 7] if p >= 8 else "" for s, p in zip(seqs, inds)]
    out = df.copy()
    out["substrate_seq_15aa"] = win
    return out.loc[[len(w) == 15 for w in win], :]


def spike_polarity(df):
    """Per unique 15-mer: positive / negative / mixed, from the `set` column."""
    g = df.groupby("substrate_seq_15aa")["set"].apply(set)
    pos = g.apply(lambda s: bool(s & {"test_pos", "train_pos"}))
    neg = g.apply(lambda s: "train_tn" in s)
    return int((pos & ~neg).sum()), int((neg & ~pos).sum()), int((pos & neg).sum())


# =========================================================================== #
# PART 1 - designed spike-ins and designed library (Table S13)
# =========================================================================== #
def part1_design(P):
    hdr("PART 1 - DESIGNED SPIKE-INS AND DESIGNED LIBRARY  (Q1, Q5)")

    st_sp = pd.read_csv(os.path.join(P, "st_spike_ins.csv"))
    y_sp = pd.read_csv(os.path.join(P, "y_spike_ins.csv"))
    sty_sp = pd.concat([st_sp, y_sp], ignore_index=True)

    n_st = st_sp["substrate_seq_15aa"].nunique()
    n_y = y_sp["substrate_seq_15aa"].nunique()
    claim("S13_spikein_ST", 465, n_st)
    claim("S13_spikein_Y", 151, n_y)
    claim("S13_spikein_total", 616, n_st + n_y)
    claim("Results_spikein_approx300", "~300", n_st + n_y, "MISMATCH",
          "Results/Methods '~300 documented kinase-substrate pairs' is wrong: "
          "616 unique spike-in windows were designed, 449 of them positive "
          "pairs. ~300 is not reproducible from any file in the design path.")

    print("\n  designed spike-in `set` composition (row level):")
    for lab, d in (("ST", st_sp), ("Y", y_sp)):
        print(f"    {lab}: {d['set'].value_counts().to_dict()}  "
              f"rows={len(d)}  unique 15-mers={d['substrate_seq_15aa'].nunique()}  "
              f"unique (15-mer,kinase) pairs="
              f"{len(d.drop_duplicates(['substrate_seq_15aa', 'kinase']))}  "
              f"target kinases={d['kinase'].nunique()}")

    pos, neg, mixed = spike_polarity(sty_sp)
    claim("S13_spikein_documented_pairs", 449, pos)
    claim("S13_spikein_documented_negatives", 165, neg)
    claim("S13_spikein_both_directions", 2, mixed)

    # designed library / oligo pool
    st_ol = pd.read_csv(os.path.join(P, "st_oligo_full_codon.csv"))
    y_ol = pd.read_csv(os.path.join(P, "y_oligo_full_codon.csv"))
    claim("S13_designed_rows_ST", 1978, len(st_ol))
    claim("S13_designed_rows_Y", 1950, len(y_ol))
    claim("S13_designed_rows_total", 3928, len(st_ol) + len(y_ol))

    s_st, s_y = set(st_ol["substrate_seq_15aa"]), set(y_ol["substrate_seq_15aa"])
    claim("S13_distinct_sequences", 3918, len(s_st | s_y))
    claim("S13_sequences_in_both_libraries", 10, len(s_st & s_y))

    # class counts, two defensible conventions
    nb = pd.concat([st_ol[["substrate_seq_15aa", "set"]],
                    y_ol[["substrate_seq_15aa", "set"]]], ignore_index=True)
    nb_counts = nb["set"].value_counts().to_dict()
    nb_psp = nb_counts.get("psp_random", 0)
    nb_animal = nb_counts.get("animal_random", 0)
    nb_spike = nb_counts.get("spike_in", 0)
    print(f"\n  design-notebook `set` column (animal label applied after psp, "
          f"cell 27): psp={nb_psp} animal={nb_animal} spike_in={nb_spike}")

    # build_table_s13.py convention: psp membership wins over animal
    st_psp = set(pd.read_csv(os.path.join(P, "st_sub_psp.csv")).iloc[:, 0])
    y_psp = set(pd.read_csv(os.path.join(P, "y_sub_psp.csv")).iloc[:, 0])
    spike_st = set(st_sp["substrate_seq_15aa"])
    spike_y = set(y_sp["substrate_seq_15aa"])

    def reclass(df, psp_pool, spikes):
        out = []
        for s in df["substrate_seq_15aa"]:
            if s in spikes:
                out.append("spike_in")
            elif s in psp_pool:
                out.append("psp_random")
            else:
                out.append("animal_random")
        return pd.Series(out)

    g_st = reclass(st_ol, st_psp, spike_st).value_counts().to_dict()
    g_y = reclass(y_ol, y_psp, spike_y).value_counts().to_dict()
    g_psp = g_st.get("psp_random", 0) + g_y.get("psp_random", 0)
    g_animal = g_st.get("animal_random", 0) + g_y.get("animal_random", 0)
    print(f"  build_table_s13.py convention (psp pool wins over animal): "
          f"psp={g_psp} animal={g_animal}   ST={g_st}  Y={g_y}")

    claim("S13_candidates_total", 3312, nb_psp + nb_animal)
    claim("S13_candidates_PSP", 2753, g_psp, None,
          f"2,753 reproduces only under build_table_s13.py's class priority "
          f"(PhosphoSitePlus-pool membership overrides the animal pool). The "
          f"design notebook's own `set` column gives {nb_psp}; the 10-sequence "
          f"difference is windows present in BOTH pools.")
    claim("S13_candidates_animal", 559, g_animal, None,
          f"design notebook's `set` column gives {nb_animal} (same 10 windows)")

    # central residue
    st_ctr = st_ol["substrate_seq_15aa"].str[7].value_counts().to_dict()
    y_ctr = y_ol["substrate_seq_15aa"].str[7].value_counts().to_dict()
    print(f"\n  central residue  ST lib: {st_ctr}   Y lib: {y_ctr}")
    claim("S13_ST_lib_all_SerThr_centred",
          "all ST windows S/T", "yes" if set(st_ctr) <= {"S", "T"} else "no",
          "MATCH" if set(st_ctr) <= {"S", "T"} else "MISMATCH")
    y_stc = y_ol[y_ol["substrate_seq_15aa"].str[7].isin(["S", "T"])]
    claim("S13_Y_lib_SerThr_centred_windows", 45, len(y_stc))
    claim("S13_Y_lib_SerThr_all_from_PSP", "all PhosphoSitePlus",
          "/".join(sorted(y_stc["set"].unique())),
          "MATCH" if set(y_stc["set"]) == {"psp_random"} else "MISMATCH")

    # Levenshtein thresholds, read out of the notebook source
    claim("S13_levenshtein_spikein_dedup", 7, 7, "MATCH",
          "230310_select_substrate_oligo.ipynb cells 6 and 34: DIST_THRESH = 7. "
          "sanitize_redundancy() DROPS members with distance < thresh, so the "
          "retained set has pairwise Levenshtein >= 7.")
    claim("S13_levenshtein_PSP_pool", 9, 9, "MATCH",
          "cells 13 and 41: sanitize_redundancy(thresh=9); cell 21 re-applies "
          "thresh=9 to the combined PSP+animal Y pool")
    claim("S13_levenshtein_animal_pool", 11, 11, "MATCH",
          "cell 18: sanitize_redundancy(thresh=11), Y library only")
    claim("Methods_levenshtein_wording", "spike-ins '< 7', candidates '> 8'",
          "spike-ins >= 7, PSP pool >= 9, animal pool >= 11", "MISMATCH",
          "Methods inverts the spike-in threshold ('removal of redundant "
          "sequences (Levenshtein distance < 7)' describes what was removed, "
          "not what was kept) and collapses two different candidate thresholds "
          "(9 for PhosphoSitePlus, 11 for animal) into one '> 8'.")

    return st_sp, y_sp, sty_sp, st_ol, y_ol


# =========================================================================== #
# PART 2 - 15-mer -> protein mapping (Table S13 ambiguity claims)
# =========================================================================== #
def part2_mapping(P, A):
    hdr("PART 2 - 15-MER -> SOURCE PROTEIN MAPPING  (Q5)")

    bg = pd.read_pickle(os.path.join(A, "substrates_background_12345.pkl"))
    st = pd.read_csv(os.path.join(P, "st_sub_all.csv"))
    y = pd.read_csv(os.path.join(P, "y_sub_all.csv"))
    st["library"], y["library"] = "ST", "Y"
    rows = pd.concat([st, y], ignore_index=True)          # 3,928 ordered rows
    uniq = rows.drop_duplicates("substrate_seq_15aa")      # 3,918 distinct

    mapped = set(bg["substrate_seq_15aa"])
    unmapped = sorted(set(uniq["substrate_seq_15aa"]) - mapped)
    claim("S13_unmappable_count", 1, len(unmapped))
    claim("S13_unmappable_sequence", "VGAPIAVSAFYAESC",
          ";".join(unmapped) or "(none)")
    if unmapped:
        print(f"      (that window is itself a spike-in: design_set=spike_in, "
              f"set=train_tn, target kinase P53350)")

    lib_windows = set(uniq["substrate_seq_15aa"])
    nprot = (bg[bg["substrate_seq_15aa"].isin(lib_windows)]
             .groupby("substrate_seq_15aa")["substrate"].nunique())
    rows = rows.merge(nprot.rename("n_protein_matches"),
                      left_on="substrate_seq_15aa", right_index=True, how="left")
    rows["n_protein_matches"] = rows["n_protein_matches"].fillna(0).astype(int)

    ambig_rows = int((rows["n_protein_matches"] > 1).sum())
    ambig_uniq = int((nprot.reindex(uniq["substrate_seq_15aa"]).fillna(0) > 1).sum())
    claim("S13_multimapping_windows", 2069, ambig_rows, None,
          f"2,069 is a ROW count over the 3,928 ordered rows "
          f"({100 * ambig_rows / len(rows):.1f}%), matching build_table_s13.py. "
          f"At the level of the 3,918 DISTINCT windows it is {ambig_uniq} "
          f"({100 * ambig_uniq / len(uniq):.1f}%). Either way 53% is right, but "
          f"the caption should say which denominator it uses.")
    claim("S13_multimapping_pct", "53%", f"{100 * ambig_rows / len(rows):.1f}%",
          "MATCH", "52.7% rounds to 53%")
    print(f"      max distinct source proteins for a single window: "
          f"{int(nprot.max())}")
    return rows


# =========================================================================== #
# PART 3 - reproduce the Table S10 benchmark (403 pairs) and show its provenance
# =========================================================================== #
def part3_benchmark(KM, A, S):
    hdr("PART 3 - WHERE THE 403 BENCHMARK PAIRS COME FROM  (Q2)")
    print("""  Code path (ksi_analysis_2025/8_2batch_blib_pipeline_v2.ipynb):
    cell 14  loads {st,y}_spike_ins.csv ...
    cell 17  ... and only PRINTS them.  The designed spike-ins are NEVER used
             in the confusion matrix.  <-- confirms the author's statement
    cell 10  interaction_classed_221017_wsource.pkl  -> y_true = 1
             (the curated KSI database = KINBERT's positive training corpus)
    cell 11  kinbertv1_pred/kinbertv1_DB_Test.pkl    -> y_true as stored
             (KINBERT v1's held-out DB test split, incl. its sampled negatives)
    cell 12  data_anno = concat([interaction_classed, test_pred_v1])
    cell 21  drop EXCLUDE_SOURCES                    -> train_dat_pos
    cell 22  dat_in_screen = screen x blib kinases, drop EXCLUDE_KINASES
    cell 24  curr_df_all = INNER MERGE(dat_in_screen, train_dat_pos)
             on (kinase_acc, substrate 15-mer)       <-- the benchmark set
             'set' is then re-derived from y_true and mislabelled
             'train_pos'/'train_tn', reusing the spike-in vocabulary.
""")

    human_kinome = pd.read_pickle(
        os.path.join(KM, "data", "human_kinome_seq_230301.pkl"))

    ic = add_window_15(pd.read_pickle(os.path.join(
        KM, "data", "interactions", "interaction_classed_221017_wsource.pkl")))
    ic = (ic[["kinase", "substrate_seq_15aa", "source_all"]]
          .drop_duplicates().rename(columns={"source_all": "source"}))
    ic["y_true"] = 1
    print(f"  interaction DB unique KSIs: {len(ic)}")

    tp = add_window_15(pd.read_pickle(os.path.join(
        KM, "data", "kinbertv1_pred", "kinbertv1_DB_Test.pkl")))
    tp = (tp[["kinase_seq", "substrate_seq_15aa", "y_true", "source"]]
          .merge(human_kinome, on="kinase_seq")
          [["kinase", "substrate_seq_15aa", "source", "y_true"]])
    print(f"  kinbertv1 DB_Test rows after kinome merge: {len(tp)}  "
          f"{tp['y_true'].value_counts().to_dict()}")

    data_anno = pd.concat([ic, tp], ignore_index=True)
    tdp = data_anno.loc[~data_anno["source"].isin(EXCLUDE_SOURCES), :]
    print(f"  high-confidence annotation rows: {len(tdp)}  "
          f"{tdp['y_true'].value_counts(dropna=False).to_dict()}")
    print("  NOTE: after the source filter, every surviving POSITIVE comes from\n"
          "        interaction_classed (curated DB) and every surviving NEGATIVE\n"
          "        comes from kinbertv1_DB_Test's source=NaN rows, i.e. KINBERT's\n"
          "        1-negative-per-positive RANDOMLY SAMPLED decoys - not\n"
          "        experimentally documented non-substrates.")
    print("  NOTE: EXCLUDE_SOURCES is matched with .isin() against a comma-joined\n"
          "        string ('deepkinzero,phosphositeplus,kinasephos'), so it only\n"
          "        drops rows whose source_all is EXACTLY one excluded name. It\n"
          "        happens to do the intended job for the common combinations,\n"
          "        but it is not a real per-source filter.")

    blib = pd.read_csv(os.path.join(S, "blib_kinases_with_acc.csv"), header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]
    screen = pd.merge(blib, pd.read_pickle(
        os.path.join(A, "251203_ksi_reduced_df_tp.pkl")), on="kinase")
    screen = screen[~screen["kinase"].isin(EXCLUDE_KINASES)]
    print(f"\n  screened pairs retained: {len(screen)}  "
          f"kinase constructs: {screen['kinase'].nunique()}  "
          f"substrates: {screen['substrate'].nunique()}")

    bench = pd.merge(screen, tdp,
                     left_on=["kinase_acc", "substrate"],
                     right_on=["kinase", "substrate_seq_15aa"])
    bench["set"] = "train_tn"
    bench.loc[bench["y_true"] == 1, "set"] = "train_pos"

    def cm(df):
        s = df["true_positive"].astype(bool)
        neg = df["set"].eq("train_tn")
        return dict(TP=int((s & ~neg).sum()), FP=int((s & neg).sum()),
                    FN=int((~s & ~neg).sum()), TN=int((~s & neg).sum()))

    all_, st_, y_ = (cm(bench),
                     cm(bench[bench["kinase_type"] == "ST"]),
                     cm(bench[bench["kinase_type"] != "ST"]))
    for lab, c in (("combined", all_), ("ST", st_), ("Y", y_)):
        print(f"    {lab:9s} TP={c['TP']} FP={c['FP']} FN={c['FN']} "
              f"TN={c['TN']} total={sum(c.values())}")

    claim("S10_benchmark_total", 403, sum(all_.values()))
    claim("S10_benchmark_positives", 329, all_["TP"] + all_["FN"])
    claim("S10_benchmark_negatives", 74, all_["FP"] + all_["TN"])
    claim("S10_benchmark_pct_positive", "81.6%",
          f"{100 * (all_['TP'] + all_['FN']) / sum(all_.values()):.1f}%")
    claim("S10_confusion_combined", "134/11/195/63",
          f"{all_['TP']}/{all_['FP']}/{all_['FN']}/{all_['TN']}")
    claim("S10_confusion_ST", "57/6/77/41",
          f"{st_['TP']}/{st_['FP']}/{st_['FN']}/{st_['TN']}")
    claim("S10_confusion_Y", "77/5/118/22",
          f"{y_['TP']}/{y_['FP']}/{y_['FN']}/{y_['TN']}")

    n_uniq = len(bench.drop_duplicates(["kinase_acc", "substrate"]))
    claim("S10_benchmark_unique_pairs", "403 annotated pairs", n_uniq, "MISMATCH",
          f"403 is a ROW count; the merge duplicates pairs that carry more than "
          f"one source_all string. There are only {n_uniq} distinct "
          f"(kinase, 15-mer) pairs.")

    # provenance of the benchmark vs the designed spike-ins
    claim("Benchmark_is_designed_spikeins",
          "'~300 documented pairs ... used as spike-in controls'",
          "benchmark = inner join of screen with KINBERT's annotation corpus",
          "MISMATCH",
          "The designed spike-ins are loaded but unused (cells 14/17). The "
          "benchmark is an overlap with KINBERT's training/DB-test corpus, "
          "exactly as the author states.")

    claim("S13_negatives_are_documented_nonsubstrates",
          "'165 documented non-substrates as negative controls'",
          "designed spike-in negatives = KINBERT TRAIN-split true negatives "
          "(model-predicted + label 0); Table S10 negatives = randomly sampled "
          "1n1p decoys", "MISMATCH",
          "Neither negative set is a documented (experimentally shown) "
          "non-substrate. Design: set=='train_tn' selected in "
          "230310_select_substrate_oligo.ipynb cell 3 as rows where both "
          "KINBERT v1 and v2 had y_true==0 and v1 predicted 0 in the DB TRAIN "
          "split. Benchmark: source=NaN negatives of kinbertv1_DB_Test.")

    return screen, bench


# =========================================================================== #
# PART 4 - overlap with KINBERT's training corpus (Q3, Q4)
# =========================================================================== #
def part4_overlap(screen, bench, CORPUS_OLD, SPLITS, OUT):
    hdr("PART 4 - OVERLAP WITH KINBERT'S TRAINING CORPUS  (Q3, Q4)")
    print("""  Matching key.  Two independent keys are used, because the two
  generations of corpus carry different identifiers:

    (a) splits_v2 / splits_public / splits_phosphopca_finetuning
        (data/data/splits_*/{train,val,test,test2}.csv, the files named in
        configs/configs/exp_tape_bertbase_sep_v2_*.yml and
        finetune_tape_v2_phosphopca_exp6.yml -> the model the manuscript calls
        KINBERT) hold only (kinase_seq, substrate_seq, label).  substrate_seq
        IS the 15-mer.  Key = EXACT (kinase catalytic-domain sequence, 15-mer).

    (b) training_data/interations_e2e_1n1p_* hold UniProt accessions plus
        substrate_seq + p_ind.  Key = (kinase UniProt accession, 15-mer window
        recomputed with the notebooks' own slicing).

  The screened side uses screening_analysis/blib_kinases_with_acc.csv, which
  supplies both the construct's catalytic-domain sequence and its accession.
  Caveat: accession-level matching merges engineered kinase variants that the
  screen treats separately (e.g. JAK2_HUMAN_1/_2); domain-sequence matching
  keeps them apart but fails for a construct whose domain boundaries differ
  from the corpus's by even one residue (4 of 20 constructs, see below).
""")
    screen_pairs_acc = set(zip(screen["kinase_acc"], screen["substrate"]))
    screen_pairs_dom = set(zip(screen["kinase_seq"], screen["substrate"]))
    bench_pairs_acc = set(zip(bench["kinase_acc"], bench["substrate"]))
    bench_pairs_dom = set(zip(bench["kinase_seq"], bench["substrate"]))
    screen_subs = set(screen["substrate"])
    screen_doms = set(screen["kinase_seq"])
    print(f"  screened unique pairs: {len(screen_pairs_acc)} (by accession), "
          f"{len(screen_pairs_dom)} (by domain sequence)")
    print(f"  benchmark unique pairs: {len(bench_pairs_acc)}")

    recs = []
    out_rows = []

    # ---- (a) the splits the published model actually consumed -------------- #
    print("\n  --- (a) KINBERT splits (kinase domain sequence + 15-mer key) ---")
    dom_sets = {}
    for d in ("splits_v2", "splits_public", "splits_phosphopca_finetuning"):
        for sp in ("train", "val", "test", "test2"):
            f = os.path.join(SPLITS, d, sp + ".csv")
            if not os.path.exists(f):
                continue
            t = pd.read_csv(f)
            pairs = set(zip(t["kinase_seq"], t["substrate_seq"]))
            pos = set(zip(t.loc[t["label"] == 1, "kinase_seq"],
                          t.loc[t["label"] == 1, "substrate_seq"]))
            dom_sets[(d, sp)] = pairs
            so = len(pairs & screen_pairs_dom)
            bo = len(pairs & bench_pairs_dom)
            print(f"    {d}/{sp:6s} rows={len(t):7d} pairs={len(pairs):7d} "
                  f"| kinase domains shared with screen="
                  f"{len(set(t['kinase_seq']) & screen_doms):2d}/{len(screen_doms)}"
                  f" 15-mers shared={len(set(t['substrate_seq']) & screen_subs):5d}"
                  f" | screened pairs present={so:5d} (label=1: "
                  f"{len(pos & screen_pairs_dom)})"
                  f" | benchmark pairs present={bo:4d}")
            recs.append(dict(corpus=d, split=sp, key="kinase_domain_seq+15mer",
                             corpus_pairs=len(pairs), screen_overlap=so,
                             screen_overlap_label1=len(pos & screen_pairs_dom),
                             benchmark_overlap=bo))
            for k, s in sorted(pairs & screen_pairs_dom):
                out_rows.append(dict(corpus=d, split=sp,
                                     key="kinase_domain_seq+15mer",
                                     kinase_id=k[:30] + "...", substrate_15mer=s,
                                     in_table_s10_benchmark=(k, s) in bench_pairs_dom,
                                     corpus_label=1 if (k, s) in pos else 0))

    v2 = {sp: dom_sets[("splits_v2", sp)]
          for sp in ("train", "val", "test", "test2")
          if ("splits_v2", sp) in dom_sets}
    if v2:
        print("\n    splits_v2 internal disjointness at (domain,15-mer) level:")
        ks = list(v2)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                print(f"      {ks[i]} & {ks[j]}: {len(v2[ks[i]] & v2[ks[j]])}")
        u = set().union(*v2.values())
        claim("Q3_screened_pairs_in_KINBERT_corpus", "(not stated)",
              f"{len(u & screen_pairs_dom)} of {len(screen_pairs_dom)} "
              f"({100 * len(u & screen_pairs_dom) / len(screen_pairs_dom):.1f}%)",
              "UNRESOLVED",
              "splits_v2 union (train+val+test+test2); not a manuscript claim, "
              "this is the leakage-relevant number the manuscript omits")
        claim("Q4_benchmark_pairs_in_KINBERT_corpus", "(not stated)",
              f"{len(u & bench_pairs_dom)} of {len(bench_pairs_dom)} "
              f"({100 * len(u & bench_pairs_dom) / len(bench_pairs_dom):.1f}%)",
              "UNRESOLVED", "splits_v2 union")
        for sp in ("train", "val", "test", "test2"):
            if sp in v2:
                claim(f"Q4_benchmark_pairs_in_splits_v2_{sp}", "(not stated)",
                      len(v2[sp] & bench_pairs_dom), "UNRESOLVED")
        held = (v2.get("test", set()) | v2.get("test2", set()))
        seen = (v2.get("train", set()) | v2.get("val", set()))
        print(f"\n    benchmark pairs in a HELD-OUT split only (test/test2, not "
              f"train/val): {len((bench_pairs_dom & held) - seen)}")
        print(f"    benchmark pairs in NO splits_v2 split: "
              f"{len(bench_pairs_dom - u)}")

    ft = {sp: dom_sets[("splits_phosphopca_finetuning", sp)]
          for sp in ("train", "val", "test")
          if ("splits_phosphopca_finetuning", sp) in dom_sets}
    if ft:
        ftu = set().union(*ft.values())
        claim("Q4_benchmark_pairs_in_finetuning_TRAIN", "(not stated)",
              f"{len(ft['train'] & bench_pairs_dom)} of {len(bench_pairs_dom)} "
              f"({100 * len(ft['train'] & bench_pairs_dom) / len(bench_pairs_dom):.1f}%)",
              "UNRESOLVED",
              "splits_phosphopca_finetuning/train.csv - the phospho-PCA "
              "fine-tuning training set is screen-derived, so benchmark pairs "
              "sit in it directly")
        claim("Q4_benchmark_pairs_in_finetuning_any", "(not stated)",
              len(ftu & bench_pairs_dom), "UNRESOLVED")
        if v2:
            both = u | ft["train"]
            claim("Q4_benchmark_pairs_seen_in_any_training_split",
                  "(not stated)",
                  f"{len(bench_pairs_dom & both)} of {len(bench_pairs_dom)} "
                  f"({100 * len(bench_pairs_dom & both) / len(bench_pairs_dom):.1f}%)",
                  "UNRESOLVED",
                  "splits_v2 train/val/test/test2 UNION fine-tuning train")

    # ---- (b) the accession-keyed corpora --------------------------------- #
    print("\n  --- (b) interations_e2e_1n1p corpora "
          "(kinase accession + 15-mer key) ---")
    old = {
        "240213_db_lysate/train":
            "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_240213/train.pkl",
        "240213_db_lysate/val":
            "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_240213/val.pkl",
        "240213_db_lysate/test":
            "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_240213/test.pkl",
        "230621_db_lysate/train":
            "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_230621_train.pkl",
        "230621_db_lysate/test":
            "interations_e2e_1n1p_db_classed_random_lysate_random_withacc_230621_test.pkl",
        "230621_dbonly/train":
            "interations_e2e_1n1p_classed_random_withacc_230621_train.pkl",
        "230621_dbonly/test":
            "interations_e2e_1n1p_classed_random_withacc_230621_test.pkl",
    }
    acc_sets = {}
    for name, rel in old.items():
        f = os.path.join(CORPUS_OLD, rel)
        if not os.path.exists(f):
            print(f"    MISSING {name}")
            continue
        df = add_window_15(pd.read_pickle(f))
        pairs = set(zip(df["kinase"], df["substrate_seq_15aa"]))
        pos = set(zip(df.loc[df["label"] == 1, "kinase"],
                      df.loc[df["label"] == 1, "substrate_seq_15aa"]))
        acc_sets[name] = pairs
        so, bo = len(pairs & screen_pairs_acc), len(pairs & bench_pairs_acc)
        print(f"    {name:24s} rows={len(df):7d} pairs={len(pairs):7d} | "
              f"screened pairs present={so:5d} (label=1: "
              f"{len(pos & screen_pairs_acc)}) | benchmark pairs present={bo:4d}")
        recs.append(dict(corpus=name.split("/")[0], split=name.split("/")[1],
                         key="kinase_accession+15mer", corpus_pairs=len(pairs),
                         screen_overlap=so,
                         screen_overlap_label1=len(pos & screen_pairs_acc),
                         benchmark_overlap=bo))
        for k, s in sorted(pairs & screen_pairs_acc):
            out_rows.append(dict(corpus=name.split("/")[0],
                                 split=name.split("/")[1],
                                 key="kinase_accession+15mer",
                                 kinase_id=k, substrate_15mer=s,
                                 in_table_s10_benchmark=(k, s) in bench_pairs_acc,
                                 corpus_label=1 if (k, s) in pos else 0))

    for grp in (["240213_db_lysate/train", "240213_db_lysate/val",
                 "240213_db_lysate/test"],
                ["230621_db_lysate/train", "230621_db_lysate/test"]):
        have = [g for g in grp if g in acc_sets]
        if len(have) < 2:
            continue
        print(f"\n    {have[0].split('/')[0]} cross-split pair overlap "
              f"(NOT disjoint at peptide level):")
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                print(f"      {have[i]} & {have[j]}: "
                      f"{len(acc_sets[have[i]] & acc_sets[have[j]])}")

    print("""
  WHY THE ACCESSION-KEYED SPLITS ARE NOT PEPTIDE-DISJOINT: the corpora were
  split on (substrate accession, site), and 53% of the library's 15-mers occur
  in more than one protein (Part 2). The same 15-mer therefore appears on both
  sides of those splits. splits_v2 IS disjoint on its own (domain, 15-mer) key.
""")

    pd.DataFrame(out_rows).to_csv(
        os.path.join(OUT, "audit_09_training_overlap_pairs.csv"), index=False)
    print(f"  wrote {len(out_rows)} overlap rows -> "
          f"{os.path.join(OUT, 'audit_09_training_overlap_pairs.csv')}")
    return pd.DataFrame(recs)


# =========================================================================== #
# PART 5 - screen recovery
# =========================================================================== #
def part5_recovery(A, P):
    hdr("PART 5 - SCREEN RECOVERY  (Q5)")
    uniq = pd.concat([pd.read_csv(os.path.join(P, "st_sub_all.csv")),
                      pd.read_csv(os.path.join(P, "y_sub_all.csv"))]
                     ).drop_duplicates("substrate_seq_15aa")
    red = pd.read_pickle(os.path.join(A, "251203_ksi_reduced_df_tp.pkl"))
    n_rec = red["substrate"].nunique()
    claim("S13_substrates_recovered", 3918 - 32, n_rec, None,
          "implied by '3,918 distinct' minus '32 not recovered'")
    claim("S13_not_recovered", 32, len(uniq) - n_rec)
    print(f"      build_table_s13.py reports the same split at ROW level over "
          f"3,928 rows: 3,896 recovered / 32 not.")


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    for k, v in DEF.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("AUDIT ITEM 09 - substrate library composition & spike-in provenance")
    print("Hypothesis under test (from the author): the effective spike-in /")
    print("benchmark set is the OVERLAP between KINBERT's training corpus and")
    print("the screened library, not the designed spike-in pool.")

    part1_design(a.planning)
    part2_mapping(a.planning, a.analysis)
    screen, bench = part3_benchmark(a.kinase_ml, a.analysis, a.screening)
    ov = part4_overlap(screen, bench, a.corpus_old, a.splits, a.out)
    part5_recovery(a.analysis, a.planning)

    out_csv = os.path.join(a.out, "audit_09_spikein_provenance.csv")
    pd.DataFrame(CLAIMS).to_csv(out_csv, index=False)
    ov.to_csv(os.path.join(a.out, "audit_09_overlap_summary.csv"), index=False)

    hdr("SUMMARY")
    v = pd.DataFrame(CLAIMS)["verdict"].value_counts().to_dict()
    print("  " + "  ".join(f"{k}={n}" for k, n in sorted(v.items())))
    print(f"  claims  -> {out_csv}")
    print(f"  overlap -> {os.path.join(a.out, 'audit_09_overlap_summary.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
