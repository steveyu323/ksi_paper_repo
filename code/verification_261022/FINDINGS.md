# FINDINGS — numeric audit of the KSI manuscript

Audit started 2026-10-22 against `manuscript/261014/KSI_Paper.zip`.
Each item: what the manuscript says, what the code says, and **which value to use**.

Status: items 03, 04, 10 complete. Items 01, 02, 05, 06, 07, 08, 09 in progress.

---

## Item 03 — Kinase panel size: **23, not 24** ✅ RESOLVED

**Use 23.** The author's recollection is confirmed by the screening data.

| source | value |
|---|---|
| main.tex L138 (SUMMARY) | "24 kinases" ❌ |
| main.tex L149 (INTRODUCTION) | "24 representative kinases" ❌ |
| Table S12 title + 15+8 arithmetic | 23 constructs ✅ |
| `dat_all_blib_2batch.csv` (265,601 rows) | **23** ✅ |
| `8_2batch_blib_pipeline_v3.ipynb` cells 19/20/25 | prints `ST n_kinase 15 / Y n_kinase 8` ✅ |

Reproduced independently by the raw merged CSV, two downstream pickles
(`251203_ksi_reduced_df_tp.pkl`, `251130_ksi_inference_scores.pkl`) and the notebook's own
printed output.

**23 constructs = 21 distinct kinase identities** after merging PLK1/PLK1-T210D (P53350)
and KS6B1-wt/KS6B1-T252D (P23443). Keep the word **"constructs"** wherever 23 appears;
use 21 only if a distinct-gene count is needed. Table S12's caption already handles this
correctly. (Its "three constructs are phosphomimetic or otherwise engineered" = IKKβ
S177E/S181E + the two T→D variants; internally consistent.)

### DYRK4 — designed, not screened, but *not* literally absent

- **Design side:** `screening_analysis/blib_kinases_with_acc.csv` has **24** constructs,
  row 15 = `ST,DYRK4_HUMAN,Q9NR20` (290 aa); `wetlab_planning/230504_primers.csv` has 24 rows.
- **Screening side:** zero occurrences in `dat_all_blib_2batch.csv` and absent from every
  downstream product (reduced dfs, inference scores, fine-tuning data, rescore results).
- **Nuance:** in `lookup_tables/st.lookup.corrected.pkl` an error-tolerant probe finds
  **43 DYRK4 reads vs 1,837 for TITIN** (the weakest real ST kinase) — 0.036% of the
  library. **0 reads match the designed construct exactly:** all 13 DYRK4 variants start
  `WKG(S|F)FGQVAKCL…` while the annotated construct starts `GKGSFGQVAKCL…`. The
  kinase-domain boundary was mis-annotated by ~1–8 residues, so the exact-sequence join
  never succeeded and DYRK4 contributes no KSIs.

**Wording to use:** "designed but recovered at <0.1% of library representation and
excluded" is accurate. "Designed but never constructed" would **not** be.

**Dead code worth knowing about:** `1_parse_lookup_tables.ipynb` cell 5 carries
`# 251210: fixing the missing DYRK4 issue…` and patches `kinase_aa` for DYRK4, writing
`lookup_tables/blib_kinases_with_acc.csv`. **That rescue did not work** — the patched
sequence still matches 0 reads, and `dat_all_blib_2batch.pkl/.csv` (2025-11-26) predates it
and was never regenerated. Note this so nobody re-runs notebooks 1→3 expecting DYRK4.

---

## Item 04 — Titin arm: **serine/threonine. Methods is wrong.** ✅ RESOLVED

**Use Ser/Thr.** Table S12 is correct; `main.tex` L1012 (Methods) is the only place Titin
sits in the tyrosine arm.

Over-determined by four independent signals:
1. `kinase_type == 'ST'` for all **3,032** TITIN rows; no row carries `Y`.
2. **PAABD readers observed with TITIN: `1433`, `fha`, `ww`** (1,206 / 856 / 970 rows) and
   **zero `sh2` rows.** `sh2` is used by all 8 Tyr-arm kinases and nobody else; `ww`/`fha`/
   `1433` by all 15 ST-arm kinases and nobody else. The reader set alone settles it.
3. TITIN appears only in `st.lookup.corrected.pkl` (1,805 exact-match PacBio reads); the Y
   lookup has a single stray cross-contaminating read.
4. Design reference `blib_kinases_with_acc.csv` line 8 = `ST,TITIN_HUMAN,Q8WZ42`.

This matters exactly as suspected: the Tyr placement would imply Titin was read by Src-SH2
against the 1,950-peptide Tyr pool under the Tyr false-positive rules — none of which
happened. The Results' description of Titin as a pseudokinase yielding the fewest hits is
consistent with its ST position (lowest ST read count).

**Table S12 agrees with the screening data for all 23 kinases — no other arm mismatch.**

---

## Item 01 — KINBERT corpus: **the review notes had this backwards** 🔴 DECISION NEEDED

**The code says 2,021 kinases, not 2,304.** `review_notes/remaining_issues.md` (L162, L281)
concluded "the text is correct, Figure 4 panel A is wrong — fix the figure, not the text."
The kinase half of that is **wrong**, and so was my earlier item #1.

### 2,304 is a double-counting bug

`eval_0_datasource_stats.ipynb` cell 20:
```python
all_kinase_train = lysate_ksis["kinase"].drop_duplicates().tolist() \
                 + db_ksis["kinase"].drop_duplicates().tolist()
n_kinase = len(all_kinase_train)      # -> 2304
```
`len()` of two **concatenated lists** is a sum, not a union: 331 (lysate) + 1,973 (db) =
2,304, double-counting a **304-kinase overlap**. True union = 2,000; the final corpus holds
**2,021** distinct accessions. Cell 18 (the v1 path) dedupes correctly, so the bug is
specific to the v2 cell.

Corroborating: the Results' "2.9× the 800 in Phosformer, 6.7× the 343 in DeepKinZero"
reproduce *exactly* from 2,304 (2.88, 6.72) — confirming 2,304 is the value in use, and
that correcting it changes those multipliers too.

**The thesis `tab:ch2:1` also says 2,304, so the thesis inherited the same bug** — which is
why the earlier cross-check against the thesis appeared to vindicate the text.

### 147,488 vs 162,447 — both real, different units

Both come from the *same* file (`data/data/pairs_aug_v2.csv`, label==1), under different
aggregation keys. Not an earlier build.

| key | count |
|---|---|
| distinct (substrate_seq_15aa, kinase_domain_seq) — **sequence pairs** | **147,488** |
| distinct (substrate, kinase, p_aa, p_ind) — **site-level interactions** | **162,447** |
| distinct kinase accessions | **2,021** |
| distinct `kinase_domain_seq` | 1,604 |
| organisms | **92** |

**Use 147,488 as "training pairs."** Decisive: `data_processing/make_splits_v2_no_leak.py`
and `prepare_splits.py` key on `KEY_COLS_DEFAULT = ["kinase_seq", "substrate_seq"]`, so
147,488 is what the model actually sees as distinct examples; it never sees 162,447 items.
If the 162,447 framing is wanted in the figure, label it "annotated kinase–substrate–**site**
interactions".

**92 organisms survives** restriction to the corpus, but is strictly a **lower bound**: only
1,971 of the 2,021 corpus kinases appear in the organism annotation table, so 50 are
unannotated.

### Recommended authoritative set to propagate

> **2,021 kinases / 92 organisms / 147,488 unique kinase-domain–substrate-peptide sequence
> pairs (162,447 annotated kinase–substrate–site interactions)**

This changes **Results, Table S1, Fig S1a and Table S3**. Figure 2's kinase and interaction
numbers then need **no change** — only the organism swap and rounding below.

⚠️ **This is a substantive change to a headline number and needs your sign-off**, since it
lowers the corpus-scale claim and contradicts both the thesis and the prior review notes.

### Downstream edits if 2,021 is adopted

| location | from | to |
|---|---|---|
| Results | "2.9× the 800 in Phosformer" | **2.5×** |
| Results | "6.7× the 343 in DeepKinZero" | **5.9×** |

### Two new errors found in passing

1. **Fig 2 artwork: the Phospho.ELM / PhosphoSitePlus organism values are swapped.** Truth
   is **Phospho.ELM = 11, PhosphoSitePlus = 10**. `upsetplot` sorts rows by set size
   independently per panel, so the organism panel's row order differs from the kinase
   panel's, and the organism column `1, 10, 11, 92` was transcribed onto the kinase panel's
   label order. PDF text geometry confirms it (y=53.2 carries "Phospho.ELM / 336 / 10";
   y=61.9 carries "PhosphoSitePlus / 818 / 11"). Per-source **kinase** counts
   (331 / 336 / 818 / 2,277) are all correct.
2. **Results L199 credits DB+lysate numbers to the database-only set.** The sentence
   introduces only the four curated databases, states "The corpus reaches 147,488
   interactions across 2,304 kinases and 92 organisms", and *then* says "On top of this
   **database-only** set, we added a Titania LC-MS/MS dataset…". Both quoted numbers are
   DB+lysate. The true database-only corpus is **26,267 sequence pairs (29,894 site pairs)
   across 1,995 kinases**. As written it credits the lysate contribution to the databases
   and then adds it a second time. **Not in the review notes.**

Also: the artwork renders Phosformer's 24,534 as **"24.9K"**; correct rounding is **24.5K**.

### Unresolvable on this tree

The comparator figures (Phosformer 800 / 13 / 24,534; DeepKinZero 343 / 1 / 13,426) are
prose transcriptions in `eval_0_datasource_stats.ipynb` **cell 14 (markdown)**, quoted from
the source publications. No code here recomputes them — they must be checked against those
papers directly. 6 claims flagged UNRESOLVED.

---

## Item 02 — Table S3: **the printed totals are correct** ✅ RESOLVED

All 12 cells reproduce **exactly**. Do not change the bin counts.

| | 1–5 | 5–20 | 20–50 | 50–100 | 100–200 | >200 | total |
|---|---|---|---|---|---|---|---|
| DB+lysate | 1129 | 290 | 79 | 63 | 66 | 189 | **1816** ✅ |
| DB only | 1211 | 375 | 113 | 44 | 24 | 23 | **1790** ✅ |

None of the suspected causes apply. The bins are **exhaustive and non-overlapping** —
`pd.cut(bins=[0,5,20,50,100,200,inf], right=True, include_lowest=True)` yields (0,5],
(5,20], … so the label "5–20" actually means **6–20**; the labels overlap, the binning does
not. **0 kinases fall in a NaN bin**, and no kinase has 0 substrates (a `groupby` only emits
keys with ≥1 row).

**The 488-kinase gap decomposes exactly:**
- **283** = 2,304 − 2,021 — the double-counting bug in item 01;
- **205** = 2,021 − 1,816 — accessions dropped by sequence dedup. Cells 22/23 bin
  `drop_duplicates(['substrate_seq_15aa','kinase_domain_seq'])` and *then* `groupby("kinase")`.
  Because 2,021 accessions share only **1,604** distinct domain sequences, a paralogue whose
  domain sequence duplicates an already-seen one loses every row and vanishes from the table.

**Fix:** keep 1,816 / 1,790 and add a one-line note that sequence deduplication is why the
rows do not total the corpus kinase count. Do **not** inflate the bins.

If you would rather S3 totalled the full kinase count, the site-level histogram (no sequence
dedup) is available: DB+lysate 1274 / 325 / 86 / 63 / 67 / 206 = **2,021**; DB-only
1351 / 412 / 123 / 54 / 30 / 25 = **1,995**.

Separately, restate the bin **labels** as 1–5, 6–20, 21–50, … to match the actual edges.

---

## Item 07 — PER2 site: **Ser662. The figure is the typo.** ✅ RESOLVED

**Use Ser662.** Every pipeline table carries `p_ind = 662`:
`260130_ptmvar_nr_finalized.csv:18` (CK1, PER2, 662, S, rs121908635, Pathogenic, Advanced
sleep phase syndrome 1, Tier 1), plus the curated, consensus-tier and psite-overlap tables.
Kinases are CK1δ and CK1ε (collapsed to `CK1` in the finalized table).

The plotting code can only emit `p662`: `generate_all_figures_final.py:640` draws
`f'p{p_ind_str}'` straight from `p_ind`, and `474` assigns `'Circadian': ['PER2']` so PER2
is the only substrate in that cluster. **`562` appears nowhere** in any `.py` or `.ipynb`
under `evaluation/`. Not isoform numbering — the tables use canonical O15055 (Toh 2001
S662G) numbering. Conclusion: off-by-100 transcription error in hand-redrawn artwork.

### 🔴 Three more orphan artwork labels — same error class, not previously flagged

Every panel-B `pNNN` label was checked against `p_ind`. Four orphans, each one typo from a
real row:

| figure label | cluster | should be |
|---|---|---|
| `p562` | Circadian (CK1) | **p662** — PER2 Ser662 |
| `p11` | Metabolism (PIM1) | **p111** — VHL Ser111 (dropped digit) |
| `p109` | p53 pathway (MET) | **p106** — TP53 Ser106 |
| `p152` | NF-κB/Immune (IKK) | **p151** — TBK1 Ser151 |

### 🔴 Substantive text fix: β-catenin GSK3 sites

The Results sentence "CK1-family priming of β-catenin at Ser45 precedes GSK3A/GSK3B
phosphorylation at **Ser33, Ser37, and Thr41**" is correct canonical biology but is **not
what this pipeline's tier-1 set contains**:

| site | tier-1 kinases in the output |
|---|---|
| Ser45 | CK1α/α2/δ/ε/γ1/γ2/γ3 ✅ |
| Ser33 | GSK3A + GSK3B ✅ |
| **Ser37** | **JNK1/JNK2** — no GSK3 row ❌ |
| **Thr41** | **no tier-1 at all** (CK1δ/ε/γ tier 3, p38α tier 2) ❌ |

Fix: soften to "GSK3A/GSK3B phosphorylation at Ser33 within the Ser33/Ser37/Thr41 degron",
or state plainly that the Ser37/Thr41 part is literature rather than model output.

### Other Results sites — mostly clean, two to soften

PTPS Thr58 (AMPKα1) ✅ · TP53 Ser215 (AurA+AurB) ✅ · cyclin F Ser621 (CK2α1) ✅ ·
KIT Ser821 (PKCβ) ✅ · CBL Tyr371 tier 2 ✅.
Two minor over-distributions: **TP53 Ser269 is AurB only** (the "Aurora-A and Aurora-B …
Ser215 and Ser269" phrasing over-distributes), and **NCC Thr60 SPAK/STLK3 is tier 2**, not
tier 1 (OSR1 is tier 1).

### Fig 7 caption counts: numerically right, wrongly named

`260130_ptmvar_nr_finalized.csv` has exactly **40 onsite / 25 flanked rows** ✅ (panel-B
insets also check out: onsite Tier1/Tier2 = 11/29, flanked = 12/13). But these are
**kinase–substrate–site rows, not distinct variants** — only **20** distinct dbSNP IDs
onsite and **17** flanked. "40 on-phosphosite mutations" overstates the mutation count ~2×.
Say "kinase–site pairs".

---

## Item 08 — M1 Ser53: **ERK7 is correct. The caption and panel B are wrong.** ✅ RESOLVED

**Use ERK7 (MAPK15), logit 4.6875, rank 1.** PKCε does not appear anywhere in M1 Ser53's
top 10. Confirmed in three independent artifacts: `250118_virus_against_kinome_hits_with_drug.csv`,
`260119_demo_kinase_virus_drug_pairs.csv` rows 57–58, and the exported panel-C grids
`eval_virus_dms_{kinase,drug}_psite_demo.csv`.

The confusion is a **swap with the H3N2 HA nomination**:

| | site | kinase | logit | drugs |
|---|---|---|---|---|
| influenza M1 | **Ser53** | **ERK7** | 4.6875 | aspirin, minocycline, fostamatinib |
| influenza H3N2 HA | **Thr329** | **PKCε** | 5.0625 | Bryostatin 1, KAI-1455, + 3 |

`table_05_virus_demo.py` adds the influenza rows back as `top_df.iloc[[1, 6, 9]]`, which is
exactly M1 S53 (ERK7) and H3N2 T329 (PKCε) — two separate nominations.

Verdict on the four accounts: **Results ✅ correct · caption ❌ · panel B ❌ · panel C ✅.**

**Panel B is self-contradictory** (read from PDF text spans): its heading says
`Influenza H3N2 · HA`, kinase `PKCε`, drugs `Bryostatin 1` / `KAI-1455` — all the H3N2 HA
T329 nomination — while only the protein/site labels say `M1` / `S53`. **Fix: relabel that
node `HA` / `T329`.** If M1 Ser53 belongs in panel B it needs its own ERK7 entry.

### 🔴 The GSK-3β / N Ser202 claim fails the pipeline's own filter

The Results say "KINBERT identified a two-kinase pattern … SRPK1/2 to Ser206 and GSK-3β to
the adjacent Ser202". For **N Ser202** the ranking is NEK10 3.4531, ICK 3.1406,
PCTAIRE1 3.0625, MOK 3.0, PCTAIRE2 2.7812, **GSK3B 2.7031 (rank 6)**, GSK3A 2.5469 — and
`passing_kinases` is **`['ICK', 'PCTAIRE1']` only**. GSK-3β is a rank-6 call that did **not**
pass the selectivity filter; it reaches the highlight list via the literature-evidence join,
not the pipeline. This is the weakest claim in that paragraph. **Soften to "GSK-3β also
scores at the adjacent Ser202 (rank 6)" or drop it.**

### Other Figure 8 nominations — three partials

| claim | code says | fix |
|---|---|---|
| CK1ε, Zika E Ser72 | CK1α 3.8438 is **rank 1**; CK1ε 3.2656 is rank 4. CK1ε was highlighted because it is the one carrying drugs | say "a CK1-family kinase (CK1α/δ/ε)" |
| Zika E Ser72 drugs | pipeline gives CK1ε → **Seliciclib; Umbralisib**. **Purvalanol is not mapped here** (it goes to SRPK2 at Zika E T76/Y90 and N S206) | Purvalanol is literature context — fine, but note Umbralisib is the pipeline's second drug |
| SRPK1/2, N Ser206 | SRPK2 rank 1 (6.5625) ✅; **SRPK1 rank 2 but filtered out**. PFTAIRE2 (5.0938) also passes, unmentioned | "SRPK1/2" defensible by rank; only SRPK2 survived the filter |
| ALK/FMS/KDR/FLT4, ORF9b Tyr42 | all four pass, but **9 kinases pass** and FGFR4 (8.0), LTK (8.0), EphB4/FGFR3/FGFR1 (7.9375) score equal or higher and are unnamed | "strong RTK consensus" is right; the named four are a **drug-driven subset**, not the top four |
| Lorlatinib via ALK, Vatalanib via FLT4 | both in `passing_drugs` ✅ | — |

### 🔴 Drug mapping is diluting the shortlist — confirmed mechanism

Aspirin and minocycline are genuinely in the matrix for M1 Ser53. ERK7 = MAPK15 (Q8TD08);
`drug_kinase_uniprot.pkl` has exactly three rows against it:
- **Acetylsalicylic acid** — 11 annotated kinase targets (AMPKα1/α2, Erk1–5, Erk7, IKKβ, RSK2): a pan-MAPK/AMPK/IKK annotation, not an ERK7 ligand.
- **Minocycline** — 13 targets (Erk1–5, Erk7, JNK1–3, p38α/β/δ/γ): the entire MAPK family.
- **Fostamatinib** — **288** kinases; attached to 6 of M1 Ser53's 10 kinases and dominating most sites in the matrix.

So **the ERK7 nomination is real; its drug-repurposing candidates are not interpretable as
on-target.** The caption already hedges "not established to act on-target" — worth one
explicit sentence about annotation promiscuity.

### Caveats recorded

- `virus_pipeline.py` warns that re-deriving from the raw model pickles changes 28 of 96
  rows (unstable tie sort). All logits above come from the cached
  `250118_…pkl` / `_with_drug.csv`, which the scripts state is the exact object the figures
  were built from. **Tie ordering within equal logits is not reproducible** (e.g. HIPK2/HIPK4
  at 4.4375; the ORF9b RTKs at 7.9375) — but ERK7's 4.6875 and PKCε's 5.0625 are unique at
  rank 1, so neither headline call is affected.
- `drug_kinase_uniprot.pkl` is a cached merge and the upstream drug databases are not on
  this machine, so the aspirin→MAPK15 and minocycline→MAPK15 **edges are certain but their
  source database is not**.

---

## Item 05 — SPARK p-values: **two different tests, not two runs** ✅ RESOLVED (+1 new error)

Both sets of numbers are printed by **the same cell of the same notebook from the same
data**: `cellsam-internal/spark_overlay_patched.ipynb` cell 21 (`execution_count=269`), the
only cell in the entire tree containing `mannwhitneyu`.

| comparison | section 2: one-sided MW (**main.tex**) | section 4: two-sided + Bonferroni k=6 (**artwork**) |
|---|---|---|
| treated-all vs control | **1.708e-07** | p_raw 3.417e-07 → **p_adj 2.050e-06** |
| non-zero vs control | **4.239e-63** | p_raw 8.477e-63 → **p_adj 5.086e-62** |
| selected vs control | **2.088e-59** | p_raw 4.176e-59 → **p_adj 2.506e-58** |

Ratio p_adj / p_one-sided = 12.002, 11.998, 12.002 — exactly **2 × 6** (one- vs two-sided ×
Bonferroni k=6). Neither set is stale.

**The artwork's malformed value is 2.1 × 10⁻⁶, not 2.1 × 10⁻⁵.** The figure's text layer
draws `p=21×10-6` — a dropped decimal point in 2.1×10⁻⁶ (= 2.050e-6). The review notes'
guess of 2.1×10⁻⁵ was wrong.

**This is editorial, not numerical:** the Fig 6 caption already says the middle panel shows
"Bonferroni-corrected comparisons", so caption and artwork agree. Only the juxtaposition
with the Results' "one-sided Mann–Whitney U" reads as a contradiction. State which test each
number is.

### 🔴 NEW: the quoted n and the quoted p describe different pools

Cell 21's own banner prints **`IKK-SPK-novel excluded`**. Its n was recovered by inverting
`r = 1 − 2U/(n1·n2)` from the printed (U, r) pairs — two integer solutions, only one fits
inside the real pools:

| pool | n used by the p-values | n quoted in the caption | difference |
|---|---|---|---|
| control | **442** | 488 | − 46 (IKK-SPK-novel basal: 16+30) |
| treated-all | **413** | 540 | − 127 (IKK-SPK-novel activated: 61+16+50) |
| non-zero | **153** | 207 | − 54 |
| selected | **111** | 138 | − 27 (IKK-SPK-novel selected: 14+3+10) |

All three subtractions land **exactly** on the IKK-SPK-novel per-condition counts from cell
27 — so the identification is solid, not coincidence. The quoted **p-values, r, and medians
all come from a 7-reporter-minus-IKK-SPK-novel analysis**, while the quoted
n = 488/540/207/138 come from the all-reporters analysis.

**Action: either re-run the tests on the full pools, or state that the test excludes
IKK-SPK-novel and give n = 442/413/153/111.** Not previously flagged anywhere.

### Counts, r and medians

Verified against the notebook's stored outputs: control **488** ✅, treated-all **540** ✅,
selected **138** ✅, non-zero **207** ✅ (printed directly by cell 20; the 263 not-all-zero
cells narrow to 207 under the extra `peak_delta > 0` filter). `ACTIVATION_THRESH = 0.01` ✅,
`BASAL_PEAK_CAP = 0.05` ✅, "~98% of basal below 0.05" = **477/488 = 97.75%** ✅.
Median ΔSPARK non-zero **0.02252 → 0.023** ✅, control **0.000** ✅ (both from the excluded
pool; the all-reporters median is 0.01818).

⚠️ **The rank-biserial sign is inverted.** The code computes `r = 1 − 2U/(n1·n2)` with `U`
from `alternative='greater'`, which flips the conventional sign. Magnitude 0.81 is right,
but the printed **−0.81 actually indicates treated > control** — the opposite of what a
reader will infer. Report **+0.81** or state the convention.

### Ruled out, and one provenance caveat

The manual-curation CSVs (`ikk_novel_polyIC_manual.csv` 6 cells,
`tbk1_novel_diABZI_manual.csv` 4 cells) load into `df_manual` and are consumed **only** by
the trace-overlay cells, never entering `df_stats` — so they cannot move any p-value. (They
do explain the panel-B legend counts: TBK1-novel diABZI 34 = 30 automated + 4 manual.)

**Caveat:** cell 21 ran at `execution_count=269` while cells 19/20 ran later (280/305), and
cell 21 references variables no surviving cell defines. The pool-construction code for the
published p-values **has been overwritten**; re-running the notebook today would give the
IKK-novel-*included* numbers, not the published ones.

**Not attempted:** recomputation from raw per-cell traces. They live in
`spark_examples/spark_results.h5` and **h5py is not installed** on this machine; that H5 is
also dated 2026-03-20 while the figure session ran to 2026-04-08. Flagged rather than
guessed.

---

## Item 06 — Table S11: **the post-FT column is correct. My earlier diagnosis was wrong.** 🔴

**"Logit" means the raw positive-class logit z1 — not log-odds.** From
`biosensor_scoring/score_pairs_kinbert.py`, in `score()`:

```python
logits = out[0].float().cpu()
probs.extend(F.softmax(logits, dim=-1)[:, 1].numpy())
logits_out.extend(logits[:, 1].numpy())
```

Both columns come from the **same forward pass on the same tensor**. So
`z1 − log-odds = z0`, which has no reason to be zero — and the "~0.22 offset" **is z0**:

| row | z1 (printed) | log-odds | z0 |
|---|---|---|---|
| TBK-SPARK-novel post | 1.426901 | 1.187011 | **+0.2399** |
| IKK-SPARK-novel post | 1.420063 | 1.205464 | **+0.2146** |
| TBK-SPARK-novel pre | −1.320312 | −1.380 | +0.060 |
| IKK-SPARK-novel pre | 3.000000 | 2.987 | +0.013 |

The pre-FT column only *looked* self-consistent because that checkpoint's z0 is near zero
(and bf16 quantises the logit to ~1/128). **It is the same convention throughout.** This
corrects review-notes §B3, triage item 8, refine.ink #13 — and my own item D in the previous
message. Consistent with the standing caution that `logit` comments in this area are
unreliable and must be verified from the code path, which is what was done here.

### The rows that *are* wrong are the two `-1` reporters

| row | printed | actual source | verdict |
|---|---|---|---|
| TBK-SPARK-novel pre | 0.201 / −1.320 | `..._early_loss__ksi_full.csv` row 25400: 0.20055768 / −1.3203125 | ✅ exact, one run |
| TBK-SPARK-novel post | 0.766 / 1.427 | `..._phosphopca_finetuning.csv` row 779: 0.7662061 / 1.4269006 | ✅ exact |
| IKK-SPARK-novel pre | 0.952 / 3.000 | `..._early_loss__ksi_full.csv` row 24100: 0.9522171 / 3.0 | ✅ exact |
| IKK-SPARK-novel post | 0.769 / 1.420 | `..._phosphopca_finetuning.csv` row 3288: 0.7694954 / 1.4200631 | ✅ exact |
| **TBK-SPARK-1 "pre"** | 0.803 / 1.623 | `eval_2_finetune_phosphoPCA.ipynb` cell 4, var `tape_fintuned` | 🔴 **the POST-FT pair printed in the pre-FT column** |
| **IKKB-SPARK-1 "pre"** | 0.990 / 4.620 | `wetlab_planning/biosensor_design/ikka_tbk1_substrates.ipynb` cell 24 | 🔴 **2023 v1 model, and self-consistent only as (p, log-odds)** — a different logit definition |

So Table S11 as printed **mixes three scoring runs and two definitions of "logit"**.

Independent corroboration that prob and z1 are one run: the fp32 re-score in
`rescore_results.csv` reproduces **both** numbers of each saved post-FT row
(Δprob 2.5e-07 / Δlogit 1.1e-06 for TBK-novel).

### Corrected Table S11 — only the bottom two rows change

| Reporter | Kinase | Substrate (15 aa) | Phospho-PCA | pre-FT Prob | pre-FT Logit | post-FT Prob | post-FT Logit | PAABD |
|---|---|---|---|---|---|---|---|---|
| TBK-SPARK-novel | TBK1_HUMAN | QGPSQTSTPTDVTAI | 1.241 | 0.201 | −1.320 | 0.766 | 1.427 | FHA |
| IKK-SPARK-novel | IKKB_HUMAN | PHEEPSNSLMAIECR | 1.623 | 0.952 | 3.000 | 0.769 | 1.420 | WW |
| TBK-SPARK-1 | TBK1_HUMAN | DHPYASLTLPDDAAA | — | **0.952** | **3.016** | **0.803** | **1.623** | FHA |
| IKKB-SPARK-1 | IKKB_HUMAN | NPQERTLTLVDTGIG | — | **0.999** | **7.500** | **0.992** | **4.846** | FHA |

Three things to state with it:
1. **Add a footnote defining Logit as the raw positive-class logit z1** (so log-odds =
   z1 − z0). This single fix dissolves the entire objection.
2. The `-1` reporters' pre-FT numbers are **bf16 re-scores** (~3e-2 on prob, ~1/128 logit
   granularity — hence `7.500` and `3.016`); the post-FT numbers are fp32 and reproduce to
   ~4e-6. These four numbers exist **only as re-scores** — the pairs appear in no saved
   screen output. Consider printing fewer digits, or re-scoring in fp32.
3. **The caption's claim that the `-1` reporters were "not rescored after fine-tuning" is
   now false** — they have been. Either print the post-FT values above or reword.

### Phospho-PCA column ✅ matches exactly

`binding_max` in `..._phosphopca_finetuning.csv`: TBK1 × QGPSQTSTPTDVTAI (row 779)
**1.2412884 → 1.241** ✅; IKKB × PHEEPSNSLMAIECR (row 3288) **1.6234228 → 1.623** ✅. Both
`label=1`, `true_positive=True`. Neither `-1` pair appears in that file, consistent with the
`—` entries. The column is **kinase-pair-specific**, not per-substrate (the same TBK
substrate scores 0.299 against PDK1, 0.226 against TAK1).

---

## Item 09 — Spike-in provenance: **your recollection is confirmed verbatim in code** 🔴 MOST CONSEQUENTIAL

**The designed spike-ins were never used.** `8_2batch_blib_pipeline_v2.ipynb` (the notebook
that writes `confusion_matrix_summary.csv`) loads `{st,y}_spike_ins.csv` in **cell 14** and
only **prints** them in cell 17. They never enter the confusion matrix.

The benchmark is built instead in **cell 24**:
```python
curr_df_all = pd.merge(dat_in_screen, train_dat_pos,
                       left_on=['kinase_acc','substrate'],
                       right_on=['kinase','substrate_seq_15aa'])
```
— **an inner join of the screen against KINBERT's annotation corpus.** Positives come from
`interaction_classed_221017_wsource.pkl` (KINBERT's positive training corpus); negatives from
`kinbertv1_DB_Test.pkl`. The result is then relabelled `train_pos`/`train_tn`, **reusing the
spike-in vocabulary** — almost certainly why the draft text conflated the two.

Confusion matrix reproduces cell-for-cell: combined **134/11/195/63 = 403**; ST 57/6/77/41 =
181; Y 77/5/118/22 = 222. ✅

### 🔴 Leakage: the benchmark is largely *inside* KINBERT's training corpus

Matched on exact (kinase catalytic-domain sequence, 15-mer) against `splits_v2` — the files
named in `exp_tape_bertbase_sep_v2_*.yml`:

| split | screened pairs present | benchmark pairs present (of 385) |
|---|---|---|
| train | 540 | **205** |
| val | 84 | 29 |
| test | 76 | 38 |
| test2 | 64 | 38 |
| **union** | **764 / 29,799 (2.6%)** | **310 / 385 (80.5%)** |

- **127 of 385 (33%)** are in the **train** split of the phospho-PCA fine-tuning set; 153 in
  that corpus overall. Essentially the whole fine-tuning set is screen-derived (9,013 of
  9,634 pairs).
- **338 of 385 (88%)** sit on the training side of one model or the other. Only 40 are in
  neither corpus; 76 are in a held-out split only.

**Consequence: Table S10 is a screen-vs-database *concordance* measurement, not a held-out
KINBERT evaluation.** This must be stated. It is the concrete answer to the review notes'
block E leakage questions — and it is worse than those notes assumed.

**Extra finding:** the accession-keyed corpora are **not disjoint at the peptide level**
(240213 train∩val = 2,459 pairs; train∩test = 2,594; 230621 train∩test = 40,869), because
they were split on (substrate accession, site) while **53% of these 15-mers occur in more
than one protein**. `splits_v2` *is* disjoint on its own key.

### 🔴 "Documented negatives" are not documented

Both negative sets are **database/sampled decoys, not experimentally demonstrated
non-substrates**:
- Table S10's **74 "documented negatives"**: after the source filter, every surviving
  negative is a `kinbertv1_DB_Test` row with `source = NaN` — KINBERT's
  one-negative-per-positive **randomly sampled decoys**.
- Table S13's **165 "documented non-substrates"**: selected in
  `230310_select_substrate_oligo.ipynb` cell 3 as rows where both KINBERT v1 and v2 had
  `y_true == 0` and v1 predicted 0 — **in the DB TRAIN split**.

This changes a claim about experimental provenance, not just a count. **Worth confirming
independently before rewriting.**

### Counts: "~300" is wrong; everything else in Table S13 is right

Designed spike-ins, from `{st,y}_spike_ins.csv`:

| | rows | unique 15-mers | target kinases |
|---|---|---|---|
| ST | 1,058 | **465** | 6 |
| Y | 158 | **151** | 3 |
| total | 1,216 | **616** | 9 |

Polarity at window level: **449 positive / 165 negative / 2 both** — exact match to Table
S13. **"~300" is not reproducible from any file in the design path.** Delete it.

**403 is a ROW count — there are only 385 distinct (kinase, 15-mer) pairs.** The merge
duplicates pairs carrying more than one `source_all` string.

**Everything else in the Table S13 caption reproduces exactly:** 3,928 / 1,978 / 1,950 ·
616 (465+151) · 3,312 · 2,753 · 559 · 449/165/2 · 3,918 distinct + 10 shared · 45
Ser/Thr-centred Tyr windows · 2,069 multi-mapping (53%) · `VGAPIAVSAFYAESC` unmappable ·
32 not recovered · Levenshtein 7/9/11. ✅

### Smaller qualifications

1. **Levenshtein wording is backwards.** `sanitize_redundancy()` *drops* members at distance
   < threshold, so retained sets are **≥7** (controls), **≥9** (PSP pool), **≥11** (animal
   pool). Methods' "< 7" describes removal, and its single "> 8" collapses two thresholds.
2. **2,753 / 559 is convention-dependent.** The Table S13 generator sets `animal_random` as
   default then overrides with PSP membership → 2,753/559. The design notebook applies
   `animal_random` *after* `psp_random` → **2,743/569**. The 10-sequence difference is
   windows in both pools. Say which convention is used.
3. **2,069 needs its denominator:** 2,069 of 3,928 ordered rows; 2,063 of 3,918 distinct
   windows. Max 68 distinct source proteins for one window.
4. The unmappable window `VGAPIAVSAFYAESC` is itself a spike-in (`train_tn`, target P53350).
5. **Code bug worth knowing:** `EXCLUDE_SOURCES` is applied with `.isin()` against a
   comma-joined string, so it only drops rows whose `source_all` is *exactly* one excluded
   name. It happens to do the intended job for common combinations but is not a real
   per-source filter.

### Where Table S13 came from — and it's ahead of the caption

`ksi_analysis_2025/manuscript_261014/tableS13_substrate_library/` (`build_table_s13.py`,
`config.py`, `outputs/summary_stats.txt`). **Its README already lists 6 of the 7
contradictions above** — the generator (Oct 14) is ahead of the draft caption, which was
only partly updated from it. The README also flags that the Zenodo DOI was never verified
and the CSV will not match the deposit unless re-uploaded.

Corrected drafts for the Results sentence, the Methods paragraph (plus a new
"Benchmarking against database annotations" sub-paragraph) and the Table S13 caption are in
`out/audit_09_corrected_wording.txt`.

### Caveat on corpus identity

Which KINBERT generation backs the manuscript's reported numbers is **inferred from the
config files, not stated in the manuscript**. `splits_v2` was used because
`exp_tape_bertbase_sep_v2_*.yml` points at it. Also, **4 of 20 screened kinase constructs
have no byte-identical domain sequence in the corpus**, so the domain-keyed overlap is a
**lower bound**. If the manuscript's KINBERT is a different checkpoint, re-run against that
config's `train_csv`.

---

## Item 10 — Supp Tables S8/S9 internal arithmetic ⚠️ ONE DEFINITIONAL ISSUE

24 of 28 checks pass. Verified exactly: designed pairs = kinases × designed substrates for
both arms (8×1,950 = 15,600; 15×1,978 = 29,670); monotonic attrition at every stage;
kinase total 23; constructed total 40,087; screened total 35,915.

**Table S8 "Mean reads per barcode" is not the ratio of its two neighbouring columns.**

| library | After-ORF-QC ÷ Final unique barcodes | printed | implied denominator | implied assignment rate |
|---|---|---|---|---|
| ST (blib5) | 1,631,649 / 121,070 = **13.48** | **11.27** | ~144,778 | **83.6%** |
| Y (blib1) | 1,286,517 / 205,272 = **6.27** | **5.31** | ~242,282 | **84.7%** |

The printed means imply a *larger* barcode denominator, and the implied assignment rate is
~84% in **both** libraries — consistent, not coincidental. Reading: the mean is taken over
**all observed barcodes**, before the ≥90% unambiguous-assignment filter that produces the
"Final unique barcodes" column.

**Action:** the numbers are probably right; the column juxtaposition misleads. Clarify in
the caption, and consider stating the ~84% confident-assignment rate explicitly — it is a
useful QC figure currently invisible in the table. Confirm the denominator against the
lookup-table code before printing it.

**Trivial:** ST overall pair attrition is 23.5%, printed as 23% in SuppFig_library
(truncated, not rounded). Y is 15.3% → 15%, fine.

---

## Item 11 — Regenerated SF5/SF6/SF7 (`ksi_analysis_2025/SF5_SF6_SF7/`)

Script: `scripts/audit_11_sf5_sf6_sf7.py`. **35 claims: 26 MATCH, 8 MISMATCH, 1 UNRESOLVED.**

### 11a — n = 403 vs n = 258: same matrix, two denominators ✅ EXPLAINED

Both figures draw the **identical** confusion matrix. They differ only in whether the 145
false negatives the assay **could not have detected** (120 no-PAABD reader + 25 QC-removed)
are in the denominator. 50 + 120 + 25 = 195 ✅.

| | SF5c / Table S10 (**nominal** FN = 195) | main Fig 2 panel D (**evaluable** FN = 50) |
|---|---|---|
| Total | **403** ✅ | **258** ✅ |
| Accuracy | **0.489** ✅ | **0.764** ✅ |
| Recall | **0.407** ✅ | **0.728** ✅ |
| Precision | 0.924 ✅ | 0.924 ✅ |

Every one of those eight values reproduces exactly. **Nothing is miscalculated** — but the
same benchmark is reported at **accuracy 0.489 in the supplement and 0.764 in the main
figure**, and neither panel says which denominator it used.

**Precision is denominator-invariant** (134/145 either way), which is exactly why this never
got caught: the number everyone checks is the one number that cannot move.

**Fix:** label both. Main Fig 2 panel D → "n = 258 screening-evaluable pairs; accuracy and
recall over evaluable pairs". SF5c / Table S10 → "n = 403 rows (385 distinct pairs); accuracy
and recall over all documented interactions". Showing accuracy at all is arguably unhelpful
here — it is the metric most distorted by the undetectable FNs. Consider dropping it from one
of the two, or printing both pairs side by side.

### 11b — LFSR ≤ 0.05 is correctly applied throughout ✅

| script | hit key | source | max `lfsr_best_hit` |
|---|---|---|---|
| `sf5_library.py` | `df["true_positive"]` | `251203_ksi_reduced_df_tp.pkl` | **0.05** ✅ |
| `sf6_false_positive.py` | `df["positive"]` | `251203_ksi_reduced_df.pkl` | **0.05** ✅ |
| `sf7_decoupling.py` | *no threshold* | `decoupling_out/251125_pymc_*` | n/a ✅ correct — S7 plots distributions and median abundances, not hit calls |

No LFSR 0.01 contamination in any of the three. `positive` = **8,890** pairs, matching the
changelog exactly.

🔴 **But a fourth "hit" quantity surfaced:** `true_positive` = **8,100** pairs (= `positive`
minus false-positive substrates). See the cross-cutting table below — the manuscript now uses
the bare word "KSIs" for at least four distinct quantities.

### 11c — SF6: the "16× conflict" is a FALSE ALARM ✅ DISSOLVED

Review-notes §B5 / refine.ink #31 claimed panels c and d disagree by 16×. They do not —
they have **different denominators**:

- **Panel c** plots, at each LFC threshold, the fraction of the substrates **flagged at that
  threshold** with promiscuity > 0.5. The denominator **varies along the x-axis** (1,325 at
  LFC = −4; **33** at LFC = 1.5).
- **Panel d** plots all control substrates, with promiscuity filled to 0 for unflagged ones.

Recomputed: at LFC = 1.5, **25 of 33 flagged substrates exceed 0.5 → 75.8%**, and **33
substrates have non-zero promiscuity**. Panel c reports the 25/33; panel d reports the 33.
**Perfectly consistent.** The caption's wording is what misleads — panel c says "Fraction of
control substrates", implying all 2,753, when it means *flagged* substrates.

**This also rescues the QC calibration story.** Methods' "~75% of flagged substrates were
phosphorylated by more than half of all kinases" is **correct** (75.76%). Review-notes
triage item 13 ("the tyrosine calibration as stated does not hold") was built partly on B5
and should be re-examined before rewriting.

### 11d 🔴 NEW: the volcano counts barcodes, not substrates — and the README's fix is also wrong

`sf6d_volcano.csv` has **2,753 rows but only 2,602 unique `Gene` values**: the volcano draws
**one marker per sgrna/barcode**, not per substrate.

| statement | source | correct? |
|---|---|---|
| "46 of 2754 substrates" | S6 caption (shipped) | ❌ counts the legend swatch *and* uses the wrong noun |
| "45 of 2753 substrates" | `SF5_SF6_SF7/README.md` | ❌ swatch fixed, **noun still wrong** |
| **"33 of 2,602 substrates"** (= 45 of 2,753 barcodes) | recomputed | ✅ |

So the README is right that the shipped caption over-counts by one, but its own correction
keeps the barcode denominator while calling them substrates. Use **33 of 2,602 substrates**,
or say "45 of 2,753 barcoded substrate constructs" if the per-barcode framing is wanted.

### 11e — SF6 panel c reference line

| statement | value |
|---|---|
| S6 caption provenance line: "recovered at … 75.00%" | ❌ **wrong** — it is **75.76%** |
| artwork: "76 % at the chosen LFC = 1.5 cut-off" | ✅ |
| Methods: "~75%" | ✅ |
| caption: "never falls below 27%" | ✅ (26.94%) |

Only the caption's claim of exact recovery at 75.00% needs fixing.

### 11f — SF5 a/b and SF7 are clean

SF5c combined counts 134/11/195/63/403 all ✅. Attrition 23.5% (ST) / 15.3% (Y) — printed as
23%/15%, truncated not rounded (trivial).

SF7: **45** median-abundance values ✅; depleted group (ADCK1, MAST1, TITIN) negative across
all three PAABDs ✅; enriched group (KKCC1, KS6B1, TBK1) positive across all three ✅; WW
gives smaller effects both directions ✅ (mean |median| 0.313 vs FHA 0.744, 14-3-3 0.843).

The two "0.01 discrepancies" the README flags are **rounding-boundary artifacts**, not real
disagreements: IKKB FHA = −0.8546 (deviation **0.0054** from −0.86) and PDPK1 WW = −0.3849
(deviation **0.0051** from −0.39). Both straddle the 2-dp boundary. The caption's "largest
deviation 0.005" is therefore only *marginally* exceeded — say "≤0.006" and the claim is
sound.

---

## Cross-cutting: the four different "hit" quantities

From `CHANGELOG_lfsr0.05_2026-10-21.md`, confirmed canonical at b ≥ 0.1, LFSR ≤ 0.05:

| quantity | value | where it belongs |
|---|---|---|
| substrate × kinase × PAABD **rows** selected | **12,936** of 61,979 | Fig 2 caption (already correct) |
| **`positive`** (substrate, kinase) pairs | **8,890** | — currently unstated |
| **`true_positive`** pairs (= positive − FP substrates) | **8,100** | — currently unstated; this is what SF5c and the biological analyses use |
| total (substrate, kinase) pairs in the reduced df | 34,767 | — |
| KSIs **constructed** | 40,087 | Fig 1A / Fig 2A (label as constructed) |
| KSIs **screened** | 35,915 | Results |

The distinction between **8,890 `positive`** and **8,100 `true_positive`** matters: the
false-positive-substrate filter removes 790 pairs, and the downstream biology (dark kinases,
motifs, tree, fine-tuning set) all keys on `true_positive`, while SF6's promiscuity analysis
keys on `positive` — correctly, since it is *measuring* the promiscuity that defines the
filter.

Figure 1's "high-confidence interaction matrices of 25,214 serine/threonine and 14,873
tyrosine KSIs" conflates coverage with positive calls — those are *constructed* totals, not
hits. Three distinct quantities are currently used interchangeably across Fig 1, Fig 2 and
Results.
