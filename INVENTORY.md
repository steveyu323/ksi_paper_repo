# KSI Paper — reproducibility package inventory

**Scope: the final manuscript**, `KSI_Paper_261022_CY/` as of 2026-10-23 —
8 main figures, 12 supplementary figures (S1–S12), 13 supplementary tables
(S1–S13). The figure PDFs and `.tex` in `figures/` are byte-identical to that
manuscript; `figures/CHANGES_261022.md` is its revision log.

This supersedes the 2026-10-14 inventory. The 12 figure PDFs that changed since
are kept in `_superseded_261014/figures/` for diffing only.

> **Figure files are not numbered like the printed figures.** `Fig 05` prints as
> Figure 2, `Fig 06` as Figure 3, `Fig 02` as Figure 4, `Fig 07` as Figure 5,
> `Fig 08` as Figure 6, `Fig 03` as Figure 7, `Fig 04` as Figure 8. Every table
> below is keyed on the **printed** number first. Each `.ai` also carries a
> baked-in caption with its *file* number, which is wrong in the printed
> document — see Blocker 1.

## Source trees

| Section | Tree |
|---|---|
| phospho-PCA screen | `${KSI_REPO_ROOT}/ksi_analysis_2025` |
| KINBERT | `${KSI_ML_ROOT}` (live) + `${KSI_RESTORE_ROOT}` |
| SPARK | `${KSI_REPO_ROOT}/cellsam-internal` |
| Variant + viral | `${KSI_REPO_ROOT}/kinase_ml` |

Neither KINBERT tree is a superset of the other: the live tree has the final
eval notebooks, `eval_7_separate_tests.ipynb`, `hf_kinbert/`; the tarball tree
has `biosensor_scoring/`, `eval_tables/`, `eval_table_scripts/`.

## Main figures

Every main figure is an **Illustrator composite**. The panels come from code;
the assembly, labelling and layout do not. None of the eight is regenerable by
running a script — this is the honest answer to "code + data render the exact
results", and it is Blocker 2.

| Printed | File | Panel → code | Data |
|---|---|---|---|
| **Fig 1** | `Fig 01 - Overview` | Schematic; no code. Quoted counts from stage 01 (barcode QC) and `8_2batch_blib_pipeline_v3.ipynb` §5 | PacBio HiFi + count tables |
| **Fig 2** | `Fig 05 - MATH` | **A** `8_…v3.ipynb` §5 + `build_table_s13.py` · **B** `figures/mock_boltzmann_to_logfc.py` · **C** stage 05 (`05_fit_decoupling_model.py`, `decoupling_model.py`) + `figures/decoupling_supp_figures.py` → `figS_hit_selection_binding_lfsr` (b ≥ 0.1, LFSR ≤ 0.05 → 12,936 of 61,979) · **D** `8_…v3.ipynb` §7–8 + `10_comparison_to_invitro_v2.ipynb` (84.9 % = 45/53, `INVITRO_THRESH = 20`) | `raw_screen_count_mapped.csv`, `251130_ksi_inference_scores.pkl`, `invitro_overlap_summary.csv` |
| **Fig 3** | `Fig 06 - Biological insights` | **A** selectivity tree `fig03_tyrosine_tree/screening_tree_optA_reordered_renamed.nwk` (rotation-only relabel of `screening_tree_optA.newick`); Kendall τ = 0.50, p = 1.1 × 10⁻¹⁸ and the acid:base corroboration from `fig03_tyrosine_tree/figure3_tyrosine_revision.ipynb` · **B** `12_dark_kinase_analysis.ipynb` + `figures/information_content_calculator.py`; GO from `12_2_dark_kinase_go_analysis.ipynb` | canonical hit table (`251203_ksi_reduced_df_tp.pkl`); tyrosine PSSMs from the `kinase-library` PyPI package |
| **Fig 4** | `Fig 02 - KINBERT` | **A** `eval_0_datasource_stats.ipynb` cells 18–24, with the 2,304 → **2,021** de-duplication fix (`verification_261022/scripts/audit_01_corpus_stats.py`) · **B** `model_architecture_figures/make_fig.py` (schematic) · **C** `eval_2_v2test_dark_nonhuman.ipynb` (live) + `eval_7_separate_tests.ipynb` (LinkPhinder panel) | `pairs_aug_v{1,2}.csv`, `eval_outputs/*.pkl` |
| **Fig 5** | `Fig 07 - Fine tuning…` | **A** `eval_2_finetune_phosphoPCA.ipynb` · **B** `13_kinbert_phosphoPCA_link.ipynb` + `_source_scripts/reporter_panel.py` (supersedes the hand-typed values in notebook 13); the 0.5 decision boundary drawn here · **C** `9_screening_tp_to_reporter_candidates.ipynb` + `kinase_ml/.../ikka_tbk1_substrates.ipynb`; rescoring `make_rescore_request.py` → `score_pairs_kinbert.py` | KINBERT checkpoints pre-FT and fine-tuned; `rescore_results.csv` |
| **Fig 6** | `Fig 08 - SPARK Results` | **A, B** `spark_overlay_patched.ipynb` cells 18–23, 30–34 (cell 21 holds the published Mann–Whitney tests) · upstream `spark_predict_patched.ipynb` → `spark_reprocess_ikknovel.ipynb` → `260327_curated_ikk_novel_dynamics.ipynb` · **C** FIJI strips via `assemble_frame_strips.ipynb`, provenance recovered by `261001_recover_fiji_normalization.ipynb` | raw `.tif` **absent** — Blocker 3 |
| **Fig 7** | `Fig 03 - Network rewiring` | **A** the 2026 chain: `_source_eval/eval_5_prediction_against_kinome_ptmvar.ipynb`, `eval_5_2_…_flanks.ipynb` and `eval_5_3_…_curated_net.ipynb` read the `exp_tape_bertbase_sep_v2_long__ptmvar_{on_psite,flank_wt,flank_mut}__6eda71d75e35.pkl` outputs (one per `-Copy1/2/3` config) and produce `260130_ptmvar_nr_finalized.csv`; `generate_all_figures_final.py` draws the panels · **B** laid out by hand from `260130_ptmvar_nr_finalized.csv` (65 rows = 40 on-site + 25 flanking pairs, 34 site nodes), staged in `data/results/variant_viral/`; all 34 labels re-verified against it by `verification_261022` | `PTMVar.csv` (PSP 2022-10) |
| **Fig 8** | `Fig 04 - Viruses` | **A** `collect_and_inspect_prefs.ipynb` → `230105_dms_psi_inference.ipynb` · **B, C** the 2026 chain: `_source_eval/eval_6_prediction_against_kinome_virus_{demo,full}.ipynb` read `exp_tape_bertbase_sep_v2_long__virus{,_dms}__53ceb023016e.pkl`, giving `250118_virus_against_kinome_hits_with_drug.csv` and `eval_tables/eval_virus_dms_*_demo.csv` (the exact 42 × 19 panel C frame). The 2023 `virus_predictions_evaluation.ipynb` is retained as provenance but is **not** the source of the printed values | 4 DMS studies, `virus_psites/covid_psite_1_min.pkl`, PKIDB, DrugBank |

## Supplementary figures — reproducibility verified

**9 of 12 are exactly reproducible.** Verified by regenerating or by comparing
the shipped PDF against the script output.

| Printed | File | Generator | Verdict |
|---|---|---|---|
| **S1** | `SuppFig_corpus` | `figures/SF1_SF5_SF6_SF7/sf1_corpus.py` | ✅ **byte-identical** |
| **S2** | `SuppFig_benchmark` | **none** — redrawn by hand from Table S4 | ❌ Blocker 4 |
| **S3** | `SuppFig_generalizability` | **none** — redrawn by hand from Table S5 | ❌ Blocker 4 |
| **S4** | `SuppFig_finetuning` | **none** — redrawn by hand from Table S6; `eval_2_finetune_phosphoPCA.ipynb` has all 6 `savefig` calls commented out | ❌ Blocker 4 |
| **S5** | `SuppFig_library` | `SF1_SF5_SF6_SF7/sf5_library.py` | ✅ **byte-identical** |
| **S6** | `SuppFig_falsepositive` | `SF1_SF5_SF6_SF7/sf6_false_positive.py` | ✅ **byte-identical** |
| **S7** | `SuppFig_decoupling` | `SF1_SF5_SF6_SF7/sf7_decoupling.py` | ✅ **byte-identical** |
| **S8** | `SuppFig_replicate_counts` | `figures/seq_replicate_qc/01_raw_replicate_concordance.py` | ✅ **byte-identical** |
| **S9** | `SuppFig_replicate_binding` | `seq_replicate_qc/04_conditional_replicate_posteriors.py` (needs 02, 03 first) | ✅ **byte-identical** |
| **S10** | `SuppFig_convergence` | `figures/bayes_fit_qc/01_convergence_diagnostics.py` | ✅ text layer identical (PDF metadata differs only) |
| **S11** | `SuppFig_hyperparameters` | `bayes_fit_qc/01_convergence_diagnostics.py` | ✅ text layer identical |
| **S12** | `SuppFig_sampler` | `bayes_fit_qc/01_convergence_diagnostics.py` | ✅ text layer identical |

Each S1/S5/S6/S7 panel also writes its plotted values to
`data/results/phosphoPCA/sf_source_data/*.csv`, so every point is traceable
without rerunning.

## Supplementary tables

| Printed | Label | Source |
|---|---|---|
| S1 | `tab:scope` | KINBERT row from `audit_01_corpus_stats.py` (**2,021** / 92 / 147,488); DeepKinZero and Phosformer are published values — **not independently verified**, see Blocker 6 |
| S2 | `tab:coverage` | literature-derived, no data dependency |
| S3 | `tab:ipk` | `eval_0_datasource_stats.ipynb` cells 22–23; totals confirmed by `audit_02` |
| S4 | `tab:arch` | `eval_table_scripts/table_02_model_structure.py` → `eval_tables/benchmark_model_structure_*.csv` |
| S5 | `tab:gen` | `table_03_generalizability.py` + `table_04_ksi_screen.py` |
| S6 | `tab:ft` | `eval_2_finetune_phosphoPCA.ipynb` |
| S7 | `tab:viral` | `kinase_ml` viral inputs. **Attribution unresolved** — the printed source is Bouhaddou 2020, but the input file `covid_psite_1_min.pkl` records no provenance (Blocker 7) |
| S8 | `tab:libstats` | stage 01 (`01_parse_lookup_tables.py`); arithmetic checked by `audit_10_library_tables.py` |
| S9 | `tab:libdesign` | `8_2batch_blib_pipeline_v3.ipynb`; checked by `audit_10` |
| S10 | `tab:spike` | `8_2batch_blib_pipeline_v2.ipynb` **cell 24** — an inner join of the screen against KINBERT's annotation corpus, *not* the designed spike-ins. Provenance and the 80.5 % training overlap from `audit_09_spikein_provenance.py` |
| S11 | `tab:spark` | `score_pairs_kinbert.py` → `rescore_results.csv`; the two `-1` rows corrected per `audit_06_tableS11_logits.py` |
| S12 | `tab:kinases` | screen kinase table; 23 constructs confirmed by `audit_03_kinase_panel.py`. JAK2 construct is **JH1 (catalytic)** |
| S13 | `tab:substrates` | `_source_scripts/tableS13_substrate_library/build_table_s13.py` |

## Main-text and caption numbers

The headline numbers that changed in this revision each trace to a script in
`code/verification_261022/`, whose `FINDINGS.md` gives the manuscript value, the
recomputed value and which to use.

| Claim | Script |
|---|---|
| corpus 2,021 kinases / 147,488 sequence pairs / 162,447 site interactions; DB-only 26,267 / 29,894 / 1,995 | `audit_01_corpus_stats.py` |
| Table S3 bin totals 1,816 / 1,790 | `audit_02` |
| 23 kinase constructs = 21 distinct; DYRK4 excluded | `audit_03_kinase_panel.py` |
| Titin is serine/threonine | `audit_04` |
| SPARK p-values, n pools, rank-biserial sign | `audit_05_spark_pvalues.py` |
| Table S11 logits (`z₁`, not log-odds) | `audit_06_tableS11_logits.py` |
| PER2 Ser662 and the three other Fig 7 labels | `audit_07_per2_site.py` |
| influenza M1 Ser53 → ERK7; H3N2 HA Thr329 → PKCε | `audit_08_m1_ser53.py` |
| substrate-library provenance; 403 rows / 385 pairs; 80.5 % overlap | `audit_09_spikein_provenance.py` |
| Tables S8/S9 arithmetic | `audit_10_library_tables.py` |
| S5/S6/S7 panel values (35 claims) | `audit_11_sf5_sf6_sf7.py` |
| benchmark class balance (positive fractions 0.31–0.73) | see `figures/CHANGES_261022.md` §2.1 — recomputed from the eval files; **script not yet added to this package**, Blocker 5 |
| Kendall τ = 0.50, p = 1.1 × 10⁻¹⁸ | `figures/fig03_tyrosine_tree/figure3_tyrosine_revision.ipynb` |

## Verification harness

`code/common/verify_expected_values.py` re-derives the printed values from the
staged data and diffs every script-generated figure against the shipped PDF.
As of 2026-10-23 it reports **15 passed, 0 failed, 3 skipped** (the three skips
are S2/S3/S4, which have no generator). Run it before any deposit.

One live defect it would have caught: the package's
`code/01_phosphoPCA/config.py` still had `LFSR_THRESH = 0.01`, so stage 06 and
the decoupling figures would have regenerated at the pre-2026-10-21 threshold
(9,966 rows instead of 12,936). Fixed, and the `_source_scripts/` provenance
copies were refreshed from the patched sources.

## What was removed, and what was deliberately kept

Audited 2026-10-23 against the final manuscript: every candidate was checked by
grepping the printed `.tex` and figure text layers for the values it produces.

**Removed (26 paths).** Each survives in its source tree; nothing below
contributes a value, panel or curve to the final manuscript.

| Removed | Why |
|---|---|
| `figures/bayes_fit_qc/02_posterior_predictive.py` (+ the `_source_scripts/` copy) | Produces the posterior-predictive and variance-partition outputs. None of its distinctive values appears in either `.tex`: Bayesian R² 0.971/0.962/0.964/0.815, variance shares, calibration R² 0.984/0.993 — 0 hits each. Supplementary Note 1's "posterior predictive"/PIT mentions are the **held-out cross-replicate** check from `seq_replicate_qc/03`, not this script's in-sample one |
| `_source_notebooks/11_novel_ksi_analysis.ipynb` | Superseded by `11_ksi_tree_comparison_v2`, which reproduces the deposited `screening_tree.newick` byte-for-byte |
| `_source_notebooks/11_ksi_motif_analysis_ylib.ipynb` | Y-library IC analysis; all save calls commented out and Fig 3B contains only the three S/T dark-kinase heatmaps |
| `03_spark/_source_notebooks/ikk_spk_novel_fixed.ipynb` | Dev iteration. Its overlay has five conditions including GSK8612; the printed IKK-β novel panel has four (Blank 29, BMS-345541 16, TNFα 14, polyIC 6), matching `spark_overlay_patched` cells 33/34. None of its five fixes reached the published figure |
| 8 `configs/eval_configs/*.yml` — `exp1_esmc_frozen`, `exp2_esmc_frozen_xattn`, `exp3_esm2t12_unfrozen` (**not** the `_v2`), `exp4_esm2t12_unfrozen_xattn`, `exp_esm2_t8_fullfinetune_eos`, `exp_tape_bertbase_sep` (**not** the `_v2`), `exp_esm2_t12_fullfinetune_eos_{dark,human}` | v1-corpus and ESM-ablation runs. "ESMC", "XAttn", "FullFT", "T8" appear 0 times in either `.tex`; Table S5 prints only the **TAPE** ablations |
| `eval_tables/eval_v1_test_{bar,metrics_full}.csv`, `VERIFICATION_01_v1_models.csv`, `eval_table_scripts/table_01_v1_models.py` | The v1 model comparison. None of its eight model names occurs in the manuscript |
| `eval_tables/eval_virus_dms_*_fulldrug.csv` (3), `VERIFICATION_06_virus_full.csv`, `table_06_virus_full.py` | The uncurated full-kinome virus matrix (262 kinases × 67 sites). Printed Fig 8C is the curated 42 × 19 matrix |
| `eval_tables/ks_pair_upset_{set_sizes,intersections,reconstruction_gap}.csv` | The **pair-level** UpSet. Its values (2,299 / 22,073 / 32,154 / 132,284) appear in no `.tex` and in no figure text layer. The `_nkinase_` and `_norganism_` siblings are **kept** — their set sizes are printed verbatim in Fig 4B |
| `_source_scripts/ww_normalization/` | "Not used by any manuscript figure" by its own README; also the last place still carrying the superseded LFSR 0.01 |

`run_all_tables.py` had its `table_01` and `table_06` entries removed to match.
`virus_pipeline.py` and `table_00_datasource_upset.py` are **kept** — both are
shared with scripts that are still used.

**Kept, though they look unused.** These were the near-misses:

- `eval_configs/exp_tape_bertbase_sep_v2_long-Copy{1,2,3}.yml` — not editor
  duplicates. Each enables exactly one PTMVar dataset
  (`ptmvar_on_psite`, `ptmvar_flank_wt`, `ptmvar_flank_mut`) and together they
  are the three inference legs behind printed Fig 7, including the 25 flanking
  dotted edges.
- `exp_tape_bertbase_sep_v2_early_loss.yml` — reads like an aborted run; it is
  the pre-fine-tuning checkpoint printed in Table S11 and Fig 5B.
- `exp3_esm2t12_unfrozen_v2.yml`, `exp_esmt12_frozen_v2.yml` — opaque names for
  KINBERT:ESM-CONCAT and KINBERT:ESM-FROZEN in Table S4.
- `eval_tables/eval_ksi_*.csv` — this is the **phospho-PCA column of Table S5**
  and the fourth facet of Figure S3, which `benchmark_generalizability_*` does
  not contain. See the discrepancy note below.
- `eval_tables/eval_virus_dms_*_demo.csv` — `drug_psite_demo` is the exact
  Fig 8C plotting frame (42 drugs × 19 sites, matched one-for-one against the
  PDF's axis labels); `audit_08` reads two of them by name.
- `VERIFICATION_0{0,2,3,4,5,7}_*.csv` — no printed value of their own, but they
  are the evidence for `eval_tables/README.md`'s 390/392 match claim.
- `_source_notebooks/4_filter_unmapped_substrates.ipynb` — superseded by stage
  04 except for the `substrates_background_12345.pkl` construction, which
  exists nowhere else.
- `_source_notebooks/11_2_kinase_tree_analysis.ipynb` — `figure3_tyrosine_revision.ipynb`
  explicitly recomputes *this* notebook's method for the printed τ.
- `03_spark/_source_notebooks/260318_find_spark_files.ipynb` — appears in no
  inventory table, but is the sole writer of the four `tif_file_mapping*.csv`
  that `spark_predict_patched.ipynb` reads.
- `figures/seq_replicate_qc/02` and `03` — produce no figure, but `04`
  (Figure S9) needs them and `03` is the source of Supplementary Note 1's
  held-out coverage numbers.

**All four open decisions closed 2026-10-25.**

| # | Decision | Evidence | Outcome |
|---|---|---|---|
| 1 | `_source_notebooks/11_ksi_motif_analysis-stlib.ipynb` | Regenerated panel B with `--motif-kinase-set dark` into a scratch dir: all six deposited `IC_matrix_{aa,prop}_{ADCK1,KKCC1,PHKG1}.csv` reproduce **bit-identically** (max abs diff 0.0e+00), and the printed values follow exactly (CaMKK1 Basic@$-$3 = 0.475, ADCK1 Basic@$-$6 = 0.755, PHKG1 max Basic at $-$3 = 0.718). The deposited Fig 3B panels come from the dark path | **Deleted** |
| 2 | `eval_configs/exp_esm2_t12_fullfinetune_eos_v2_early_loss.yml` | Six of its seven references in `eval_2_v2test_dark_nonhuman.ipynb` are commented out. The seventh (cell 112) loads the predictions, but **cell 114 — the cell that builds `prediction_dfs` — has both early-loss entries commented out**, leaving only `'KINBERT:TAPE': tape_db_lysate_yeast`. Nothing printed derives from it | **Deleted.** The **TAPE** `..._v2_early_loss.yml` is a different file and is **kept** — it is the pre-fine-tuning checkpoint in Table S11 and Fig 5B |
| 3 | `04_variant_viral/.../binary_tape_eval_kinbertv2_VSHumanKinome.ipynb` | `audit_08_m1_ser53.py` validates every printed Fig 8 label against the **2026** chain — `250118_virus_against_kinome_hits_with_drug.csv` and `eval_virus_dms_{kinase,drug}_psite_demo.csv` — and never reads a `kinbertv1_*` logit. Printed values reproduce: H3N2 HA Thr329 $\rightarrow$ PKC$\epsilon$ 5.0625, M1 Ser53 $\rightarrow$ ERK7 4.6875 | **Deleted** |
| 4 | `eval_ksi_bar.csv` 0.880 / 0.758 vs Table S5's 0.874 / 0.754 | Author decision: `early_loss` and `v2` are one model family and the 0.005 is not worth re-rendering built figures | **Not reconciled by design.** A note was added to the Table S5 caption instead: the KINBERT:TAPE values are from the checkpoint that rendered the panels, an independent re-run in the same family differs by $\leq$ 0.006, and the deposited table records the re-run values |

## Blockers, in order of how much they affect the deposit

1. **The 8 main figures carry baked-in captions with the wrong figure number**,
   and because each artboard is taller than the text block the real LaTeX
   captions do not print (only Figure 1's does). Fix: delete the embedded
   caption, crop the artboard, re-export at 176 mm. Per-figure size budgets in
   `figures/CHANGES_261022.md` §13. **Do this before the deposit**, or the
   deposited figures disagree with the deposited `.tex`.
2. **No main figure is script-reproducible.** All eight are Illustrator
   composites. The panels are code-derived; the assembly is not. State this
   plainly in the deposit README rather than implying a one-command rebuild.
3. **Raw SPARK microscopy is absent.** 200 `.tif` FOVs live only on madrox, and
   `spark_results.h5` is stale (2026-03-20 vs the 2026-04-08 figure run), so
   Fig 6's n = 488/540/207/138 will not reproduce from the local copy.
4. **S2, S3 and S4 have no generator.** They were redrawn by hand from Tables
   S4–S6. The values are all in `eval_tables/*.csv`, so writing `sf2`/`sf3`/`sf4`
   in the idiom of `sf1_corpus.py` is a bounded job that would take the package
   from 9/12 to 12/12 reproducible supplementary figures.
5. **The benchmark class-balance recount has no script in the package.** The
   positive fractions now printed in the Tables S4–S6 captions were computed ad
   hoc. Add the script.
6. **Phosformer 800 / 13 / 24,534 and DeepKinZero 343 / 1 / 13,426** are prose
   transcriptions from the source papers with no code behind them; they are the
   denominators of the 2.5× and 5.9× claims.
7. **Table S7 attribution.** `covid_psite_1_min.pkl` records no provenance; the
   manuscript cites Bouhaddou 2020 on the author's say-so.
8. **Zenodo `10.5281/zenodo.19616157`** was never verified as published and
   public, and `build_table_s13.py`'s README warns the regenerated CSV will not
   match that deposit unless re-uploaded.
9. **`model.py:108` hard-codes another user's home directory**
   (another user's `~/.cellsam_assets/...`). Harmless for the SPARK runs, breaks a
   naive reviewer run.
10. **Reporter naming is inconsistent** across `tif_file_mapping{,_v2,_v3,_v4}.csv`
    (`tbk1` / `TBK-SPK-1` / `TBK-SPARK-1`). The manuscript is now unified on
    `TBK-SPARK-1` / `TBK-SPARK-novel` / `IKK-SPARK-1` / `IKK-SPARK-novel`.
11. **`spark_overlay_patched.ipynb` cell 32** writes an `IKK-SPK-1` overlay to
    `spark_IKKnovel_overlay.pdf`. If Fig 6B's IKK panels were picked by
    filename, one may be mislabelled.

### Closed since 2026-10-14

- *S5/S6/S7 have no generating code* — **closed**, `SF1_SF5_SF6_SF7/` produces
  all three plus S1, byte-identically.
- *84.9 % vs ~75 % in-vitro confirmation* — **closed**, the author adopted
  **84.9 % (45 of 53)** and the manuscript now says so throughout. One tidy-up
  remains: `10_comparison_to_invitro_v2.ipynb` cell 12 (markdown) still says
  "signal > 100" while cell 13 sets `INVITRO_THRESH = 20`, and cell 21's prose
  still says "~75 %". The deposited numbers come from threshold 20.
- *616 spike-ins vs ~300* and *constructed vs screened KSI counts* — **closed**,
  both are now stated correctly in the manuscript.
