# Source notebooks — what became of each

These 18 notebooks are the provenance record for `code/01_phosphoPCA` and are
kept unchanged. They are already de-duplicated to the FINAL version of each
analysis (earlier `8_2batch_blib_pipeline.ipynb` / `_v2`,
`10_comparison_to_invitro.ipynb` and similar are not in this folder).

Outputs are named as they appear in `../` (the package code directory).

| notebook | became | notes |
|---|---|---|
| `1_parse_lookup_tables.ipynb` | stage `01_parse_lookup_tables.py` | whole notebook; nothing left behind |
| `2_generate_lfc_table.ipynb` | stage `02_generate_lfc_table.py` | whole notebook. Two kinds of cruft removed: the batch-merge block appears twice, identically, and one cell writes `bliballvariant2blib5.count.txt` from the *batch-1* table before a later cell overwrites it from the real batch-2 run — only the later one is kept (see the `NOTE:` in the script). The designed-oligo coverage check is kept as `--skip-coverage-qc`-able QC output |
| `3_integrate_lfc_lookup.ipynb` | stage `03_integrate_lfc_lookup.py` | the four plotnine LFC-histogram cells and the single-barcode spot checks were exploratory and are left behind |
| `4_filter_unmapped_substrates.ipynb` | stage `04_filter_unmapped_substrates.py` | the filter and its QC counts are in the script. **Left behind, and not reproducible from the notebook:** the cells that build `substrates_background_12345.pkl` refer to two frames (`sub1234`, `sty_background`) and a helper (`assign_15aa`) that were defined in an earlier session of the same kernel, so the file is treated as a deposited input. Its provenance is documented in the stage docstring and in `../_source_scripts/tableS13_substrate_library/config.py` |
| `5_enrichment_analysis.ipynb` | stage `05_fit_decoupling_model.py` + `../decoupling_model.py` | the model is now a single shared module, imported by stage 05 and by every QC refit. The `pip uninstall h5py / install netCDF4` cell became an environment note in `../requirements.txt` |
| `6_inspect_decoupled_binding.ipynb` | stage `06_call_hits.py` | the hit-calling and table assembly are in the script. The two ad-hoc supplementary figures in it (`fig1_activity_abundance.pdf`, `fig2_abundance_consistency.pdf`) are superseded by `figures/decoupling_supp_figures.py`, which draws the same quantities for the deposited supplementary figure; the plotnine histogram cells are left behind |
| `7_identifiy_false_positives.ipynb` | stage `07_filter_false_positives.py` + `figures/suppfig_falsepositive.py` | split: the substrate lists and the canonical hit table in the stage script, the two plotting cells in the figure script. The interpretation written in its markdown cells is condensed into the stage docstring |
| `8_2batch_blib_pipeline_v3.ipynb` | `figures/fig05_math.py` (sections 7–8 → panel D; section 5 → `--library-stats`) | **Left behind:** section 6b's full kinase × substrate plotnine heatmaps (already a standalone script, `../_source_scripts/plot_ksi_heatmaps.py`), the Venn-diagram imports that are never used, the `plot_db_vs_screening_confusion_evaluable()` variant (a second rendering of the same counts, with the excluded FNs annotated in the FN cell), the stacked-bar and step-plot views of the same FN decomposition, and the `.to_latex()` calls — the tables are written as CSV instead |
| `9_screening_tp_to_reporter_candidates.ipynb` | exploratory record | nominates SPARK reporter candidates from the hit table; feeds the KINBERT/SPARK sections (Fig 07), not Fig 05/06. Its two sharp edges are already captured as standalone scripts in `../_source_scripts/`: `check_selection_per_paabd.py` (what happens if the `binding_max` argmax is not used to pin each candidate's pAABD) and `reporter_panel.py` (the candidate table, looked up rather than hand-typed) |
| `10_comparison_to_invitro_v2.ipynb` | `figures/fig05_math.py` (panel D, right) | whole analysis extracted. **Left behind:** the per-kinase signal-density plot and the `venn2` version of the overlap (the manuscript uses the pie). Two mismatches between its prose and its code are preserved as code, with `NOTE:` comments: the markdown says the in-vitro threshold is 100 (the code uses 20) and that ~75% of hits were confirmed (the code and the figure give 84.9%). It also redefines the screening call as `binding_max > 0` for this comparison instead of the canonical `true_positive`; kept, and flagged |
| `11_ksi_tree_comparison_v2.ipynb` | `figures/fig06_biological_insights.py` (panel A, `--panel a-trees`) | tree building, Newick export and the leaf-order alignment are extracted; the regenerated `screening_tree.newick` is byte-identical to the deposited one. **Left behind:** the four tree-topology similarity metrics (Robinson–Foulds, matching-split, matching-pair, mutual clustering information), their comparison report, and the hand-drawn `seq_tree.svg` rendering with kinase-family colouring — none of them is used by Fig 06A |
| `11_2_kinase_tree_analysis.ipynb` | `figures/fig06_biological_insights.py` (panel A, `--panel a-cross`) | whole analysis. Its `KINOME_THRESHOLD`/`SCREEN_THRESHOLD` comments say "median" but the code uses the mean of each distance vector; the code is kept, with a `NOTE:` |
| ~~`11_ksi_motif_analysis-stlib.ipynb`~~ | **removed 2026-10-25** | Verified unused: regenerating panel B with `--motif-kinase-set dark` reproduces all six deposited `IC_matrix_{aa,prop}_{ADCK1,KKCC1,PHKG1}` CSVs **bit-identically** (max abs diff 0.0e+00), and the printed IC values follow (CaMKK1 Basic@-3 = 0.475, ADCK1 Basic@-6 = 0.755, PHKG1 max Basic at -3). The deposited Fig 3B panels therefore come from the dark path, not from this notebook. Its parameters remain reachable as `fig06_biological_insights.py --panel b-motifs --motif-kinase-set stlib` |
| `11_ksi_motif_analysis_ylib.ipynb` | exploratory record | the same for the Y library; all save calls commented out, and no Y motif panel appears in Fig 06 |
| `11_novel_ksi_analysis.ipynb` | exploratory record (superseded) | the first version of the tree analysis. Superseded by `11_ksi_tree_comparison_v2.ipynb`; see the disagreement note below |
| `12_dark_kinase_analysis.ipynb` | `figures/fig06_biological_insights.py` (panel B, `--panel b-motifs`) | whole analysis; `information_content_calculator.py` is imported, not inlined. **Left behind:** the final cells that cross-check the dark-kinase hits against `pairs_aug_v2.csv` labels (a KINBERT corpus lookup, printed and not saved) |
| `12_2_dark_kinase_go_analysis.ipynb` | `figures/fig06_biological_insights.py` (panel B, `--panel b-go`) | whole analysis; reproduces the deposited `260112_dark_kinase_go_results.csv` exactly (56 rows, identical p and q). **Left behind:** the interactive `filter_by_go_term()` helpers used to look up individual GO terms, and the `reduce_to_seq` / `collapse_to_substrate_seq_15aa` / `use_promiscuity_thresh` branches that were not taken |
| `13_kinbert_phosphoPCA_link.ipynb` | exploratory record | links KINBERT scores to the phospho-PCA hits and writes the fine-tuning dataset; belongs to the KINBERT section (Fig 07, `tab:ft`), not to this pipeline. Its hand-typed candidate table is already replaced by `../_source_scripts/reporter_panel.py` |

## Where two notebooks disagreed, and how it was resolved

* **Tree input filters — `11_novel_ksi_analysis` vs `11_ksi_tree_comparison_v2`.**
  The older notebook keeps substrates measured against > 6 (Y) or > 11 (S/T)
  kinases and filters on `true_positive`; v2 uses > 3 and > 6 and filters on
  `lfsr_best_hit` instead. Both write `screening_tree.newick`. Resolved by
  regenerating the tree from v2 and comparing with the deposited file: v2's
  filters reproduce it byte-for-byte, so v2 is the production path and the
  older notebook is retained only as a record.

* **An operator-precedence accident in v2, deliberately preserved.** The Y
  branch of that filter is written
  `y.loc[y['substrate'].isin(well_rep_y) & y['lfsr_best_hit'] < 0.001, :]`.
  `&` binds tighter than `<`, so it evaluates as
  `(isin(...) & lfsr_best_hit) < 0.001` and keeps the *complement* of the
  intended filter — 11,657 of 13,943 Y rows, where the parenthesised version
  would keep 1,058. The S/T branch on the next line does have the parentheses.
  Since the asymmetric version is what produced the deposited tree, it is
  reproduced verbatim with a `NOTE:`, not corrected.

* **`assign_paabd()` has four versions across the notebooks.** The motif
  definitions differ: SH2 is pY alone in notebooks 4, 9 and 13 but pY plus a
  hydrophobic at +3 in 8_2; FHA allows D/I/S/Y at +3 in notebook 4 and only D
  elsewhere; WW allows P/L/V/I at +1 in notebook 9 and only P elsewhere.
  Resolved per consumer rather than globally: `fig05_math.py` carries the 8_2
  definitions, because those are the ones behind the confusion-matrix numbers
  in the manuscript, and says so in a `NOTE:`. No stage script needs the
  function — stage 04's only use of it was in building the substrate
  background, which is a deposited input.

* **Count-table paths.** `2_generate_lfc_table.ipynb` writes some tables to
  `${KSI_DATA_ROOT}/processed_count_table/` and others to
  `${DATA_MOUNT}/processed_count_table/`. These are the same directory under
  two different bind mounts, used in sessions months apart; there is one count
  table directory on disk. Both collapse to `config.COUNT_DIR`. Likewise its
  `sequencing_data_1204` / `sequencing_data_1217` are today's
  `sequencing_data_231204` / `sequencing_data_231217`.

## Could not be cleanly extracted

* `substrates_background_12345.pkl` (stage 04) — see the table entry for
  notebook 4. Read as a deposited input; its construction is only partly in
  the notebook.
* `screening_tree_reordered_renamed_newick.nwk` (Fig 06A cross-tree distances)
  — the leaf-order-aligned tree with leaves renamed to the kinome tree's
  labels. The reordering is reproduced by `--panel a-trees`, but the renaming
  was done by hand in the figure draft, so the file is read as an input.
* The IC panels in Fig 06B for the non-dark kinases (PDPK1, KS6B1, PLK1, KGP1
  and the two phosphomimetic variants) — the notebook that renders them
  (`11_ksi_motif_analysis-stlib.ipynb`, removed 2026-10-25) had all of its save calls commented
  out, so the exact filter set behind the deposited panels cannot be confirmed
  from the notebook alone. `--motif-kinase-set stlib` runs that notebook's
  parameters as written.
