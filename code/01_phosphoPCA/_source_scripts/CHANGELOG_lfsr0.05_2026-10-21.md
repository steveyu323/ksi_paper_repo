# Hit threshold made consistent at b ≥ 0.1, LFSR ≤ 0.05 (2026-10-21)

## Why

The canonical hit set `ksi_analysis_2025/251203_ksi_reduced_df_tp.pkl` was built at
b ≥ 0.1, LFSR ≤ 0.05 (verified: recomputing from `251130_ksi_inference_scores.pkl` at
0.05 reproduces its 8,890 positive pairs with 100% agreement). It is read by 15 notebooks
and 3 scripts, so it was **kept unchanged**. Several scripts, notebook 6 cell 15 and the
manuscript text stated or used LFSR ≤ 0.01 instead. Everything below was brought into
line with 0.05.

Unchanged by decision: the pickle; the Table S10 / Figure S-library confusion matrix
(already at 0.05: TP 134, FP 11, FN 195, TN 63); the Kinexus figure 84.9% (45/53) from
`10_comparison_to_invitro_v2.ipynb`.

## Code changes (`ksi_analysis_2025/`)

| File | Change | Rerun |
|---|---|---|
| `6_inspect_decoupled_binding.ipynb` cell 15 | `lfsr_thresh=0.01` → `0.05` | no (now reproduces the shipped pickle) |
| `decoupling_supp_figures.py` L43 | `LFSR_THRESH = 0.01` → `0.05` | yes |
| `seq_replicate_qc/03_compare_replicate_fits.py` L17 (docstring), L62 | 0.01 → 0.05 | yes |
| `seq_replicate_qc/04_conditional_replicate_posteriors.py` L72 | 0.01 → 0.05 | yes |
| `bayes_fit_qc/qc_common.py` L33 | 0.01 → 0.05 | (via 01) |
| `bayes_fit_qc/01_convergence_diagnostics.py` L22 (docstring) | 0.01 → 0.05 | yes |
| `ww_normalization/gain_common.py` L39 | 0.01 → 0.05 | (via 03) |
| `ww_normalization/03_compare_current_vs_normalized.py` L23 (docstring) | 0.01 → 0.05 | yes |

Not changed: `seq_replicate_qc/04` L20 ("shift of < 0.01", not a threshold); axis tick
labels in `decoupling_supp_figures.py`; the secondary "supported" criterion
(lfsr ≤ 0.05, b > 0) in 03/04.

## Regenerated outputs

- `supp_plot_outputs/decoupling/`: all figures rewritten. Threshold-dependent:
  `figS_hit_selection_binding_lfsr{,_linear}.{pdf,svg,png}` (12,936 of 61,979 rows
  selected, was 9,966), `figS_example1_baseline_sets_calls.*`,
  `figS_example2_substrate_rank_flip.*`, `tableS_decoupling_per_pair_stats.csv`,
  `tableS_decoupling_rank_flips.csv` (857 substrates, was 842), `scores_with_naive.pkl`.
- `seq_replicate_qc/out/`: `conditional_summary.csv`, `conditional_pairs_*.csv`,
  `replicate_fit_*.csv`, `fig_conditional_replicates.*`, `fig_conditional_hits.*`,
  `fig_replicate_b_scatter.*`, `fig_replicate_calibration_hits.*`,
  `fig_replicate_vs_production.*`.
- `bayes_fit_qc/out/`: `chain_split_hits.csv` (Jaccard 0.97–0.98, was 0.95–0.96),
  `flagged_rhat_params.csv` (1 of 9 flagged parameters is now a hit),
  new `fig_chain_split_agreement.*`. `fig_convergence_rhat_ess.pdf`,
  `fig_hyperparameter_traces.pdf`, `fig_sampler_energy.pdf` were rewritten but render
  pixel-identical to before (threshold does not enter them).
- `ww_normalization/out/`: `comparison_summary.csv`, `comparison_per_kinase.csv`,
  `hit_changes_ww.csv`, `fig_ww_current_vs_normalized.*` (exploratory; not in manuscript).

## Documentation updated

- `supp_plot_outputs/decoupling/FIGURE_LEGENDS.md`, `DECOUPLING_NOTES.md`: threshold and
  all hit-dependent counts (12,936 hits; 3,175 (24.5%) naive-negative; 33 of 53 pairs;
  KGP1 269 of 427; PHKG1 532, PLK1 623 admitted; 2,343 confidently negative; 857 rank flips).
- `seq_replicate_qc/README.md`, `bayes_fit_qc/README.md`: new values.
- `ww_normalization/README.md`: new values. **Conclusion changed:** at 0.05 the current
  fit's replicate hit concordance already meets the model expectation (0.36 vs 0.34), so
  "normalization fixes replicate concordance" no longer holds; the held-out-prediction
  improvement is unaffected.
- `supp_S20-S24/supp_text_S20-S24.md` (this folder, plus copies in
  `ksi_analysis_2025/supp_S20-S24/` and `ksi_analysis_2025/supp_text_S20-S24.md`).

## Manuscript changes (`KSI_Paper.zip`, full diff in `lfsr0.05_tex_changes.patch`)

main.tex
- Results (benchmark paragraph): precision CI 0.87–0.96 and specificity 0.851 (0.75–0.92)
  added; "precision above 0.90 means a hit can be pursued with high confidence" replaced by
  an explanation that 0.924 is a benchmark operating point at 81.6% positives and does not
  give map-wide precision; Kinexus ~75% → 84.9% (45 of 53).
- Fig 5 caption: LFSR ≤ 0.01 → 0.05; 9,966 → 12,936 rows; "database spike-ins" →
  "screened pairs with curated database annotations (81.6% documented positives)" with the
  prevalence caveat; Kinexus ~75% → 84.9% (45 of 53).
- Discussion (limitations): Kinexus ~75% → 84.9% (45 of 53); precision qualified as
  specific to the benchmark's positive fraction.
- Methods (statistical decoupling): hit rule stated (b ≥ 0.1, LFSR ≤ 0.05 in at least one
  PAABD; used for every hit call).

supplementary.tex
- Library figure title/caption: "spike-in validation" → "database benchmark"; precision
  caveat added.
- Table S10 caption: renamed; composition (329/74, 81.6%), threshold, prevalence caveat,
  specificity; "over all spike-ins" → "over all documented interactions".
- Replicate note: grid-vs-NUTS hit Jaccard 0.97–0.98 → 0.98–0.99; replicate direction
  96.8/99.4/98.4/93.8% at 0.01 → 95.1/98.5/97.2/93.5% at 0.05.
- MCMC note: "none was a called hit" → one is (KGP1 × THEFKPQSGAEIKEG, WW; b 0.27,
  LFSR 0.014).

Figures in the zip
- `figures_supp/SuppFig_replicate_binding.pdf` replaced (= `seq_replicate_qc/out/fig_conditional_replicates.pdf`).
- Other SuppFigs unchanged (S22–S24 pixel-identical; the rest are not threshold-dependent).

## Confirmed: no main-figure pipeline was affected

The pickle was not modified, so every downstream analysis is bit-identical to before.
All of them take hits as `true_positive` straight from the pickle (b ≥ 0.1, LFSR ≤ 0.05)
and apply no threshold of their own, so a single study-wide 0.05 describes them correctly:

| Notebook | Feeds | Hit criterion |
|---|---|---|
| `9_screening_tp_to_reporter_candidates.ipynb` | reporter nomination (Fig 7/8) | `true_positive`, ranked by `binding_max` |
| `11_ksi_motif_analysis-stlib.ipynb`, `_ylib` | motif / substrate analysis | `true_positive` |
| `11_novel_ksi_analysis.ipynb` | novel KSIs | `true_positive` |
| `12_dark_kinase_analysis.ipynb`, `12_2_..._go_analysis.ipynb` | dark kinases (Fig 6) | `true_positive` |
| `13_kinbert_phosphoPCA_link.ipynb` | fine-tuning set (Fig 7) | `true_positive` |
| `11_ksi_tree_comparison_v2.ipynb` | kinase tree (Fig 3) | stricter `lfsr_best_hit < 0.001`, see below |

Before this change, LFSR 0.01 appeared only in model-QC scripts, the decoupling
supplementary figures and Fig 5 panel D — never in a biological result.

## Open issue, not caused by this change

`11_ksi_tree_comparison_v2.ipynb` cell 13, tyrosine arm:

```python
blib_screen_y['substrate'].isin(well_rep_y) & blib_screen_y['lfsr_best_hit']<0.001
```

`&` binds tighter than `<`, so this is `(isin & lfsr_best_hit) < 0.001`, which casts to
bool and inverts; it is exactly `~(isin & lfsr.astype(bool))`. It selects 11,657 rows
(1,936 substrates) of which only 727 are hits, instead of the intended 1,058 rows
(605 substrates, all hits). The S/T line below is correctly parenthesized and unaffected.
The 0.001 cutoff itself is a separate, deliberate choice and is not changed by this work.
Preliminary re-analysis only; nothing in the project has been edited.

## Manual step still needed

**Main Fig 5 panel D** still shows "lfsr ≤ 0.01" and the old counts. The figure is an
Illustrator composite (`figures/Fig 05 - MATH.pdf`, author Mariya Khan) and cannot be
regenerated by script. Replacement panel: `Fig05_panelD_update/Fig05D_hit_selection_lfsr0.05.{pdf,svg,png}`
(= `supp_plot_outputs/decoupling/figS_hit_selection_binding_lfsr.*`).

## Not touched

- `ksi_analysis_2025/manuscript_261014/` (dated snapshot of the 261014 manuscript).
- `review_notes/` inside the zip (still mention 0.01).
- `10_comparison_to_invitro_v2.ipynb` (hit definition `binding_max > 0` kept; 84.9% used).
