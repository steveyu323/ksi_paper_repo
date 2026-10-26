# Supplementary Figures S5–S7: reproducible rebuilds

The S5–S7 PDFs in `manuscript/261014/KSI_Paper.zip` (`figures_supp/SuppFig_{library,falsepositive,decoupling}.pdf`)
were redrawn from the thesis tables and figures, and two were recovered from vector geometry, so no code produced them.
These scripts rebuild each one from the primary data. They reuse the logic of the notebooks that
made the originals.

| Fig | Script | Output | Original source in `ksi_analysis_2025/` |
|---|---|---|---|
| S1 | `sf1_corpus.py` | `out/SuppFig_corpus.pdf` | a, b: `eval_0_datasource_stats.ipynb` cells 18–23 (KINBERT row recomputed from `pairs_aug_v{1,2}.csv` with the 2,304 → 2,021 kinase de-duplication fix from `audit_01_corpus_stats.py`; DeepKinZero/Phosformer are published values). Resolves X1 of `KSI_Paper_261022_CY/CHANGES_261022.md` |
| S5 | `sf5_library.py` | `out/SuppFig_library.pdf` | a: `1_parse_lookup_tables.ipynb` cells 2, 6 (QC read counts). b: `8_2batch_blib_pipeline_v2.ipynb` cell 18 (`stats_df`). c: same notebook, cells 24–33 (`confusion_matrix_summary.csv`) |
| S6 | `sf6_false_positive.py` | `out/SuppFig_falsepositive.pdf` | a, b: `7_identifiy_false_positives.ipynb` cell 2 → `supp_plot_outputs/lfc_distribution.pdf`. c, d: cells 8, 11, 36 → `supp_plot_outputs/fp_identification_panels.pdf` |
| S7 | `sf7_decoupling.py` | `out/SuppFig_decoupling.pdf` | a, b: `6_inspect_decoupled_binding.ipynb` cell 11 → `supp_plot_outputs/fig1_activity_abundance.pdf`. c: cell 12 → left heatmap of `supp_plot_outputs/fig2_abundance_consistency.pdf` |

Run each script with `python <script>` from any directory, using the base miniconda env
(pandas 2.3, matplotlib 3.10, seaborn 0.13, scipy, biopython). Each script also writes the
plotted values to `out/source_data/`.
- S5 panel a reads about 5.4 M PacBio reads, which takes about 6 min and needs 2.4 GB of RAM. The counts are cached in
  `out/source_data/sf5a_ccs_qc_counts.csv`; pass `--recount` to rebuild them.

## Agreement with the manuscript

- **S5**: every number matches Tables S8–S10 exactly. Pair attrition is 23.5 % for ST and 15.3 % for Y.
- **S6**: the curve sits at 26.9 % at its lowest and 75.8 % at LFC = 1.5, which matches the figure's "76 %".
  The manuscript caption says **"46 of 2754 substrates"**, but the true count is **45 of 2753**. The vector recovery
  counted the legend swatch of the original PDF as a data point.
  The volcano is coloured with the cell-11 promiscuity table. That is the S/T-kinase promiscuity of the control
  substrates flagged at LFC > 1.5 and FDR < 0.1, and it is the only table that reproduces the shipped PDF. In the
  notebook, cell 36 picked up whichever `fp_stats` was in memory.
- **S7**: 43 of the 45 median abundances match. IKKB FHA (−0.85 vs −0.86) and PDPK1 WW (−0.38 vs −0.39)
  differ by 0.01, from reading values off the vector geometry. The rebuild takes them straight from the posteriors.
