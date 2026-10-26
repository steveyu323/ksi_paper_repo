# 01_phosphoPCA — pooled phospho-PCA screen

Reproducibility package for the phospho-PCA screen: barcoded kinase–substrate
fusions selected in yeast on methotrexate, sequenced before and after
selection, and turned into per-pair binding scores by a hierarchical
negative-binomial model that separates kinase abundance from substrate
selectivity.

Everything is driven by `config.py`, which holds every input and output path
and every analysis constant. Set these environment variables (defaults in
`config.py`; see its docstring for the full description):

| variable | what it points at | default |
|---|---|---|
| `KSI_DATA_ROOT` | sequencing + PacBio data root (the container's `${DATA_MOUNT}/<user>`) | *no usable default - set it* |
| `KSI_REPO_ROOT` | analysis tree holding `ksi_analysis_2025/`, `kinase_ml/`, `screening_analysis/` (read only) | *no usable default - set it* |
| `KSI_COUNT_DIR` | MAGeCK count/test outputs | `$KSI_DATA_ROOT/processed_count_table` |
| `KSI_OUT_DIR` | everything this pipeline writes | `<package>/data/results/phosphoPCA/rerun` |
| `KSI_FIG_DIR` | figure outputs | `$KSI_OUT_DIR/figures` |

`../../data/results/phosphoPCA/` is the deposited-results view: a flat
directory of symlinks into the original analysis directory. It is an input, not
an output. **Do not set `KSI_OUT_DIR` to it, or to `ksi_analysis_2025`** — a
stage would then write through those symlinks and overwrite the deposited
results. `cp -L` the inputs you need into a fresh `KSI_OUT_DIR` instead.

Two environments are needed and they are not interchangeable — see
`requirements.txt`. The short version: stages 01–04, 06–07 and the figure
scripts run in the python 3.11 analysis environment; **stage 05 and both QC
folders need the PyMC environment** (`pymc 5.20.1`, `pytensor 2.27.1`,
`arviz<0.22`), which resolves to numpy 1.26 and therefore cannot read the
numpy-2 pickles the other environment writes. That is why stage 05 and the QC
scripts only ever read `.nc` and `.csv`.

## Pipeline

Run in order. Times are rough, on 4–8 cores.

| stage | reads | writes | cost |
|---|---|---|---|
| `00_reads_to_counts.sh` | Illumina FASTQ of the four sequencing runs | `$KSI_COUNT_DIR/*.count.txt` (de-novo MAGeCK counts), then the per-sublibrary `mageck test` results `*_rra.{gene,sgrna}_summary.txt` | hours, I/O bound. Needs `fastp` + `mageck` 0.5.9.5, not python |
| `01_parse_lookup_tables.py` | PacBio HiFi FASTA of the two plasmid pools; `blib_kinases_with_acc.csv` | `lookup_tables/{y,st}_pacbio_lookup_raw.pkl`, `{y,st}.lookup.corrected.pkl`, `blib_kinases_with_acc.csv` | ~10 min, ~16 GB (5.4 M long reads held as dataframes). `--skip-parse` reuses the raw pickles |
| `02_generate_lfc_table.py` | the raw MAGeCK count tables | per-sublibrary, per-batch count tables and the two-batch `bliballvariant12*`/`blib_fpb12*` tables | ~10 min, ~16 GB |
| `03_integrate_lfc_lookup.py` | `*_2batch_rra.{gene,sgrna}_summary.txt`, the two lookup tables, the two-batch counts | `dat_all_blib_2batch.pkl` / `.csv` | ~3 min, ~4 GB |
| `04_filter_unmapped_substrates.py` | `dat_all_blib_2batch.csv`, `substrates_background_12345.pkl` | `decoupling_out/raw_screen_count.pkl`, `raw_screen_count_mapped.pkl` / `.csv` | < 1 min |
| `05_fit_decoupling_model.py` | `decoupling_out/raw_screen_count_mapped.csv` | `decoupling_out/251125_pymc_{paabd}.nc`, `_{paabd}.csv`, `_binding_stats_{paabd}.csv` | **PyMC environment.** ~1–1.5 h per pAABD (4 chains in 4 processes), 4 pAABDs; ~16 GB; each `.nc` is ~3 GB. `--stats-only` re-derives the CSVs from saved posteriors |
| `06_call_hits.py` | the four fits' CSVs, `raw_screen_count_mapped.pkl` | `251130_ksi_inference_scores.pkl`, `251203_ksi_reduced_df.pkl` | ~2 min |
| `07_filter_false_positives.py` | `251203_ksi_reduced_df.pkl`, `blib{7,10}_2batch_rra.sgrna_summary.txt` | **`251203_ksi_reduced_df_tp.pkl`** (the canonical hit table), `fp_substrates.csv`, `fp_promiscuity_sweep.csv` | ~2 min. `--skip-sweep` drops the figure input |

`decoupling_model.py` holds the model itself (priors, centring, NUTS settings,
seed) and is imported by stage 05 and by every QC refit, so none of them can
drift from the fit that produced the deposited posteriors.

What each stage is for, in one line each: 01 assigns every barcode a
(kinase, substrate) identity from long reads; 02 sanitises the de-novo count
tables and merges the two sequencing batches; 03 joins enrichment to identity;
04 drops substrates with no protein identity; 05 splits enrichment into a
kinase-global term and a kinase-centred binding score; 06 calls hits at
lfsr ≤ 0.01 and b ≥ 0.1; 07 removes substrates that grow without a kinase.

## Figures

All under `figures/`. Each script takes `--help`.

| script | produces | manuscript item | cost |
|---|---|---|---|
| `fig05_math.py --panel b` | `mock_boltzmann_to_logfc.{png,pdf}` | **Fig 05** panel B | seconds |
| `fig05_math.py --panel c` | `decoupling/figS_hit_selection_binding_lfsr.*` | **Fig 05** panel C | ~1 min |
| `fig05_math.py --panel d` | `fig05D_confusion_matrix_{y,st,sty}.pdf`, `fig05D_confusion_summary.csv`, `fig05D_fn_decomposition.csv`, `fig05D_invitro_pie.pdf`, `fig05D_invitro_overlap_summary.csv` | **Fig 05** panel D; the FN decomposition is the spike-in performance table (`tab:spike`) | ~5 min, ~8 GB (loads the 1.6 M-row annotation tables) |
| `fig05_math.py --library-stats` | `fig05A_library_stats.csv` | the numbers on **Fig 05** panel A, and `tab:libdesign` | ~2 min |
| `fig06_biological_insights.py --panel a-trees` | `fig06A_screening_tree.newick`, `fig06A_kinase_seq_tree.newick`, `fig06A_aligned_dendrograms.pdf` | **Fig 06** panel A | ~2 min; needs `mafft` on `PATH` |
| `fig06_biological_insights.py --panel a-cross` | `fig06A_cross_tree_scatter.pdf`, `fig06A_cross_tree_results.csv` | **Fig 06** panel A annotations (conservation / divergence / convergence) | ~2 min; needs `ete3` |
| `fig06_biological_insights.py --panel b-motifs` | `fig06B_IC_heatmap_{aa,prop}_{kinase}.png`, `fig06B_IC_matrix_*.csv` | **Fig 06** panel B, left | ~1 min |
| `fig06_biological_insights.py --panel b-go` | `fig06B_go_{kinase}.png`, `fig06B_go_results.csv` | **Fig 06** panel B, right | ~3 min |
| `suppfig_falsepositive.py` | `suppfig_fp_lfc_distribution.pdf`, `suppfig_fp_identification_panels.pdf` | **SuppFig_falsepositive** | < 1 min |
| `decoupling_supp_figures.py` | `decoupling/figS_*` and two tables | **SuppFig_decoupling** (`figS_decoupling_schematic`), plus the per-pair pre/post histograms and `tableS_decoupling_per_pair_stats.csv` | ~3 min |

Shared, imported, not run directly: `information_content_calculator.py`
(positional information content) and `mock_boltzmann_to_logfc.py` (which also
runs standalone).

### QC folders

`figures/bayes_fit_qc/` and `figures/seq_replicate_qc/` are kept as the scripts
they already were; each has its own README with the results in full. Only
their path constants were changed, to come from `config.py`, and
`seq_replicate_qc` now imports the shared `decoupling_model.py` instead of its
own copy. **Both need the PyMC environment.**

| script | produces | manuscript item | cost |
|---|---|---|---|
| `bayes_fit_qc/01_convergence_diagnostics.py` | `fig_convergence_rhat_ess`, `fig_hyperparameter_traces`, `fig_sampler_energy`, `fig_chain_split_agreement`, 5 CSVs | **SuppFig_convergence**, **SuppFig_hyperparameters**, **SuppFig_sampler** | ~15 min, **~40 GB RAM** — it loads the full `b` posterior draws |
| `seq_replicate_qc/01_raw_replicate_concordance.py` | `raw_concordance_*.csv`, `fig_raw_*` | **SuppFig_replicate_counts** | < 1 min |
| `seq_replicate_qc/02_refit_per_replicate.py --all` | `out/refits/{paabd}_rep{r}_*` | input to `03` | ~50 min on 32 cores (8 NUTS fits in parallel) |
| `seq_replicate_qc/03_compare_replicate_fits.py` | `replicate_fit_*.csv`, `fig_replicate_*` | independent-refit comparison | < 1 min, needs `02` |
| `seq_replicate_qc/04_conditional_replicate_posteriors.py` | `conditional_summary.csv`, `conditional_pairs_*.csv`, `fig_conditional_replicates` | **SuppFig_replicate_binding** | ~5 min |

## Provenance

* `_source_notebooks/` — the 18 notebooks this code was extracted from,
  unchanged. `_source_notebooks/NOTES.md` says what became of each one.
* `_source_scripts/` — the standalone scripts as they were found, including the
  original `bayes_fit_qc/`, `seq_replicate_qc/` (with their result `out/`
  directories) and two further self-contained folders that were left in place
  because they are already clean and have their own README and config:
  * `tableS13_substrate_library/` — builds the candidate substrate-library
    manifest (`tab:substrates`); has its own `config.py` keyed on
    `KSI_REPO_ROOT`.
  Also here: `dockerfiles/` (the three container definitions behind
  `requirements.txt`), `check_selection_per_paabd.py`, `plot_ksi_heatmaps.py`,
  `reporter_panel.py`, and the `plot_/validate_table2_1_training_data.py` pair,
  which belong to the KINBERT and SPARK sections rather than to this pipeline.
