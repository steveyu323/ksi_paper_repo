# eval_tables

One CSV table for every figure in `../eval_figs`, holding the numbers that
figure plots.

Each table is produced by a script in `../eval_table_scripts`. The scripts
call the same functions the notebooks called, on the same inputs, so a table
cannot drift from its figure by construction. Every script also re-checks its
output against numbers recorded in the source notebook or embedded in the
figure file, and writes the result to a `VERIFICATION_*.csv`.

Run everything with:

```
python3 eval_table_scripts/run_all_tables.py
```

A full run takes about fifteen minutes, almost all of it loading the large
prediction pickles.

## Verification result

| | |
|---|---|
| Values checked against the figures | 392 |
| Values that match | 390 |
| Values that do not match | 2 |

Both mismatches are missing input data, not calculation errors. They are
described under "Known gaps" below.

### Two kinds of check

Most checks compare a regenerated table with numbers the source notebook
printed in the same cell that saved the figure. That proves the table and the
notebook agree.

`verify_bars_against_pdf.py` goes further for the two summary bar figures. It
measures them. Both are vector PDFs, so every bar is a rectangle with an exact
height and the axis ticks give the scale; bar colour identifies the model and
position identifies the facet and metric. All 70 bars across
`benchmark_model_structure_bar.pdf` and `benchmark_generalizability_bar.pdf`
agree with the tables to within 0.002, against an axis where one point of
geometry is about 0.0034. The three UpSet tables are read from their PDFs the
same way.

## Path translation

The notebooks were written on a machine where the project lived at `/ksi_ml`,
with an older copy at `/kinase_ml`. On this restored tree the same content
sits under one root, so `eval_table_utils.rp()` rewrites those prefixes:

| Notebook path | Restored path |
|---|---|
| `/ksi_ml/code/...` | `<root>/code/code/...` |
| `/ksi_ml/data/...` | `<root>/data/data/...` |
| `/ksi_ml/runs/...` | `<root>/runs/runs/...` |

## What each table contains

Benchmark figures come in bar and ROC pairs, so each gets three tables:

- `<name>_bar.csv` is the bar figure's own plotting frame, one row per model
  and metric.
- `<name>_roc.csv` is the ROC figure's own plotting frame, the false and true
  positive rate at every threshold, per model.
- `<name>_metrics_full.csv` is the complete metric set behind both, including
  the decision threshold and the confusion matrix.

## Figure to table map

### Data sources (`table_00_datasource_upset.py`, from `eval_0_datasource_stats.ipynb`)

| Figure | Tables | Status |
|---|---|---|
| `ks_pair_upset.pdf` | `ks_pair_upset_set_sizes.csv`, `ks_pair_upset_intersections.csv` | read from the figure |
| `ks_pair_upset_nkinase.pdf` | `ks_pair_upset_nkinase_*.csv` | read from the figure |
| `ks_pair_upset_norganism.pdf` | `ks_pair_upset_norganism_*.csv` | read from the figure |

These three are the exception to the rule above. The notebook builds them from
four per-database files under `/kinase_ml/data/raw/raw` plus an identifier
mapping table, and none of those survive on the restored tree. The closest
substitute, `interaction_classed_221017_wsource.pkl`, is the merged and
filtered union rather than the raw per-database sets, and gives visibly smaller
counts.

So the script decodes the figures instead. They are vector PDFs: every count is
real text, and every matrix dot records membership in its fill opacity. The
extraction is self-checking, because each source's set size must equal the sum
of the intersections that source takes part in, and for all three figures it
does. A `*_reconstruction_gap.csv` records how far the surviving data falls
short: 0 of 4 sources for the pair figure, 1 of 4 for kinases, 4 of 4 for
organisms.

### v1 models (`table_01_v1_models.py`, from `eval_1_v1_models.ipynb`)

| Figure | Tables | Status |
|---|---|---|
| `eval_v1_test_bar.png` | `eval_v1_test_bar.csv`, `eval_v1_test_metrics_full.csv` | 7 of 8 models |

This figure is not scored from saved predictions. It reads each training run's
metrics log, keeps the epoch with the best validation PR-AUC, and plots that
epoch's accuracy, ROC-AUC and PR-AUC. The seven runs that survive reproduce the
model ordering the notebook printed, exactly.

### Model architecture (`table_02_model_structure.py`, from `eval_2_v2test_dark_nonhuman.ipynb`)

| Figure | Tables | Status |
|---|---|---|
| `evalv2_test_bar.png`, `evalv2_test_roc.png` | `evalv2_test_*.csv` | exact |
| `evalv2_test2_bar.png`, `evalv2_test2_roc.png` | `evalv2_test2_*.csv` | exact |
| `eval_phosformer_train_bar.png`, `_roc.png` | `eval_phosformer_train_*.csv` | exact |
| `eval_phosformer_test_bar.png`, `_roc.png` | `eval_phosformer_test_*.csv` | exact |
| `benchmark_model_structure_bar.{png,pdf}`, `_roc.{png,pdf}` | `benchmark_model_structure_bar.csv`, `_roc.csv`, `_summary_wide.csv` | exact, and measured from the PDF |

Six models on four test sets. The combined figure facets those four side by
side, so its table is the four stacked, the way the notebook stacked them.
All 182 checked values match, including every figure in the notebook's own
summary tables.

The combined bar figure is checked three ways: each of its 48 bars is verified
individually through the four constituent benchmarks, its four facet means are
verified against the notebook's summary, and all 48 bars are then measured
straight off `benchmark_model_structure_bar.pdf` and compared with the table.

### Generalizability (`table_03_generalizability.py`, same notebook)

| Figure | Tables | Status |
|---|---|---|
| `eval_dark_test_bar.png`, `_roc.png` | `eval_dark_test_*.csv` | exact |
| `eval_nonhuman_test_bar.png`, `_roc.png` | `eval_nonhuman_test_*.csv` | exact |
| `eval_yeast_bar.png`, `_roc.png` | `eval_yeast_*.csv` | exact |
| `benchmark_generalizability_bar.{png,pdf}`, `_roc.{png,pdf}` | `benchmark_generalizability_bar.csv`, `_roc.csv`, `_summary_wide.csv` | exact, and measured from the PDF |

Each holdout asks whether a model that never saw a class of kinase in training
still scores it. The ablated models appear only in their own holdout, which is
why the wide summary has blanks.

The yeast benchmark is built by two notebooks. The figure now in `eval_figs` is
the three-model version from this notebook, which is what the table reproduces.
The four-model version in `eval_3_yeast_human_benchmarks.ipynb` was overwritten.

### KSI screen (`table_04_ksi_screen.py`, from `eval_3_yeast_human_benchmarks.ipynb`)

| Figure | Tables | Status |
|---|---|---|
| `eval_ksi_bar.png`, `eval_ksi_roc.png` | `eval_ksi_*.csv` | exact |

The one benchmark that includes the geometry-augmented model.

### Virus, curated (`table_05_virus_demo.py`, from `eval_6_..._virus_demo.ipynb`)

| Figure | Table | Status |
|---|---|---|
| `eval_virus_dms_kinase_psite_demo.{png,pdf,svg}` | `eval_virus_dms_kinase_psite_demo.csv` | exact |
| `eval_virus_dms_drug_kinase_demo.{png,pdf,svg}` | `eval_virus_dms_drug_kinase_demo.csv` | axis pinned to the figure |
| `eval_virus_dms_drug_psite_demo.{png,pdf,svg}` | `eval_virus_dms_drug_psite_demo.csv` | exact |

Three heatmaps over a hand-curated set of viral phospho-sites. The pipeline
enters at the cached `250118_virus_against_kinome_hits.pkl` rather than at the
raw model output, deliberately: re-deriving that file changes 28 of its 96 rows,
because the top-ten-kinase step sorts on logits full of ties with a non-stable
sort. The three external drug databases the notebook merges are absent, but
their only product is `drug_kinase_uniprot.pkl`, which is cached here and
reproduces the notebook's recorded output exactly.

One input deserves care: `260119_demo_kinase_virus_drug_pairs.csv` is the
plotting frame of the drug-by-site figure, captured just before the grid is
completed. Nothing in the repository writes it, and nothing regenerates it, so
it is the only surviving copy. Recomputing it from the pipeline yields the same
rows but in a permuted order, which would permute the figure's x axis, so the
script reads the file rather than rebuilding it. Do not delete it.

The drug-by-kinase figure has a second such sort, on a column where most values
tie, which decides its drug axis. A free re-run recovers 28 of the figure's 44
drugs. The table therefore pins the axis to the 44 drugs the figure shows,
recovered from its vector artifact and the exported pair list; the free re-run
is kept alongside as `..._free_rerun.csv` so the drift stays visible. All 25
checks on the pinned grid pass, including its full XL820 column.

### Virus, full kinome (`table_06_virus_full.py`, from `eval_6_..._virus_full.ipynb`)

| Figure | Table | Status |
|---|---|---|
| `eval_virus_dms_kinase_psite_fulldrug.png` | `eval_virus_dms_kinase_psite_fulldrug.csv` | exact |
| `eval_virus_dms_drug_kinase_fulldrug.png` | `eval_virus_dms_drug_kinase_fulldrug.csv` | approximate |
| `eval_virus_dms_drug_psite_fulldrug.png` | `eval_virus_dms_drug_psite_fulldrug.csv` | approximate |

The same three heatmaps without the curation step. The first regenerates
exactly, at the recorded 262 kinases by 67 sites. The other two cannot: the
notebook's last run used a drug-target table that is not on disk, and the
cached version places Seliciclib two rows off the recorded position in one
direction while an older cached variant misses in the other. Every logit and
probability still matches; the drift is confined to which drugs land on the
axis, and shows up as 10,680 cells against the recorded 10,613.

## Known gaps

Two checked values do not match, both because an input is missing.

1. **`eval_v1_test_bar.png` is missing one of eight bars.** The run directory
   `exp4_esm2t12_unfrozen_xattn`, behind DB-ESM2-T12-Unfrozen-XAttn, is not on
   the restored tree. A directory named `exp4_esm2t12_unfrozen_xattn_v2` does
   exist, but it is a different experiment tag and is deliberately not
   substituted. The row is emitted with empty metrics and an `available` flag.

2. **`eval_virus_dms_drug_psite_fulldrug.png` regenerates 10,680 cells against
   the recorded 10,613.** Cause described in the section above.

## Stale figures

`eval_v2_test_bar.png` and `eval_v2_test_roc.png` have no table. Nothing in the
repository writes those filenames, including the checkpoint copies of the
notebooks. They are January artifacts superseded in February by
`evalv2_test_bar.png` and `evalv2_test_roc.png`, which cover the same benchmark
and do have tables.
