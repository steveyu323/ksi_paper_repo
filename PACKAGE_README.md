# KSI paper — code and data package

Non-redundant code and data for the **final** manuscript,
`KSI_Paper_261022_CY/` as of 2026-10-23: 8 main figures, 12 supplementary
figures (S1-S12), 13 supplementary tables (S1-S13). The figure PDFs and `.tex`
in `figures/` are byte-identical to that manuscript, and
`figures/CHANGES_261022.md` is its revision log. Superseded 2026-10-14 figure
PDFs are kept in `_superseded_261014/` for diffing only.

`INVENTORY.md` is the authoritative map: figure panel → code chain → raw data,
plus the blockers. Read it first. This file covers layout, provenance of the
sources, and what still has to happen before a Zenodo deposit.

## Layout

```
code/01_phosphoPCA/     pooled phospho-PCA screen (Fig 01, 05, 06; most supp figs)
code/02_kinbert/        KINBERT training, benchmarking, prediction (Fig 02, 07A; supp)
code/03_spark/          SPARK imaging + reporter nomination (Fig 07B/C, Fig 08)
code/04_variant_viral/  pathogenic-variant and viral downstream (Fig 03, Fig 04)
code/common/            make_manifest.py
code/verification_261022/  the numeric audit behind the corrected caption and
                        text numbers (scripts + outputs + FINDINGS.md)
data/raw/               symlinks to raw inputs, by section
data/results/           result tables and checkpoints destined for Zenodo
data/manifests/         resolved path, size, mtime (+ optional sha256) per file
figures/                the figure PDFs as they appear in the manuscript, plus the .tex
```

Each `code/*/` holds `_source_notebooks/` and `_source_scripts/` — the
de-duplicated **final** versions of the original notebooks, kept unchanged as
the provenance record — alongside cleaned pipeline scripts where those exist.

## Source trees

The work spans four trees, not the three originally identified. Figures 03 and
04 come from `kinase_ml/`, which was not in the original list.

| Section | Tree |
|---|---|
| phospho-PCA | `${KSI_REPO_ROOT}/ksi_analysis_2025` |
| KINBERT | `${KSI_ML_ROOT}` (**live, canonical**) + `${KSI_RESTORE_ROOT}` |
| SPARK | `${KSI_REPO_ROOT}/cellsam-internal` |
| Variant + viral | `${KSI_REPO_ROOT}/kinase_ml` |

### The two KINBERT trees — neither is a superset

This caught us out once, so it is worth stating plainly. The restored tarball
tree is **not** the final state of the code; its evaluation layer froze around
2026-02/03, while the live `ksi_ml` tree continued to 2026-06.

| Only in the live tree | Only in the tarball tree |
|---|---|
| final eval notebooks (`eval_2_v2test_dark_nonhuman.ipynb` 2026-06-14, 1,061,038 B vs 600,390 B) | `biosensor_scoring/` (2026-09-17, the Fig 07B re-scoring harness) |
| `eval_7_separate_tests.ipynb` + the LinkPhinder benchmark tables and figures | `eval_tables/` and `eval_table_scripts/` (**absent from the live tree entirely**) |
| `hf_kinbert/` (the published model) | |
| `eval_figs/kinbert_performance_*_thesis_pre.*` (likely the real Fig 02C panel) | |

This package takes the eval notebooks and LinkPhinder artefacts from the live
tree, and `eval_tables/` plus `biosensor_scoring/` from the tarball tree.

Ignore the `*.tar` archives in the tarball tree (22 + 61 + 17 + 54 GB); the
extracted directories hold the same content. Note the extracted dirs are
nested one level deeper (`tarballs/code/code/`, `tarballs/runs/runs/`, and
`training_data/` three deep).

## Raw data

Staged as symlinks plus a manifest, not copied. Regenerate with:

```
python3 code/common/make_manifest.py .
python3 code/common/make_manifest.py . --checksum    # once, just before upload
```

Current totals:

| Tree | Section | Files | Size |
|---|---|---|---|
| raw | phosphoPCA | 268 | 101.2 G |
| raw | kinbert | 107 | 3.0 G |
| raw | spark | 533 | 2.8 G |
| raw | variant_viral | 618 | 177.9 M |
| results | kinbert | 116 | 2.4 G |
| results | phosphoPCA | 105 | 133.0 M |
| results | variant_viral | 9 | 83.2 M |
| results | spark | 9 | 38.6 M |

`data/raw/` deliberately excludes things that are large and either
regenerable or out of scope: PacBio `subreads.bam`/`scraps.bam` (~960 G),
`pairs_aug_ptmvar_flank.csv` (40 G), the `*_withmif_*` structure-featurized
training sets (26 G), `SARS2-mut-fitness/` (239 M, read by no figure code),
and the `.nc` posterior traces (10.6 G — summarise instead).

## Verifying that code + data still give the printed results

```
python3 code/common/verify_expected_values.py . \
        --outputs /path/to/ksi_analysis_2025
```

Checks the canonical hit counts against the staged pickles and compares every
script-generated supplementary figure with the shipped PDF. Current status:

```
15 passed, 0 failed, 3 skipped
```

The 15 passes are the six hit-calling / Table S10 values (61,979 rows;
12,936 called hit; 8,890 `positive`; 8,100 `true_positive`; 134/11/195/63;
403 total) and the nine reproducible supplementary figures, six of them
byte-identical and three identical in the text layer with only PDF metadata
differing. The 3 skips are S2, S3 and S4, which have no generator.

If "substrate x kinase x PAABD rows called hit" ever returns 9,966, the
`LFSR_THRESH` in `code/01_phosphoPCA/config.py` has reverted to 0.01; the
study-wide value is **0.05**.

## A partial Zenodo deposit already exists

`${KSI_DATA_ROOT2}/ksi_paper_zenodo/` was started on 2026-04-16. **Extend
it rather than rebuilding**, but note it is substantially incomplete and
internally inconsistent:

- `kinbert_training/` — 4 pair tables (1.28 G). Missing all splits, both
  corpus pickles, every checkpoint, the `eval_outputs` subset and `eval_tables/`.
- `long_read_seq/` — 2 files, in **mismatched formats**: the Tyr library as a
  demultiplexed `.fastq.gz` but the Ser/Thr library as a CCS `.fasta.gz`.
  Deposit both libraries in the same format.
- `short_read_seq/` — only **7 of 32** fastp-merged FASTQs. Present: 3 of the 8
  replicate-1 variant samples (26227–26229) and the 4 replicate-1 FP controls.
  Missing: 26230–26234, the rest of the 231217 run, and **all of replicate 2**
  (both the 240424 and 240508 runs).

## Library naming — read this before touching the lookup tables

The `blib` numbers mean **different things on the two sequencing platforms**,
because the PacBio tubes were labelled in reverse. Getting this wrong
cross-wires the Tyr and Ser/Thr maps.

| Illumina (`3_integrate_lfc_lookup.ipynb`) | PacBio (`1_parse_lookup_tables.ipynb`) |
|---|---|
| blib1 → FHA, Ser/Thr | blib1-ksb.CY → **Tyr** lookup ("labeled reversed so Y") |
| blib2 → 14-3-3, Ser/Thr | blib5-ksb.CY → **Ser/Thr** lookup ("labeled reversed so ST") |
| blib3 → WW, Ser/Thr | |
| blib5 → Src-SH2, **Tyr** | |
| blib7, blib10 → substrate-only FP controls | |

The supplementary table's "ST (blib5) / Y (blib1)" labels follow the PacBio
convention and are correct for the lookup tables.

## Third-party code and models — cite, do not redistribute

- `cellsam-internal` is a vendored clone of
  `github.com/vanvalenlab/cellsam-internal` at commit
  `1253ff67bb9dd4cf1c4350f63d546f3b3de6c534`, and it in turn vendors Meta's
  `segment_anything/` and `AnchorDETR/`. Strip `.git/` if any of it ships.
- SPARK segmentation weights `mixed_model_inference_balanced.pth` (784 MB) are
  distributed out-of-band by the upstream project, so they are not ours to
  redistribute. Thresholds used: `bbox 0.38095 / mask 0.07921 / iou 0.47619`.
- Baselines: Phosformer (env file vendored), Kolossus (**no env file** — only
  hash-named run dirs), LinkPhinder / Novacek et al. 2020.
- MAGeCK v0.5.9.5; PyMC 5.20.1 / pytensor 2.27.1; PKIDB release 2025-04-15.

## Environments

| Stage | Env |
|---|---|
| phospho-PCA decoupling + QC | `bayes_fit_qc/README.md`: python 3.11, pymc 5.20.1, pytensor 2.27.1, arviz<0.22, matplotlib 3.9 |
| reads → counts | MAGeCK v0.5.9.5 + fastp (version not retained in the HTML reports) |
| KINBERT 2024-era training | `code/02_kinbert/requirements_2024_training.txt` + `Dockerfile_2024_training` |
| KINBERT 2025–26 runs | **no pinned env exists.** Logs show python 3.11 / torch ≥2.x / bf16 |
| virus DMS prefs | `code/04_variant_viral/requirements_dms.txt` (needs `dms_tools2`) |
| Fig 04C plotting | needs `plotnine`, which no requirements file lists |

## Known code defects

Verified directly in the source, not inferred. Each will bite someone
re-running this.

1. `binary_tape_eval_kinbertv2_VSHumanKinome.ipynb` loops
   `for i in range(len(dat_tags)-1)` over
   `["virPTM","influenza","covid-psite1","covid-psite-2","PTMVar"]`, so the
   **PTMVar leg never runs**. Its v2 output is 172 KB against 42 MB for the v1
   equivalent. Use the **v1** `ptmvar_logits_{1,2}.pkl` for Fig 03.
2. `collect_and_inspect_prefs.ipynb` reads `BF520_rhesus_prefsdf` from
   `results/human_prefs/summary_avgprefs.csv` — the *human* file. The
   `rhesus_prefs/` directory exists and is described in the notebook's own
   markdown but never read, so the `BF520_rhesus` rows are duplicates of
   `BF520_human`. Affects the HIV rows of Fig 04A/C.
3. `230105_link_dms_inference_to_drug_dbs.ipynb` has **no `savefig`/`ggsave`**
   (0 occurrences); Fig 04C exists only as inline notebook output and must be
   re-rendered.
4. `eval_2_finetune_phosphoPCA.ipynb` has **all 6 save calls commented out**,
   in the newest copy too. SuppFig_finetuning / Fig 07A has no saved figure and
   no `eval_tables` CSV — only stored cell output. Same pattern in
   `eval_7_separate_tests.ipynb`.
5. `virus_predictions_evaluation.ipynb` calls `savefig('SRPK2.png')` three
   times with the same filename, so only the last of three panels survives.
6. `spark_overlay_patched.ipynb` cell 32 writes an `IKK-SPK-1` overlay to
   `spark_IKKnovel_overlay.pdf`.
7. `mapped_PTM_var.ipynb` and the SPARK notebooks hard-code the dead
   `${SPARK_ACQUISITION_MOUNT}/...` mount; phospho-PCA notebooks 2, 3 and 7 hard-code the
   container path `${DATA_MOUNT}/...`.
8. `model.py:108` hard-codes another user's `~/.cellsam_assets/...`.

## Figures with no generating code

Hand-drawn or digitized; state the provenance rather than implying they are
reproducible. Keyed on **printed** figure number:

- Fig 1 (whole), the Fig 7A and Fig 8A pipeline schematics, the Fig 5C
  pipeline schematic and the Fig 6C construct schematics.
- **All eight main figures are Illustrator composites.** Panels are
  code-derived; assembly, labelling and layout are not. No main figure is
  regenerable by running a script.
- `SuppFig_{benchmark,generalizability,finetuning}` (= S2, S3, S4) were
  redrawn by hand from Tables S4-S6. Their values are all in
  `eval_tables/*.csv`, so writing `sf2`/`sf3`/`sf4` in the idiom of
  `SF1_SF5_SF6_SF7/sf1_corpus.py` would take the package from 9/12 to 12/12
  reproducible supplementary figures.

**No longer in this list:** `SuppFig_{corpus,library,falsepositive,decoupling}`
(= S1, S5, S6, S7) are now generated by `SF1_SF5_SF6_SF7/`, and the shipped
PDFs are byte-identical to the script output.

## Does not exist and must be made

- ~~**A curated variant table**~~ -- **closed.**
  `code/code/evaluation/260130_ptmvar_nr_finalized.csv` is exactly that table:
  65 rows (40 on-phosphosite + 25 flanking kinase-site pairs, 20 and 17
  distinct dbSNP variants), with `tier`, `dbSNP`, `Clinical_Significance`,
  `Disease/Phenotype`, `justification`, `literature` and `direct_quote`
  columns. It is the node/edge list behind Fig 7B, and the final manuscript
  carries no separate variant supplementary table, so nothing is missing.
- **The Fig 03A filters** (`wt logit ≥ 5`, selectivity, substrate-share) and
  the `Δlogit > 2.5` call, plus the OmniPath / PhosphoNetworks / STRING and
  dbSNP / ClinVar cross-validation. No code.
- **`covid_psite_*_min.pkl` / `influenza_psite_cleaned.pkl` prep** from the
  published CSVs. No code.
- **The SARS-CoV-2 logits → PKIDB/DrugBank join** behind the Fig 04B/C covid
  rows. Unscripted, likely done by hand.
- **A v2 drug-annotated table.** `matched_drug` exists only on the v1 scores,
  while the plotted matrix is built from v2; the v2+drug table is never
  persisted.

## Numeric discrepancies to resolve

- SuppFig_generalizability caption says KINBERT:TAPE ROC-AUC **0.754** on the
  phospho-PCA holdout; `eval_tables/eval_ksi_bar.csv` says **0.7585**. Related:
  `eval_tables/` was regenerated from the February notebooks, so it may be one
  version behind the June figures, and the live notebook's facet list no longer
  includes `eval_ksi`. Re-check before deposit.
- Spike-in counts (616 vs ~300, of which 165 are documented non-substrates) and
  constructed-vs-screened KSI counts (25,214 / 14,873 vs 22,695 / 13,220).

Two conflicts listed in `tableS13_substrate_library/README.md` were checked
against this manuscript and **do not apply** — the substrate count is
self-consistent (1,978 + 1,950 = 3,928) and the "84.9% of 45 pairs" problem
belongs to an earlier draft; the current text says ~75% throughout. Details in
`INVENTORY.md`.
