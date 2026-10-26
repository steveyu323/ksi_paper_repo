# ksi_paper_repo

Analysis code for the KSI manuscript (KINBERT + phospho-PCA). This is the
**code half** of the reproducibility package; the raw and processed data are
deposited separately (Zenodo), and the manuscript sources are not here.

## Layout

```
code/01_phosphoPCA/        pooled phospho-PCA screen: reads -> counts -> decoupling -> hits
code/02_kinbert/           KINBERT training, benchmarking, prediction, biosensor scoring
code/03_spark/             SPARK imaging analysis and reporter nomination
code/04_variant_viral/     pathogenic-variant and viral host-kinase downstream
code/verification_261022/  numeric audit behind the manuscript's corrected values
code/common/               manifest builder and the reproducibility harness
```

`INVENTORY.md` is the map: every printed figure, supplementary figure and
supplementary table, keyed on its **printed** number, with the code and data
behind it. Read it first. `PACKAGE_README.md` covers environments, source-tree
provenance and known code defects.

## What is and is not reproducible

| | Status |
|---|---|
| 9 of 12 supplementary figures | regenerate **byte-identically** from the scripts here |
| Supplementary Figures S2, S3, S4 | no generator — redrawn by hand from Tables S4–S6 |
| All 8 main figures | Illustrator composites. Panels are code-derived; assembly is not |

Check it yourself, against a copy of the data:

```
python3 code/common/verify_expected_values.py . --outputs /path/to/ksi_analysis_2025
```

It re-derives the printed hit counts and diffs each script-generated figure
against the shipped PDF. Expected: `15 passed, 0 failed, 3 skipped`.

## Notes for anyone re-running this

- No lab filesystem path is hard-coded anywhere here. **`PATHS.md` is the
  reference**: which environment variables the code reads, and which `${VAR}`
  strings are inert placeholders in the archived notebooks that you must
  substitute by hand before re-running them.
- `PATHS.md` also explains why `sf5_library.py` and `sf6_false_positive.py`
  cannot be run from where they sit in this repo, and how to run them.
- The study-wide hit threshold is **b ≥ 0.1, LFSR ≤ 0.05**. If a rerun reports
  9,966 selected rows instead of 12,936, `LFSR_THRESH` has reverted to 0.01.
- `_source_notebooks/` and `_source_scripts/` are the provenance record, kept
  as they ran except that lab filesystem paths were replaced with `${VAR}`
  placeholders; the cleaned, parameterised versions sit alongside them.
- Third-party code is **not** vendored here. CellSAM, its segmentation weights,
  Phosformer, KolossuS and LinkPhinder are cited, not redistributed.
