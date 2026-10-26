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

## Notes for anyone re-running this

- No lab filesystem path is hard-coded anywhere here. **`PATHS.md` is the
  reference**: which environment variables the code reads, and which `${VAR}`
  strings are inert placeholders in the archived notebooks that you must
  substitute by hand before re-running them.
- `PATHS.md` also explains why `sf5_library.py` and `sf6_false_positive.py`
  cannot be run from where they sit in this repo, and how to run them.
- The study-wide hit threshold is **b ≥ 0.1, LFSR ≤ 0.05**. 
- Third-party code is **not** provided here. CellSAM, its segmentation weights,
  Phosformer, KolossuS and LinkPhinder are cited, not redistributed.
