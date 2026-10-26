# Paths

Nothing in this repo contains a lab filesystem path. Every absolute path that
used to be hard-coded is now either read from an environment variable or left
in the source as a `${VAR}` placeholder with a comment naming it.

Two kinds of occurrence, and the difference matters:

| | Behaviour |
|---|---|
| **Live** — read with `os.environ.get` | resolves at runtime once you export the variable |
| **Placeholder** — the literal text `${VAR}` inside a string | **does not expand.** It is a provenance record. Substitute by hand before running |

Shell-style `${VAR}` does not interpolate in Python, so a placeholder left in a
string will fail with `FileNotFoundError` naming the placeholder itself. That is
intentional: it fails loudly instead of silently reading the wrong file.

## Live variables

Set these and the parameterised code runs. Where a default is listed the
variable is optional.

| Variable | What it points at | Default |
|---|---|---|
| `KSI_DATA_ROOT` | sequencing/PacBio data root: raw run directories, `processed_count_table/`, the designed-substrate reference CSVs | none — must be set |
| `KSI_REPO_ROOT` | analysis code/data root holding `ksi_analysis_2025/`, `kinase_ml/`, `screening_analysis/`. Read only | none — must be set |
| `KSI_COUNT_DIR` | MAGeCK count/RRA tables | `$KSI_DATA_ROOT/processed_count_table` |
| `KSI_OUT_DIR` | everything the phospho-PCA pipeline writes | `<repo>/data/results/phosphoPCA/rerun` |
| `KSI_FIG_DIR` | figure outputs | `$KSI_OUT_DIR/figures` |
| `KSI_RESTORE_ROOT` | manuscript restore tree, for the spike-in provenance audit | none — must be set |
| `KSI_DATA_ROOT2` | second data mount; parent of the restore tree | none — must be set |
| `KSI_TREE_GT` | the published (uncorrected) newick tree directly | `$KSI_DATA_ROOT2/vulcan_bkup_04-09-26/vanvalen_restore/tarballs/manuscript/261014/screening_tree_gt.newick` |

Read by:

```
code/01_phosphoPCA/config.py                         KSI_DATA_ROOT KSI_REPO_ROOT KSI_COUNT_DIR KSI_OUT_DIR KSI_FIG_DIR
code/01_phosphoPCA/figures/SF1_SF5_SF6_SF7/sf5_library.py        KSI_DATA_ROOT
code/01_phosphoPCA/figures/SF1_SF5_SF6_SF7/sf6_false_positive.py KSI_DATA_ROOT KSI_COUNT_DIR
code/01_phosphoPCA/figures/fig03_tyrosine_tree/figure3_tyrosine_revision.ipynb  KSI_TREE_GT KSI_DATA_ROOT2
code/01_phosphoPCA/_source_scripts/tableS13_substrate_library/config.py         KSI_DATA_ROOT
code/verification_261022/scripts/audit_09_spikein_provenance.py                 KSI_REPO_ROOT KSI_RESTORE_ROOT
```

**Do not** set `KSI_OUT_DIR` to `<repo>/data/results/phosphoPCA` or to
`$KSI_REPO_ROOT/ksi_analysis_2025`. The first is a symlink view onto the
second, which is the read-only deposited analysis directory; writing there
overwrites deposited artifacts through the symlinks. `cp -L` the inputs you
need into a fresh `KSI_OUT_DIR` instead.

## Placeholder-only variables

These appear as inert `${VAR}` text in the archived notebooks. They record
where the data sat when the notebook ran; substitute them before re-running.

| Variable | Was | Appears in |
|---|---|---|
| `SPARK_ACQUISITION_ROOT` | the microscope acquisition user directory on the imaging mount | 11 SPARK/KINBERT source notebooks (384 occurrences) |
| `SPARK_ACQUISITION_ROOT_ALT` | a second operator's directory on the same mount | SPARK source notebooks |
| `SPARK_ACQUISITION_MOUNT` | the imaging mount root | SPARK source notebooks |
| `DATA_MOUNT`, `DATA_MOUNT2` | the two container bind-mount roots | `docker run` lines and container-absolute paths |
| `CONDA_ROOT` | the conda prefix | two stored notebook *outputs* (log text only, not code) |
| `KSI_ML_ROOT` | the KINBERT working tree | documentation only |

`DATA_MOUNT` and `DATA_MOUNT2` are the same physical count-table directory seen
through two bind mounts; see the header of `code/01_phosphoPCA/config.py`.

## Running the supplementary-figure generators

`sf5_library.py` and `sf6_false_positive.py` resolve their *other* inputs
relative to the script's own parent directory, because they were written to sit
in `ksi_analysis_2025/SF5_SF6_SF7/`. In this repo they sit three levels deeper,
so running them in place fails on the sibling inputs (`251203_ksi_reduced_df.pkl`,
`decoupling_out/`, `lookup_tables/`, `screening_analysis/`) regardless of how
the environment variables are set. To re-run them, copy the script back beside
those inputs:

```bash
export KSI_DATA_ROOT=/path/to/sequencing_data_root
cp code/01_phosphoPCA/figures/SF1_SF5_SF6_SF7/sf6_false_positive.py \
   $KSI_REPO_ROOT/ksi_analysis_2025/SF5_SF6_SF7/
cd $KSI_REPO_ROOT/ksi_analysis_2025/SF5_SF6_SF7 && python3 sf6_false_positive.py
```

Both were verified to produce output byte-identical to the shipped PDFs after
parameterisation (identical text layer, identical glyph coordinates, identical
bytes once the PDF creation timestamp is excluded). `sf1_corpus.py` and
`sf7_decoupling.py` needed no path changes and are byte-identical to the
originals.
