# Table S13 — candidate substrate library manifest

Regenerates the full substrate-library table from the original wet-lab
planning files and the phosphosite identity mapping.

```bash
cd tableS13_substrate_library
python3 build_table_s13.py        # -> outputs/tableS13_substrate_library.csv
python3 make_latex_excerpt.py     # -> outputs/tableS13_excerpt.tex
```

Requires `pandas` only. Runtime ~1 min, dominated by loading the 156 MB
phosphosite background table.

## Outputs

| File | Contents |
| --- | --- |
| `outputs/tableS13_substrate_library.csv` | Full manifest, 3,928 rows |
| `outputs/tableS13_representative_rows.csv` | Two rows per library × class |
| `outputs/tableS13_excerpt.tex` | LaTeX rows for the manuscript table |
| `outputs/summary_stats.txt` | Counts reproduced on each run |

## Provenance

Every field traces to a file already in the repo; nothing is hand-entered.

| Input | Role | Written by |
| --- | --- | --- |
| `kinase_ml/code/wetlab_planning/{st,y}_sub_all.csv` | the designed library (what was ordered) | `230310_select_substrate_oligo.ipynb` cells 32, 47 |
| `{st,y}_spike_ins.csv` | documented kinase–substrate pairs + their target kinase, reporter domain, DB split | cells 10, 39 |
| `{st,y}_sub_psp.csv` | redundancy-filtered PhosphoSitePlus candidate pool | cells 14, 43 |
| `substrates_background_12345.pkl` | 15-mer → UniProt accession + site index | `4_filter_unmapped_substrates.ipynb` |
| `decoupling_out/raw_screen_count.pkl` | which substrates were recovered in the screen | `4_filter_unmapped_substrates.ipynb` |

**Class assignment** follows the design notebook's own precedence (cells 27,
44), where `spike_in` is applied last and so overrides `psp_random` /
`animal_random`.

**Design thresholds**, copied into `config.py` from the notebook:
spike-in dedup `DIST_THRESH = 7`; PSP pool `thresh = 9`; animal pool
`thresh = 11`. These are recorded, not re-run — the shipped library is the
one the sampling actually produced, and re-running `sanitize_redundancy`
(which consumes an unseeded `.sample()`) would not reproduce it.

**Reproducibility check**: the script recovers 3,927 mapped / 1 unmapped,
and the single unmapped sequence is `VGAPIAVSAFYAESC` — exactly the row
`4_filter_unmapped_substrates.ipynb` flagged independently.

## Composition as built

| Library | Class | n |
| --- | --- | ---: |
| ST | Control (documented) | 465 |
| ST | Candidate (PhosphoSitePlus) | 1,513 |
| Y | Control (documented) | 151 |
| Y | Candidate (PhosphoSitePlus) | 1,240 |
| Y | Candidate (animal phosphoproteome) | 559 |
| | **total rows** | **3,928** |

3,918 of these are distinct sequences; 10 appear in both the ST and Y
libraries and are kept as separate rows because they were ordered, barcoded
and screened separately in each.

## Points where the data contradicts the draft caption

These need a decision before the caption ships — the generator reports what
is in the files rather than matching the draft.

1. **Size.** The caption says ~2,000 substrates (~300 + ~1,700). The designed
   library is **3,928 rows / 3,918 unique**, roughly double. ~2,000 is close
   to the size of *one* library (ST 1,978, Y 1,950), so the draft may have
   described a single library and been read as the pool.

2. **Control count.** 616 documented spike-ins, not ~300.

3. **Controls are not all positives.** 165 of the 616 come from the
   `train_tn` split — documented *non*-substrates included as negative
   controls. Calling the whole block "documented kinase–substrate pairs"
   overstates it. The `control_polarity` column separates
   positive (449) / negative (165) / mixed (2).

4. **Candidate source.** The caption attributes candidates to "animal
   phosphoproteomic data". Most are PhosphoSitePlus-derived (2,753); only 559
   come from the animal phosphosite pool, and only in the Y library.

5. **Identity is often ambiguous.** 2,069 of 3,928 windows (53%) occur in
   more than one protein — identical 15-mers across paralogues, isoforms and
   orthologues. A single "Source protein" column cannot be literally correct
   for these. The CSV gives a deterministic `uniprot_acc` (lowest accession,
   then lowest site index) plus `n_protein_matches` and the full
   `all_uniprot_acc` list. The LaTeX excerpt marks ambiguous rows with †.

6. **The Y library is not purely Tyr.** 45 Y-library windows are Ser/Thr
   centred, all from the PSP candidate pool. A caption saying each Y window
   is centred on a Tyr phosphoacceptor is wrong for these 45.

7. **32 designed substrates were never recovered** in the screen
   (`recovered_in_screen == False`). Whether the deposited manifest should
   describe the designed library or the screened subset is a judgement call;
   the column lets it go either way without regenerating.

Separately, the Zenodo DOI in the caption (10.5281/zenodo.19616157) was not
verified here, and this CSV will not match an existing deposit unless it is
re-uploaded.

## Column reference

| Column | Meaning |
| --- | --- |
| `substrate_id` | `SUB-<LIB>-####`, controls first then alphabetical; stable across runs |
| `library` | `ST` or `Y` |
| `uniprot_acc` | primary accession (deterministic pick among matches) |
| `all_uniprot_acc` | all matching accessions, `;`-joined |
| `n_protein_matches` | number of distinct source proteins; `>1` = ambiguous |
| `phosphosite_position` | residue index in the primary protein |
| `phosphoacceptor` | S, T or Y; central residue of the window |
| `window_15mer` | 15-residue window, phosphoacceptor at position 8 |
| `substrate_class` | manuscript-facing class |
| `design_set` | raw label: `spike_in` / `psp_random` / `animal_random` |
| `control_polarity` | spike-ins only: `positive` / `negative` / `mixed` |
| `spikein_kinase_acc` | documented target kinase(s) for spike-ins |
| `spikein_reporter_domain` | PAABD reporter: SH2, b14-3-3, WW, FHA |
| `spikein_db_split` | `test_pos` / `train_pos` / `train_tn` |
| `mapping_status` | `mapped` / `unmapped` (unmapped keeps empty identity fields) |
| `recovered_in_screen` | recovered in the barcode screen |
| `n_kinases_recovered` | number of distinct kinases it was recovered against |
