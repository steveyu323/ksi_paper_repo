# Sequencing replicates give the same answer

The decoupling model (`5_enrichment_analysis.ipynb`) pools two sequencing replicates:
**rep 1** = sequencing batch 1 (`blib_fp`) and **rep 2** = batch 2 (`blib_fp_b2`), merged in
`2_generate_lfc_table.ipynb` as `pre_1/mtx_1` and `pre_2/mtx_2` in
`decoupling_out/raw_screen_count_mapped.csv`. This folder asks whether each replicate,
analysed on its own, gives the same answer. All values are in natural-log units, the same
as `b_sk`.

## Run

```bash
conda activate ksi_bayes_qc            # see ../bayes_fit_qc/README.md
python 01_raw_replicate_concordance.py # < 1 min
python 02_refit_per_replicate.py --all # 8 NUTS fits in parallel, ~50 min on 32 cores
python 03_compare_replicate_fits.py    # < 1 min, needs 02
python 04_conditional_replicate_posteriors.py   # ~5 min
```
`decoupling_model.py` is the model code copied from notebook 5, so the refits use exactly
the production model, priors, sampler settings and seed.

## Results

### 1. Raw counts agree, and agreement rises with read depth (`01`)

At the (kinase, substrate) pair level:

| | SH2 | FHA | 14-3-3 | WW |
|---|---|---|---|---|
| pre-selection counts, r (log CPM) | 0.89 | 0.77 | 0.88 | 0.79 |
| post-selection counts, r | 0.94 | 0.84 | 0.92 | 0.79 |
| log enrichment, r | 0.97 | 0.96 | 0.96 | 0.81 |
| kinase-centred log enrichment, r | 0.74 | 0.87 | 0.86 | 0.67 |
| ...same, pairs with ≥ 1,000 input reads in both reps | 0.87–0.88 | 0.92–0.95 | 0.92–0.97 | 0.78–0.90 |

Raw log-enrichment correlation is dominated by the kinase baseline. The kinase-centred
version, the naive analogue of `b_sk`, is the stricter comparison. Its agreement climbs
steadily with depth (`fig_raw_concordance_by_depth`), which is the signature of counting
noise rather than a systematic batch difference. Per-kinase correlations are lowest for
kinases with little binding signal (few true binders), where the centred enrichment is
mostly noise.

### 2. Binding estimates per replicate, with the assay noise model fixed (`04`, primary test)

Each `b_sk` is estimated from one replicate's count. Only the quantities that describe the
assay, not individual pairs, are held at their production posterior means: NB dispersion
`alpha`, prior scale `sigma_b`, and per-kinase baselines `g_k + delta_kr`, each fit from
more than 1,000 observations. Each pair's posterior is then one-dimensional and is computed
exactly on a grid.

*Method check:* the same grid calculation using both replicates reproduces the production
NUTS fit, with `b` r ≥ 0.99999, max |Δb| ≤ 0.0074, and hit-call Jaccard 0.97–0.98.

| | SH2 | FHA | 14-3-3 | WW |
|---|---|---|---|---|
| r(b̂ rep 1, b̂ rep 2), **observed** | 0.755 | 0.873 | 0.866 | 0.679 |
| r expected if replicates agree perfectly under the model | 0.742 | 0.860 | 0.862 | 0.673 |
| scale: TLS slope rep 2 on rep 1 | 1.01 | 1.03 | 1.00 | **1.69** |
| production hits with b̂ > 0 in **both** replicates | 96.8% | 99.4% | 98.4% | 93.8% |
| hits per replicate: Jaccard observed (expected) | 0.36 (0.35) | 0.57 (0.51) | 0.57 (0.53) | 0.31 (0.39) |
| held-out 90% predictive interval coverage | 92.4% | 92.3% | 91.9% | 92.7% |
| held-out 50% predictive interval coverage | 62.4% | 59.6% | 60.3% | 68.6% |

- **The replicates agree as well as they possibly can.** The "expected" rows come from
  drawing `b` from the joint posterior, re-simulating both replicates' counts at their
  real depths, and repeating the per-replicate analysis. That is what perfectly
  consistent replicates would give, limited only by counting noise at half the reads.
  Observed correlation matches or slightly exceeds it in every pAABD, so there is no
  detectable replicate-specific disagreement beyond sampling noise.
- **Hit calls.** Single replicates have half the reads, so they call different hit sets
  even when perfectly consistent; the expected Jaccard is only 0.35–0.53. Observed
  overlap matches that for SH2, FHA and 14-3-3. Directionally, 94–99% of production hits
  have positive `b` in both replicates estimated separately, and 91–99% of one replicate's hits
  are positive in the other (`fig_conditional_hits`).
- **Uncertainty is honest to slightly conservative.** When rep 1's posterior predicts rep 2's
  counts (and vice versa), 90% intervals cover 92–93% of held-out counts. 50% intervals
  over-cover (60–69%), so the single shared NB dispersion slightly overstates noise for
  typical pairs. Errors in the posterior go toward too much caution, not false confidence,
  so lfsr-based calls are on the safe side.
- **WW is the exception: same ranking, different dynamic range.** In SH2, FHA and 14-3-3
  the two batches put `b` on the same scale (slope 1.00–1.03). In WW, batch-2 effects are
  about 1.7× batch-1 effects: median `b` of production hits is 0.30 in rep 1 vs 0.52 in
  rep 2, while posterior SDs are the same. Rank and sign agree, but rep 1 calls fewer hits
  (593 vs 1,163), and hit overlap falls below expectation (0.31 vs 0.39). The model shares
  one `b_sk` across replicates and has no replicate-specific gain on `b`, so WW's
  production `b` averages a compressed and an expanded batch. This is the likely source of
  WW's excess dispersion (1.13–1.22) in `../bayes_fit_qc`. WW hit calls are robust in
  direction (93.8% positive in both batches), but WW effect sizes should be read as
  batch-averaged. A replicate-specific scale on `b` would be the model change if this
  matters downstream.

Figures: `fig_conditional_replicates` (rep 1 vs rep 2 `b`, method check, held-out PIT),
`fig_conditional_hits`.

### 3. Fully independent single-replicate refits (`02`, `03`, secondary)

The production model was also refit by NUTS on each replicate alone, with nothing shared
between the fits. Kinase baselines agree almost perfectly (r = 0.983–0.997 across kinases),
and `b` agrees at r = 0.68–0.87, the same level as in section 2. This test is weaker,
though, for a structural reason. With a single replicate each (substrate, kinase) has one
observation, so binding `b_sk` and NB overdispersion cannot be told apart. The fits trade
one for the other:

- `alpha` drops from 25–37 (both replicates) to 6–24, so more variance is called noise and
  `b` is over-shrunk. As a result, single-replicate fits recover only 2–70% of production
  hits, and the z-scores of replicate differences have SD 0.33–0.92 instead of 1 (inflated
  posterior SDs).
- The sampler struggles along that ridge: `sigma_b` / `alpha` R̂ reaches 1.05–1.13 with ESS
  24–93 in SH2/FHA, E-BFMI falls to 0.04–0.15, and there are 0–6 divergences. `b` and the
  kinase baselines themselves still have R̂ ≤ 1.012 (`replicate_refit_convergence.csv`).
  `sigma_g`/`sigma_delta` and raw `g`/`delta` are only prior-identified with one replicate,
  so their R̂ is not meaningful.

So this test mostly shows that **the second replicate is what makes the model
identifiable**. Replicates are needed to separate reproducible binding from noise, which is
why the production model pools them. The fair same-answer test is section 2.

Figures: `fig_replicate_b_scatter`, `fig_replicate_calibration_hits`, `fig_replicate_vs_production`.

## Output files (`out/`)

| file | content |
|---|---|
| `raw_concordance_summary.csv`, `_by_depth.csv`, `_per_kinase.csv` | model-free replicate correlations |
| `conditional_summary.csv` | section 2 per pAABD, including all `expected_*` values |
| `conditional_pairs_{paabd}.csv` | per pair: b̂/sd/lfsr from rep 1, rep 2, both (grid) and production; held-out PIT |
| `refits/{paabd}_rep{r}_*.csv/.nc/.log` | independent single-replicate NUTS fits (b summaries, kinase/hyper summaries, sampler stats) |
| `replicate_fit_summary.csv`, `replicate_fit_pairs_{paabd}.csv`, `replicate_fit_kinase.csv`, `replicate_fit_hyper.csv` | section 3 comparisons |
| `replicate_refit_convergence.csv` | convergence of the 8 refits |
