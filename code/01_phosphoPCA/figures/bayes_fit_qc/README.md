# Quality of the Bayesian decoupling fit

Checks of the hierarchical negative-binomial decoupling model fit in
`5_enrichment_analysis.ipynb` (`decoupling_out/251125_pymc_{sh2,fha,1433,ww}.nc`):
does the sampler converge, are the draws good enough for the downstream calls,
and how much of the variance does the model explain?

```
y_skr ~ NB(mu_skr, alpha)
log mu_skr = log_offset_skr + g_k + delta_kr + b_sk        (b centred per kinase)
```
NUTS, 4 chains × 1000 draws after 1000 tuning steps, `target_accept = 0.9`.

## Run

```bash
conda activate ksi_bayes_qc   # python 3.11, pymc 5.20.1, arviz 0.21, h5netcdf
python 01_convergence_diagnostics.py   # ~15 min, ~40 GB RAM (loads full b draws)
```
Posterior paths come from `../../config.py` (`KSI_OUT_DIR`); results are written under
this folder's `out/`.
The env was created with
`conda create -n ksi_bayes_qc -c conda-forge python=3.11 pymc=5.20.1 pytensor=2.27.1 "arviz<0.22" matplotlib=3.9 xarray=2025.1 h5netcdf netcdf4 pandas scipy seaborn`
(newer conda-forge pytensor 3.x builds fail to import, and arviz 0.21 needs matplotlib < 3.10).
It has numpy 1.26, so it cannot read the project's numpy-2 `.pkl` files; the scripts only read `.nc`/`.csv`.

## Results

### 1. The sampler converged (`01_convergence_diagnostics.py`)

| pAABD | divergences | max-tree-depth hits | E-BFMI (per chain) | max R̂ (all params) | params R̂ > 1.01 | min bulk ESS | min tail ESS |
|---|---|---|---|---|---|---|---|
| SH2   | 0 / 4000 | 0 | 0.72–0.82 | 1.0075 | 0 / 13,260 | 1,013 | 870 |
| FHA   | 0 / 4000 | 0 | 0.71–0.82 | 1.0086 | 0 / 18,531 | 532   | 933 |
| 14-3-3| 0 / 4000 | 0 | 0.67–0.79 | 1.0115 | 3 / 21,279 | 1,540 | 1,549 |
| WW    | 0 / 4000 | 0 | 0.68–0.77 | 1.0125 | 6 / 19,181 | 1,821 | 1,730 |

Conventional targets: R̂ < 1.01, ESS > 400, E-BFMI > 0.3, no divergences (Vehtari et al. 2021).
Every parameter that enters the likelihood is included: hyperparameters, `g`, `delta`,
the kinase baseline `g + mean_r(delta)`, and `b` for every observed (substrate, kinase) pair.

- **The R̂ warning in the original 14-3-3 / WW runs** comes from 3 and 6 binding
  parameters with R̂ 1.010–1.012. That is more than perfectly mixed chains of the same
  length would give (an iid null with the same shape has max R̂ ≈ 1.004 and none above
  1.01), so it is a real but tiny effect. For every flagged pair the four chain means
  agree to within 0.02–0.07 posterior SDs. Their bulk ESS is 9,000–12,000, and none of
  them is a hit (`out/flagged_rhat_params.csv`). ArviZ's R̂ is the max of the bulk and
  folded (tail) R̂, so these flags reflect slightly different tail widths between chains,
  not a location disagreement.
- **Monte Carlo error is negligible.** MCSE / posterior SD ≤ 0.044 for every parameter
  (≤ 0.023 for every `b`).
- **Independent chains reach the same conclusions.** Rerunning the inference with chains
  1–2 versus chains 3–4 gives `b` posterior means that correlate at r ≥ 0.99994, with a
  largest absolute difference of 0.015 (`fig_chain_split_agreement`). Hit calls
  (lfsr ≤ 0.01, b ≥ 0.1) overlap with Jaccard 0.95–0.96. The ~5% that differ are pairs
  whose lfsr sits next to 0.01, where each half has only 2,000 draws. The production
  calls use all 4,000.

Figures: `fig_convergence_rhat_ess`, `fig_hyperparameter_traces` (trace + rank plots),
`fig_sampler_energy`, `fig_chain_split_agreement`.

### 2. The model explains the variance down to the noise floor

> **Not in this package.** `02_posterior_predictive.py` was removed: none of its
> outputs is printed in the final manuscript. The section below is retained as a
> record of what it found. The script itself is in the source tree,
> `ksi_analysis_2025/bayes_fit_qc/`.

The variance of the observed log enrichment `z = log(y + 0.5) − log_offset` is split
sequentially into the model's components (posterior means):

| pAABD | kinase baseline | replicate shift | binding `b` | residual | Bayesian R² (94% int.) | residual / expected NB noise |
|---|---|---|---|---|---|---|
| SH2    | 87.7% | 0.8% |  9.8% |  1.7% | 0.971 (0.970–0.972) | 1.16 |
| FHA    | 71.6% | 0.5% | 25.8% |  2.0% | 0.962 (0.961–0.963) | 1.03 |
| 14-3-3 | 74.0% | 0.1% | 23.9% |  2.0% | 0.964 (0.963–0.964) | 1.06 |
| WW     | 44.2% | 0.8% | 43.2% | 11.7% | 0.815 (0.812–0.819) | 1.13 |

- The replicate-specific kinase shift `delta_kr` explains under 1% of the variance, so
  the two sequencing batches differ by little more than a per-kinase scale factor (see
  `../seq_replicate_qc`).
- **The residual is sampling noise.** The last column compares the residual variance
  with what the fitted NB predicts for in-sample residuals. A value near 1 means nothing
  systematic is left to explain. WW's larger residual share comes from WW having about
  4× less total variance in `z` (0.21 vs 0.8–1.2). Its noise level in absolute terms is
  similar to the other pAABDs.
- **Calibration.** Observed vs posterior-mean predicted counts give R² = 0.984–0.993. The
  degrees-of-freedom–adjusted Pearson dispersion is 1.01 (SH2), 0.92 (FHA), 0.96 (14-3-3)
  and 1.22 (WW); 1 means the NB variance matches the data.

**Caveat on in-sample checks.** Each `b_sk` is estimated from just its two replicate
observations, so the fit uses about half the degrees of freedom (`dof_fraction` ≈ 0.50 in
`ppc_summary.csv`). In-sample predictive intervals therefore over-cover: 50% intervals
cover 80–83% and 90% intervals cover 97–99%. The PIT histogram has a central hump and the
raw Pearson dispersion is about 0.5. All of this is expected from the d.o.f. effect, not a
sign of a mis-specified model, which is why the dispersion and residual above are
d.o.f.-adjusted. The proper out-of-sample calibration test (fit one replicate, check
against the other) is in `../seq_replicate_qc/03_compare_replicate_fits.py`.

**Where the NB is only approximate.** One dispersion `alpha` is shared by all observations.
Residual variance runs somewhat above expectation for the lowest-count observations (predicted < ~100 post-selection reads)
in every pAABD, and in WW also for the highest-count pairs (right panel of
`fig_variance_explained`). This is WW's ~13–20% excess dispersion. The effect is mild, and
the fitted `alpha` (25–37, i.e. ~16–20% extra-Poisson CV) already makes the posterior
conservative relative to pure counting noise. Pure Poisson noise would be 7–14× smaller
(`noise_floor_var_poisson`). The extra-Poisson variance plausibly includes input-count
noise, since `pre` enters the offset as if it were known exactly.

Figures: `fig_ppc_calibration` (observed vs predicted, PIT, density overlays, per-kinase
interval coverage), `fig_variance_explained`.

## Output files (`out/`)

| file | content |
|---|---|
| `convergence_overview.csv` | per pAABD × parameter group: max R̂, # > 1.01, ESS min/median, max MCSE/SD, plus the iid R̂ null |
| `convergence_params_{paabd}.csv` | every parameter: mean, sd, R̂, bulk/tail ESS, MCSE |
| `flagged_rhat_params.csv` | the 9 parameters with R̂ > 1.01, with per-chain means/SDs, lfsr, hit status |
| `sampler_health.csv` | per chain: divergences, tree depth, acceptance, step size, E-BFMI |
| `chain_split_hits.csv` | `b` and hit calls from chains 1–2 vs 3–4 |
| `ppc_summary.csv` | coverage, PIT KS distance, Pearson dispersion (raw / d.o.f.-adj.), R², noise floors |
| `ppc_per_kinase.csv` | the same per kinase |
| `variance_partition.csv` | sequential variance fractions |
| `bayes_r2_draws.csv` | Bayesian R² per posterior draw |
