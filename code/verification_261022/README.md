# Numeric audit — KSI Paper (started 2026-10-22)

Purpose: trace every disputed number in `manuscript/261014/KSI_Paper.zip`
(main.tex, supplementary.tex, figure artwork) back to the code/data that
produced it, and record which value is correct.

## Layout
- `scripts/audit_<id>_<topic>.py` — one script per audit item. Each takes its
  input roots as argv, writes a CSV/JSON to `../out/`, and prints a PASS/FAIL
  verdict line per claim. Run with `python3 -I` (data roots are not on sys.path).
- `out/` — machine-readable results.
- `FINDINGS.md` — the verdict table (what the manuscript says / what the code
  says / which to use).

## Source roots
- MS   = manuscript/261014/        (main.tex, supplementary.tex, figures*/)
- EVAL = tarballs/code/code/evaluation/
- TD   = tarballs/training_data/
- WET  = ${KSI_REPO_ROOT}/kinase_ml/code/wetlab_planning/
- KSI  = ${KSI_REPO_ROOT}/ksi_analysis_2025/
- CS   = ${KSI_REPO_ROOT}/cellsam-internal/

## Audit items — all complete 2026-10-22. Verdicts in `FINDINGS.md`.
| id | claim under audit | verdict |
|----|-------------------|---------|
| 01 | KINBERT corpus: 2,304 / 147,488 (text) vs 2,021 / 162,447 (Fig 4 artwork) | 🔴 **artwork right, text wrong** — 2,304 is a `len(list+list)` double-count bug. Use 2,021. Both pair counts real (sequence vs site). Needs author sign-off |
| 02 | Table S3 bins sum to 1,816 / 1,790 vs 2,304 kinases | ✅ **printed totals correct**, 12/12 cells exact. Gap = 283 (bug) + 205 (sequence dedup) |
| 03 | Kinase panel: 23 constructs, DYRK4 dropped; Methods placeholders | ✅ **23 confirmed** (= 21 distinct). DYRK4 at 0.036%, 0 exact matches. Methods text in `out/audit_03_methods_replacement.txt` |
| 04 | Titin arm: Ser/Thr (Table S12) vs tyrosine (Methods) | ✅ **Ser/Thr** — over-determined. Table S12 right, Methods wrong |
| 05 | Fig 6 (SPARK) Mann-Whitney p-values: artwork vs text | ✅ **both correct** — one-sided (text) vs two-sided × Bonferroni k=6 (artwork), ratio exactly 12. 🔴 but quoted n is from a different pool than quoted p |
| 06 | Table S11 post-fine-tuning prob/logit mismatch (~0.22) | 🔴 **no mismatch** — "logit" = raw positive-class z1, offset is z0. The wrong rows are the two `-1` reporters (3 runs, 2 definitions) |
| 07 | PER2 site: p562 (Fig 7B) vs Ser662 (Results) | ✅ **Ser662** — artwork typo. Plus 3 more orphan labels: p11→p111, p109→p106, p152→p151 |
| 08 | Influenza M1 Ser53: ERK7 (Results) vs PKCe (Fig 8) | ✅ **ERK7** (4.6875). Caption/panel B swapped the H3N2 HA T329→PKCε nomination onto M1 |
| 09 | Table S13 spike-in provenance | 🔴 **author confirmed verbatim in code.** Benchmark = inner join of screen × KINBERT corpus. 310/385 (80.5%) of benchmark pairs are in KINBERT's corpus, 205 in train |
| 10 | Supp Tables S8/S9 internal arithmetic (added) | ⚠️ 24/28 pass. S8 "mean reads per barcode" uses a pre-filter denominator (~84% assignment rate) |

## Headline
Nothing found challenges the underlying measurements. Two numbers in print are wrong at
source (**2,304 kinases**; the **Fig 8 M1 Ser53 kinase**), two objections dissolve on
inspection (**Table S11 logits**, **SPARK p-values**), and the most consequential finding is
provenance rather than arithmetic: **Table S10 is a screen-vs-database concordance
measurement, not a held-out KINBERT evaluation** (item 09).
