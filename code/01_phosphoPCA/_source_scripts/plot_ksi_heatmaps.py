#!/usr/bin/env python
"""
Full KSI screening landscape — Y and ST heatmaps.

Standalone version of the heatmap cell in 8_2batch_blib_pipeline_v3.ipynb
(section 6b). Renders one heatmap per library with

    y = kinase (common gene name)   x = substrate_15_aa   fill = binding score

and writes each figure as both .png and .pdf.

Usage
-----
    python plot_ksi_heatmaps.py                     # defaults
    python plot_ksi_heatmaps.py --value lfsr_min    # colour by a different column
    python plot_ksi_heatmaps.py --row-h 0.10        # thinner kinase rows
    python plot_ksi_heatmaps.py --outdir figures    # write elsewhere
"""

import argparse
import os

import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, scale_fill_gradient2, labs, theme_minimal, theme,
    element_blank, element_text, element_rect,
)

# ── Paths (resolved relative to this script, so it runs from anywhere) ────────
HERE = os.path.dirname(os.path.abspath(__file__))

KINOME_PKL   = os.path.join(HERE, '../kinase_ml/data/human_kinome_seq_230301.pkl')
BLIB_KIN_CSV = os.path.join(HERE, '../screening_analysis/blib_kinases_with_acc.csv')
SCREEN_PKL   = os.path.join(HERE, './251203_ksi_reduced_df_tp.pkl')

# Figure output directory
OUTDIR       = os.path.join(HERE, 'supp_plot_outputs')

# Kinases dropped from all downstream analysis (matches the notebook)
EXCLUDE_KINASES = {'KS6B1_HUMAN_active', 'KS6B1_HUMAN_wt', 'PLK1_HUMAN_active'}

# ── Defaults (mirror the notebook cell) ──────────────────────────────────────
HEAT_VALUE = 'binding_max'   # continuous binding score of each kinase–substrate pair
HEAT_ROW_H = 0.16            # figure inches per kinase row (tile height)
HEAT_WIDTH = 12.0            # figure inches, full width
HEAT_CLIP  = (0.01, 0.99)    # quantiles for the colour-scale limits


def load_screen():
    """Load the processed screening hits, annotated with kinase type/accession."""
    blib_kinases_with_acc = pd.read_csv(BLIB_KIN_CSV, header=None)
    blib_kinases_with_acc.columns = ['kinase_type', 'kinase', 'kinase_seq', 'kinase_acc']

    dat_in_screen = pd.read_pickle(SCREEN_PKL)
    dat_in_screen = pd.merge(blib_kinases_with_acc, dat_in_screen, on='kinase')
    dat_in_screen = dat_in_screen.loc[~dat_in_screen['kinase'].isin(EXCLUDE_KINASES), :]
    return dat_in_screen


def load_common_names():
    """UniProt accession → common gene name (e.g. KSYK_HUMAN / P43405 → SYK)."""
    human_kinome = pd.read_pickle(KINOME_PKL)
    return (
        human_kinome.loc[:, ['kinase', 'kinase_gene']]
        .drop_duplicates('kinase')
        .set_index('kinase')['kinase_gene']
        .to_dict()
    )


def plot_ksi_heatmap(df, lib_label, common_names, value=HEAT_VALUE, clip=HEAT_CLIP,
                     row_h=HEAT_ROW_H, width=HEAT_WIDTH, save=True, outdir=OUTDIR):
    """
    Full kinase x substrate screening landscape as a plotnine heatmap.

    Both axes are ordered by mean signal so structure is visible; the colour
    scale is clipped at robust quantiles so a few extreme tiles do not wash out
    the rest. Pairs absent from the screen stay blank (panel background).
    With save=True the figure is written to `outdir` as both .png and .pdf.
    """
    d = (
        df.loc[:, ['kinase', 'kinase_acc', 'substrate', value]]
          .dropna(subset=[value])
          .rename(columns={'substrate': 'substrate_15_aa'})
          .copy()
    )

    # ── common gene name on the y axis (fall back to stripping _HUMAN) ───────
    d['kinase'] = [
        common_names.get(acc, str(k).replace('_HUMAN', ''))
        for acc, k in zip(d['kinase_acc'], d['kinase'])
    ]

    # ── order axes by mean signal ────────────────────────────────────────────
    sub_order = d.groupby('substrate_15_aa')[value].mean().sort_values().index.tolist()
    kin_order = d.groupby('kinase')[value].mean().sort_values().index.tolist()
    d['substrate_15_aa'] = pd.Categorical(d['substrate_15_aa'], categories=sub_order, ordered=True)
    d['kinase'] = pd.Categorical(d['kinase'], categories=kin_order, ordered=True)

    # ── symmetric, quantile-clipped colour limits ────────────────────────────
    lo, hi = d[value].quantile(list(clip)).values
    lim = float(max(abs(lo), abs(hi)))
    d[value] = d[value].clip(-lim, lim)

    n_kin, n_sub = len(kin_order), len(sub_order)
    height = row_h * n_kin + 1.2

    p = (
        ggplot(d, aes('substrate_15_aa', 'kinase', fill=value))
        + geom_tile()
        + scale_fill_gradient2(
            low='#2166ac', mid='#f7f7f7', high='#b2182b',
            midpoint=0, limits=(-lim, lim), name='binding\nscore',
        )
        + labs(
            title=f'{lib_label} library — full KSI screening landscape '
                  f'({n_kin} kinases x {n_sub} substrates)',
            x=f'substrate_15_aa (n={n_sub}, ordered by mean {value})',
            y='kinase',
        )
        + theme_minimal()
        + theme(
            figure_size=(width, height),
            axis_text_x=element_blank(),
            axis_ticks_major_x=element_blank(),
            axis_text_y=element_text(size=7),
            panel_grid=element_blank(),
            panel_background=element_rect(fill='#e8e8e8'),
        )
    )

    if save:
        os.makedirs(outdir, exist_ok=True)
        stem = os.path.join(outdir, f'ksi_heatmap_full_{lib_label}')
        for ext, kw in [('png', {'dpi': 300}), ('pdf', {})]:
            p.save(f'{stem}.{ext}', width=width, height=height, verbose=False, **kw)
        print(f'saved {stem}.png / {stem}.pdf  ({n_kin} kinases x {n_sub} substrates)')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--value', default=HEAT_VALUE,
                    help=f'column to colour tiles by (default: {HEAT_VALUE})')
    ap.add_argument('--row-h', type=float, default=HEAT_ROW_H,
                    help=f'figure inches per kinase row (default: {HEAT_ROW_H})')
    ap.add_argument('--width', type=float, default=HEAT_WIDTH,
                    help=f'figure width in inches (default: {HEAT_WIDTH})')
    ap.add_argument('--clip', type=float, nargs=2, default=list(HEAT_CLIP),
                    metavar=('LO', 'HI'),
                    help=f'quantiles for colour limits (default: {HEAT_CLIP[0]} {HEAT_CLIP[1]})')
    ap.add_argument('--outdir', default=OUTDIR,
                    help='directory for the .png/.pdf output (default: ./supp_plot_outputs)')
    ap.add_argument('--libs', nargs='+', default=['Y', 'ST'],
                    help='kinase_type values to plot (default: Y ST)')
    args = ap.parse_args()

    dat_in_screen = load_screen()
    common_names = load_common_names()
    print(f'Screening hits retained: {dat_in_screen.shape[0]}')

    for lib in args.libs:
        sub = dat_in_screen.loc[dat_in_screen['kinase_type'] == lib, :]
        if sub.empty:
            print(f'[warn] no rows with kinase_type == {lib!r}; skipped')
            continue
        plot_ksi_heatmap(sub, lib, common_names, value=args.value,
                         clip=tuple(args.clip), row_h=args.row_h,
                         width=args.width, outdir=os.path.abspath(args.outdir))


if __name__ == '__main__':
    main()
