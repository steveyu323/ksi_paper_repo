#!/usr/bin/env python3
"""
Annotated bar plots of the *corrected* Table 2.1 training-data scope, drawn with
plotnine.

Input
-----
    table2_1_validation/table2_1_corrected.csv

produced by `validate_table2_1_training_data.py`, which recounts every cell from
the raw DeepKinZero / Phosformer / KINBERT dataset files.  This script never
hardcodes a count: run the validator first, then run this.

    python validate_table2_1_training_data.py
    python plot_table2_1_training_data.py

Figures
-------
fig_table2_1_scope        Three facets (kinases / organisms / KSIs), one
                          horizontal bar per model, direct-labelled at the tip.
                          Carries the corrected counts only; pass
                          --show-published-notes to annotate each disagreeing bar
                          with the value printed in Table 2.1.
fig_table2_1_delta        The disagreement on its own: percent change from the
                          published figure to the corrected one, diverging about
                          zero, with the absolute difference labelled.

Each figure is written as PDF (vector, for the manuscript) and PNG (for review),
plus a CSV of exactly the rows plotted, so every mark has a readable twin.

Design notes
------------
* Bars carry one hue: the count is a magnitude, and the model is already named on
  the axis, so colouring each bar differently would double-encode bar length.
* The value axis is dropped in the scope figure - with only four bars per facet,
  direct labels at the tips are more precise than gridlines and much quieter.
  Facets use free value scales because the three statistics are different
  measures; they are never overlaid on a shared axis.
* Palette slots are the documented reference instance (categorical slot 1 blue /
  slot 2 orange; diverging blue<->red about a neutral gray), used unmodified.
  That pair is documented as clearing the adjacent-pair CVD and normal-vision
  gates in both light and dark modes.  Shape reinforces hue here anyway - bars
  versus diamonds - so identity never rests on colour alone.
* Text wears ink tokens, never a series colour.

Usage
-----
    python plot_table2_1_training_data.py
    python plot_table2_1_training_data.py --theme dark
    python plot_table2_1_training_data.py --corrected-csv path/to.csv --out-dir figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_blank,
    element_line,
    element_rect,
    element_text,
    facet_wrap,
    geom_col,
    geom_hline,
    geom_text,
    ggplot,
    guides,
    guide_legend,
    labs,
    scale_fill_manual,
    scale_y_continuous,
    theme,
    theme_minimal,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Palette - reference instance, light and dark steps of the same hues.
# ---------------------------------------------------------------------------
PALETTES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "ink_muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series_corrected": "#2a78d6",  # categorical slot 1, blue
        "series_published": "#eb6834",  # categorical slot 2, orange
        "diverging_neg": "#2a78d6",     # corrected below published
        "diverging_pos": "#e34948",     # corrected above published
        "diverging_mid": "#c3c2b7",     # exact match
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "ink_muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series_corrected": "#3987e5",
        "series_published": "#d95926",
        "diverging_neg": "#3987e5",
        "diverging_pos": "#e66767",
        "diverging_mid": "#383835",
    },
}

FONT = "DejaVu Sans"

STATISTICS = [
    ("n_kinases", "Kinases"),
    ("n_organisms", "Organisms"),
    ("n_ksis", "Kinase–substrate interactions"),
]

# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------
def load_long(corrected_csv: Path) -> tuple[pd.DataFrame, list[str]]:
    """Reshape the corrected table to one row per (model, statistic)."""
    wide = pd.read_csv(corrected_csv)

    records = []
    for row in wide.itertuples():
        for field, label in STATISTICS:
            corrected = getattr(row, field)
            published = getattr(row, f"published_{field}")
            records.append(
                {
                    "model": row.model,
                    "statistic": label,
                    # float, so an absent value is NaN and the scale stays
                    # continuous; pd.NA would make the column object dtype.
                    "corrected": float("nan") if pd.isna(corrected) else float(corrected),
                    "published": float("nan") if pd.isna(published) else float(published),
                    "reported": not pd.isna(corrected),
                }
            )
    long = pd.DataFrame(records)

    long["delta"] = long["corrected"] - long["published"]
    long["pct_delta"] = (
        100.0 * long["delta"] / long["published"].where(long["published"] != 0)
    )
    long["changed"] = long["reported"] & long["delta"].fillna(0).ne(0)

    # Largest dataset at the top; models with nothing to report sink to the
    # bottom. coord_flip puts the last category level at the top, so order the
    # levels bottom-to-top.
    order = (
        long.loc[long["statistic"] == STATISTICS[2][1]]
        .sort_values("corrected", na_position="first")["model"]
        .tolist()
    )
    long["model"] = pd.Categorical(long["model"], categories=order, ordered=True)
    long["statistic"] = pd.Categorical(
        long["statistic"], categories=[label for _, label in STATISTICS], ordered=True
    )
    return long.sort_values(["statistic", "model"]), order


def add_facet_geometry(long: pd.DataFrame) -> pd.DataFrame:
    """Per-facet label offsets, since each facet has its own value range."""
    long = long.copy()
    span = long.groupby("statistic", observed=True)[["corrected", "published"]].max()
    span = span.max(axis=1).rename("facet_max")
    long = long.join(span, on="statistic")
    long["pad"] = 0.025 * long["facet_max"]

    long["tip_label"] = long["corrected"].map(
        lambda v: "" if pd.isna(v) else f"{int(v):,}"
    )
    # Labels sit clear of the bar tip. Rendered text cannot be measured here, so
    # the layout is built from facet-relative offsets that make a collision
    # impossible rather than merely unlikely.
    long["tip_x"] = long["corrected"] + long["pad"]

    # Annotation for the cells whose published value was wrong.
    long["delta_label"] = [
        f"published {int(p):,}  ({int(d):+,})" if changed else ""
        for p, d, changed in zip(long["published"], long["delta"], long["changed"])
    ]
    # The note sits on the same row, clear of the tip label. Facets are stacked
    # one per row so each panel is full-width, which is what makes the room.
    long["delta_x"] = long["corrected"] + 0.17 * long["facet_max"]
    long["missing_label"] = [
        "" if reported else "not reported" for reported in long["reported"]
    ]
    return long


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def base_theme(pal: dict[str, str], figure_size: tuple[float, float]) -> theme:
    return theme_minimal(base_family=FONT) + theme(
        figure_size=figure_size,
        plot_background=element_rect(fill=pal["surface"], color="none"),
        panel_background=element_rect(fill=pal["surface"], color="none"),
        panel_grid_major_y=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.06,
        strip_background=element_rect(fill=pal["surface"], color="none"),
        strip_text=element_text(
            size=9.5, color=pal["ink"], weight="bold", ha="left", margin={"b": 6}
        ),
        axis_title=element_blank(),
        axis_text_y=element_text(size=9, color=pal["ink"]),
        axis_ticks=element_blank(),
        legend_position="top",
        legend_direction="horizontal",
        legend_title=element_blank(),
        legend_text=element_text(size=8.5, color=pal["ink_secondary"]),
        legend_box_spacing=0.01,
        legend_key_spacing_x=6,
        plot_title=element_text(
            size=12, color=pal["ink"], weight="bold", ha="left", margin={"b": 2}
        ),
        plot_subtitle=element_text(
            size=9, color=pal["ink_secondary"], ha="left", margin={"b": 10}
        ),
        plot_caption=element_text(
            size=7.5, color=pal["ink_muted"], ha="left", margin={"t": 10}
        ),
    )


# ---------------------------------------------------------------------------
# Figure 1 - corrected scope, with the published value annotated where it differs
# ---------------------------------------------------------------------------
def figure_scope(
    long: pd.DataFrame, pal: dict[str, str], show_published_notes: bool = False
) -> ggplot:
    """Corrected counts, one bar per model.

    By default the panel carries only the corrected numbers, so the figure reads
    as the table it replaces.  ``show_published_notes`` restores the grey
    "published N (delta)" annotation beside each disagreeing bar; the same
    information is in `fig_table2_1_delta`, drawn to scale.
    """
    bars = long.loc[long["reported"]].copy()
    notes = long.loc[long["changed"]].copy()
    missing = long.loc[~long["reported"]].copy()

    plot = (
        ggplot()
        # One series, one hue: the count is a magnitude and the model is already
        # named on the axis, so a per-model colour would only restate bar length.
        # A single series needs no legend - the title says what is plotted.
        + geom_col(bars, aes("model", "corrected"), fill=pal["series_corrected"], width=0.42)
        + geom_text(
            bars,
            aes("model", "tip_x", label="tip_label"),
            ha="left",
            size=8.5,
            color=pal["ink"],
            fontweight="bold",
        )
        + geom_text(
            missing,
            aes("model", 0, label="missing_label"),
            ha="left",
            size=8,
            fontstyle="italic",
            color=pal["ink_muted"],
            nudge_y=0.0,
        )
        + coord_flip()
        + facet_wrap("~statistic", scales="free_x", ncol=1)
        # Headroom is sized for whatever the longest label on a row is: the
        # published note when it is shown, otherwise just the tip value.
        + scale_y_continuous(expand=(0, 0, 0.55 if show_published_notes else 0.12, 0))
        + labs(
            title="Training-data scope across KSI prediction methods",
            subtitle=(
                "Counts recomputed from each dataset's own release files "
                "(train + validation + test)."
            ),
            caption=(
                (
                    "Grey notes give the value printed in Table 2.1 where it "
                    "disagrees with the recount, and the difference. KolossuS\n"
                    "reports no training-set composition. "
                    if show_published_notes
                    else "KolossuS reports no training-set composition. "
                )
                + "Value scales differ per panel; the three statistics are never "
                "compared on a shared axis."
            ),
        )
        + base_theme(pal, (8.6, 6.2))
        + theme(axis_text_x=element_blank(), panel_grid_major_x=element_blank())
    )

    if show_published_notes:
        # The published value rides the same row as a note. It is deliberately
        # NOT drawn as a marker on the scale: every disagreement is within a few
        # percent of the bar tip, so a marker there is inseparable from the tip
        # label.
        plot += geom_text(
            notes,
            aes("model", "delta_x", label="delta_label"),
            ha="left",
            size=7.4,
            color=pal["ink_secondary"],
        )
    return plot


# ---------------------------------------------------------------------------
# Figure 2 - the disagreement, on its own
# ---------------------------------------------------------------------------
def figure_delta(long: pd.DataFrame, pal: dict[str, str]) -> ggplot:
    data = long.loc[long["reported"]].copy()
    data["direction"] = [
        "Corrected count is higher"
        if d > 0
        else ("Corrected count is lower" if d < 0 else "Reproduces exactly")
        for d in data["delta"]
    ]
    data["direction"] = pd.Categorical(
        data["direction"],
        categories=[
            "Corrected count is lower",
            "Reproduces exactly",
            "Corrected count is higher",
        ],
        ordered=True,
    )
    reach = max(1.0, float(data["pct_delta"].abs().max()))
    data["label"] = [
        "exact match" if d == 0 else f"{int(d):+,}"
        for d in data["delta"]
    ]
    # Always outside the bar end, never over the fill: a positive bar labels to
    # its right, a negative bar to its left, a zero-length bar to the right of
    # the baseline.
    gap = 0.035 * reach
    data["label_x"] = [
        (p + gap) if d > 0 else ((p - gap) if d < 0 else gap)
        for d, p in zip(data["delta"], data["pct_delta"])
    ]

    plot = (
        ggplot(data, aes("model", "pct_delta"))
        + geom_hline(yintercept=0, color=pal["axis"], size=0.5)
        + geom_col(aes(fill="direction"), width=0.42)
        + geom_text(
            data.loc[data["delta"] >= 0],
            aes("model", "label_x", label="label"),
            ha="left",
            size=7.4,
            color=pal["ink_secondary"],
        )
        + geom_text(
            data.loc[data["delta"] < 0],
            aes("model", "label_x", label="label"),
            ha="right",
            size=7.4,
            color=pal["ink_secondary"],
        )
        + coord_flip()
        + facet_wrap("~statistic", nrow=1)
        + scale_fill_manual(
            values=[pal["diverging_neg"], pal["diverging_mid"], pal["diverging_pos"]],
            drop=False,
        )
        + scale_y_continuous(
            expand=(0.22, 0, 0.32, 0),
            labels=lambda values: [f"{v:+.0f}%" for v in values],
        )
        + guides(fill=guide_legend(override_aes={"size": 0.6}))
        + labs(
            title="Where Table 2.1 disagrees with the datasets",
            subtitle=(
                "Percent change from the published figure to the recounted one; "
                "labels give the absolute difference."
            ),
            caption=(
                "The published KINBERT kinase count summed the database and lysate "
                "tables without de-duplicating the kinases they share.\n"
                "The two KSI shortfalls track dataset release differences, not "
                "counting errors."
            ),
        )
        + base_theme(pal, (11.0, 3.4))
        + theme(
            axis_text_x=element_text(size=8, color=pal["ink_muted"]),
            panel_grid_major_x=element_line(color=pal["grid"], size=0.4),
        )
    )
    return plot


# ---------------------------------------------------------------------------
def save(plot: ggplot, out_dir: Path, stem: str, size: tuple[float, float]) -> None:
    for suffix, dpi in ((".pdf", 300), (".png", 200)):
        plot.save(
            out_dir / f"{stem}{suffix}",
            width=size[0],
            height=size[1],
            dpi=dpi,
            verbose=False,
        )
    print(f"  {out_dir / (stem + '.pdf')}")
    print(f"  {out_dir / (stem + '.png')}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot the corrected Table 2.1 training-data scope.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corrected-csv",
        type=Path,
        default=HERE / "table2_1_validation" / "table2_1_corrected.csv",
        help="output of validate_table2_1_training_data.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=HERE / "table2_1_validation",
        help="directory for the figures and the plotted-data CSV",
    )
    parser.add_argument(
        "--show-published-notes",
        action="store_true",
        help=(
            "annotate each disagreeing bar with the value printed in Table 2.1 "
            "and the difference; off by default, since fig_table2_1_delta shows "
            "the same information to scale"
        ),
    )
    parser.add_argument(
        "--theme",
        choices=sorted(PALETTES),
        default="light",
        help="palette mode; dark uses the same hues stepped for a dark surface",
    )
    args = parser.parse_args(argv)

    if not args.corrected_csv.exists():
        parser.error(
            f"{args.corrected_csv} not found - run "
            "validate_table2_1_training_data.py first"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pal = PALETTES[args.theme]

    long, order = load_long(args.corrected_csv)
    long = add_facet_geometry(long)
    print(f"model order (top to bottom): {', '.join(reversed(order))}")

    # The table-view twin: exactly the rows the figures draw.
    plotted = long[
        ["model", "statistic", "corrected", "published", "delta", "pct_delta"]
    ].copy()
    plotted.to_csv(args.out_dir / "table2_1_plotted.csv", index=False)

    suffix = "" if args.theme == "light" else f"_{args.theme}"
    print("figures:")
    save(
        figure_scope(long, pal, args.show_published_notes),
        args.out_dir,
        f"fig_table2_1_scope{suffix}",
        (8.6, 6.2),
    )
    save(figure_delta(long, pal), args.out_dir, f"fig_table2_1_delta{suffix}", (11.0, 3.4))
    print(f"  {args.out_dir / 'table2_1_plotted.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
