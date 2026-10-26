"""
Check the two combined benchmark tables against the figures themselves.

The other verification in this directory compares a regenerated table with
numbers the source notebook printed.  That proves the table and the notebook
agree, but it trusts that the notebook's printed numbers and its saved figure
came from the same object.

This script closes that gap for the two summary bar figures by measuring the
figures.  Both are vector PDFs, so each bar is a rectangle with an exact
height, and the y-axis tick labels give the scale.  Bar colour identifies the
model, and position identifies the facet and the metric.  The measured value is
then compared with the corresponding row of the table.

    python3 eval_table_scripts/verify_bars_against_pdf.py

Precision: the axis spans roughly 290 points for a score of 1.0, so a point of
geometry is about 0.0034.  Agreement is required to 0.002.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pymupdf

from eval_table_utils import FIG_DIR, table_path, write_verification, Checker

TOLERANCE = 0.002

# plotnine assigned these colours in the order the models first appear.
PALETTE = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6",
           "#1ABC9C", "#E91E63", "#FF5722", "#00BCD4", "#8BC34A"]

FIGURES = {
    "benchmark_model_structure_bar": [
        "Phosformer", "Kolossus", "KINBERT:TAPE", "KINBERT:ESM",
        "KINBERT:ESM-CONCAT", "KINBERT:ESM-FROZEN",
    ],
    "benchmark_generalizability_bar": [
        "Phosformer", "Kolossus", "KINBERT:TAPE", "KINBERT:TAPE-No Dark",
        "KINBERT:TAPE-Non Human",
    ],
}


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(round(int(h[i:i + 2], 16) / 255, 3) for i in (0, 2, 4))


def read_bars(pdf_path, models):
    """Measure every bar in a faceted bar figure."""
    page = pymupdf.open(pdf_path)[0]
    colour_to_model = {hex_to_rgb(PALETTE[i]): m for i, m in enumerate(models)}

    words = page.get_text("words")

    # --- y scale, from the two extreme tick labels on the value axis ---
    ticks = {}
    for x0, y0, x1, y1, text, *_ in words:
        try:
            value = float(text)
        except ValueError:
            continue
        if 0.0 <= value <= 1.0 and x1 < 70:
            ticks[value] = (y0 + y1) / 2
    lo, hi = min(ticks), max(ticks)
    scale = (ticks[lo] - ticks[hi]) / (hi - lo)
    baseline = ticks[lo]

    # --- metric groups: one label per metric per facet ---
    groups = [((x0 + x1) / 2, text) for x0, y0, x1, y1, text, *_ in words
              if text in ("PR_AUC", "ROC_AUC")]
    groups.sort()

    # --- facet titles, sitting in the strip above each panel ---
    strips = {}
    for x0, y0, x1, y1, text, *_ in words:
        if 35 < y0 < 60:
            strips.setdefault(round((y0 + y1) / 2), []).append((x0, x1, text))
    parts = sorted(strips.items())[0][1]
    panels = []
    for x0, x1, text in sorted(parts):
        if panels and x0 - panels[-1][1] < 12:
            panels[-1] = (panels[-1][0], x1, panels[-1][2] + " " + text)
        else:
            panels.append((x0, x1, text))
    panel_centres = [((a + b) / 2, name) for a, b, name in panels]

    def nearest(value, options):
        return min(options, key=lambda o: abs(o[0] - value))

    # --- the bars ---
    rows = []
    for item in page.get_drawings():
        fill = item.get("fill")
        if fill is None:
            continue
        key = tuple(round(v, 3) for v in fill)
        if key not in colour_to_model:
            continue
        rect = item["rect"]
        if abs(rect.y1 - baseline) > 1.5:
            continue  # a legend key: every real bar is rooted on the axis
        centre = (rect.x0 + rect.x1) / 2
        _, metric = nearest(centre, groups)
        _, panel = nearest(centre, panel_centres)
        rows.append({
            "Test_Dataset": panel,
            "Metric": metric,
            "Model": colour_to_model[key],
            "measured": (baseline - rect.y0) / scale,
        })
    return pd.DataFrame(rows)


def main():
    checks = []
    for name, models in FIGURES.items():
        pdf = os.path.join(FIG_DIR, f"{name}.pdf")
        print(f"\n=== {name}.pdf ===")
        measured = read_bars(pdf, models)
        table = pd.read_csv(table_path(f"{name}.csv"))

        # Facet titles in the figure are wrapped; match on a squashed form.
        def squash(s):
            return "".join(str(s).split()).lower()

        table["_key"] = (table["Test_Dataset"].map(squash) + "|" +
                         table["Metric"] + "|" + table["Model"])
        measured["_key"] = (measured["Test_Dataset"].map(squash) + "|" +
                            measured["Metric"] + "|" + measured["Model"])

        merged = measured.merge(table[["_key", "Score"]], on="_key", how="left")
        print(f"  {len(measured)} bars measured, "
              f"{len(table)} rows in the table, "
              f"{int(merged['Score'].notna().sum())} matched up")

        c = Checker(f"{name}.pdf (measured from the figure)")
        for _, r in merged.iterrows():
            dataset, metric, model = r["_key"].split("|")
            c.check(f"{r['Test_Dataset']} / {metric} / {model}",
                    float(r["measured"]),
                    float(r["Score"]) if pd.notna(r["Score"]) else None,
                    TOLERANCE)
        checks.append(c.report())

        unmatched = set(table["_key"]) - set(measured["_key"])
        if unmatched:
            print(f"  table rows with no bar in the figure: {len(unmatched)}")
            for k in sorted(unmatched):
                print(f"    {k}")

    write_verification(checks, "VERIFICATION_07_bars_measured_from_pdf.csv")


if __name__ == "__main__":
    main()
