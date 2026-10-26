"""
Tables for the data-source UpSet figures.

Source notebook: ``eval_0_datasource_stats.ipynb`` (cells 15, 27, 28).

Figures covered
---------------
  eval_figs/ks_pair_upset.pdf            kinase-substrate pairs per source
  eval_figs/ks_pair_upset_nkinase.pdf    distinct kinases per source
  eval_figs/ks_pair_upset_norganism.pdf  distinct organisms per source

Why this script reads the figures instead of the data
-----------------------------------------------------
The notebook builds these plots from four per-database files under
``/kinase_ml/data/raw/raw`` (Phospho.ELM, PhosphoSitePlus, KinasePhos 3.0 and
the cell-lysate screen), plus an identifier mapping table.  None of those
survive on the restored tree, and the closest available substitute,
``interaction_classed_221017_wsource.pkl``, is the merged and filtered union
rather than the raw per-database sets.  It gives visibly smaller counts, so
recomputing from it would produce a table that silently disagrees with the
figure.

The figures themselves are vector PDFs and carry their contents exactly: each
count is real text, and each matrix dot records membership in its fill opacity
(solid means the source is in that intersection, faded means it is not).  This
script decodes that, so the tables it writes are the figures' own numbers.

It also runs the reconstruction from the surviving data and writes it beside
the extracted table, with a column giving the difference, so the size of the
gap is on record rather than assumed.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import pymupdf

from eval_table_utils import Checker, FIG_DIR, RESTORE_ROOT, table_path, write_verification

FIGURES = {
    "ks_pair_upset": "kinase-substrate pairs",
    "ks_pair_upset_nkinase": "distinct kinases",
    "ks_pair_upset_norganism": "distinct organisms",
}

# Filled dots are drawn at full opacity, unfilled ones at 0.18.
FILLED_OPACITY_MIN = 0.5


def extract_upset(pdf_path):
    """Decode an UpSet figure into its category, set-size and intersection tables."""
    page = pymupdf.open(pdf_path)[0]
    drawings = page.get_drawings()

    # --- the matrix dots: circles, one per (intersection, category) ---
    dots = [d for d in drawings
            if any(it[0] == "c" for it in d["items"]) and d["rect"].width < 20]
    if not dots:
        raise RuntimeError(f"no matrix dots found in {pdf_path}")

    def cluster(values, tol=2.0):
        out = []
        for v in sorted(values):
            if not out or abs(v - out[-1][-1]) > tol:
                out.append([v])
            else:
                out[-1].append(v)
        return [sum(g) / len(g) for g in out]

    col_x = cluster([(d["rect"].x0 + d["rect"].x1) / 2 for d in dots])
    row_y = cluster([(d["rect"].y0 + d["rect"].y1) / 2 for d in dots])

    def nearest(value, centres):
        return min(range(len(centres)), key=lambda i: abs(centres[i] - value))

    membership = {}
    for d in dots:
        cx = (d["rect"].x0 + d["rect"].x1) / 2
        cy = (d["rect"].y0 + d["rect"].y1) / 2
        opacity = d.get("fill_opacity")
        opacity = 1.0 if opacity is None else opacity
        membership[(nearest(cx, col_x), nearest(cy, row_y))] = opacity >= FILLED_OPACITY_MIN

    # --- category names, from the labels beside the set-size bars ---
    words = page.get_text("words")
    labels = {}
    for x0, y0, x1, y1, text, *_ in words:
        cy = (y0 + y1) / 2
        r = nearest(cy, row_y)
        if abs(row_y[r] - cy) < 8 and not text.replace(".", "").isdigit():
            labels.setdefault(r, []).append((x0, text))
    categories = {r: " ".join(t for _, t in sorted(v)) for r, v in labels.items()}

    # --- set sizes: the number printed beside each row's bar ---
    # Taken from whole words rather than loose digits, so that the "3.0" inside
    # a name like Kinasephos3.0 cannot be mistaken for part of a count.
    matrix_left = min(col_x) - 20
    set_sizes = {}
    for x0, y0, x1, y1, text, *_ in words:
        cy = (y0 + y1) / 2
        if not text.isdigit() or x1 >= matrix_left:
            continue
        r = nearest(cy, row_y)
        if abs(row_y[r] - cy) < 8:
            set_sizes[r] = int(text)

    # --- intersection counts: the number printed above each bar ---
    # Neighbouring labels can touch horizontally, so digits are grouped into
    # runs first: characters of one number share a baseline and sit flush
    # against each other, while separate labels differ in height (each label
    # rides on top of its own bar) or are separated by a visible gap.
    spacing = (max(col_x) - min(col_x)) / max(len(col_x) - 1, 1)
    plot_left = min(col_x) - spacing / 2
    bar_top = min(row_y) - 15

    digits = []
    raw = page.get_text("rawdict")
    for block in raw["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                for ch in span.get("chars", []):
                    if ch["c"].isdigit():
                        x0, y0, x1, y1 = ch["bbox"]
                        digits.append({"c": ch["c"], "x0": x0, "x1": x1,
                                       "y": (y0 + y1) / 2})

    digits = [d for d in digits if d["y"] < bar_top]
    digits.sort(key=lambda d: (round(d["y"], 1), d["x0"]))

    runs = []
    for d in digits:
        if runs:
            prev = runs[-1][-1]
            gap = d["x0"] - prev["x1"]
            same_label = abs(d["y"] - prev["y"]) < 1.0 and 0.0 <= gap < 3.0
        else:
            same_label = False
        if same_label:
            runs[-1].append(d)
        else:
            runs.append([d])

    counts = {}
    for run in runs:
        centre = (run[0]["x0"] + run[-1]["x1"]) / 2
        if centre < plot_left:
            continue  # a y-axis tick label, not a bar label
        column = nearest(centre, col_x)
        value = int("".join(d["c"] for d in run))
        if column in counts:
            raise RuntimeError(
                f"two labels resolved to bar {column} in {os.path.basename(pdf_path)}: "
                f"{counts[column]} and {value}")
        counts[column] = value

    # --- assemble ---
    cat_rows = pd.DataFrame([
        {"category": categories.get(r, f"row{r}"),
         "set_size": set_sizes.get(r),
         "matrix_row": r}
        for r in range(len(row_y))
    ])

    inter_rows = []
    for i in range(len(col_x)):
        members = [categories.get(r, f"row{r}") for r in range(len(row_y))
                   if membership.get((i, r))]
        row = {"intersection_index": i,
               "degree": len(members),
               "members": " & ".join(sorted(members)),
               "intersection_size": counts.get(i)}
        for r in range(len(row_y)):
            row[categories.get(r, f"row{r}")] = int(bool(membership.get((i, r))))
        inter_rows.append(row)
    inter = pd.DataFrame(inter_rows)
    return cat_rows, inter


# --------------------------------------------------------------------------
# The reconstruction attempt, for the record
# --------------------------------------------------------------------------

INTERACTIONS = os.path.join(RESTORE_ROOT, "data", "data", "interactions", "interactions")
SOURCE_KEYS = {"Phospho.ELM": "phosphoelm",
               "PhosphoSitePlus": "phosphositeplus",
               "Kinasephos3.0": "kinasephos"}


def reconstruct_set_sizes(kind):
    """Recompute each source's set size from the data that did survive.

    Returns ``None`` when the inputs for that figure are not available.
    """
    ws_path = os.path.join(INTERACTIONS, "interaction_classed_221017_wsource.pkl")
    lysate_path = os.path.join(INTERACTIONS, "lysate_ksi_cleaned_221115.pkl")
    if not (os.path.exists(ws_path) and os.path.exists(lysate_path)):
        return None

    ws = pd.read_pickle(ws_path).copy()
    lysate = pd.read_pickle(lysate_path).loc[:, ["kinase", "substrate", "p_aa", "p_ind"]].copy()

    def ks_pair(df):
        return (df["substrate"].astype(str) + "-" + df["kinase"] + "-" +
                df["p_aa"].astype(str) + "-" + df["p_ind"].astype(str))

    ws["ks_pair"] = ks_pair(ws)
    lysate["ks_pair"] = ks_pair(lysate)

    if kind == "organism":
        org_path = os.path.join(INTERACTIONS, "all_kinase_organism_info_old.tsv")
        if not os.path.exists(org_path):
            return None
        org = pd.read_csv(org_path, sep="\t").loc[:, ["From", "Organism"]]
        org.columns = ["kinase", "organism"]
        ws = pd.merge(ws, org, on="kinase")
        lysate = pd.merge(lysate, org, on="kinase")
        col = "organism"
    elif kind == "kinase":
        col = "kinase"
    else:
        col = "ks_pair"

    out = {}
    for label, key in SOURCE_KEYS.items():
        out[label] = int(ws.loc[ws["source_all"].str.contains(key), col].nunique())
    out["Cell Lysate LC-MS"] = int(lysate[col].nunique())
    return out


KIND_BY_FIGURE = {"ks_pair_upset": "pair",
                  "ks_pair_upset_nkinase": "kinase",
                  "ks_pair_upset_norganism": "organism"}


def main():
    checks = []

    for name, what in FIGURES.items():
        pdf = os.path.join(FIG_DIR, f"{name}.pdf")
        print(f"\n=== {name}.pdf  ({what}) ===")
        cats, inter = extract_upset(pdf)

        p = table_path(f"{name}_set_sizes.csv")
        cats.to_csv(p, index=False)
        print(f"  wrote {os.path.basename(p)}  ({len(cats)} sources)")

        p = table_path(f"{name}_intersections.csv")
        inter.to_csv(p, index=False)
        print(f"  wrote {os.path.basename(p)}  ({len(inter)} intersections)")

        # The extraction is self-checking: every source's set size must equal
        # the sum of the intersections that source takes part in.
        c = Checker(f"{name}.pdf")
        for _, row in cats.iterrows():
            category = row["category"]
            total = int(inter.loc[inter[category] == 1, "intersection_size"].sum())
            c.check(f"{category}: intersections sum to its set size",
                    float(total), float(row["set_size"]), 0)
        c.check("every intersection has a count",
                float(inter["intersection_size"].notna().sum()), float(len(inter)), 0)
        c.check("every intersection has at least one member",
                float((inter["degree"] > 0).sum()), float(len(inter)), 0)
        checks.append(c.report())

        # Reconstruction from surviving data, recorded next to the figure values.
        recon = reconstruct_set_sizes(KIND_BY_FIGURE[name])
        if recon is None:
            print("  reconstruction skipped: source data not on this machine")
            continue
        comp = cats.copy()
        comp["reconstructed_set_size"] = comp["category"].map(recon)
        comp["difference"] = comp["reconstructed_set_size"] - comp["set_size"]
        p = table_path(f"{name}_reconstruction_gap.csv")
        comp.to_csv(p, index=False)
        n_match = int((comp["difference"] == 0).sum())
        print(f"  wrote {os.path.basename(p)}: "
              f"{n_match}/{len(comp)} sources reproduce from surviving data")

    write_verification(checks, "VERIFICATION_00_datasource_upset.csv")


if __name__ == "__main__":
    main()
