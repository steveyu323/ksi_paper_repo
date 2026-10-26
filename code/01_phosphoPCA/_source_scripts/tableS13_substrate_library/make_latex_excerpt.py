#!/usr/bin/env python3
"""Emit the representative-rows LaTeX block for Table S13.

Replaces the placeholder rows in the manuscript table with real substrates,
keeping the existing column structure. Writes to outputs/tableS13_excerpt.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg  # noqa: E402

CSV = cfg.OUT_DIR / "tableS13_substrate_library.csv"


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main() -> int:
    df = pd.read_csv(CSV)

    # Two representative rows per (library, class), in table order.
    order = [
        ("Y", "Control (documented)"),
        ("Y", "Candidate (PhosphoSitePlus)"),
        ("Y", "Candidate (animal phosphoproteome)"),
        ("ST", "Control (documented)"),
        ("ST", "Candidate (PhosphoSitePlus)"),
    ]

    rows = []
    for lib, cls in order:
        sel = df[(df.library == lib) & (df.substrate_class == cls)].head(1)
        for _, r in sel.iterrows():
            pos = (
                ""
                if pd.isna(r.phosphosite_position)
                else f"{r.phosphoacceptor}{int(r.phosphosite_position)}"
            )
            acc = "---" if pd.isna(r.uniprot_acc) else esc(r.uniprot_acc)
            amb = (
                r"$^{\dagger}$" if r.n_protein_matches > 1 else ""
            )
            rows.append(
                f"{esc(r.substrate_id)} & {acc}{amb} & {pos} & "
                f"\\texttt{{{r.window_15mer}}} & {esc(r.substrate_class)} \\\\"
            )

    total = len(df)
    body = "\n".join(rows)
    body += (
        "\n\\multicolumn{5}{@{}l}{\\textit{[" + f"{total:,} substrate rows "
        "--- complete manifest in the deposited CSV]}} \\\\"
    )

    out = cfg.OUT_DIR / "tableS13_excerpt.tex"
    out.write_text(body + "\n")
    print(body)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
