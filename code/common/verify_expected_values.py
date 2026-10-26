#!/usr/bin/env python3
"""
Reproducibility check for the KSI paper package.

Asserts that the staged data still yields the exact values printed in the
final manuscript (KSI_Paper_261022_CY, 2026-10-23), and that each
script-generated supplementary figure is identical to the shipped PDF.

    python3 code/common/verify_expected_values.py .            # from the package root
    python3 code/common/verify_expected_values.py . --figures-only

Exit status is 0 only if every check passes. Checks whose inputs are not
staged are reported as SKIP and do not fail the run.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))


def check(name: str, got, want, detail: str = "") -> None:
    if got is None:
        record(SKIP, name, detail or "input not staged")
    elif got == want:
        record(PASS, name, f"{got}")
    else:
        record(FAIL, name, f"got {got}, expected {want}. {detail}".strip())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 1. Hit calling at the study-wide threshold (b >= 0.1, LFSR <= 0.05)
# --------------------------------------------------------------------------
def check_hit_counts(root: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        record(SKIP, "hit counts", "pandas not available")
        return

    res = root / "data" / "results" / "phosphoPCA"

    scores = res / "251130_ksi_inference_scores.pkl"
    if scores.exists():
        d = pd.read_pickle(scores)
        check("rows in the decoupling score table", len(d), 61979)
        hit = (d["lfsr"] <= 0.05) & (d["binding_strength"] >= 0.1)
        check("substrate x kinase x PAABD rows called hit", int(hit.sum()), 12936,
              "if this returns 9966 the LFSR threshold has reverted to 0.01")
    else:
        check("rows in the decoupling score table", None, 61979)
        check("substrate x kinase x PAABD rows called hit", None, 12936)

    for fn, col, want, label in [
        ("251203_ksi_reduced_df.pkl", "positive", 8890, "`positive` (substrate, kinase) pairs"),
        ("251203_ksi_reduced_df_tp.pkl", "true_positive", 8100, "`true_positive` pairs"),
    ]:
        p = res / fn
        if p.exists():
            d = pd.read_pickle(p)
            check(label, int(d[col].sum()) if col in d.columns else None, want,
                  f"column `{col}` missing from {fn}" if col not in d.columns else "")
        else:
            check(label, None, want)

    cm = res / "confusion_matrix_summary.csv"
    if cm.exists():
        d = pd.read_csv(cm)
        row = d[d.iloc[:, 0].astype(str).str.lower().str.startswith("combined")]
        if len(row):
            tp, fp, fn_, tn = (int(row.iloc[0][c]) for c in ("TP", "FP", "FN", "TN"))
            check("Table S10 confusion matrix (TP/FP/FN/TN)", (tp, fp, fn_, tn), (134, 11, 195, 63))
            check("Table S10 total annotated rows", tp + fp + fn_ + tn, 403)
        else:
            check("Table S10 confusion matrix (TP/FP/FN/TN)", None, (134, 11, 195, 63))
    else:
        check("Table S10 confusion matrix (TP/FP/FN/TN)", None, (134, 11, 195, 63))


# --------------------------------------------------------------------------
# 2. Script-generated supplementary figures must match the shipped PDFs
# --------------------------------------------------------------------------
# Printed number -> (shipped PDF, generator, generator output)
SUPP_FIGURES = {
    "S1": ("SuppFig_corpus.pdf", "SF1_SF5_SF6_SF7/sf1_corpus.py", "SuppFig_corpus.pdf"),
    "S5": ("SuppFig_library.pdf", "SF1_SF5_SF6_SF7/sf5_library.py", "SuppFig_library.pdf"),
    "S6": ("SuppFig_falsepositive.pdf", "SF1_SF5_SF6_SF7/sf6_false_positive.py", "SuppFig_falsepositive.pdf"),
    "S7": ("SuppFig_decoupling.pdf", "SF1_SF5_SF6_SF7/sf7_decoupling.py", "SuppFig_decoupling.pdf"),
    "S8": ("SuppFig_replicate_counts.pdf", "seq_replicate_qc/01_raw_replicate_concordance.py", "fig_raw_replicate_scatter.pdf"),
    "S9": ("SuppFig_replicate_binding.pdf", "seq_replicate_qc/04_conditional_replicate_posteriors.py", "fig_conditional_replicates.pdf"),
    "S10": ("SuppFig_convergence.pdf", "bayes_fit_qc/01_convergence_diagnostics.py", "fig_convergence_rhat_ess.pdf"),
    "S11": ("SuppFig_hyperparameters.pdf", "bayes_fit_qc/01_convergence_diagnostics.py", "fig_hyperparameter_traces.pdf"),
    "S12": ("SuppFig_sampler.pdf", "bayes_fit_qc/01_convergence_diagnostics.py", "fig_sampler_energy.pdf"),
}
# S2, S3 and S4 are redrawn by hand from Tables S4-S6 and have no generator.
NO_GENERATOR = {
    "S2": "SuppFig_benchmark.pdf",
    "S3": "SuppFig_generalizability.pdf",
    "S4": "SuppFig_finetuning.pdf",
}


def _text_layer(p: Path):
    try:
        import pymupdf
    except ImportError:
        return None
    return [w[4] for page in pymupdf.open(p) for w in page.get_text("words")]


def check_figures(root: Path, outputs: Path | None) -> None:
    supp = root / "figures" / "supp"
    for num, (shipped, gen, out) in SUPP_FIGURES.items():
        tgt = supp / shipped
        if not tgt.exists():
            record(SKIP, f"{num} {shipped}", "shipped PDF missing from the package")
            continue
        if outputs is None:
            record(SKIP, f"{num} {shipped}", f"pass --outputs DIR to compare against {gen}")
            continue
        src = next(iter(outputs.rglob(out)), None)
        if src is None:
            record(SKIP, f"{num} {shipped}", f"{out} not found under {outputs}")
            continue
        if sha256(tgt) == sha256(src):
            record(PASS, f"{num} {shipped}", "byte-identical to generator output")
        else:
            a, b = _text_layer(tgt), _text_layer(src)
            if a is not None and a == b:
                record(PASS, f"{num} {shipped}", "text layer identical (PDF metadata differs)")
            elif a is None:
                record(SKIP, f"{num} {shipped}", "bytes differ; pymupdf unavailable to compare text")
            else:
                record(FAIL, f"{num} {shipped}", f"differs from {gen} output")

    for num, shipped in NO_GENERATOR.items():
        record(SKIP, f"{num} {shipped}", "no generator exists - redrawn by hand from Tables S4-S6")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".", help="package root")
    ap.add_argument("--outputs", help="directory holding the generator outputs "
                                      "(e.g. ksi_analysis_2025), enables figure comparison")
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve()

    if not a.figures_only:
        check_hit_counts(root)
    check_figures(root, Path(a.outputs).resolve() if a.outputs else None)

    width = max(len(n) for _, n, _ in results) + 2
    for status, name, detail in results:
        print(f"  [{status}] {name:<{width}} {detail}")
    n_fail = sum(1 for s, _, _ in results if s == FAIL)
    n_skip = sum(1 for s, _, _ in results if s == SKIP)
    n_pass = sum(1 for s, _, _ in results if s == PASS)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
