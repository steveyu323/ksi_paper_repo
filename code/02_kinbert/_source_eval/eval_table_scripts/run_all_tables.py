"""
Regenerate every table under ``evaluation/eval_tables`` and summarise how well
each one matches its figure.

Usage
-----
    python3 eval_table_scripts/run_all_tables.py            # everything
    python3 eval_table_scripts/run_all_tables.py 02 03      # only those scripts

Runtime is dominated by the benchmark scripts, which load several very large
prediction files; expect roughly fifteen minutes for a full run.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from eval_table_utils import SCRIPT_DIR, table_path

# key, script, description, verification file it writes.
# Steps 01 (v1 model comparison) and 06 (full-kinome virus matrix) were
# dropped from this package because nothing they produce is printed; see
# INVENTORY.md. The keys below are therefore not contiguous.
SCRIPTS = [
    ("00", "table_00_datasource_upset.py", "data-source UpSet figures",
     "VERIFICATION_00_datasource_upset.csv"),
    ("02", "table_02_model_structure.py", "model-architecture benchmarks",
     "VERIFICATION_02_model_structure.csv"),
    ("03", "table_03_generalizability.py", "generalizability holdouts",
     "VERIFICATION_03_generalizability.csv"),
    ("04", "table_04_ksi_screen.py", "KSI screening benchmark",
     "VERIFICATION_04_ksi_screen.csv"),
    ("05", "table_05_virus_demo.py", "curated virus figures",
     "VERIFICATION_05_virus_demo.csv"),
    # Runs last: it measures the finished figures and compares them with the
    # tables the steps above wrote.
    ("07", "verify_bars_against_pdf.py", "summary bars measured from the PDFs",
     "VERIFICATION_07_bars_measured_from_pdf.csv"),
]


def main(selected):
    failures = []
    for key, script, description, _ in SCRIPTS:
        if selected and key not in selected:
            continue
        print("\n" + "=" * 70, flush=True)
        print(f"[{key}] {description}  ({script})", flush=True)
        print("=" * 70, flush=True)
        result = subprocess.run(
            [sys.executable, "-W", "ignore::FutureWarning",
             os.path.join(SCRIPT_DIR, script)],
            cwd=os.path.dirname(SCRIPT_DIR),
        )
        if result.returncode != 0:
            failures.append(script)

    # Roll the per-script verification files into one summary.
    parts = []
    for key, script, _, verification in SCRIPTS:
        path = table_path(verification)
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.insert(0, "script", script)
            parts.append(df)

    if parts:
        summary = pd.concat(parts, ignore_index=True)
        out = table_path("VERIFICATION_summary.csv")
        summary.to_csv(out, index=False)
        n_ok = int(summary["match"].sum())
        print("\n" + "=" * 70)
        print(f"{n_ok}/{len(summary)} checked values match their figure")
        print(f"wrote {os.path.basename(out)}")
        mismatches = summary[~summary["match"]]
        if len(mismatches):
            print("\nvalues that do not match:")
            print(mismatches[["figure", "quantity", "regenerated",
                              "figure_value"]].to_string(index=False))

    if failures:
        print(f"\nscripts that failed: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(set(sys.argv[1:])))
