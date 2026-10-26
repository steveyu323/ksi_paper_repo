#!/usr/bin/env python3
"""Flatten spark_results.h5 into one tidy per-cell-per-frame table.

The deposited `spark_all_runs.csv` had no surviving writer cell in any of the
notebooks, so this script re-creates it from the HDF5 store. It mirrors the
layout written by `spark_utils.save_to_hdf5`: one group per run, holding a
`spark_df` subgroup with one dataset per column, a `tracked_masks` array, a
sparse `puncta_coords` group, and a JSON `metadata` attribute.

Only `spark_df` and the metadata are needed for the table; masks and puncta
coordinates are left in the HDF5.

One fix relative to the original CSV: pandas wrote the HDF5's fixed-length
byte strings through unchanged, so string columns came out as `b'tbk1'`
rather than `tbk1`. Byte columns are decoded here, which changes those cells
in the regenerated file. Pass --raw-bytes to reproduce the original.

    ./export_spark_table.py <spark_results.h5> -o spark_all_runs.csv
"""

import argparse
import json
import sys

import h5py
import numpy as np
import pandas as pd

# Column order of the originally deposited table, so a regenerated file diffs
# cleanly against it. Columns outside this list are appended alphabetically.
CANONICAL_ORDER = [
    "cell_area_px",
    "cell_id",
    "cell_thresh",
    "cell_total_int",
    "condition",
    "frame",
    "puncta_area_px",
    "puncta_count",
    "puncta_total_int",
    "reporter",
    "spark_norm",
    "spark_norm_ctrl",
    "spark_signal",
    "time_min",
    "run_key",
]


def decode(values, raw_bytes):
    """Turn an HDF5 byte-string column into real text."""
    if raw_bytes or values.dtype.kind not in ("S", "O"):
        return values
    return np.array(
        [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in values],
        dtype=object,
    )


def read_run(group, run_key, raw_bytes):
    if "spark_df" not in group:
        return None, None

    df_grp = group["spark_df"]
    data = {col: decode(df_grp[col][...], raw_bytes) for col in df_grp.keys()}
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        raise ValueError(f"{run_key}: ragged spark_df columns {lengths}")

    df = pd.DataFrame(data)
    df["run_key"] = run_key

    meta = {}
    if "metadata" in group.attrs:
        raw = group.attrs["metadata"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {}
    meta["run_key"] = run_key
    meta["n_rows"] = len(df)
    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5_path")
    ap.add_argument("-o", "--out", default="spark_all_runs.csv")
    ap.add_argument(
        "--metadata-out",
        help="also write per-run quantification parameters here (CSV)",
    )
    ap.add_argument(
        "--raw-bytes",
        action="store_true",
        help="leave byte strings undecoded, reproducing the original CSV",
    )
    args = ap.parse_args()

    frames, metas, skipped = [], [], []
    with h5py.File(args.h5_path, "r") as hf:
        for run_key in sorted(hf.keys()):
            df, meta = read_run(hf[run_key], run_key, args.raw_bytes)
            if df is None:
                skipped.append(run_key)
                continue
            frames.append(df)
            metas.append(meta)

    if not frames:
        print(f"no spark_df groups found in {args.h5_path}", file=sys.stderr)
        return 1

    table = pd.concat(frames, ignore_index=True)
    ordered = [c for c in CANONICAL_ORDER if c in table.columns]
    ordered += sorted(c for c in table.columns if c not in ordered)
    table = table[ordered]
    table.to_csv(args.out, index=False)

    print(f"{args.out}: {len(table)} rows x {len(table.columns)} cols, {len(frames)} runs")
    if skipped:
        print(f"  skipped (no spark_df): {', '.join(skipped)}")

    if args.metadata_out:
        pd.DataFrame(metas).to_csv(args.metadata_out, index=False)
        print(f"{args.metadata_out}: {len(metas)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
