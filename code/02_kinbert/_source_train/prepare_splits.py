#!/usr/bin/env python3
"""
Prepare deterministic train/val/test CSVs from a single original CSV.

- Input schema: kinase_seq, substrate_seq, label
- Output: train.csv, val.csv, test.csv in out_dir

Deterministic split: uses sha1(kinase_seq|substrate_seq|seed) -> float in [0,1)
so re-saving/shuffling the input CSV doesn't change the split.

Optionally, you can also pass --dedupe_pairs to drop exact duplicate (kinase, substrate, label) rows.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUOJ]+$")


def sanitize_seq(s: str) -> str:
    if s is None:
        return ""
    return WHITESPACE_RE.sub("", str(s).upper())


def hash_to_u01(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    # take 16 hex chars -> 64-bit int
    x = int(h[:16], 16)
    return x / float(2**64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--dedupe_pairs", action="store_true")
    ap.add_argument("--kinase_max_len", type=int, default=384)
    ap.add_argument("--substrate_max_len", type=int, default=32)
    args = ap.parse_args()

    if args.val_frac < 0 or args.test_frac < 0 or (args.val_frac + args.test_frac) >= 1.0:
        raise ValueError("Need val_frac>=0, test_frac>=0, and val_frac+test_frac < 1.0")

    df = pd.read_csv(args.input_csv)
    for c in ["kinase_seq", "substrate_seq", "label"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c}. Found: {list(df.columns)}")

    df["kinase_seq"] = df["kinase_seq"].map(sanitize_seq)
    df["substrate_seq"] = df["substrate_seq"].map(sanitize_seq)

    # basic filtering
    ok = (
        df["kinase_seq"].str.len().between(1, args.kinase_max_len)
        & df["substrate_seq"].str.len().between(1, args.substrate_max_len)
        & df["kinase_seq"].str.match(VALID_AA_RE)
        & df["substrate_seq"].str.match(VALID_AA_RE)
    )
    df = df[ok].reset_index(drop=True)

    if args.dedupe_pairs:
        df = df.drop_duplicates(["kinase_seq", "substrate_seq", "label"]).reset_index(drop=True)

    # deterministic assignment
    # IMPORTANT: do NOT include label in the key; we want the pair to deterministically land in one split.
    keys = (df["kinase_seq"] + "|" + df["substrate_seq"] + f"|seed={args.seed}").tolist()
    u = np.array([hash_to_u01(k) for k in keys], dtype=np.float64)

    test_cut = args.test_frac
    val_cut = args.test_frac + args.val_frac

    split = np.where(u < test_cut, "test", np.where(u < val_cut, "val", "train"))
    df["split"] = split

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    df_val   = df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    df_test  = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)

    df_train.to_csv(out_dir / "train.csv", index=False)
    df_val.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)

    print(f"Wrote:\n- {out_dir/'train.csv'} ({len(df_train)})\n- {out_dir/'val.csv'} ({len(df_val)})\n- {out_dir/'test.csv'} ({len(df_test)})")


if __name__ == "__main__":
    main()
