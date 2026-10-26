#!/usr/bin/env python3
"""
Generate splits_v2 from an expanded interaction CSV with columns:
  kinase_seq, substrate_seq, label

while keeping the existing splits_v1/test.csv as a fixed, separate test2 set.

What it does
- Reads expanded CSV (new pool) and splits_v1/{train,val,test}.csv
- Copies splits_v1/test.csv -> splits_v2/test2.csv
- Removes any rows from the expanded pool that overlap with test2 (by key_cols)
- Re-splits the remaining pool into splits_v2/{train,val,test}.csv
  using v1 split proportions and stratification by label
- Guarantees no leakage from test2 into train/val/test (by key_cols)

Defaults
- key_cols = ["kinase_seq","substrate_seq"] (since that's all you have)
- Stratified by label (assumes binary 0/1)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY_COLS_DEFAULT = ["kinase_seq", "substrate_seq"]
REQ_COLS_NEW = ["kinase_seq", "substrate_seq", "label"]


def read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    return df.iloc[:,1:]


def assert_has_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}\nColumns present: {list(df.columns)}")


def normalize_keys(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    k = df[key_cols].copy()
    # normalize as strings; strip to reduce spurious mismatches
    for c in key_cols:
        k[c] = k[c].astype(str).str.strip()
    return k


def key_set(df: pd.DataFrame, key_cols: list[str]) -> set[tuple]:
    k = normalize_keys(df, key_cols)
    return set(map(tuple, k.to_numpy()))


def drop_overlaps(df_pool: pd.DataFrame, df_block: pd.DataFrame, key_cols: list[str]) -> tuple[pd.DataFrame, int]:
    assert_has_cols(df_pool, key_cols, "expanded_pool")
    assert_has_cols(df_block, key_cols, "test2_df")

    block = key_set(df_block, key_cols)
    pool_k = list(map(tuple, normalize_keys(df_pool, key_cols).to_numpy()))
    keep = np.array([t not in block for t in pool_k], dtype=bool)
    dropped = int((~keep).sum())
    return df_pool.loc[keep].reset_index(drop=True), dropped


def stratified_split_indices(
    y: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def alloc(idxs: np.ndarray):
        n = len(idxs)
        frac_train = n_train / (n_train + n_val + n_test)
        frac_val = n_val / (n_train + n_val + n_test)
        n_tr = int(round(n * frac_train))
        n_va = int(round(n * frac_val))
        if n_tr + n_va > n:
            n_va = max(0, n - n_tr)
        n_te = n - n_tr - n_va
        return idxs[:n_tr], idxs[n_tr : n_tr + n_va], idxs[n_tr + n_va :]

    tr0, va0, te0 = alloc(idx0)
    tr1, va1, te1 = alloc(idx1)

    train_idx = np.concatenate([tr0, tr1])
    val_idx = np.concatenate([va0, va1])
    test_idx = np.concatenate([te0, te1])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--expanded_csv",
        type=str,
        required=True,
        help="Expanded interaction CSV with columns: kinase_seq, substrate_seq, label",
    )
    ap.add_argument(
        "--splits_v1_dir",
        type=str,
        default="splits_v1",
        help="Directory containing train.csv, val.csv, test.csv (v1).",
    )
    ap.add_argument(
        "--splits_v2_dir",
        type=str,
        default="splits_v2",
        help="Output directory for v2 splits (train/val/test/test2).",
    )
    ap.add_argument(
        "--key_cols",
        type=str,
        default=",".join(KEY_COLS_DEFAULT),
        help=f"Comma-separated columns used to define interaction identity for leakage removal. Default: {KEY_COLS_DEFAULT}",
    )
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument(
        "--dedup_before_split",
        action="store_true",
        help="Drop exact duplicate rows in expanded_csv before splitting.",
    )
    args = ap.parse_args()

    key_cols = [c.strip() for c in args.key_cols.split(",") if c.strip()]
    rng = np.random.default_rng(args.seed)

    v1_dir = Path(args.splits_v1_dir)
    v2_dir = Path(args.splits_v2_dir)
    v2_dir.mkdir(parents=True, exist_ok=True)

    # --- load v1 (for proportions + fixed test2) ---
    v1_train = read_csv(v1_dir / "train.csv")
    v1_val = read_csv(v1_dir / "val.csv")
    v1_test2 = read_csv(v1_dir / "test.csv")  # will be copied to v2/test2.csv

    # Ensure test2 has the key cols (and usually label exists too, but not required for leakage)
    assert_has_cols(v1_test2, key_cols, "splits_v1/test.csv (test2)")

    # --- load expanded pool ---
    df_pool = read_csv(Path(args.expanded_csv))
    assert_has_cols(df_pool, REQ_COLS_NEW, "expanded_csv")
    assert_has_cols(df_pool, key_cols, "expanded_csv")

    # normalize labels
    df_pool["label"] = pd.to_numeric(df_pool["label"], errors="coerce")
    if df_pool["label"].isna().any():
        raise ValueError("expanded_csv has non-numeric labels after coercion.")
    df_pool["label"] = df_pool["label"].astype(int)

    bad = df_pool.loc[~df_pool["label"].isin([0, 1]), "label"].value_counts()
    if len(bad) > 0:
        raise ValueError(f"expanded_csv has non-binary labels:\n{bad}")

    if args.dedup_before_split:
        before = len(df_pool)
        df_pool = df_pool.drop_duplicates().reset_index(drop=True)
        print(f"[info] Deduped expanded pool: {before} -> {len(df_pool)} rows")

    # --- remove overlaps with fixed test2 ---
    df_pool2, dropped = drop_overlaps(df_pool, v1_test2, key_cols=key_cols)
    print(f"[info] Dropped {dropped} rows from expanded pool due to overlap with test2 keys")

    # --- compute split sizes using v1 proportions ---
    n1_train, n1_val, n1_test = len(v1_train), len(v1_val), len(v1_test2)
    total1 = n1_train + n1_val + n1_test
    if total1 == 0:
        raise ValueError("splits_v1 appears empty (train+val+test = 0).")

    frac_train = n1_train / total1
    frac_val = n1_val / total1
    # test frac derived by remainder

    N = len(df_pool2)
    if N < 10:
        raise ValueError(f"Not enough rows left after removing test2 overlaps: N={N}")

    n_train = int(round(N * frac_train))
    n_val = int(round(N * frac_val))
    n_test = N - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        # fallback
        n_train = int(round(N * 0.8))
        n_val = int(round(N * 0.1))
        n_test = N - n_train - n_val
        print("[warn] v1 proportions produced a degenerate split; falling back to 80/10/10")

    # --- stratified split ---
    y = df_pool2["label"].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = stratified_split_indices(y, n_train, n_val, n_test, rng)

    df_train = df_pool2.iloc[train_idx].reset_index(drop=True)
    df_val = df_pool2.iloc[val_idx].reset_index(drop=True)
    df_test = df_pool2.iloc[test_idx].reset_index(drop=True)
    df_test2 = v1_test2.copy()

    # --- leakage check (by key_cols) ---
    s2 = key_set(df_test2, key_cols)
    leak_train = len(key_set(df_train, key_cols) & s2)
    leak_val = len(key_set(df_val, key_cols) & s2)
    leak_test = len(key_set(df_test, key_cols) & s2)
    if leak_train or leak_val or leak_test:
        raise RuntimeError(
            f"Leakage detected vs test2 (key_cols={key_cols}): "
            f"train={leak_train}, val={leak_val}, test={leak_test}"
        )

    # --- write outputs ---
    df_train.to_csv(v2_dir / "train.csv", index=False)
    df_val.to_csv(v2_dir / "val.csv", index=False)
    df_test.to_csv(v2_dir / "test.csv", index=False)
    df_test2.to_csv(v2_dir / "test2.csv", index=False)

    def report(name: str, df: pd.DataFrame):
        vc = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        print(f"[split] {name:6s}: n={len(df):8d} label_counts={vc}")

    print(f"[out] wrote: {v2_dir/'train.csv'}")
    print(f"[out] wrote: {v2_dir/'val.csv'}")
    print(f"[out] wrote: {v2_dir/'test.csv'}")
    print(f"[out] wrote: {v2_dir/'test2.csv'} (copied from splits_v1/test.csv)")
    report("train", df_train)
    report("val", df_val)
    report("test", df_test)
    report("test2", df_test2)


if __name__ == "__main__":
    main()
