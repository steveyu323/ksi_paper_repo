"""Does PHEEPSNSLMAIECR/IKKB survive selection if we drop binding_max?

Notebook 9 selects on binding_max = max over PAABD reporters, then merges the
per-PAABD inference table on mean == binding_max, which silently pins each
candidate's PAABD to the argmax.  This re-runs the same pipeline scoring each
PAABD on its own measurement and compares.
"""

import pandas as pd

BLIB_CSV = "../screening_analysis/blib_kinases_with_acc.csv"
SCREEN_PKL = "./251203_ksi_reduced_df_tp.pkl"
INFERENCE_PKL = "./251130_ksi_inference_scores.pkl"

MIN_GAP = 0.4
FOCUS = ("IKKB_HUMAN", "PHEEPSNSLMAIECR")

# motif flag name <-> paabd_hit token
MOTIF = {"SH2": "sh2", "b14-3-3": "1433", "WW": "ww", "FHA": "fha"}


def assign_paabd(dat):
    """notebook 9's definitions (looser WW than notebook 13)"""
    s = dat["substrate_seq_15aa"]
    dat = dat.copy()
    dat["SH2"] = s.apply(lambda x: x[7] == "Y")
    dat["b14-3-3"] = s.apply(lambda x: (x[7] in "ST") and (x[4] == "R") and (x[9] == "P"))
    dat["WW"] = s.apply(lambda x: (x[7] in "ST") and (x[8] in ["P", "L", "V", "I"]))
    dat["FHA"] = s.apply(lambda x: (x[7] == "T") and (x[10] == "D"))
    return dat


def load_screen():
    blib = pd.read_csv(BLIB_CSV, header=None)
    blib.columns = ["kinase_type", "kinase", "kinase_seq", "kinase_acc"]
    screen = pd.read_pickle(SCREEN_PKL).rename(columns={"substrate": "substrate_seq_15aa"})
    d = pd.merge(blib, screen, on="kinase")
    d = assign_paabd(d)
    d = d.loc[~d["kinase"].isin(["KS6B1_HUMAN_active", "PLK1_HUMAN_active"]), :]
    return d.loc[d["true_positive"]]


def dominant(df, score_col, group_col="substrate_seq_15aa", min_gap=MIN_GAP):
    """notebook 9's pick_dominant_hit: top kinase for a substrate must beat the
    runner-up by more than min_gap."""
    df = df.copy()
    df["_rank"] = df.groupby(group_col)[score_col].rank(method="first", ascending=False)
    top = df[df["_rank"] == 1].copy()
    second = (
        df[df["_rank"] == 2][[group_col, score_col]].rename(columns={score_col: "_second"})
    )
    top = top.merge(second, on=group_col, how="left")
    top["_gap"] = top[score_col] - top["_second"]
    return top[top["_gap"] > min_gap]


def main():
    tp = load_screen()
    inference = pd.read_pickle(INFERENCE_PKL)
    kin, seq = FOCUS

    # ---- long table: one row per (kinase, substrate, paabd) that is both an
    # experimental hit for that paabd and motif-compatible with it
    frames = []
    for flag, token in MOTIF.items():
        sub = tp.loc[tp["paabd_hit"].apply(lambda s, t=token: t in s) & tp[flag], :].copy()
        sub["paabd"] = token
        frames.append(sub)
    wpaabd_long = pd.concat(frames).drop_duplicates(["kinase", "substrate_seq_15aa", "paabd"])

    long = pd.merge(
        wpaabd_long,
        inference[["kinase", "substrate", "paabd", "mean"]].rename(
            columns={"substrate": "substrate_seq_15aa", "mean": "binding_at_paabd"}
        ),
        on=["kinase", "substrate_seq_15aa", "paabd"],
    )

    print("=" * 78)
    print(f"A. All kinases measured against {seq}, per PAABD")
    print("=" * 78)
    wide = (
        inference.loc[inference["substrate"] == seq]
        .pivot_table(index="kinase", columns="paabd", values="mean")
        .sort_values("ww", ascending=False)
    )
    print(wide.to_string())

    # ---- baseline: notebook 9 as written (binding_max, argmax PAABD)
    print("\n" + "=" * 78)
    print("B. Notebook 9 as written (score = binding_max over PAABDs)")
    print("=" * 78)
    base = pd.merge(
        inference,
        tp,
        left_on=["kinase", "substrate", "mean"],
        right_on=["kinase", "substrate_seq_15aa", "binding_max"],
    )
    base_dom = dominant(base, "binding_max")
    hit = base_dom[(base_dom["kinase"] == kin) & (base_dom["substrate_seq_15aa"] == seq)]
    print(f"{seq} / {kin} selected: {len(hit) == 1}")
    if len(hit):
        print(f"  assigned paabd = {hit.iloc[0]['paabd']}, score = {hit.iloc[0]['binding_max']:.4f}")

    # ---- per-PAABD: dominance computed within each PAABD on its own numbers
    print("\n" + "=" * 78)
    print("C. Per-PAABD dominance (score = that PAABD's own measurement)")
    print("=" * 78)
    per = pd.concat(
        [dominant(g, "binding_at_paabd") for _, g in long.groupby("paabd", sort=False)]
    )
    hit = per[(per["kinase"] == kin) & (per["substrate_seq_15aa"] == seq)]
    print(f"{seq} / {kin} selected: {len(hit) > 0}")
    if len(hit):
        for _, r in hit.iterrows():
            print(f"  paabd = {r['paabd']}, score = {r['binding_at_paabd']:.4f}")
    else:
        # why did it fail?
        for p in sorted(long.loc[long["substrate_seq_15aa"] == seq, "paabd"].unique()):
            g = long[(long["substrate_seq_15aa"] == seq) & (long["paabd"] == p)]
            g = g.sort_values("binding_at_paabd", ascending=False)
            top = g.iloc[0]
            gap = top["binding_at_paabd"] - (
                g.iloc[1]["binding_at_paabd"] if len(g) > 1 else float("-inf")
            )
            print(
                f"  paabd={p}: top kinase = {top['kinase']} "
                f"({top['binding_at_paabd']:.4f}), gap to runner-up = {gap:.4f} "
                f"{'PASS' if gap > MIN_GAP else 'FAIL'}"
            )

    # ---- where does it rank among IKKB's own WW candidates?
    print("\n" + "=" * 78)
    print(f"D. {kin} WW candidates ranked by their WW measurement")
    print("=" * 78)
    ikkb_ww = long[(long["kinase"] == kin) & (long["paabd"] == "ww")].sort_values(
        "binding_at_paabd", ascending=False
    )
    ikkb_ww = ikkb_ww[["substrate_seq_15aa", "binding_at_paabd", "binding_max", "paabd_hit"]]
    ikkb_ww = ikkb_ww.reset_index(drop=True)
    ikkb_ww.index += 1
    print(f"n = {len(ikkb_ww)}")
    print(ikkb_ww.head(15).to_string())
    where = ikkb_ww.index[ikkb_ww["substrate_seq_15aa"] == seq]
    print(f"\n{seq} rank among {kin} WW candidates: {list(where)} of {len(ikkb_ww)}")

    print("\n" + "=" * 78)
    print(f"E. {kin} candidates (any PAABD) that pass per-PAABD dominance")
    print("=" * 78)
    ikkb_pass = per[per["kinase"] == kin].sort_values("binding_at_paabd", ascending=False)
    print(
        ikkb_pass[["substrate_seq_15aa", "paabd", "binding_at_paabd", "binding_max"]]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
