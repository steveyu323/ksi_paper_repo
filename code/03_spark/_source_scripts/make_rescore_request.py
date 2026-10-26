"""Emit the (kinase_seq, substrate_seq, model) rows that need KinBERT inference.

Run this here; run the scoring where the weights live.

`kinase_seq` is the kinase DOMAIN sequence, taken from pairs_aug_ksi_full.csv so
it is byte-identical to what both models were originally scored on.  Do not
substitute the full-length kinase_seq from blib_kinases_with_acc.csv.

Rows tagged purpose=validate already have a known score in the saved inference
output.  Score those first and check them against expected_prob; if they do not
reproduce, the checkpoint or the input encoding is not the one that produced the
figure's other bars, and the regenerate rows must not be used.
"""

import pandas as pd

PAIRS_AUG_CSV = "./pairs_aug_ksi_full.csv"
OUT_CSV = "./rescore_request.csv"

ORIG_MODEL = "exp_tape_bertbase_sep_v2_early_loss"
FINETUNED_MODEL = "finetune_tape_v2_phosphopca_exp6"

# 15mers from screening_analysis/240416_SPARK_candidate_selection.ipynb
# never appear in any *_ksi_full.csv, so both models must score them
REGENERATE = [
    ("TBK-SPARK-1", "TBK1_HUMAN", "DHPYASLTLPDDAAA"),
    ("IKKB-SPARK-1", "IKKB_HUMAN", "NPQERTLTLVDTGIG"),
]

# already scored -> use these to prove the checkpoint/encoding match
VALIDATE = [
    ("TBK-SPARK-novel", "TBK1_HUMAN", "QGPSQTSTPTDVTAI", 0.200558, 0.766206),
    ("IKK-SPARK-novel", "IKKB_HUMAN", "PHEEPSNSLMAIECR", 0.952217, 0.769495),
]


def domain_seqs(pairs_aug_csv=PAIRS_AUG_CSV):
    pa = (
        pd.read_csv(pairs_aug_csv, usecols=["kinase_human_name", "kinase_domain_seq"])
        .dropna()
        .drop_duplicates()
    )
    dup = pa["kinase_human_name"].duplicated(keep=False)
    if dup.any():
        raise ValueError(
            f"ambiguous domain seq for: {sorted(pa.loc[dup, 'kinase_human_name'].unique())}"
        )
    return dict(zip(pa["kinase_human_name"], pa["kinase_domain_seq"]))


def main():
    lookup = domain_seqs()
    rows = []

    for name, kinase, seq in REGENERATE:
        for model in (ORIG_MODEL, FINETUNED_MODEL):
            rows.append(
                {
                    "name": name,
                    "kinase": kinase,
                    "kinase_seq": lookup[kinase],
                    "substrate_seq": seq,
                    "model": model,
                    "purpose": "regenerate",
                    "expected_prob": "",
                }
            )

    for name, kinase, seq, exp_orig, exp_ft in VALIDATE:
        for model, expected in ((ORIG_MODEL, exp_orig), (FINETUNED_MODEL, exp_ft)):
            rows.append(
                {
                    "name": name,
                    "kinase": kinase,
                    "kinase_seq": lookup[kinase],
                    "substrate_seq": seq,
                    "model": model,
                    "purpose": "validate",
                    "expected_prob": expected,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(f"wrote {OUT_CSV}  ({len(out)} rows)")
    print(
        out[["name", "kinase", "substrate_seq", "model", "purpose", "expected_prob"]].to_string(
            index=False
        )
    )
    print("\nkinase domain sequences (length):")
    for k in sorted({r["kinase"] for r in rows}):
        print(f"  {k:12s} {len(lookup[k])} aa")


if __name__ == "__main__":
    main()
