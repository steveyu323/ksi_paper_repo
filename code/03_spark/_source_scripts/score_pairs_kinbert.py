"""Score the pairs in rescore_request.csv with KinBERT checkpoints.

Self-contained: copy this file plus rescore_request.csv to wherever the weights
live.  It needs only torch + tape; it reads no other project file.

Input encoding matches kinase_ml/code/tape_end2end/kinbert_dataset.py
(BertDataset, with_sep=True):

    <cls> kinase_seq <sep> substrate_seq <sep>

where kinase_seq is the kinase DOMAIN sequence carried in the request CSV.

Usage
-----
    python score_pairs_kinbert.py \
        --model exp_tape_bertbase_sep_v2_early_loss=/path/to/orig_ckpt \
        --model finetune_tape_v2_phosphopca_exp6=/path/to/finetuned_ckpt \
        --out rescore_results.csv

purpose=validate rows are scored first and checked against expected_prob.  If
they do not reproduce, the run aborts: scoring the DB reporters with a
non-matching checkpoint would put non-comparable bars on the same figure.
Override with --force only if you know why they differ.
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tape import ProteinBertForSequenceClassification, TAPETokenizer


def encode(tokenizer, kinase_seq, substrate_seq):
    tokens = (
        tokenizer.tokenize(kinase_seq) + ["<sep>"] + tokenizer.tokenize(substrate_seq)
    )
    tokens = tokenizer.add_special_tokens(tokens)
    return np.array(tokenizer.convert_tokens_to_ids(tokens), np.int64)


@torch.no_grad()
def score(model, tokenizer, pairs, device, batch_size=8):
    """pairs: list of (kinase_seq, substrate_seq). Returns prob, logit arrays."""
    model.eval()
    probs, logits_out = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        ids = [encode(tokenizer, k, s) for k, s in chunk]
        width = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), width), np.int64)
        input_mask = np.zeros((len(ids), width), np.int64)
        for j, x in enumerate(ids):
            input_ids[j, : len(x)] = x
            input_mask[j, : len(x)] = 1
        out = model(
            input_ids=torch.from_numpy(input_ids).to(device),
            input_mask=torch.from_numpy(input_mask).to(device),
        )
        logits = out[0].float().cpu()
        probs.extend(F.softmax(logits, dim=1)[:, 1].numpy())
        # the saved *_ksi_full.csv files store logit = logit_pos - logit_neg
        logits_out.extend((logits[:, 1] - logits[:, 0]).numpy())
    return np.array(probs), np.array(logits_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", default="rescore_request.csv")
    ap.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="map a model name in the request CSV to a checkpoint dir; repeatable",
    )
    ap.add_argument("--out", default="rescore_results.csv")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--force", action="store_true", help="score even if validation fails")
    args = ap.parse_args()

    ckpts = {}
    for spec in args.model:
        if "=" not in spec:
            raise SystemExit(f"--model expects NAME=PATH, got: {spec}")
        name, path = spec.split("=", 1)
        ckpts[name] = path

    req = pd.read_csv(args.request)
    missing = sorted(set(req["model"]) - set(ckpts))
    if missing:
        raise SystemExit(f"no --model path given for: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")
    tokenizer = TAPETokenizer(vocab="iupac")

    results, all_ok = [], True
    for model_name, group in req.groupby("model", sort=False):
        print(f"--- {model_name}")
        print(f"    {ckpts[model_name]}")
        model = ProteinBertForSequenceClassification.from_pretrained(
            ckpts[model_name]
        ).to(device)

        pairs = list(zip(group["kinase_seq"], group["substrate_seq"]))
        probs, logits = score(model, tokenizer, pairs, device)

        g = group.copy()
        g["prob"] = probs
        g["logit"] = logits
        g["checkpoint"] = ckpts[model_name]
        g["delta"] = (g["prob"] - g["expected_prob"]).abs()
        results.append(g)

        val = g[g["purpose"] == "validate"]
        for _, r in val.iterrows():
            ok = r["delta"] < args.tol
            all_ok &= bool(ok)
            print(
                f"    {'OK      ' if ok else 'MISMATCH'} {r['name']:16s} "
                f"expected={r['expected_prob']:.6f} got={r['prob']:.6f} d={r['delta']:.2e}"
            )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print()

    out = pd.concat(results, ignore_index=True)

    if not all_ok and not args.force:
        out.to_csv(args.out + ".unvalidated", index=False)
        raise SystemExit(
            f"Validation failed; wrote {args.out}.unvalidated for inspection.\n"
            "A checkpoint here does not reproduce the saved inference output, so its\n"
            "regenerate rows are not comparable to the figure's existing bars.\n"
            "Check the checkpoint identity first; re-run with --force to override."
        )

    out.to_csv(args.out, index=False)
    print(out[["name", "model", "purpose", "prob", "logit"]].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
