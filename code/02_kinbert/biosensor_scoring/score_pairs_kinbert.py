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
        --model exp_tape_bertbase_sep_v2_early_loss=/path/to/run/checkpoints/best.pt \
        --model finetune_tape_v2_phosphopca_exp6=/path/to/run/checkpoints/best.pt \
        --out rescore_results.csv

A --model PATH is the run's `best.pt` (a dir is also accepted and resolved to
`checkpoints/best.pt` inside it).  These checkpoints are
`{cfg, state_dict, epoch, best_val, threshold}` dicts, NOT HuggingFace
`from_pretrained` directories: the model is rebuilt from the TAPE `bert-base`
config and the state_dict is loaded into it, which is what the scripts that
produced the saved outputs did (evaluation/eval_utils_sep.py:412,
evaluation/predict_finetuned_tape.ipynb cell 7).

Precision is NOT uniform across these two runs and it matters more than the
tolerance suggests:

    exp_tape_bertbase_sep_v2_early_loss  scored under bf16 autocast
    finetune_tape_v2_phosphopca_exp6     scored in plain fp32

Each checkpoint's own cfg["bf16"] records this, so precision is taken from the
checkpoint (override with --precision).  Re-running the bf16 model in fp32
shifts probs by ~2e-3 on average with a systematic bias, so it is not a
harmless substitution.

purpose=validate rows are scored first and checked against expected_prob.  If
they do not reproduce, the run aborts: scoring the DB reporters with a
non-matching checkpoint would put non-comparable bars on the same figure.
Override with --force only if you know why they differ.

The per-row tolerance is precision-aware (see TOL).  A bf16 forward is not
reproducible to 1e-3: the original run batched 128 rows of pairs_ksi_full.csv
together, so padding width and reduction order differed from any re-run, and
bf16 quantises the logits to ~1/128 near |logit|=1.  Measured over 256 pairs
resampled from the saved output, the right checkpoint in bf16 gives max
|dprob|=2.8e-2 (median 1.2e-4, unbiased, corr 0.99995), while a deliberately
wrong checkpoint gives mean |dprob|=0.30 and corr 0.38 -- so a 3e-2 bf16
tolerance still separates the two decisively.  fp32 runs reproduce to ~4e-6.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tape import ProteinBertForSequenceClassification, TAPETokenizer

# per-row |prob - expected_prob| tolerance, by forward precision
TOL = {"fp32": 1e-3, "bf16": 3e-2}


def load_checkpoint(path):
    """Resolve PATH to a best.pt and load it. Returns (model, cfg)."""
    if os.path.isdir(path):
        cand = os.path.join(path, "checkpoints", "best.pt")
        path = cand if os.path.exists(cand) else os.path.join(path, "best.pt")
    if not os.path.exists(path):
        raise SystemExit(f"no checkpoint at: {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise SystemExit(f"{path} has no 'state_dict' key (got {sorted(ckpt)})")

    cfg = ckpt.get("cfg", {})
    name = cfg.get("tape_model_name", "bert-base")
    model = ProteinBertForSequenceClassification.from_pretrained(name, num_labels=2)
    # strict: these runs used no DataParallel/compile, so keys match exactly.
    model.load_state_dict(ckpt["state_dict"], strict=True)
    meta = {"path": path, "epoch": ckpt.get("epoch"), "threshold": ckpt.get("threshold")}
    return model, cfg, meta


def encode(tokenizer, kinase_seq, substrate_seq, max_len):
    tokens = (
        tokenizer.tokenize(kinase_seq) + ["<sep>"] + tokenizer.tokenize(substrate_seq)
    )
    tokens = tokenizer.add_special_tokens(tokens)
    ids = tokenizer.convert_tokens_to_ids(tokens)[:max_len]
    return np.array(ids, np.int64)


@torch.no_grad()
def score(model, tokenizer, pairs, device, bf16, max_len, batch_size=8):
    """pairs: list of (kinase_seq, substrate_seq). Returns prob, logit arrays."""
    model.eval()
    probs, logits_out = [], []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        ids = [encode(tokenizer, k, s, max_len) for k, s in chunk]
        width = max(len(x) for x in ids)
        input_ids = np.zeros((len(ids), width), np.int64)
        input_mask = np.zeros((len(ids), width), np.int64)
        for j, x in enumerate(ids):
            input_ids[j, : len(x)] = x
            input_mask[j, : len(x)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(
                input_ids=torch.from_numpy(input_ids).to(device),
                input_mask=torch.from_numpy(input_mask).to(device),
            )
        logits = out[0].float().cpu()
        probs.extend(F.softmax(logits, dim=-1)[:, 1].numpy())
        # the saved *_ksi_full.csv files store the raw positive-class logit,
        # logits[:, 1] -- not logit_pos - logit_neg (eval_utils_sep.py:308,
        # predict_finetuned_tape.ipynb cell 16).
        logits_out.extend(logits[:, 1].numpy())
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
    ap.add_argument(
        "--tol",
        type=float,
        default=None,
        help=f"per-row prob tolerance; default is precision-aware {TOL}",
    )
    ap.add_argument(
        "--precision",
        choices=["auto", "fp32", "bf16"],
        default="auto",
        help="auto = take each run's precision from its checkpoint cfg['bf16']",
    )
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
        model, cfg, meta = load_checkpoint(ckpts[model_name])
        model = model.to(device)
        resolved = meta["path"]

        if args.precision == "auto":
            prec = "bf16" if cfg.get("bf16") else "fp32"
        else:
            prec = args.precision
        bf16 = prec == "bf16" and device.type == "cuda"
        max_len = int(cfg.get("max_len", 1024))
        tol = args.tol if args.tol is not None else TOL[prec]

        print(f"    ckpt       {resolved}")
        print(f"    run_name   {cfg.get('run_name')}   epoch={meta['epoch']}")
        print(f"    precision  {prec} (autocast={bf16})   max_len={max_len}   tol={tol:.1e}")

        pairs = list(zip(group["kinase_seq"], group["substrate_seq"]))
        probs, logits = score(model, tokenizer, pairs, device, bf16, max_len)

        g = group.copy()
        g["prob"] = probs
        g["logit"] = logits
        g["checkpoint"] = resolved
        g["precision"] = prec
        g["delta"] = (g["prob"] - g["expected_prob"]).abs()
        results.append(g)

        if cfg.get("run_name") and cfg["run_name"] != model_name:
            all_ok = False
            print(f"    MISMATCH run_name is {cfg['run_name']!r}, expected {model_name!r}")

        val = g[g["purpose"] == "validate"]
        for _, r in val.iterrows():
            ok = r["delta"] < tol
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
    cols = ["name", "model", "purpose", "precision", "prob", "logit", "expected_prob", "delta"]
    print(out[cols].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
