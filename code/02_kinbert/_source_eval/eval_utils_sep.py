#!/usr/bin/env python3
"""
Evaluation utilities for TWO pair-encoding training schemes with separator tokens:

A) HF EOS separator (train_pair_interaction_esm2_eos.py / train_pair_interaction_esmpp_eos.py)
   Pair encoding (token-level):
        input_ids = [CLS] + kin_ids + [EOS] + sub_ids + [EOS]

B) TAPE ProteinBert with '<sep>' token between kin and sub (train_pair_interaction_tape.py)
   Pair encoding:
        toks = tokenize(kin) + ['<sep>'] + tokenize(sub)
        toks = add_special_tokens(toks)

Outputs:
- Writes ONE pickle per dataset with naming:
    <model_id>__<dataset_id>__<ckpt_tag>.pkl
  e.g. exp1_esmc_frozen__db_lysate_test__364ff8bf9439.pkl

Eval YAML schema:

training_command: "python train_pair_interaction_esm2_eos.py --config configs/exp_esm2_eos.yml"
output_dir: "eval_outputs"           # optional (default: directory of this eval yaml)
model_id: "exp_esm2_eos"             # optional (default: run_name from training yaml)
model_data_identifier: "suite_v1"    # optional; stored in pickle (not used for filename)

device: "cuda:0"          # optional
batch_size: 128           # optional
num_workers: 4            # optional
bf16: true                # optional (autocast on cuda)

datasets:
  - id: "test"
    csv: "splits_v1/test.csv"
  - id: "cold_kinase"
    csv: "splits_v1/test2_cold_kinase.csv"

Each dataset CSV must have columns: kinase_seq, substrate_seq, label
"""

from __future__ import annotations

import argparse
import re
import shlex
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
WHITESPACE_RE = re.compile(r"\s+")
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUOJ]+$")


def sanitize_seq(s: str) -> str:
    if s is None:
        return ""
    return WHITESPACE_RE.sub("", str(s).upper())


def sha1_12(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def load_pairs_csv(csv_path: str, max_len_kin: Optional[int] = None, max_len_sub: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["kinase_seq", "substrate_seq", "label"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {csv_path}. Found: {list(df.columns)}")
    df["kinase_seq"] = df["kinase_seq"].map(sanitize_seq)
    df["substrate_seq"] = df["substrate_seq"].map(sanitize_seq)

    ok = (
        df["kinase_seq"].str.len().ge(1)
        & df["substrate_seq"].str.len().ge(1)
        & df["kinase_seq"].str.match(VALID_AA_RE)
        & df["substrate_seq"].str.match(VALID_AA_RE)
        & df["label"].isin([0, 1, 0.0, 1.0, "0", "1"])
    )
    if max_len_kin is not None:
        ok &= df["kinase_seq"].str.len().le(int(max_len_kin))
    if max_len_sub is not None:
        ok &= df["substrate_seq"].str.len().le(int(max_len_sub))

    df = df[ok].copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def parse_training_command(training_command: str) -> Tuple[str, str]:
    toks = shlex.split(training_command)
    script = None
    cfg = None
    for t in toks:
        if t.endswith(".py"):
            script = t
            break
    for i, t in enumerate(toks):
        if t == "--config" and i + 1 < len(toks):
            cfg = toks[i + 1]
            break
    if script is None or cfg is None:
        raise ValueError(f"Could not parse training_command: {training_command}")
    return script, cfg


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_best_ckpt(train_cfg: Dict[str, Any]) -> Path:
    out_dir = Path(train_cfg.get("out_dir", "./runs"))
    run_name = train_cfg.get("run_name", None)
    if not run_name:
        raise ValueError("Training config must define run_name to locate checkpoint.")
    ckpt = out_dir / str(run_name) / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt}")
    return ckpt


# -----------------------------
# HF EOS scheme
# -----------------------------
class HFEOSPairDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame):
        self.kin = df["kinase_seq"].tolist()
        self.sub = df["substrate_seq"].tolist()
        self.y = df["label"].astype(np.int64).values

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        return self.kin[i], self.sub[i], int(self.y[i])


def make_loader_hf_eos(
    df: pd.DataFrame,
    tokenizer,
    max_len: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    ds = HFEOSPairDataset(df)
    cls_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or eos_id is None or pad_id is None:
        raise ValueError("Tokenizer must have cls_token_id, eos_token_id, pad_token_id for EOS scheme.")

    def collate(batch):
        kinseqs, subseqs, ys = zip(*batch)
        ids_list: List[np.ndarray] = []
        for kin, sub in zip(kinseqs, subseqs):
            kin_ids = tokenizer(kin, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            sub_ids = tokenizer(sub, add_special_tokens=False, return_attention_mask=False)["input_ids"]
            ids = [cls_id] + kin_ids + [eos_id] + sub_ids + [eos_id]
            ids = ids[:max_len]
            ids_list.append(np.asarray(ids, dtype=np.int64))

        Lmax = max(len(x) for x in ids_list)
        input_ids = torch.full((len(ids_list), Lmax), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(ids_list), Lmax), dtype=torch.long)
        for i, arr in enumerate(ids_list):
            n = int(arr.shape[0])
            input_ids[i, :n] = torch.from_numpy(arr)
            attention_mask[i, :n] = 1
        label = torch.tensor(ys, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label, "kin": kinseqs, "sub": subseqs}

    return torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate,
        pin_memory=True,
    )


@torch.no_grad()
def predict_hf_eos(model, loader, device: torch.device, amp_enabled: bool) -> Dict[str, Any]:
    model.eval()
    ys, logits1, probs1, kin_out, sub_out = [], [], [], [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits  # (B,2)
            loss = F.cross_entropy(logits, y, reduction="mean")
            prob = torch.softmax(logits, dim=-1)[:, 1]
            log1 = logits[:, 1]

        loss_sum += float(loss.item()) * int(y.numel())
        n += int(y.numel())
        ys.append(y.detach().cpu().numpy())
        logits1.append(log1.detach().float().cpu().numpy())
        probs1.append(prob.detach().float().cpu().numpy())
        kin_out.extend(list(batch["kin"]))
        sub_out.extend(list(batch["sub"]))

    return {
        "labels": np.concatenate(ys).astype(np.int64),
        "logits_pos": np.concatenate(logits1).astype(np.float32),
        "probs_pos": np.concatenate(probs1).astype(np.float32),
        "loss": float(loss_sum / max(1, n)),
        "kinase_seq": kin_out,
        "substrate_seq": sub_out,
    }


# -----------------------------
# TAPE <sep> scheme
# -----------------------------
class TAPEPairDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, with_sep: bool = True):
        self.pad_id = int(tokenizer.vocab.get("<pad>", 0))
        self.y = df["label"].astype(np.int64).values
        kin_list = df["kinase_seq"].tolist()
        sub_list = df["substrate_seq"].tolist()
        self.kin = kin_list
        self.sub = sub_list
        self.max_len = int(max_len)
        self.with_sep = bool(with_sep)
        if self.with_sep and "<sep>" not in tokenizer.vocab:
            raise ValueError("TAPE tokenizer vocab missing '<sep>' but with_sep=True.")

        self.ids_list: List[np.ndarray] = []
        for kin, sub in zip(kin_list, sub_list):
            kin_toks = tokenizer.tokenize(kin)
            sub_toks = tokenizer.tokenize(sub)
            toks = kin_toks + (["<sep>"] if self.with_sep else []) + sub_toks
            toks = tokenizer.add_special_tokens(toks)
            ids = tokenizer.convert_tokens_to_ids(toks)[: self.max_len]
            self.ids_list.append(np.asarray(ids, dtype=np.int64))

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        return self.ids_list[i], int(self.y[i]), self.kin[i], self.sub[i]


def make_loader_tape_sep(
    df: pd.DataFrame,
    tokenizer,
    max_len: int,
    batch_size: int,
    num_workers: int,
    with_sep: bool = True,
    shuffle: bool = False,
) -> torch.utils.data.DataLoader:
    ds = TAPEPairDataset(df, tokenizer=tokenizer, max_len=max_len, with_sep=with_sep)
    pad_id = int(ds.pad_id)

    def collate(batch):
        ids_list, ys, kins, subs = zip(*batch)
        Lmax = max(len(x) for x in ids_list)
        input_ids = torch.full((len(ids_list), Lmax), fill_value=pad_id, dtype=torch.long)
        input_mask = torch.zeros((len(ids_list), Lmax), dtype=torch.long)
        for i, arr in enumerate(ids_list):
            arr_t = torch.from_numpy(arr)
            n = int(arr_t.numel())
            input_ids[i, :n] = arr_t
            input_mask[i, :n] = 1
        y = torch.tensor(ys, dtype=torch.long)
        return {"input_ids": input_ids, "input_mask": input_mask, "label": y, "kin": kins, "sub": subs}

    kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate,
        pin_memory=True,
    )
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return torch.utils.data.DataLoader(ds, **kwargs)


@torch.no_grad()
def predict_tape_sep(model, loader, device: torch.device, amp_enabled: bool) -> Dict[str, Any]:
    model.eval()
    ys, logits1, probs1, kin_out, sub_out = [], [], [], [], []
    loss_sum, n = 0.0, 0
    for batch in tqdm(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            (loss, _metrics), logits = model(input_ids=input_ids, input_mask=input_mask, targets=y)
            prob = torch.softmax(logits, dim=-1)[:, 1]
            log1 = logits[:, 1]

        loss_sum += float(loss.item()) * int(y.numel())
        n += int(y.numel())
        ys.append(y.detach().cpu().numpy())
        logits1.append(log1.detach().float().cpu().numpy())
        probs1.append(prob.detach().float().cpu().numpy())
        kin_out.extend(list(batch["kin"]))
        sub_out.extend(list(batch["sub"]))

    return {
        "labels": np.concatenate(ys).astype(np.int64),
        "logits_pos": np.concatenate(logits1).astype(np.float32),
        "probs_pos": np.concatenate(probs1).astype(np.float32),
        "loss": float(loss_sum / max(1, n)),
        "kinase_seq": kin_out,
        "substrate_seq": sub_out,
    }


def run_eval_from_eval_yaml(eval_yaml: str, device: Optional[str] = None) -> List[str]:
    eval_cfg = _load_yaml(eval_yaml)
    training_command = eval_cfg["training_command"]

    # output dir
    output_dir = eval_cfg.get("output_dir", None)
    if output_dir is None:
        # default: alongside eval yaml
        output_dir = str(Path(eval_yaml).resolve().parent / "eval_outputs")
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    model_data_identifier = str(eval_cfg.get("model_data_identifier", "suite"))
    batch_size = int(eval_cfg.get("batch_size", 128))
    num_workers = int(eval_cfg.get("num_workers", 4))
    bf16 = bool(eval_cfg.get("bf16", True))

    datasets = eval_cfg.get("datasets", [])
    if not datasets:
        raise ValueError("Eval config must provide non-empty 'datasets' list.")

    train_script, train_yaml = parse_training_command(training_command)
    train_script_base = Path(train_script).name
    train_cfg = _load_yaml(train_yaml)

    ckpt_path = _resolve_best_ckpt(train_cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}

    # derive model_id for filename
    model_id = str(eval_cfg.get("model_id") or train_cfg.get("run_name") or "model")
    ckpt_tag = sha1_12(str(ckpt_path.resolve()))

    # runtime overrides
    if device is None:
        device = str(eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device_t = torch.device(device)
    amp_enabled = (device_t.type == "cuda") and bf16

    written: List[str] = []

    if train_script_base in ("train_pair_interaction_esm2_eos.py", "train_pair_interaction_esmpp_eos.py"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_name = ckpt_cfg.get("model_name") or train_cfg.get("model_name") or train_cfg.get("model_name") or "facebook/esm2_t12_35M_UR50D"
        max_len = int(ckpt_cfg.get("max_len") or train_cfg.get("max_len") or 512)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(device_t)
        model.eval()

        for ds in datasets:
            ds_id = str(ds["id"])
            csv = str(ds["csv"])
            df = load_pairs_csv(csv, max_len_kin=max_len, max_len_sub=max_len)
            loader = make_loader_hf_eos(df, tokenizer, max_len=max_len, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            pred = predict_hf_eos(model, loader, device=device_t, amp_enabled=amp_enabled)

            payload: Dict[str, Any] = {
                "model_id": model_id,
                "dataset_id": ds_id,
                "model_data_id": f"{model_id}__{ds_id}__{ckpt_tag}",
                "model_data_identifier": model_data_identifier,
                "training_command": training_command,
                "train_script": str(train_script),
                "train_config": str(train_yaml),
                "best_checkpoint": str(ckpt_path),
                "ckpt_cfg": ckpt_cfg,
                "meta": {"scheme": "hf_eos", "model_name": model_name, "max_len": max_len},
                **pred,
            }

            out_path = output_dir_p / f"{model_id}__{ds_id}__{ckpt_tag}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            written.append(str(out_path))

    elif train_script_base in ("train_pair_interaction_tape.py", "train_pair_interaction_tape_updated_fast.py"):
        from tape import ProteinBertForSequenceClassification, TAPETokenizer

        max_len = int(ckpt_cfg.get("max_len") or train_cfg.get("max_len") or 1024)
        with_sep = bool(ckpt_cfg.get("with_sep") if "with_sep" in ckpt_cfg else train_cfg.get("with_sep", True))
        tape_model_name = ckpt_cfg.get("tape_model_name") or train_cfg.get("tape_model_name") or "bert-base"

        tokenizer = TAPETokenizer(vocab="iupac")
        model = ProteinBertForSequenceClassification.from_pretrained(tape_model_name, num_labels=2)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.to(device_t)
        model.eval()

        for ds in datasets:
            ds_id = str(ds["id"])
            print(f'starting on: {ds_id}')
            csv = str(ds["csv"])
            df = load_pairs_csv(csv, max_len_kin=max_len, max_len_sub=max_len)
            loader = make_loader_tape_sep(df, tokenizer, max_len=max_len, batch_size=batch_size, num_workers=num_workers, with_sep=with_sep, shuffle=False)
            pred = predict_tape_sep(model, loader, device=device_t, amp_enabled=amp_enabled)

            payload: Dict[str, Any] = {
                "model_id": model_id,
                "dataset_id": ds_id,
                "model_data_id": f"{model_id}__{ds_id}__{ckpt_tag}",
                "model_data_identifier": model_data_identifier,
                "training_command": training_command,
                "train_script": str(train_script),
                "train_config": str(train_yaml),
                "best_checkpoint": str(ckpt_path),
                "ckpt_cfg": ckpt_cfg,
                "meta": {"scheme": "tape_sep", "tape_model_name": tape_model_name, "max_len": max_len, "with_sep": with_sep},
                **pred,
            }

            out_path = output_dir_p / f"{model_id}__{ds_id}__{ckpt_tag}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            written.append(str(out_path))

    else:
        raise NotImplementedError(
            f"Unsupported train script for separator-token eval: {train_script_base}. "
            "Supported: train_pair_interaction_esm2_eos.py, train_pair_interaction_esmpp_eos.py, train_pair_interaction_tape.py"
        )

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_config", required=True, help="Path to eval YAML")
    ap.add_argument("--device", default=None, help="Override device, e.g. cuda:0 or cpu")
    args = ap.parse_args()
    outs = run_eval_from_eval_yaml(args.eval_config, device=args.device)
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
