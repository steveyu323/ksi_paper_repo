#!/usr/bin/env python3
"""
Train a kinase–substrate interaction classifier using a TAPE ProteinBert backbone,
driven by a YAML config.

UPDATED (speed-focused):
- Pre-encodes each (kinase_seq, substrate_seq) pair into input_ids ONCE in the Dataset
  (instead of tokenizing inside collate every batch). This removes a major CPU bottleneck
  in the original script. :contentReference[oaicite:0]{index=0}
- DataLoader: persistent_workers + prefetch_factor when num_workers>0
- Optional TF32 for faster matmul on Ampere+ GPUs (safe for transformers)
- Keeps the same pair encoding scheme:
    toks = tokenize(kin) + ['<sep>'] + tokenize(sub); add_special_tokens(toks)

Input CSVs must contain columns: kinase_seq, substrate_seq, label

Usage:
  python train_pair_interaction_tape_updated_fast.py --config configs/configs_tape_exp1_updated.yml
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from tape import ProteinBertForSequenceClassification, TAPETokenizer


WHITESPACE_RE = re.compile(r"\s+")
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUOJ]+$")


def sanitize_seq(s: str) -> str:
    if s is None:
        return ""
    return WHITESPACE_RE.sub("", str(s).upper())


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    ks: Tuple[int, ...] = (50, 100, 200),
) -> Dict[str, float]:
    y_true = y_true.astype(np.int32)
    y_prob = y_prob.astype(np.float64)

    y_pred = (y_prob >= threshold).astype(np.int32)
    cc = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]

    acc = safe_div(tp + tn, tp + tn + fp + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    spec = safe_div(tn, tn + fp)
    bal_acc = 0.5 * (rec + spec)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_div((tp * tn - fp * fn), denom) if denom > 0 else 0.0

    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    logloss = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    brier = float(((y_prob - y_true) ** 2).mean())

    auroc = np.nan
    auprc = np.nan
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass

    order = np.argsort(-y_prob)
    total_pos = int(y_true.sum())
    out: Dict[str, float] = {}
    for K in ks:
        K_eff = min(K, len(y_true))
        top = order[:K_eff]
        tp_k = int(y_true[top].sum())
        out[f"precision@{K}"] = safe_div(tp_k, K_eff)
        out[f"recall@{K}"] = safe_div(tp_k, total_pos) if total_pos > 0 else 0.0

    out.update({
        "threshold": float(threshold),
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "balanced_acc": bal_acc,
        "mcc": mcc,
        "logloss": logloss,
        "brier": brier,
        "auroc": float(auroc) if auroc == auroc else np.nan,
        "auprc": float(auprc) if auprc == auprc else np.nan,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    })
    return out


def choose_threshold_on_val(y_true: np.ndarray, y_prob: np.ndarray, policy: str) -> float:
    if policy == "fixed_0.5":
        return 0.5
    probs = np.unique(y_prob)
    if len(probs) > 2000:
        probs = np.quantile(y_prob, np.linspace(0.0, 1.0, 2000))
        probs = np.unique(probs)

    best_t, best = 0.5, -1e9
    for t in probs:
        m = compute_metrics(y_true, y_prob, threshold=float(t), ks=())
        score = m["mcc"] if policy == "val_max_mcc" else m["f1"]
        if score > best:
            best = score
            best_t = float(t)
    return best_t


def load_and_filter(csv_path: str, max_len: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["kinase_seq", "substrate_seq", "label"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {csv_path}. Found: {list(df.columns)}")

    df["kinase_seq"] = df["kinase_seq"].map(sanitize_seq)
    df["substrate_seq"] = df["substrate_seq"].map(sanitize_seq)

    ok = (
        df["kinase_seq"].str.len().between(1, max_len)
        & df["substrate_seq"].str.len().between(1, max_len)
        & df["kinase_seq"].str.match(VALID_AA_RE)
        & df["substrate_seq"].str.match(VALID_AA_RE)
        & df["label"].isin([0, 1, 0.0, 1.0, "0", "1"])
    )
    df = df[ok].copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_and_filter_optional(csv_path: Optional[str], max_len: int) -> Optional[pd.DataFrame]:
    if csv_path is None:
        return None
    csv_path = str(csv_path).strip()
    if csv_path == "" or csv_path.lower() in ("none", "null", "na"):
        return None
    p = Path(csv_path)
    if not p.exists():
        print(f"[warn] test2_csv not found: {csv_path} (skipping test2 eval)")
        return None
    try:
        df = load_and_filter(csv_path, max_len=max_len)
        if len(df) == 0:
            print(f"[warn] test2_csv empty after filtering: {csv_path} (skipping test2 eval)")
            return None
        return df
    except Exception as e:
        print(f"[warn] failed to load test2_csv={csv_path}: {e} (skipping test2 eval)")
        return None


class PairEncodedDataset(Dataset):
    """
    Pre-encode each (kinase_seq, substrate_seq) pair into input_ids once.
    This replaces per-batch tokenization in collate (a major bottleneck in the original). :contentReference[oaicite:1]{index=1}
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: TAPETokenizer,
        max_len: int,
        with_sep: bool = True,
    ):
        self.with_sep = bool(with_sep)
        self.max_len = int(max_len)
        self.pad_id = tokenizer.vocab.get("<pad>", 0)

        if self.with_sep and "<sep>" not in tokenizer.vocab:
            raise ValueError(
                "Tokenizer vocab does not contain '<sep>'. "
                "Use TAPETokenizer(vocab='iupac') or set with_sep=False."
            )

        kin_list = df["kinase_seq"].tolist()
        sub_list = df["substrate_seq"].tolist()
        self.y = df["label"].astype(np.int64).values

        # store as numpy arrays (variable length)
        self.ids_list: List[np.ndarray] = []
        for kin, sub in zip(kin_list, sub_list):
            kin_toks = tokenizer.tokenize(kin)
            sub_toks = tokenizer.tokenize(sub)
            toks = kin_toks + (["<sep>"] if self.with_sep else []) + sub_toks
            toks = tokenizer.add_special_tokens(toks)
            ids = tokenizer.convert_tokens_to_ids(toks)[: self.max_len]
            self.ids_list.append(np.asarray(ids, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.ids_list[i], int(self.y[i])


def make_loader_encoded(
    ds: PairEncodedDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    pad_id = int(ds.pad_id)

    def collate(batch):
        ids_list, ys = zip(*batch)
        Lmax = max(len(x) for x in ids_list)

        input_ids = torch.full((len(ids_list), Lmax), fill_value=pad_id, dtype=torch.long)
        input_mask = torch.zeros((len(ids_list), Lmax), dtype=torch.long)

        for i, arr in enumerate(ids_list):
            arr_t = torch.from_numpy(arr)
            n = int(arr_t.numel())
            input_ids[i, :n] = arr_t
            input_mask[i, :n] = 1

        y = torch.tensor(ys, dtype=torch.long)
        return {"input_ids": input_ids, "input_mask": input_mask, "label": y}

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate,
        pin_memory=True,
    )
    if num_workers and int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(ds, **kwargs)


@dataclass
class Config:
    # data
    train_csv: str
    val_csv: str
    test_csv: str
    test2_csv: Optional[str] = None

    # run/logging
    run_name: str = "tape_run"
    out_dir: str = "./runs"

    # tokenization
    max_len: int = 1024
    with_sep: bool = True

    # backbone
    tape_model_name: str = "bert-base"
    freeze_backbone: bool = False
    dropout: float = 0.1

    # training
    epochs: int = 20
    batch_size: int = 16
    grad_accum: int = 1
    num_workers: int = 4
    bf16: bool = False
    allow_tf32: bool = True  # NEW: speed on Ampere+ GPUs

    # loss weighting (like newer code)
    use_pos_weight: bool = True

    # optimizer
    optimizer_type: str = "sgd"  # sgd | adamw
    lr: float = 1e-3
    weight_decay: float = 0.01
    sgd_momentum: float = 0.9
    sgd_nesterov: bool = True
    adamw_betas: Tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = 1e-8

    # scheduler (ReduceLROnPlateau)
    scheduler_type: str = "plateau"     # plateau | none
    plateau_monitor: str = "val_loss"   # val_loss | val_auprc | val_mcc | val_f1
    plateau_mode: str = "min"           # min for loss, max for metrics
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_threshold: float = 0.0
    plateau_threshold_mode: str = "rel" # rel | abs
    plateau_cooldown: int = 0
    plateau_min_lr: float = 1e-6

    # eval
    threshold_policy: str = "val_max_mcc"  # fixed_0.5 | val_max_mcc | val_max_f1
    ks: Tuple[int, ...] = (50, 100, 200)
    early_stop_metric: str = "auprc"       # auprc | mcc | f1
    patience: int = 3
    test_on_val_improve: bool = True


def dict_to_config(d: Dict[str, Any]) -> Config:
    if "ks" in d and isinstance(d["ks"], list):
        d["ks"] = tuple(int(x) for x in d["ks"])

    sch = d.pop("scheduler", None)
    if sch is not None and isinstance(sch, dict):
        d["scheduler_type"] = sch.get("type", d.get("scheduler_type", "plateau"))
        d["plateau_monitor"] = sch.get("monitor", d.get("plateau_monitor", "val_loss"))
        d["plateau_mode"] = sch.get("mode", d.get("plateau_mode", "min"))
        d["plateau_factor"] = float(sch.get("factor", d.get("plateau_factor", 0.5)))
        d["plateau_patience"] = int(sch.get("patience", d.get("plateau_patience", 2)))
        d["plateau_threshold"] = float(sch.get("threshold", d.get("plateau_threshold", 0.0)))
        d["plateau_threshold_mode"] = sch.get("threshold_mode", d.get("plateau_threshold_mode", "rel"))
        d["plateau_cooldown"] = int(sch.get("cooldown", d.get("plateau_cooldown", 0)))
        d["plateau_min_lr"] = float(sch.get("min_lr", d.get("plateau_min_lr", 1e-6)))

    opt = d.pop("optimizer", None)  # POP so Config doesn't see nested dict
    if isinstance(opt, dict):
        d["optimizer_type"] = opt.get("type", d.get("optimizer_type", "sgd"))
        if "lr" in opt:
            d["lr"] = float(opt["lr"])
        if "momentum" in opt:
            d["sgd_momentum"] = float(opt["momentum"])
        if "nesterov" in opt:
            d["sgd_nesterov"] = bool(opt["nesterov"])
        if "betas" in opt and isinstance(opt["betas"], (list, tuple)) and len(opt["betas"]) == 2:
            d["adamw_betas"] = (float(opt["betas"][0]), float(opt["betas"][1]))
        if "eps" in opt:
            d["adamw_eps"] = float(opt["eps"])

    return Config(**d)


def build_optimizer(cfg: Config, params) -> torch.optim.Optimizer:
    opt_type = (cfg.optimizer_type or "sgd").lower()
    if opt_type == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(cfg.lr),
            momentum=float(cfg.sgd_momentum),
            nesterov=bool(cfg.sgd_nesterov),
            weight_decay=float(cfg.weight_decay),
        )
    if opt_type in ("adamw", "adam_w"):
        b0, b1 = cfg.adamw_betas
        return torch.optim.AdamW(
            params,
            lr=float(cfg.lr),
            betas=(float(b0), float(b1)),
            eps=float(cfg.adamw_eps),
            weight_decay=float(cfg.weight_decay),
        )
    raise ValueError(f"Unknown optimizer_type={cfg.optimizer_type!r}. Use 'sgd' or 'adamw'.")


@torch.no_grad()
def run_eval(
    model: ProteinBertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    ys, ps = [], []
    loss_sum, n = 0.0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            (loss, _metrics), logits = model(input_ids=input_ids, input_mask=input_mask, targets=y)
            prob = torch.softmax(logits, dim=-1)[:, 1]

        loss_sum += float(loss.item()) * int(y.numel())
        n += int(y.numel())
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    cfg = dict_to_config(raw)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = (device.type == "cuda") and bool(cfg.bf16)

    if device.type == "cuda" and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg.out_dir) / cfg.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    tokenizer = TAPETokenizer(vocab="iupac")
    model = ProteinBertForSequenceClassification.from_pretrained(
        cfg.tape_model_name,
        num_labels=2,
    ).to(device)

    if cfg.freeze_backbone:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False

    # data
    df_tr = load_and_filter(cfg.train_csv, max_len=cfg.max_len)
    df_va = load_and_filter(cfg.val_csv, max_len=cfg.max_len)
    df_te = load_and_filter(cfg.test_csv, max_len=cfg.max_len)
    df_te2 = load_and_filter_optional(cfg.test2_csv, max_len=cfg.max_len)

    # IMPORTANT: pre-encode (moves tokenization cost out of the training loop)
    train_ds = PairEncodedDataset(df_tr, tokenizer, max_len=cfg.max_len, with_sep=cfg.with_sep)
    val_ds   = PairEncodedDataset(df_va, tokenizer, max_len=cfg.max_len, with_sep=cfg.with_sep)
    test_ds  = PairEncodedDataset(df_te, tokenizer, max_len=cfg.max_len, with_sep=cfg.with_sep)
    test2_ds = PairEncodedDataset(df_te2, tokenizer, max_len=cfg.max_len, with_sep=cfg.with_sep) if df_te2 is not None else None

    train_loader = make_loader_encoded(train_ds, cfg.batch_size, True,  cfg.num_workers)
    val_loader   = make_loader_encoded(val_ds,   cfg.batch_size, False, cfg.num_workers)
    test_loader  = make_loader_encoded(test_ds,  cfg.batch_size, False, cfg.num_workers)
    test2_loader = make_loader_encoded(test2_ds, cfg.batch_size, False, cfg.num_workers) if test2_ds is not None else None

    # pos_weight from TRAIN (optional)
    pos_weight = None
    if cfg.use_pos_weight:
        ytr = df_tr["label"].values.astype(np.float32)
        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    opt = build_optimizer(cfg, [p for p in model.parameters() if p.requires_grad])

    sched = None
    if cfg.scheduler_type in (None, "none", "NONE", "None"):
        sched = None
    elif cfg.scheduler_type == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=str(cfg.plateau_mode),
            factor=float(cfg.plateau_factor),
            patience=int(cfg.plateau_patience),
            threshold=float(cfg.plateau_threshold),
            threshold_mode=str(cfg.plateau_threshold_mode),
            cooldown=int(cfg.plateau_cooldown),
            min_lr=float(cfg.plateau_min_lr),
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {cfg.scheduler_type}")
    ################NEW########
    if cfg.early_stop_metric!='loss':
        best_val = -1e9
    else:
        best_val = 1e9
    
    # best_val = -1e9
    bad_epochs = 0
    global_step = 0
    metrics_path = out_dir / "metrics.csv"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        opt.zero_grad(set_to_none=True)

        running_loss, n_seen = 0.0, 0

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            input_mask = batch["input_mask"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                (loss, _metrics), logits = model(input_ids=input_ids, input_mask=input_mask, targets=y)

                # override loss (optional) to match newer code's pos_weight behavior
                if pos_weight is not None:
                    logit1 = logits[:, 1]
                    loss = F.binary_cross_entropy_with_logits(
                        logit1, y.float(), pos_weight=pos_weight, reduction="mean"
                    )

                loss = loss / max(1, cfg.grad_accum)

            loss.backward()
            running_loss += float(loss.item()) * int(y.numel()) * max(1, cfg.grad_accum)
            n_seen += int(y.numel())

            if step % cfg.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                for gi, pg in enumerate(opt.param_groups):
                    tb.add_scalar(f"train/lr_group{gi}", pg["lr"], global_step)
                tb.add_scalar("train/loss_step", float(loss.item()) * max(1, cfg.grad_accum), global_step)

            pbar.set_postfix(loss=float(loss.item()) * max(1, cfg.grad_accum))

        train_loss = running_loss / max(1, n_seen)

        # val
        yv, pv, val_loss = run_eval(model, val_loader, device=device, amp_enabled=amp_enabled)
        thr = choose_threshold_on_val(yv, pv, policy=cfg.threshold_policy)
        val_metrics = compute_metrics(yv, pv, threshold=thr, ks=cfg.ks)
        val_metrics["loss"] = float(val_loss)
        val_metrics["train_loss"] = float(train_loss)

        # scheduler step (guard NaN)
        if sched is not None:
            if cfg.plateau_monitor == "val_loss":
                monitor_value = float(val_loss)
            elif cfg.plateau_monitor == "val_auprc":
                monitor_value = float(val_metrics.get("auprc", np.nan))
            elif cfg.plateau_monitor == "val_mcc":
                monitor_value = float(val_metrics.get("mcc", np.nan))
            elif cfg.plateau_monitor == "val_f1":
                monitor_value = float(val_metrics.get("f1", np.nan))
            else:
                raise ValueError(f"Unknown plateau_monitor: {cfg.plateau_monitor}")
            if monitor_value == monitor_value:
                sched.step(monitor_value)

        key = cfg.early_stop_metric
        val_score = val_metrics.get(key, np.nan)
        
        # improved = (val_score == val_score) and (val_score > best_val)
        if key != 'loss':
            print('a')
            print(val_score)
            print(best_val)
            improved = (val_score == val_score) and (val_score > best_val)
        else:
            print('b')
            print(val_score)
            print(best_val)
            improved = (val_score == val_score) and (val_score < best_val)
        print("::" + key +"::")
        print(improved)
        # test/test2 only when val improves unless disabled
        test_metrics, test2_metrics = {}, {}
        if (not cfg.test_on_val_improve) or improved:
            yt, pt, test_loss = run_eval(model, test_loader, device=device, amp_enabled=amp_enabled)
            test_metrics = compute_metrics(yt, pt, threshold=thr, ks=cfg.ks)
            test_metrics["loss"] = float(test_loss)

            if test2_loader is not None:
                y2, p2, t2_loss = run_eval(model, test2_loader, device=device, amp_enabled=amp_enabled)
                test2_metrics = compute_metrics(y2, p2, threshold=thr, ks=cfg.ks)
                test2_metrics["loss"] = float(t2_loss)

        # tensorboard
        tb.add_scalar("train/loss_epoch", train_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)

        # epoch-level LR logging (after scheduler step)
        for gi, pg in enumerate(opt.param_groups):
            tb.add_scalar(f"epoch/lr_group{gi}", pg["lr"], epoch)

        for k2, v2 in val_metrics.items():
            if isinstance(v2, (int, float)) and k2 not in ("tp", "tn", "fp", "fn"):
                tb.add_scalar(f"val/{k2}", float(v2), epoch)
        for k2, v2 in test_metrics.items():
            if isinstance(v2, (int, float)) and k2 not in ("tp", "tn", "fp", "fn"):
                tb.add_scalar(f"test/{k2}", float(v2), epoch)
        for k2, v2 in test2_metrics.items():
            if isinstance(v2, (int, float)) and k2 not in ("tp", "tn", "fp", "fn"):
                tb.add_scalar(f"test2/{k2}", float(v2), epoch)

        # metrics.csv
        row = {"epoch": epoch, "global_step": global_step, **{f"val_{k}": v for k, v in val_metrics.items()}}
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        row.update({f"test2_{k}": v for k, v in test2_metrics.items()})
        df_row = pd.DataFrame([row])
        write_header = (not metrics_path.exists()) or (metrics_path.stat().st_size == 0)
        df_row.to_csv(metrics_path, mode="a", header=write_header, index=False)

        # checkpoint / early stop
        if improved:
            best_val = float(val_score)
            bad_epochs = 0
            ckpt = out_dir / "checkpoints" / "best.pt"
            torch.save(
                {"cfg": asdict(cfg), "state_dict": model.state_dict(),
                 "epoch": epoch, "best_val": best_val, "threshold": thr},
                ckpt,
            )
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_{key}={val_score:.4f} "
            f"val_auprc={val_metrics.get('auprc', np.nan):.4f} val_mcc={val_metrics.get('mcc', np.nan):.4f} "
            f"{'(improved)' if improved else ''}"
        )

        if bad_epochs >= cfg.patience:
            print(f"Early stopping after {cfg.patience} bad epochs.")
            break

    tb.close()


if __name__ == "__main__":
    main()
