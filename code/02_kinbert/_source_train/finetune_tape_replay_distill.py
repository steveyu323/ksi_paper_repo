#!/usr/bin/env python3
"""
Fine-tune a pretrained kinase–substrate TAPE model on new experimental data
WITHOUT forgetting the original public-dataset performance.

Three complementary mechanisms (all configurable, can be used independently):

  1. PARTIAL UNFREEZE + DISCRIMINATIVE LR
     Freeze the entire backbone, then unfreeze only the last N transformer
     blocks at a low LR.  Classifier head trains at a higher LR.
       - Param group 0 ("unfrozen_backbone"): lr * backbone_lr_mult
       - Param group 1 ("classifier_head"):   lr * head_lr_mult

  2. MIXED REPLAY
     Each training batch is assembled from TWO datasets — the new experimental
     data and a replay buffer of the original public training data.  A weighted
     sampler controls the mix ratio so the model keeps seeing old examples.
       - replay_csv:               path to original public training CSV
       - replay_ratio:             fraction of each batch that is replay data
                                   (0.0 = no replay, 0.5 = half-and-half)
       - replay_sample_strategy:   "proportional" (weight by ratio) or
                                   "upsample_exp" (oversample experimental)

  3. DISTILLATION REGULARISATION
     A frozen copy of the original model (the "teacher") produces soft targets
     on every batch.  A KL-divergence term between student and teacher output
     distributions is added to the loss:
       L_total = (1 - distill_alpha) * L_bce  +  distill_alpha * L_kl
     where L_kl = KL( softmax(student_logits/T) || softmax(teacher_logits/T) )
       - distill_alpha:      weight of distillation loss (0.0 = off, 0.5 = equal)
       - distill_temperature: temperature for softening (default 2.0)
       - pretrained_ckpt:     the teacher checkpoint (required for distillation)

Based on train_pair_interaction_tape.py.  All original eval/metrics/logging
preserved.

Usage:
  python finetune_tape_replay_distill.py --config configs/config_finetune.yml

  # override checkpoint from CLI:
  python finetune_tape_replay_distill.py --config configs/config_finetune.yml \
      --pretrained_ckpt runs/tape_run/checkpoints/best.pt
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import copy
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from tape import ProteinBertForSequenceClassification, TAPETokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Utilities (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

WHITESPACE_RE = re.compile(r"\s+")
VALID_AA_RE   = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZUOJ]+$")


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

    acc   = safe_div(tp + tn, tp + tn + fp + fn)
    prec  = safe_div(tp, tp + fp)
    rec   = safe_div(tp, tp + fn)
    f1    = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
    spec  = safe_div(tn, tn + fp)
    bal_acc = 0.5 * (rec + spec)

    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc   = safe_div((tp * tn - fp * fn), denom) if denom > 0 else 0.0

    eps = 1e-12
    p = np.clip(y_prob, eps, 1 - eps)
    logloss = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())
    brier   = float(((y_prob - y_true) ** 2).mean())

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
        out[f"recall@{K}"]    = safe_div(tp_k, total_pos) if total_pos > 0 else 0.0

    out.update({
        "threshold": float(threshold),
        "acc": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": spec, "balanced_acc": bal_acc, "mcc": mcc,
        "logloss": logloss, "brier": brier,
        "auroc": float(auroc) if auroc == auroc else np.nan,
        "auprc": float(auprc) if auprc == auprc else np.nan,
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    })
    return out


def choose_threshold_on_val(y_true: np.ndarray, y_prob: np.ndarray,
                            policy: str) -> float:
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


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

def load_and_filter(csv_path: str, max_len: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for c in ["kinase_seq", "substrate_seq", "label"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {csv_path}. "
                           f"Found: {list(df.columns)}")
    df["kinase_seq"]    = df["kinase_seq"].map(sanitize_seq)
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


def load_and_filter_optional(csv_path: Optional[str],
                             max_len: int) -> Optional[pd.DataFrame]:
    if csv_path is None:
        return None
    csv_path = str(csv_path).strip()
    if csv_path == "" or csv_path.lower() in ("none", "null", "na"):
        return None
    p = Path(csv_path)
    if not p.exists():
        print(f"[warn] csv not found: {csv_path} (skipping)")
        return None
    try:
        df = load_and_filter(csv_path, max_len=max_len)
        if len(df) == 0:
            print(f"[warn] csv empty after filtering: {csv_path} (skipping)")
            return None
        return df
    except Exception as e:
        print(f"[warn] failed to load {csv_path}: {e} (skipping)")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Dataset (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

class PairEncodedDataset(Dataset):
    """Pre-encode each (kinase_seq, substrate_seq) pair into input_ids once."""

    def __init__(self, df: pd.DataFrame, tokenizer: TAPETokenizer,
                 max_len: int, with_sep: bool = True):
        self.with_sep = bool(with_sep)
        self.max_len  = int(max_len)
        self.pad_id   = tokenizer.vocab.get("<pad>", 0)

        if self.with_sep and "<sep>" not in tokenizer.vocab:
            raise ValueError(
                "Tokenizer vocab does not contain '<sep>'. "
                "Use TAPETokenizer(vocab='iupac') or set with_sep=False.")

        kin_list = df["kinase_seq"].tolist()
        sub_list = df["substrate_seq"].tolist()
        self.y = df["label"].astype(np.int64).values

        self.ids_list: List[np.ndarray] = []
        for kin, sub in zip(kin_list, sub_list):
            kin_toks = tokenizer.tokenize(kin)
            sub_toks = tokenizer.tokenize(sub)
            toks = kin_toks + (["<sep>"] if self.with_sep else []) + sub_toks
            toks = tokenizer.add_special_tokens(toks)
            ids  = tokenizer.convert_tokens_to_ids(toks)[:self.max_len]
            self.ids_list.append(np.asarray(ids, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return self.ids_list[i], int(self.y[i])


def _make_collate_fn(pad_id: int):
    """Return a collate function closed over pad_id."""
    def collate(batch):
        ids_list, ys = zip(*batch)
        Lmax = max(len(x) for x in ids_list)
        input_ids  = torch.full((len(ids_list), Lmax), fill_value=pad_id,
                                dtype=torch.long)
        input_mask = torch.zeros((len(ids_list), Lmax), dtype=torch.long)
        for i, arr in enumerate(ids_list):
            arr_t = torch.from_numpy(arr) if isinstance(arr, np.ndarray) \
                    else torch.as_tensor(arr)
            n = int(arr_t.numel())
            input_ids[i, :n]  = arr_t
            input_mask[i, :n] = 1
        y = torch.tensor(ys, dtype=torch.long)
        return {"input_ids": input_ids, "input_mask": input_mask, "label": y}
    return collate


def make_loader_encoded(ds: PairEncodedDataset, batch_size: int,
                        shuffle: bool, num_workers: int,
                        sampler=None) -> DataLoader:
    """Build a DataLoader.  If sampler is provided, shuffle must be False."""
    pad_id = int(ds.pad_id)
    kwargs = dict(
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=_make_collate_fn(pad_id),
        pin_memory=True,
    )
    if num_workers and int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = 2
    return DataLoader(ds, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Mixed-replay DataLoader
# ═══════════════════════════════════════════════════════════════════════════

def build_mixed_replay_loader(
    exp_ds: PairEncodedDataset,
    replay_ds: PairEncodedDataset,
    batch_size: int,
    replay_ratio: float,
    num_workers: int,
    epoch_length_from: str = "experimental",  # "experimental" | "replay" | "max"
) -> DataLoader:
    """
    Build a single DataLoader over ConcatDataset([exp_ds, replay_ds]) with a
    WeightedRandomSampler that controls the mixing ratio per epoch.

    replay_ratio = 0.3 means ~30% of samples drawn per epoch come from the
    replay (original public) dataset, ~70% from the experimental dataset.

    epoch_length_from controls how many samples constitute one "epoch":
      - "experimental": len(exp_ds)          (loop once through experimental)
      - "replay":       len(replay_ds)
      - "max":          max(len(exp_ds), len(replay_ds))
    """
    n_exp    = len(exp_ds)
    n_replay = len(replay_ds)
    total    = n_exp + n_replay

    # per-sample weight so that sampling probabilities match the desired ratio
    # P(sample from replay) = replay_ratio
    # P(sample from experimental) = 1 - replay_ratio
    w_exp    = (1.0 - replay_ratio) / max(1, n_exp)
    w_replay = replay_ratio / max(1, n_replay)

    weights = [w_exp] * n_exp + [w_replay] * n_replay
    weights = torch.tensor(weights, dtype=torch.double)

    if epoch_length_from == "experimental":
        num_samples = n_exp
    elif epoch_length_from == "replay":
        num_samples = n_replay
    elif epoch_length_from == "max":
        num_samples = max(n_exp, n_replay)
    else:
        num_samples = n_exp

    sampler = WeightedRandomSampler(weights, num_samples=num_samples,
                                    replacement=True)

    combined = ConcatDataset([exp_ds, replay_ds])

    # Use the experimental dataset's pad_id (same tokenizer)
    pad_id = int(exp_ds.pad_id)
    kwargs = dict(
        batch_size=batch_size,
        sampler=sampler,
        num_workers=int(num_workers),
        collate_fn=_make_collate_fn(pad_id),
        pin_memory=True,
    )
    if num_workers and int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"]    = 2
    return DataLoader(combined, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Block identification and partial unfreeze
# ═══════════════════════════════════════════════════════════════════════════

def get_transformer_block_names(model: nn.Module) -> List[str]:
    """Discover transformer block paths: bert.encoder.layer.0, .1, ..."""
    block_names = set()
    for name, _ in model.named_parameters():
        m = re.match(r"(bert\.encoder\.layer\.\d+)\.", name)
        if m:
            block_names.add(m.group(1))
    return sorted(block_names, key=lambda s: int(s.rsplit(".", 1)[-1]))


def setup_partial_unfreeze(
    model: nn.Module,
    unfreeze_last_n: int,
) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Freeze everything, then selectively unfreeze:
      - last `unfreeze_last_n` transformer blocks  → returned as backbone_params
      - classifier / predict layers                → returned as head_params

    Returns (backbone_params, head_params).
    """
    # 1) freeze all
    for p in model.parameters():
        p.requires_grad = False

    # 2) unfreeze last N blocks
    block_names = get_transformer_block_names(model)
    blocks_to_unfreeze = block_names[-unfreeze_last_n:] if unfreeze_last_n > 0 else []
    if blocks_to_unfreeze:
        print(f"[finetune] Unfreezing last {unfreeze_last_n} transformer blocks: "
              f"{blocks_to_unfreeze}")

    backbone_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        for block in blocks_to_unfreeze:
            if name.startswith(block + "."):
                p.requires_grad = True
                backbone_params.append(p)
                break

    # 3) unfreeze classifier head
    head_params: List[nn.Parameter] = []
    for name, p in model.named_parameters():
        if "classifier" in name or "predict" in name:
            p.requires_grad = True
            head_params.append(p)

    return backbone_params, head_params


# ═══════════════════════════════════════════════════════════════════════════
# Distillation helpers
# ═══════════════════════════════════════════════════════════════════════════

def build_teacher(
    model_class,
    tape_model_name: str,
    ckpt_path: str,
    device: torch.device,
) -> nn.Module:
    """
    Build a frozen copy of the original model to serve as distillation teacher.
    """
    teacher = model_class.from_pretrained(tape_model_name, num_labels=2)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [teacher] Missing keys: "
              f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [teacher] Unexpected keys: "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    teacher = teacher.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"[finetune] Teacher model loaded from {ckpt_path} (frozen)")
    return teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    KL(softmax(student/T) || softmax(teacher/T)) * T^2

    T^2 scaling keeps gradient magnitudes comparable across temperatures.
    """
    T = temperature
    student_log_p = F.log_softmax(student_logits / T, dim=-1)
    teacher_p     = F.softmax(teacher_logits / T, dim=-1)
    # KLDivLoss expects log-probs as input, probs as target
    kl = F.kl_div(student_log_p, teacher_p, reduction="batchmean") * (T * T)
    return kl


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FinetuneConfig:
    # ── data ──
    train_csv:  str = ""              # NEW experimental training data
    val_csv:    str = ""              # experimental validation
    test_csv:   str = ""              # experimental test
    test2_csv:  Optional[str] = None  # optional second test set (e.g. original test)

    # ── run / logging ──
    run_name: str = "finetune_run"
    out_dir:  str = "./runs"

    # ── tokenization ──
    max_len:  int  = 1024
    with_sep: bool = True

    # ── backbone ──
    tape_model_name: str   = "bert-base"
    dropout:         float = 0.1

    # ── pretrained checkpoint ──
    pretrained_ckpt: Optional[str] = None  # path to best.pt from full training

    # ── partial unfreeze ──
    unfreeze_last_n_blocks: int = 2        # 0 = head only

    # ── discriminative learning rates ──
    lr:               float = 5e-4         # base LR (classifier head)
    backbone_lr_mult: float = 0.1          # unfrozen blocks get lr * this
    head_lr_mult:     float = 1.0          # head gets lr * this

    # ── mixed replay ──
    replay_csv:             Optional[str] = None   # original public training CSV
    replay_ratio:           float = 0.3            # fraction of batch from replay
    replay_epoch_length:    str   = "experimental"  # experimental | replay | max

    # ── distillation ──
    distill_alpha:       float = 0.0    # 0.0 = off;  0.3-0.5 = typical
    distill_temperature: float = 2.0    # softening temperature

    # ── training ──
    epochs:     int  = 30
    batch_size: int  = 16
    grad_accum: int  = 1
    num_workers: int = 4
    bf16:       bool = False
    allow_tf32: bool = True

    # ── loss ──
    use_pos_weight: bool = True

    # ── optimizer ──
    optimizer_type: str                    = "adamw"
    weight_decay:   float                  = 0.01
    adamw_betas:    Tuple[float, float]    = (0.9, 0.999)
    adamw_eps:      float                  = 1e-8
    sgd_momentum:   float                  = 0.9
    sgd_nesterov:   bool                   = True

    # ── scheduler ──
    scheduler_type:         str   = "plateau"
    plateau_monitor:        str   = "val_loss"
    plateau_mode:           str   = "min"
    plateau_factor:         float = 0.5
    plateau_patience:       int   = 3
    plateau_threshold:      float = 0.0
    plateau_threshold_mode: str   = "rel"
    plateau_cooldown:       int   = 0
    plateau_min_lr:         float = 1e-7

    # ── eval ──
    threshold_policy:    str            = "val_max_mcc"
    ks:                  Tuple[int,...] = (50, 100, 200)
    early_stop_metric:   str            = "auprc"
    patience:            int            = 5
    test_on_val_improve: bool           = True


def dict_to_config(d: Dict[str, Any]) -> FinetuneConfig:
    if "ks" in d and isinstance(d["ks"], list):
        d["ks"] = tuple(int(x) for x in d["ks"])

    # flatten nested scheduler dict
    sch = d.pop("scheduler", None)
    if sch is not None and isinstance(sch, dict):
        d["scheduler_type"]         = sch.get("type",           d.get("scheduler_type", "plateau"))
        d["plateau_monitor"]        = sch.get("monitor",        d.get("plateau_monitor", "val_loss"))
        d["plateau_mode"]           = sch.get("mode",           d.get("plateau_mode", "min"))
        d["plateau_factor"]         = float(sch.get("factor",   d.get("plateau_factor", 0.5)))
        d["plateau_patience"]       = int(sch.get("patience",   d.get("plateau_patience", 3)))
        d["plateau_threshold"]      = float(sch.get("threshold",d.get("plateau_threshold", 0.0)))
        d["plateau_threshold_mode"] = sch.get("threshold_mode", d.get("plateau_threshold_mode", "rel"))
        d["plateau_cooldown"]       = int(sch.get("cooldown",   d.get("plateau_cooldown", 0)))
        d["plateau_min_lr"]         = float(sch.get("min_lr",   d.get("plateau_min_lr", 1e-7)))

    # flatten nested optimizer dict
    opt = d.pop("optimizer", None)
    if isinstance(opt, dict):
        d["optimizer_type"] = opt.get("type", d.get("optimizer_type", "adamw"))
        if "lr" in opt:       d["lr"]           = float(opt["lr"])
        if "momentum" in opt: d["sgd_momentum"] = float(opt["momentum"])
        if "nesterov" in opt: d["sgd_nesterov"] = bool(opt["nesterov"])
        if "betas" in opt and isinstance(opt["betas"], (list, tuple)) and len(opt["betas"]) == 2:
            d["adamw_betas"] = (float(opt["betas"][0]), float(opt["betas"][1]))
        if "eps" in opt:      d["adamw_eps"]    = float(opt["eps"])

    return FinetuneConfig(**d)


# ═══════════════════════════════════════════════════════════════════════════
# Optimizer with param groups
# ═══════════════════════════════════════════════════════════════════════════

def build_finetune_optimizer(
    cfg: FinetuneConfig,
    backbone_params: List[nn.Parameter],
    head_params:     List[nn.Parameter],
) -> torch.optim.Optimizer:
    """
    Two param groups with discriminative LRs:
      Group 0 ("unfrozen_backbone"): lr * backbone_lr_mult
      Group 1 ("classifier_head"):   lr * head_lr_mult
    """
    param_groups = []

    if backbone_params:
        param_groups.append({
            "params": backbone_params,
            "lr":     cfg.lr * cfg.backbone_lr_mult,
            "name":   "unfrozen_backbone",
        })
    if head_params:
        param_groups.append({
            "params": head_params,
            "lr":     cfg.lr * cfg.head_lr_mult,
            "name":   "classifier_head",
        })

    if not param_groups:
        raise RuntimeError("No trainable parameters!  Check unfreeze config.")

    opt_type = (cfg.optimizer_type or "adamw").lower()
    if opt_type in ("adamw", "adam_w"):
        b0, b1 = cfg.adamw_betas
        return torch.optim.AdamW(
            param_groups, lr=cfg.lr,
            betas=(float(b0), float(b1)),
            eps=float(cfg.adamw_eps),
            weight_decay=float(cfg.weight_decay),
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            param_groups, lr=cfg.lr,
            momentum=float(cfg.sgd_momentum),
            nesterov=bool(cfg.sgd_nesterov),
            weight_decay=float(cfg.weight_decay),
        )
    else:
        raise ValueError(f"Unknown optimizer_type={cfg.optimizer_type!r}")


# ═══════════════════════════════════════════════════════════════════════════
# Eval (identical to original)
# ═══════════════════════════════════════════════════════════════════════════

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
        input_ids  = batch["input_ids"].to(device, non_blocking=True)
        input_mask = batch["input_mask"].to(device, non_blocking=True)
        y          = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=amp_enabled):
            (loss, _metrics), logits = model(
                input_ids=input_ids, input_mask=input_mask, targets=y)
            prob = torch.softmax(logits, dim=-1)[:, 1]

        loss_sum += float(loss.item()) * int(y.numel())
        n        += int(y.numel())
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), loss_sum / max(1, n)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--pretrained_ckpt", default=None,
                    help="Override: path to pretrained best.pt checkpoint")
    ap.add_argument("--device", default='cuda',
                    help="Override: cuda device")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        raw = yaml.safe_load(f)
    cfg = dict_to_config(raw)

    if args.pretrained_ckpt is not None:
        cfg.pretrained_ckpt = args.pretrained_ckpt

    if args.device is None:
        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp_enabled = (device.type == "cuda") and bool(cfg.bf16)

    if device.type == "cuda" and bool(cfg.allow_tf32):
        torch.backends.cuda.matmul.allow_tf32  = True
        torch.backends.cudnn.allow_tf32        = True

    out_dir = Path(cfg.out_dir) / cfg.run_name
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(log_dir=str(out_dir / "tb"))

    tokenizer = TAPETokenizer(vocab="iupac")

    # ── 1. Build student model ──────────────────────────────────────────
    model = ProteinBertForSequenceClassification.from_pretrained(
        cfg.tape_model_name, num_labels=2)

    # load pretrained weights (from your full training on public data)
    if cfg.pretrained_ckpt is not None:
        ckpt_path = Path(cfg.pretrained_ckpt)
        if ckpt_path.exists():
            print(f"[finetune] Loading pretrained checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sd   = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  Missing keys: "
                      f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
            if unexpected:
                print(f"  Unexpected keys: "
                      f"{unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")
            print("  Student checkpoint loaded.")
        else:
            print(f"[warn] pretrained_ckpt not found: {ckpt_path}, "
                  f"starting from TAPE base weights")

    # ── 2. Partial unfreeze + discriminative LRs ────────────────────────
    backbone_params, head_params = setup_partial_unfreeze(
        model, cfg.unfreeze_last_n_blocks)

    n_backbone  = sum(p.numel() for p in backbone_params)
    n_head      = sum(p.numel() for p in head_params)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Unfrozen backbone params : {n_backbone:,}")
    print(f"  Classifier head params   : {n_head:,}")
    print(f"  Total trainable          : {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / max(1, n_total):.1f}%)")

    model = model.to(device)

    # ── 3. Teacher model for distillation (optional) ────────────────────
    teacher = None
    use_distill = (cfg.distill_alpha > 0.0) and (cfg.pretrained_ckpt is not None)
    if use_distill:
        ckpt_path = Path(cfg.pretrained_ckpt)
        if ckpt_path.exists():
            teacher = build_teacher(
                ProteinBertForSequenceClassification,
                cfg.tape_model_name, str(ckpt_path), device)
        else:
            print(f"[warn] Cannot build teacher: {ckpt_path} not found. "
                  f"Distillation disabled.")
            use_distill = False

    if use_distill:
        print(f"[finetune] Distillation: alpha={cfg.distill_alpha}, "
              f"temperature={cfg.distill_temperature}")
    else:
        print(f"[finetune] Distillation: OFF")

    # ── 4. Data ─────────────────────────────────────────────────────────
    df_tr  = load_and_filter(cfg.train_csv,  max_len=cfg.max_len)
    df_va  = load_and_filter(cfg.val_csv,    max_len=cfg.max_len)
    df_te  = load_and_filter(cfg.test_csv,   max_len=cfg.max_len)
    df_te2 = load_and_filter_optional(cfg.test2_csv, max_len=cfg.max_len)

    exp_train_ds = PairEncodedDataset(df_tr, tokenizer,
                                      max_len=cfg.max_len, with_sep=cfg.with_sep)
    val_ds       = PairEncodedDataset(df_va, tokenizer,
                                      max_len=cfg.max_len, with_sep=cfg.with_sep)
    test_ds      = PairEncodedDataset(df_te, tokenizer,
                                      max_len=cfg.max_len, with_sep=cfg.with_sep)
    test2_ds     = (PairEncodedDataset(df_te2, tokenizer,
                                       max_len=cfg.max_len, with_sep=cfg.with_sep)
                    if df_te2 is not None else None)

    # ── 5. Mixed replay loader (or plain experimental loader) ───────────
    replay_ds = None
    df_replay = load_and_filter_optional(cfg.replay_csv, max_len=cfg.max_len)

    if df_replay is not None and cfg.replay_ratio > 0.0:
        replay_ds = PairEncodedDataset(df_replay, tokenizer,
                                       max_len=cfg.max_len, with_sep=cfg.with_sep)
        train_loader = build_mixed_replay_loader(
            exp_ds=exp_train_ds,
            replay_ds=replay_ds,
            batch_size=cfg.batch_size,
            replay_ratio=cfg.replay_ratio,
            num_workers=cfg.num_workers,
            epoch_length_from=cfg.replay_epoch_length,
        )
        print(f"[finetune] Mixed replay: {len(exp_train_ds)} experimental + "
              f"{len(replay_ds)} replay samples, "
              f"ratio={cfg.replay_ratio:.2f}")
    else:
        train_loader = make_loader_encoded(
            exp_train_ds, cfg.batch_size, True, cfg.num_workers)
        print(f"[finetune] No replay — training on {len(exp_train_ds)} "
              f"experimental samples only")

    val_loader   = make_loader_encoded(val_ds,   cfg.batch_size, False,
                                       cfg.num_workers)
    test_loader  = make_loader_encoded(test_ds,  cfg.batch_size, False,
                                       cfg.num_workers)
    test2_loader = (make_loader_encoded(test2_ds, cfg.batch_size, False,
                                        cfg.num_workers)
                    if test2_ds is not None else None)

    # ── 6. pos_weight (computed on experimental train) ──────────────────
    pos_weight = None
    if cfg.use_pos_weight:
        ytr   = df_tr["label"].values.astype(np.float32)
        n_pos = float((ytr == 1).sum())
        n_neg = float((ytr == 0).sum())
        if n_pos > 0:
            pos_weight = torch.tensor([n_neg / max(1.0, n_pos)], device=device)

    # ── 7. Optimizer ────────────────────────────────────────────────────
    opt = build_finetune_optimizer(cfg, backbone_params, head_params)
    for i, pg in enumerate(opt.param_groups):
        name = pg.get("name", f"group_{i}")
        np_  = sum(p.numel() for p in pg["params"])
        print(f"  Param group {i} ({name}): lr={pg['lr']:.2e}, #params={np_:,}")

    # ── 8. Scheduler ────────────────────────────────────────────────────
    sched = None
    if cfg.scheduler_type in (None, "none", "NONE", "None"):
        pass
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
    elif cfg.scheduler_type == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.epochs, eta_min=float(cfg.plateau_min_lr))
    else:
        raise ValueError(f"Unknown scheduler_type: {cfg.scheduler_type}")

    # ── 9. Training loop ────────────────────────────────────────────────
    if cfg.early_stop_metric != "loss":
        best_val = -1e9
    else:
        best_val = 1e9

    bad_epochs   = 0
    global_step  = 0
    metrics_path = out_dir / "metrics.csv"

    # save config
    with open(out_dir / "finetune_config.yml", "w") as f:
        yaml.dump(asdict(cfg), f, default_flow_style=False)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        opt.zero_grad(set_to_none=True)

        running_loss_bce, running_loss_kl, n_seen = 0.0, 0.0, 0

        for step, batch in enumerate(pbar, start=1):
            input_ids  = batch["input_ids"].to(device, non_blocking=True)
            input_mask = batch["input_mask"].to(device, non_blocking=True)
            y          = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=amp_enabled):

                # ── student forward ──
                (loss_ce, _metrics), student_logits = model(
                    input_ids=input_ids, input_mask=input_mask, targets=y)

                # optional: override with pos-weighted BCE
                if pos_weight is not None:
                    logit1  = student_logits[:, 1]
                    loss_ce = F.binary_cross_entropy_with_logits(
                        logit1, y.float(), pos_weight=pos_weight,
                        reduction="mean")

                # ── distillation loss ──
                loss_kl = torch.tensor(0.0, device=device)
                if use_distill and teacher is not None:
                    with torch.no_grad():
                        (_, _), teacher_logits = teacher(
                            input_ids=input_ids, input_mask=input_mask,
                            targets=y)
                    loss_kl = distillation_loss(
                        student_logits, teacher_logits,
                        temperature=cfg.distill_temperature)

                # ── combined loss ──
                alpha = cfg.distill_alpha if use_distill else 0.0
                loss  = (1.0 - alpha) * loss_ce + alpha * loss_kl
                loss  = loss / max(1, cfg.grad_accum)

            loss.backward()

            bs = int(y.numel())
            running_loss_bce += float(loss_ce.item()) * bs
            running_loss_kl  += float(loss_kl.item()) * bs
            n_seen           += bs

            if step % cfg.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

                for gi, pg in enumerate(opt.param_groups):
                    gname = pg.get("name", f"group{gi}")
                    tb.add_scalar(f"train/lr_{gname}", pg["lr"], global_step)
                tb.add_scalar("train/loss_step",
                              float(loss.item()) * max(1, cfg.grad_accum),
                              global_step)
                if use_distill:
                    tb.add_scalar("train/loss_bce_step",
                                  float(loss_ce.item()), global_step)
                    tb.add_scalar("train/loss_kl_step",
                                  float(loss_kl.item()), global_step)

            pbar.set_postfix(
                loss=float(loss.item()) * max(1, cfg.grad_accum),
                bce=float(loss_ce.item()),
                kl=float(loss_kl.item()) if use_distill else 0.0)

        train_loss_bce = running_loss_bce / max(1, n_seen)
        train_loss_kl  = running_loss_kl  / max(1, n_seen)
        train_loss     = (1.0 - (cfg.distill_alpha if use_distill else 0.0)) \
                         * train_loss_bce \
                         + (cfg.distill_alpha if use_distill else 0.0) \
                         * train_loss_kl

        # ── validation ──────────────────────────────────────────────────
        yv, pv, val_loss = run_eval(model, val_loader, device=device,
                                    amp_enabled=amp_enabled)
        thr         = choose_threshold_on_val(yv, pv, policy=cfg.threshold_policy)
        val_metrics = compute_metrics(yv, pv, threshold=thr, ks=cfg.ks)
        val_metrics["loss"]       = float(val_loss)
        val_metrics["train_loss"] = float(train_loss)
        if use_distill:
            val_metrics["train_loss_bce"] = float(train_loss_bce)
            val_metrics["train_loss_kl"]  = float(train_loss_kl)

        # scheduler step
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitor_map = {
                    "val_loss":  float(val_loss),
                    "val_auprc": float(val_metrics.get("auprc", np.nan)),
                    "val_mcc":   float(val_metrics.get("mcc", np.nan)),
                    "val_f1":    float(val_metrics.get("f1", np.nan)),
                }
                monitor_value = monitor_map.get(cfg.plateau_monitor, np.nan)
                if monitor_value == monitor_value:  # NaN guard
                    sched.step(monitor_value)
            else:
                sched.step()

        key       = cfg.early_stop_metric
        val_score = val_metrics.get(key, np.nan)

        if key != "loss":
            improved = (val_score == val_score) and (val_score > best_val)
        else:
            improved = (val_score == val_score) and (val_score < best_val)

        # test / test2 (only when val improves, unless disabled)
        test_metrics, test2_metrics = {}, {}
        if (not cfg.test_on_val_improve) or improved:
            yt, pt, test_loss = run_eval(model, test_loader, device=device,
                                         amp_enabled=amp_enabled)
            test_metrics         = compute_metrics(yt, pt, threshold=thr,
                                                   ks=cfg.ks)
            test_metrics["loss"] = float(test_loss)

            if test2_loader is not None:
                y2, p2, t2_loss = run_eval(model, test2_loader, device=device,
                                           amp_enabled=amp_enabled)
                test2_metrics         = compute_metrics(y2, p2, threshold=thr,
                                                        ks=cfg.ks)
                test2_metrics["loss"] = float(t2_loss)

        # ── tensorboard ─────────────────────────────────────────────────
        tb.add_scalar("train/loss_epoch", train_loss, epoch)
        tb.add_scalar("val/loss", val_loss, epoch)
        if use_distill:
            tb.add_scalar("train/loss_bce_epoch", train_loss_bce, epoch)
            tb.add_scalar("train/loss_kl_epoch",  train_loss_kl,  epoch)

        for gi, pg in enumerate(opt.param_groups):
            gname = pg.get("name", f"group{gi}")
            tb.add_scalar(f"epoch/lr_{gname}", pg["lr"], epoch)

        for prefix, mdict in [("val", val_metrics), ("test", test_metrics),
                               ("test2", test2_metrics)]:
            for k2, v2 in mdict.items():
                if isinstance(v2, (int, float)) and k2 not in \
                        ("tp", "tn", "fp", "fn"):
                    tb.add_scalar(f"{prefix}/{k2}", float(v2), epoch)

        # ── metrics.csv ─────────────────────────────────────────────────
        row = {"epoch": epoch, "global_step": global_step,
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        row.update({f"test2_{k}": v for k, v in test2_metrics.items()})
        df_row       = pd.DataFrame([row])
        write_header = (not metrics_path.exists()) or \
                       (metrics_path.stat().st_size == 0)
        df_row.to_csv(metrics_path, mode="a", header=write_header, index=False)

        # ── checkpoint / early stop ─────────────────────────────────────
        if improved:
            best_val   = float(val_score)
            bad_epochs = 0
            ckpt       = out_dir / "checkpoints" / "best.pt"
            torch.save({
                "cfg":        asdict(cfg),
                "state_dict": model.state_dict(),
                "epoch":      epoch,
                "best_val":   best_val,
                "threshold":  thr,
            }, ckpt)
        else:
            bad_epochs += 1

        print(
            f"[epoch {epoch}] "
            f"train_loss={train_loss:.4f} "
            f"(bce={train_loss_bce:.4f}"
            f"{f', kl={train_loss_kl:.4f}' if use_distill else ''}) "
            f"val_{key}={val_score:.4f} "
            f"val_auprc={val_metrics.get('auprc', np.nan):.4f} "
            f"val_mcc={val_metrics.get('mcc', np.nan):.4f} "
            f"{'✓ improved' if improved else ''}")

        if bad_epochs >= cfg.patience:
            print(f"Early stopping after {cfg.patience} epochs "
                  f"without improvement.")
            break

    tb.close()
    print(f"\n[finetune] Done. Best val_{key}={best_val:.4f}")
    print(f"  Checkpoint : {out_dir / 'checkpoints' / 'best.pt'}")
    print(f"  Metrics    : {metrics_path}")


if __name__ == "__main__":
    main()
