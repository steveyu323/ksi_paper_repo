#!/usr/bin/env python3
"""
Inference script for cached kinase-substrate interaction models.

This script loads a trained model checkpoint and adds predictions (logits and probabilities)
to an input CSV file containing kinase-substrate pairs.

Usage:
  python infer_cached_interaction.py --eval_config configs/infer_config.yml
  python infer_cached_interaction.py --eval_config configs/infer_config.yml --device cuda:0
"""

from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


# ----------------------------
# Model (copied from train_pair_interaction_cached.py)
# ----------------------------
def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean pooling over sequence dimension."""
    mask_f = mask.float().unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return (x * mask_f).sum(dim=1) / denom


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, q_mask: torch.Tensor, k_mask: torch.Tensor):
        # q_mask unused by torch MHA; kept for parity
        key_pad = ~k_mask
        out, _ = self.attn(q, k, k, key_padding_mask=key_pad, need_weights=False)
        return self.ln(q + self.drop(out))


class PairHead(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, pk: torch.Tensor, ps: torch.Tensor):
        z = torch.cat([pk, ps, (pk - ps).abs(), pk * ps], dim=-1)
        return self.mlp(z).squeeze(-1)


class CachedInteractionModel(nn.Module):
    """
    If use_cross_attn == False:
        uses pooled buffers: kin_emb (Nk,D), sub_emb (Ns,D)
    If use_cross_attn == True:
        uses token buffers: kin_tok (Nk,Lk,D), kin_mask (Nk,Lk), sub_tok (Ns,Ls,D), sub_mask (Ns,Ls)
    """
    def __init__(
        self,
        *,
        use_cross_attn: bool,
        dim: int,
        dropout: float,
        xattn_layers: int,
        xattn_heads: int,
        kin_emb: Optional[torch.Tensor] = None,
        sub_emb: Optional[torch.Tensor] = None,
        kin_tok: Optional[torch.Tensor] = None,
        kin_mask: Optional[torch.Tensor] = None,
        sub_tok: Optional[torch.Tensor] = None,
        sub_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.use_cross_attn = bool(use_cross_attn)

        if not self.use_cross_attn:
            assert kin_emb is not None and sub_emb is not None
            self.register_buffer("kin_emb", kin_emb)  # (Nk,D)
            self.register_buffer("sub_emb", sub_emb)  # (Ns,D)
            self.head = PairHead(dim, dropout)
            self.xattn = None
        else:
            assert kin_tok is not None and kin_mask is not None and sub_tok is not None and sub_mask is not None
            self.register_buffer("kin_tok", kin_tok)    # (Nk,Lk,D)
            self.register_buffer("kin_mask", kin_mask)  # (Nk,Lk) bool
            self.register_buffer("sub_tok", sub_tok)    # (Ns,Ls,D)
            self.register_buffer("sub_mask", sub_mask)  # (Ns,Ls) bool

            self.xattn = nn.ModuleList([
                CrossAttentionBlock(dim, xattn_heads, dropout)
                for _ in range(int(xattn_layers))
            ])
            self.head = PairHead(dim, dropout)

    def forward(self, kin_idx: torch.Tensor, sub_idx: torch.Tensor) -> torch.Tensor:
        if not self.use_cross_attn:
            pk = self.kin_emb.index_select(0, kin_idx)
            ps = self.sub_emb.index_select(0, sub_idx)
            return self.head(pk, ps)
        else:
            kin_seq = self.kin_tok.index_select(0, kin_idx)
            kin_m = self.kin_mask.index_select(0, kin_idx)
            sub_seq = self.sub_tok.index_select(0, sub_idx)
            sub_m = self.sub_mask.index_select(0, sub_idx)

            for blk in self.xattn:
                kin_seq = blk(kin_seq, sub_seq, kin_m, sub_m)

            pk = masked_mean_pool(kin_seq, kin_m)
            ps = masked_mean_pool(sub_seq, sub_m)
            return self.head(pk, ps)


# ----------------------------
# Dataset for inference
# ----------------------------
class InferencePairDataset(Dataset):
    def __init__(self, kin_idx: torch.Tensor, sub_idx: torch.Tensor):
        self.kin_idx = kin_idx
        self.sub_idx = sub_idx

    def __len__(self):
        return len(self.kin_idx)

    def __getitem__(self, i):
        return self.kin_idx[i], self.sub_idx[i]


# ----------------------------
# Main inference function
# ----------------------------
def run_inference_from_yaml(eval_config_path: str, device: Optional[str] = None) -> list:
    """
    Load inference config, checkpoint, and data, then add predictions to input CSV.
    
    Expected eval_config YAML format:
    ```yaml
    checkpoint_path: "runs/my_run/checkpoints/best_cached_xattn.pt"
    cache_root: "cache_v1"
    run_name: "my_run"
    input_csv: "data/pairs_to_predict.csv"
    output_csv: "predictions/pairs_with_predictions.csv"
    
    # Optional
    batch_size: 128
    num_workers: 4
    ```
    
    Input CSV must contain at minimum:
    - kin_idx: kinase index (integer)
    - sub_idx: substrate index (integer)
    
    Output CSV will be input CSV + two new columns:
    - logit: raw model output
    - prob: sigmoid(logit), probability between 0 and 1
    
    Returns:
        List of output file paths that were written
    """
    eval_config_path = Path(eval_config_path)
    with open(eval_config_path, "r") as f:
        eval_cfg = yaml.safe_load(f)
    
    # Required fields
    checkpoint_path = Path(eval_cfg["checkpoint_path"])
    cache_root = Path(eval_cfg["cache_root"])
    run_name = eval_cfg["run_name"]
    input_csv = Path(eval_cfg["input_csv"])
    output_csv = Path(eval_cfg["output_csv"])
    
    # Optional fields
    batch_size = eval_cfg.get("batch_size", 128)
    num_workers = eval_cfg.get("num_workers", 4)
    
    # Device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract config from checkpoint
    train_cfg = ckpt.get("cfg", {})
    use_cross_attn = train_cfg.get("use_cross_attn", False)
    dropout = train_cfg.get("dropout", 0.1)
    xattn_layers = train_cfg.get("xattn_layers", 1)
    xattn_heads = train_cfg.get("xattn_heads", 8)
    
    print(f"Model mode: {'cross-attention' if use_cross_attn else 'no cross-attention'}")
    
    # Load cached embeddings/tokens
    cache_dir = cache_root / run_name
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")
    
    amp_enabled = train_cfg.get("bf16", True) and device.type == "cuda"
    
    if use_cross_attn:
        print("Loading cached token embeddings...")
        kin_pack = torch.load(cache_dir / "kinase_tok.pt", map_location="cpu")
        sub_pack = torch.load(cache_dir / "substrate_tok.pt", map_location="cpu")
        
        kin_tok = kin_pack["tok"].to(torch.float32)
        kin_mask = kin_pack["mask"].to(torch.bool)
        sub_tok = sub_pack["tok"].to(torch.float32)
        sub_mask = sub_pack["mask"].to(torch.bool)
        dim = int(kin_tok.shape[-1])
        
        model = CachedInteractionModel(
            use_cross_attn=True,
            dim=dim,
            dropout=dropout,
            xattn_layers=xattn_layers,
            xattn_heads=xattn_heads,
            kin_tok=kin_tok,
            kin_mask=kin_mask,
            sub_tok=sub_tok,
            sub_mask=sub_mask,
        ).to(device)
    else:
        print("Loading cached pooled embeddings...")
        kin_pack = torch.load(cache_dir / "kinase_emb.pt", map_location="cpu")
        sub_pack = torch.load(cache_dir / "substrate_emb.pt", map_location="cpu")
        
        kin_emb = kin_pack["emb"].to(torch.float32)
        sub_emb = sub_pack["emb"].to(torch.float32)
        dim = int(kin_emb.shape[-1])
        
        model = CachedInteractionModel(
            use_cross_attn=False,
            dim=dim,
            dropout=dropout,
            xattn_layers=0,
            xattn_heads=0,
            kin_emb=kin_emb,
            sub_emb=sub_emb,
        ).to(device)
    
    # Load model weights
    # model.load_state_dict(ckpt["state_dict"])
    state = ckpt["state_dict"]

    # Drop cached buffers that are universe-specific
    DROP = {
        "kin_emb", "sub_emb",
        "kin_tok", "kin_mask", "sub_tok", "sub_mask",
    }
    state = {k: v for k, v in state.items() if k not in DROP}
    
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load] missing (expected):", [k for k in missing if k in DROP])
    print("[load] unexpected:", unexpected)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    
    # Load input CSV
    print(f"Loading input CSV from {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check if we have indices or sequences
    has_indices = "kin_idx" in df.columns and "sub_idx" in df.columns
    has_sequences = "kinase_seq" in df.columns and "substrate_seq" in df.columns
    
    if not has_indices and not has_sequences:
        raise ValueError(
            "Input CSV must contain either:\n"
            "  (1) 'kin_idx' and 'sub_idx' columns, OR\n"
            "  (2) 'kinase_seq' and 'substrate_seq' columns"
        )
    
    # If sequences are provided, map them to indices using cache metadata
    if has_sequences and not has_indices:
        print("Input contains sequences - attempting to map to cached indices...")
        
        # Try to load sequence-to-index mapping from cache
        kin_seqs = kin_pack.get("seqs", None) if 'kin_pack' in locals() else None
        sub_seqs = sub_pack.get("seqs", None) if 'sub_pack' in locals() else None
        
        if kin_seqs is None or sub_seqs is None:
            raise ValueError(
                "Cache files do not contain sequence metadata ('seqs' field).\n"
                "Please either:\n"
                "  (1) Provide 'kin_idx' and 'sub_idx' columns in your CSV instead of sequences, OR\n"
                "  (2) Use a cache that was created with sequence metadata saved\n"
                "\n"
                "Note: Your cache files contain these keys:\n"
                f"  Kinase cache: {list(kin_pack.keys())}\n"
                f"  Substrate cache: {list(sub_pack.keys())}"
            )
        
        print(f"Found {len(kin_seqs)} kinase sequences and {len(sub_seqs)} substrate sequences in cache")
        
        # Create sequence to index mappings
        kin_seq_to_idx = {seq: idx for idx, seq in enumerate(kin_seqs)}
        sub_seq_to_idx = {seq: idx for idx, seq in enumerate(sub_seqs)}
        
        # Map sequences to indices
        kin_idx_list = []
        sub_idx_list = []
        valid_rows = []
        skipped_count = 0
        
        for i, row in df.iterrows():
            k_seq = row["kinase_seq"]
            s_seq = row["substrate_seq"]
            
            if k_seq in kin_seq_to_idx and s_seq in sub_seq_to_idx:
                kin_idx_list.append(kin_seq_to_idx[k_seq])
                sub_idx_list.append(sub_seq_to_idx[s_seq])
                valid_rows.append(i)
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Only print first 5 warnings
                    missing = []
                    if k_seq not in kin_seq_to_idx:
                        missing.append(f"kinase '{k_seq[:30]}...'")
                    if s_seq not in sub_seq_to_idx:
                        missing.append(f"substrate '{s_seq[:30]}...'")
                    warnings.warn(f"Row {i}: Skipping - {', '.join(missing)} not in cache")
        
        if skipped_count > 5:
            print(f"  ... and {skipped_count - 5} more pairs skipped (not shown)")
        
        if not valid_rows:
            raise ValueError(
                f"No valid pairs found - all {len(df)} sequences are missing from cache.\n"
                "This usually means your input sequences don't match the training data."
            )
        
        # Filter dataframe to valid rows only
        df = df.iloc[valid_rows].reset_index(drop=True)
        kin_idx = torch.tensor(kin_idx_list, dtype=torch.long)
        sub_idx = torch.tensor(sub_idx_list, dtype=torch.long)
        
        if skipped_count > 0:
            print(f"Mapped {len(valid_rows)}/{len(df) + skipped_count} pairs (skipped {skipped_count} with unknown sequences)")
        else:
            print(f"Mapped all {len(valid_rows)} pairs successfully")
    else:
        # Use provided indices
        kin_idx = torch.from_numpy(df["kin_idx"].values).long()
        sub_idx = torch.from_numpy(df["sub_idx"].values).long()
        print(f"Using provided indices for {len(df)} pairs")
    
    print(f"Predicting {len(df)} pairs")
    
    dataset = InferencePairDataset(kin_idx, sub_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Run inference
    print("Running inference...")
    all_logits = []
    
    with torch.no_grad():
        for ki, si in tqdm(loader, desc="Predicting"):
            ki = ki.to(device)
            si = si.to(device)
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(ki, si)
            
            all_logits.append(logits.cpu().float().numpy())
    
    # Concatenate predictions
    logits = np.concatenate(all_logits, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    
    # Add predictions to dataframe
    df["logit"] = logits
    df["prob"] = probs
    
    # Save output
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    print(f"\nSaved predictions to {output_csv}")
    print(f"  Added columns: 'logit', 'prob'")
    print(f"  Logit range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    
    return [output_csv]


def main():
    ap = argparse.ArgumentParser(
        description="Run inference with trained kinase-substrate interaction model"
    )
    ap.add_argument(
        "--eval_config",
        required=True,
        help="Path to inference config YAML file",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="Device to use (e.g., 'cuda:0', 'cpu'). Defaults to cuda if available.",
    )
    args = ap.parse_args()
    
    written = run_inference_from_yaml(args.eval_config, device=args.device)
    for p in written:
        print(f"[ok] wrote {p}")


if __name__ == "__main__":
    main()
