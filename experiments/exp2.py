"""
exp2_scaling_factor.py
Section 2.2 — Ablation: The Scaling Factor 1/sqrt(dk)
Trains two models:
  1. With scaling factor    : Attention(Q,K,V) = softmax(QKᵀ / √dk) · V
  2. Without scaling factor : Attention(Q,K,V) = softmax(QKᵀ) · V
Logs train_loss, val_loss, and gradient norms of Q/K weights
for the first 1000 steps to W&B.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dataset import get_dataloaders
from model import (
    Transformer, make_src_mask, make_tgt_mask,
    PositionalEncoding, PositionwiseFeedForward,
    EncoderLayer, DecoderLayer, Encoder, Decoder,
)
from train import LabelSmoothingLoss, run_epoch
from lr_scheduler import NoamScheduler


# ── Config ───────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "batch_size":   64,
    "num_epochs":   10,
    "d_model":      256,
    "N":            3,
    "num_heads":    8,
    "d_ff":         1024,
    "dropout":      0.1,
    "warmup_steps": 4000,
}

WANDB_PROJECT = "da6401-a3"
WANDB_API_KEY = "your_api_key_here"   # ← paste your key
GRAD_LOG_STEPS = 1000                  # log grad norms for first N steps


# ── Patched attention (scaling toggle) ───────────────────────────────────────
def scaled_dot_product_attention_no_scale(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention WITHOUT the 1/sqrt(dk) scaling factor."""
    attn_logits = torch.matmul(Q, K.transpose(-2, -1))   # no division
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    return torch.matmul(attn_weights, V), attn_weights


class MultiHeadAttentionNoScale(nn.Module):
    """MHA without the sqrt(dk) scaling — for ablation."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_k        = d_model // num_heads
        self.W_q        = nn.Linear(d_model, d_model)
        self.W_k        = nn.Linear(d_model, d_model)
        self.W_v        = nn.Linear(d_model, d_model)
        self.W_o        = nn.Linear(d_model, d_model)
        self.dropout    = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

    def _combine_heads(self, x):
        B, _, L, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, self.d_model)

    def forward(self, query, key, value, mask=None):
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))
        attended, _ = scaled_dot_product_attention_no_scale(q, k, v, mask)
        attended = self.dropout(attended)
        return self.W_o(self._combine_heads(attended))


class EncoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        return self.norm2(x + self.drop2(self.ffn(x)))


class DecoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.drop1      = nn.Dropout(dropout)
        self.drop2      = nn.Dropout(dropout)
        self.drop3      = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop2(self.cross_attn(x, memory, memory, src_mask)))
        return self.norm3(x + self.drop3(self.ffn(x)))


def build_no_scale_transformer(src_vocab_size, tgt_vocab_size, config, device):
    """Build a Transformer that uses no scaling in attention."""
    import copy

    class TransformerNoScale(nn.Module):
        def __init__(self):
            super().__init__()
            d_model   = config["d_model"]
            N         = config["N"]
            num_heads = config["num_heads"]
            d_ff      = config["d_ff"]
            dropout   = config["dropout"]

            self.src_embedding      = nn.Embedding(src_vocab_size, d_model)
            self.tgt_embedding      = nn.Embedding(tgt_vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, dropout)

            enc_layer = EncoderLayerNoScale(d_model, num_heads, d_ff, dropout)
            dec_layer = DecoderLayerNoScale(d_model, num_heads, d_ff, dropout)
            self.encoder  = Encoder(enc_layer, N)
            self.decoder  = Decoder(dec_layer, N)
            self.generator = nn.Linear(d_model, tgt_vocab_size)
            self.dropout  = nn.Dropout(dropout)
            self.d_model  = d_model

            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def encode(self, src, src_mask):
            x = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
            return self.encoder(x, src_mask)

        def decode(self, memory, src_mask, tgt, tgt_mask):
            x = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
            return self.generator(self.decoder(x, memory, src_mask, tgt_mask))

        def forward(self, src, tgt, src_mask, tgt_mask):
            return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    return TransformerNoScale().to(device)


# ── Gradient norm logging ─────────────────────────────────────────────────────
def get_qk_grad_norms(model):
    """Compute mean gradient norm of all Q and K weight matrices."""
    q_norms, k_norms = [], []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if "W_q.weight" in name:
            q_norms.append(param.grad.norm().item())
        elif "W_k.weight" in name:
            k_norms.append(param.grad.norm().item())
    q_mean = sum(q_norms) / len(q_norms) if q_norms else 0.0
    k_mean = sum(k_norms) / len(k_norms) if k_norms else 0.0
    return q_mean, k_mean


# ── Training with grad norm logging ──────────────────────────────────────────
def train_with_grad_logging(
    run_name, model, optimizer, scheduler,
    loss_fn, train_loader, val_loader,
    config, device,
):
    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=config,
        reinit=True,
    )

    global_step = 0
    print(f"\n{'='*50}\n  Starting: {run_name}\n{'='*50}")

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss, total_batches = 0.0, 0
        from tqdm import tqdm

        for src, tgt in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src)
            tgt_mask   = make_tgt_mask(tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Log Q/K grad norms for first GRAD_LOG_STEPS steps
            if global_step < GRAD_LOG_STEPS:
                q_norm, k_norm = get_qk_grad_norms(model)
                wandb.log({
                    "step":       global_step,
                    "q_grad_norm": q_norm,
                    "k_grad_norm": k_norm,
                    "step_loss":  loss.item(),
                })

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss   += loss.item()
            total_batches += 1
            global_step  += 1

        train_loss = total_loss / max(total_batches, 1)

        # Validation
        model.eval()
        val_total, val_batches = 0.0, 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt   = src.to(device), tgt.to(device)
                tgt_input  = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                src_mask   = make_src_mask(src)
                tgt_mask   = make_tgt_mask(tgt_input)
                logits     = model(src, tgt_input, src_mask, tgt_mask)
                loss       = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                val_total  += loss.item()
                val_batches += 1
        val_loss = val_total / max(val_batches, 1)

        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    run.finish()
    print(f"Run '{run_name}' complete.\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading dataset...")
    train_loader, val_loader, _, assets = get_dataloaders(batch_size=BASE_CONFIG["batch_size"])
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]
    print(f"Vocab — src: {len(src_vocab)}  tgt: {len(tgt_vocab)}")

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab["<pad>"],
        smoothing=0.1,
    )

    # ── Run 1: WITH scaling ───────────────────────────────────────────────────
    model_scaled = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=BASE_CONFIG["d_model"],
        N=BASE_CONFIG["N"],
        num_heads=BASE_CONFIG["num_heads"],
        d_ff=BASE_CONFIG["d_ff"],
        dropout=BASE_CONFIG["dropout"],
        checkpoint_path=None,
    ).to(device)

    opt_scaled = torch.optim.Adam(model_scaled.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sch_scaled = NoamScheduler(opt_scaled, d_model=BASE_CONFIG["d_model"], warmup_steps=BASE_CONFIG["warmup_steps"])

    train_with_grad_logging(
        run_name="with-scaling-1-sqrt-dk",
        model=model_scaled,
        optimizer=opt_scaled,
        scheduler=sch_scaled,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config={**BASE_CONFIG, "scaling": True},
        device=device,
    )

    # ── Run 2: WITHOUT scaling ────────────────────────────────────────────────
    model_no_scale = build_no_scale_transformer(len(src_vocab), len(tgt_vocab), BASE_CONFIG, device)

    opt_no_scale = torch.optim.Adam(model_no_scale.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sch_no_scale = NoamScheduler(opt_no_scale, d_model=BASE_CONFIG["d_model"], warmup_steps=BASE_CONFIG["warmup_steps"])

    train_with_grad_logging(
        run_name="without-scaling",
        model=model_no_scale,
        optimizer=opt_no_scale,
        scheduler=sch_no_scale,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config={**BASE_CONFIG, "scaling": False},
        device=device,
    )

    print("Experiment 2 complete! Compare runs on wandb.ai")


if __name__ == "__main__":
    main()