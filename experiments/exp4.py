"""
exp4_pe_vs_learned.py
Section 2.4 — Positional Encoding vs. Learned Embeddings
Trains two models:
  1. Sinusoidal PE (fixed, from paper)
  2. Learned PE (torch.nn.Embedding)
Logs train_loss, val_loss, val_BLEU and PE visualizations to W&B.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from dataset import get_dataloaders
from model import (
    Transformer, make_src_mask, make_tgt_mask,
    PositionwiseFeedForward, EncoderLayer, DecoderLayer,
    Encoder, Decoder,
)
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu
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


# ── Learned Positional Encoding ───────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    """Learnable positional embeddings via nn.Embedding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ── Transformer with learned PE ───────────────────────────────────────────────
class TransformerLearnedPE(nn.Module):
    """Full Transformer but with learned positional embeddings."""

    def __init__(self, src_vocab_size, tgt_vocab_size, config):
        super().__init__()
        d_model   = config["d_model"]
        N         = config["N"]
        num_heads = config["num_heads"]
        d_ff      = config["d_ff"]
        dropout   = config["dropout"]

        self.src_embedding       = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding       = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = LearnedPositionalEncoding(d_model, dropout)
        self.d_model             = d_model

        enc_layer    = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer    = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder  = Encoder(enc_layer, N)
        self.decoder  = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.dropout  = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask):
        x = self.positional_encoding(
            self.src_embedding(src) * math.sqrt(self.d_model)
        )
        return self.encoder(x, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        x = self.positional_encoding(
            self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        )
        return self.generator(self.decoder(x, memory, src_mask, tgt_mask))

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)


# ── Sinusoidal PE visualization ───────────────────────────────────────────────
def plot_sinusoidal_pe(d_model=256, max_len=100):
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Heatmap
    im = axes[0].imshow(pe[:50, :64].numpy(), cmap="RdBu", aspect="auto",
                         vmin=-1, vmax=1)
    axes[0].set_xlabel("Embedding Dimension", fontsize=11)
    axes[0].set_ylabel("Position", fontsize=11)
    axes[0].set_title("Sinusoidal PE — Heatmap (first 50 pos, 64 dims)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[0])

    # Line plot for a few positions
    for pos in [0, 5, 10, 20, 50]:
        axes[1].plot(pe[pos, :64].numpy(), label=f"pos={pos}", alpha=0.8)
    axes[1].set_xlabel("Embedding Dimension", fontsize=11)
    axes[1].set_ylabel("PE Value", fontsize=11)
    axes[1].set_title("Sinusoidal PE — Values Across Dimensions", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_learned_pe(model, max_len=50):
    """Visualize learned PE weights after training."""
    weights = model.positional_encoding.embedding.weight[:max_len, :64].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im = axes[0].imshow(weights, cmap="RdBu", aspect="auto", vmin=-0.1, vmax=0.1)
    axes[0].set_xlabel("Embedding Dimension", fontsize=11)
    axes[0].set_ylabel("Position", fontsize=11)
    axes[0].set_title("Learned PE — Heatmap (first 50 pos, 64 dims)", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=axes[0])

    for pos in [0, 5, 10, 20, 50]:
        if pos < max_len:
            axes[1].plot(weights[pos, :64], label=f"pos={pos}", alpha=0.8)
    axes[1].set_xlabel("Embedding Dimension", fontsize=11)
    axes[1].set_ylabel("PE Value", fontsize=11)
    axes[1].set_title("Learned PE — Values Across Dimensions", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_bleu_comparison(results):
    """Bar chart comparing BLEU scores."""
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(results.keys())
    bleus   = list(results.values())
    colors  = ["#2196F3", "#FF9800"]

    bars = ax.bar(methods, bleus, color=colors, width=0.4, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, bleus):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("BLEU Score", fontsize=12)
    ax.set_title("Sinusoidal PE vs Learned PE — Validation BLEU", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(bleus) * 1.2)
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_loss_curves(history_sin, history_learned):
    """Overlay loss curves for both methods."""
    epochs = range(len(history_sin["train"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Train loss
    axes[0].plot(epochs, history_sin["train"],    "b-o", label="Sinusoidal PE", markersize=4)
    axes[0].plot(epochs, history_learned["train"],"r-s", label="Learned PE",    markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # Val loss
    axes[1].plot(epochs, history_sin["val"],    "b-o", label="Sinusoidal PE", markersize=4)
    axes[1].plot(epochs, history_learned["val"],"r-s", label="Learned PE",    markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Sinusoidal PE vs Learned PE — Loss Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Training helper ───────────────────────────────────────────────────────────
def train_model(run_name, model, config, device,
                train_loader, val_loader, test_loader,
                src_vocab, tgt_vocab):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"]
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab["<pad>"],
        smoothing=0.1,
    )

    history = {"train": [], "val": []}
    best_val = float("inf")

    print(f"\n{'='*50}\n  {run_name}\n{'='*50}")
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch, is_train=True,  device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None,      None,
                               epoch, is_train=False, device=device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}  lr={lr:.6f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss

    # Final BLEU
    # Attach vocab for evaluate_bleu
    model.src_vocab    = src_vocab
    model.tgt_vocab    = tgt_vocab
    model.device       = device
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=50)
    print(f"  → Final BLEU: {bleu:.2f}")
    wandb.log({"final_bleu": bleu})

    return history, bleu


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    #wandb.login(key=WANDB_API_KEY)

    print("Loading dataset...")
    train_loader, val_loader, test_loader, assets = get_dataloaders(
        batch_size=BASE_CONFIG["batch_size"]
    )
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]

    # ── Log sinusoidal PE visualization (no training needed) ──────────────────
    run_viz = wandb.init(project=WANDB_PROJECT, name="pe-visualization", reinit=True)
    fig_sin_pe = plot_sinusoidal_pe(d_model=BASE_CONFIG["d_model"])
    wandb.log({"sinusoidal_pe_visualization": wandb.Image(fig_sin_pe)})
    plt.close(fig_sin_pe)
    run_viz.finish()

    # ── Run 1: Sinusoidal PE ──────────────────────────────────────────────────
    run1 = wandb.init(
        project=WANDB_PROJECT,
        name="sinusoidal-pe",
        config={**BASE_CONFIG, "pe_type": "sinusoidal"},
        reinit=True,
    )
    model_sin = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=BASE_CONFIG["d_model"],
        N=BASE_CONFIG["N"],
        num_heads=BASE_CONFIG["num_heads"],
        d_ff=BASE_CONFIG["d_ff"],
        dropout=BASE_CONFIG["dropout"],
        checkpoint_path=None,
    ).to(device)
    model_sin.src_vocab = src_vocab
    model_sin.tgt_vocab = tgt_vocab

    history_sin, bleu_sin = train_model(
        "Sinusoidal PE", model_sin, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    )
    run1.finish()

    # ── Run 2: Learned PE ─────────────────────────────────────────────────────
    run2 = wandb.init(
        project=WANDB_PROJECT,
        name="learned-pe",
        config={**BASE_CONFIG, "pe_type": "learned"},
        reinit=True,
    )
    model_learned = TransformerLearnedPE(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        config=BASE_CONFIG,
    ).to(device)
    model_learned.src_vocab = src_vocab
    model_learned.tgt_vocab = tgt_vocab

    history_learned, bleu_learned = train_model(
        "Learned PE", model_learned, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    )

    # Visualize learned PE after training
    fig_learned_pe = plot_learned_pe(model_learned)
    wandb.log({"learned_pe_visualization": wandb.Image(fig_learned_pe)})
    plt.close(fig_learned_pe)
    run2.finish()

    # ── Summary run with comparison plots ─────────────────────────────────────
    run_summary = wandb.init(
        project=WANDB_PROJECT,
        name="pe-comparison-summary",
        reinit=True,
    )

    # Loss curves overlay
    fig_loss = plot_loss_curves(history_sin, history_learned)
    wandb.log({"loss_curves_comparison": wandb.Image(fig_loss)})
    plt.close(fig_loss)

    # BLEU comparison bar chart
    fig_bleu = plot_bleu_comparison({
        "Sinusoidal PE": bleu_sin,
        "Learned PE":    bleu_learned,
    })
    wandb.log({"bleu_comparison": wandb.Image(fig_bleu)})
    plt.close(fig_bleu)

    # Summary table
    wandb.log({
        "summary/sinusoidal_bleu": bleu_sin,
        "summary/learned_bleu":    bleu_learned,
        "summary/bleu_difference": bleu_sin - bleu_learned,
    })

    print(f"\n{'='*50}")
    print(f"  Sinusoidal PE BLEU : {bleu_sin:.2f}")
    print(f"  Learned PE BLEU    : {bleu_learned:.2f}")
    print(f"  Difference         : {bleu_sin - bleu_learned:+.2f}")
    print(f"{'='*50}")

    run_summary.finish()
    print("\nExp 4 complete! Check wandb.ai for all plots.")


if __name__ == "__main__":
    main()