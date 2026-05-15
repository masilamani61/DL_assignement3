"""
exp5_label_smoothing.py
Section 2.5 — Decoder Sensitivity: Label Smoothing
Trains two models:
  1. Label smoothing ε = 0.1  (paper default)
  2. Label smoothing ε = 0.0  (standard cross-entropy)
Logs train_loss, val_loss, prediction confidence, and perplexity to W&B.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

from dataset import get_dataloaders
from model import Transformer, make_src_mask, make_tgt_mask
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


# ── Prediction confidence tracker ────────────────────────────────────────────
def compute_prediction_confidence(model, data_loader, tgt_vocab, device, n_batches=20):
    """
    Compute mean softmax probability assigned to the correct token.
    Returns list of per-batch confidence values.
    """
    pad_idx = tgt_vocab["<pad>"]
    model.eval()
    confidences = []

    with torch.no_grad():
        for i, (src, tgt) in enumerate(data_loader):
            if i >= n_batches:
                break
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask   = make_tgt_mask(tgt_input, pad_idx=pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            probs  = F.softmax(logits, dim=-1)

            # Gather probability of correct token
            target_probs = probs.gather(
                2, tgt_output.unsqueeze(-1)
            ).squeeze(-1)                # (B, T)

            # Mask padding
            non_pad = (tgt_output != pad_idx).float()
            mean_conf = (target_probs * non_pad).sum() / non_pad.sum().clamp_min(1)
            confidences.append(mean_conf.item())

    return confidences


def compute_perplexity(model, data_loader, loss_fn, device, n_batches=50):
    """Compute perplexity on a subset of data."""
    pad_idx = loss_fn.pad_idx
    model.eval()
    total_loss, total_batches = 0.0, 0

    with torch.no_grad():
        for i, (src, tgt) in enumerate(data_loader):
            if i >= n_batches:
                break
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask   = make_tgt_mask(tgt_input, pad_idx=pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss   = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss   += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return math.exp(avg_loss)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_loss_curves(history_smooth, history_ce):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(len(history_smooth["train"]))

    axes[0].plot(epochs, history_smooth["train"], "b-o", label="ε=0.1 (smoothed)", markersize=4)
    axes[0].plot(epochs, history_ce["train"],     "r-s", label="ε=0.0 (CE)",       markersize=4)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title("Training Loss", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history_smooth["val"], "b-o", label="ε=0.1 (smoothed)", markersize=4)
    axes[1].plot(epochs, history_ce["val"],     "r-s", label="ε=0.0 (CE)",       markersize=4)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].set_title("Validation Loss", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Label Smoothing ε=0.1 vs ε=0.0 — Loss Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_confidence_curves(conf_smooth, conf_ce):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(conf_smooth, "b-o", label="ε=0.1 (smoothed)", markersize=5, alpha=0.8)
    ax.plot(conf_ce,     "r-s", label="ε=0.0 (CE)",       markersize=5, alpha=0.8)

    # Horizontal reference lines
    ax.axhline(y=np.mean(conf_smooth), color="blue",  linestyle="--", alpha=0.5,
               label=f"Mean ε=0.1: {np.mean(conf_smooth):.3f}")
    ax.axhline(y=np.mean(conf_ce),     color="red",   linestyle="--", alpha=0.5,
               label=f"Mean ε=0.0: {np.mean(conf_ce):.3f}")

    ax.set_xlabel("Batch", fontsize=11)
    ax.set_ylabel("Mean Prediction Confidence\n(P(correct token))", fontsize=11)
    ax.set_title("Prediction Confidence — Label Smoothing vs Standard CE",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig


def plot_confidence_histogram(conf_smooth, conf_ce):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(conf_smooth, bins=15, color="#2196F3", edgecolor="black",
                 alpha=0.8, linewidth=0.7)
    axes[0].axvline(np.mean(conf_smooth), color="red", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(conf_smooth):.3f}")
    axes[0].set_xlabel("Prediction Confidence", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("ε=0.1 (Label Smoothing)", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].hist(conf_ce, bins=15, color="#F44336", edgecolor="black",
                 alpha=0.8, linewidth=0.7)
    axes[1].axvline(np.mean(conf_ce), color="blue", linestyle="--", linewidth=2,
                    label=f"Mean: {np.mean(conf_ce):.3f}")
    axes[1].set_xlabel("Prediction Confidence", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("ε=0.0 (Standard Cross-Entropy)", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Prediction Confidence Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_perplexity_bleu_comparison(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = list(results.keys())
    colors  = ["#2196F3", "#F44336"]

    # Perplexity
    perps = [results[m]["perplexity"] for m in methods]
    bars1 = axes[0].bar(methods, perps, color=colors, width=0.4,
                         edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars1, perps):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Perplexity", fontsize=12)
    axes[0].set_title("Validation Perplexity\n(lower is better)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.4)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # BLEU
    bleus = [results[m]["bleu"] for m in methods]
    bars2 = axes[1].bar(methods, bleus, color=colors, width=0.4,
                         edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars2, bleus):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("BLEU Score", fontsize=12)
    axes[1].set_title("Validation BLEU\n(higher is better)", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.4)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.suptitle("Label Smoothing ε=0.1 vs ε=0.0 — Summary",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_label_distribution():
    """Visualize how label smoothing changes the target distribution."""
    vocab_size = 20
    true_idx   = 5

    # Hard targets (CE)
    hard = np.zeros(vocab_size)
    hard[true_idx] = 1.0

    # Smoothed targets
    eps = 0.1
    smooth = np.full(vocab_size, eps / (vocab_size - 1))
    smooth[true_idx] = 1.0 - eps

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    x = np.arange(vocab_size)

    axes[0].bar(x, hard, color=["#F44336" if i == true_idx else "#90CAF9" for i in x],
                edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Vocabulary Index", fontsize=11)
    axes[0].set_ylabel("Target Probability", fontsize=11)
    axes[0].set_title("ε=0.0 — One-hot (Hard) Targets\nModel forced to be 100% confident",
                      fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, smooth, color=["#F44336" if i == true_idx else "#90CAF9" for i in x],
                edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Vocabulary Index", fontsize=11)
    axes[1].set_ylabel("Target Probability", fontsize=11)
    axes[1].set_title(f"ε=0.1 — Smoothed Targets\nCorrect token gets {1-eps:.1f}, rest share {eps:.1f}",
                      fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Effect of Label Smoothing on Target Distribution",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Training helper ───────────────────────────────────────────────────────────
def train_run(run_name, smoothing, config, device,
              train_loader, val_loader, test_loader,
              src_vocab, tgt_vocab):

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,
    ).to(device)
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.device    = device

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"]
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab["<pad>"],
        smoothing=smoothing,
    )

    history = {"train": [], "val": []}
    print(f"\n{'='*50}\n  {run_name}  (ε={smoothing})\n{'='*50}")

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch, is_train=True,  device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None,      None,
                               epoch, is_train=False, device=device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}")
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
        })

    # Final metrics
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=50)
    perp = compute_perplexity(model, val_loader, loss_fn, device)
    conf = compute_prediction_confidence(model, val_loader, tgt_vocab, device)

    print(f"  BLEU={bleu:.2f}  Perplexity={perp:.2f}  Mean Confidence={sum(conf)/len(conf):.3f}")
    wandb.log({"final_bleu": bleu, "final_perplexity": perp})

    return model, history, bleu, perp, conf


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

    # ── Log label distribution visualization first ────────────────────────────
    run_viz = wandb.init(project=WANDB_PROJECT, name="label-smoothing-visualization", reinit=True)
    fig_dist = plot_label_distribution()
    wandb.log({"label_distribution": wandb.Image(fig_dist,
               caption="How label smoothing changes target distribution")})
    plt.close(fig_dist)
    run_viz.finish()

    # ── Run 1: Label smoothing ε=0.1 ─────────────────────────────────────────
    run1 = wandb.init(project=WANDB_PROJECT, name="label-smoothing-eps-0.1",
                      config={**BASE_CONFIG, "smoothing": 0.1}, reinit=True)
    model_smooth, hist_smooth, bleu_smooth, perp_smooth, conf_smooth = train_run(
        "Label Smoothing ε=0.1", 0.1, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    )
    run1.finish()

    # ── Run 2: Standard CE ε=0.0 ─────────────────────────────────────────────
    run2 = wandb.init(project=WANDB_PROJECT, name="label-smoothing-eps-0.0",
                      config={**BASE_CONFIG, "smoothing": 0.0}, reinit=True)
    model_ce, hist_ce, bleu_ce, perp_ce, conf_ce = train_run(
        "Standard CE ε=0.0", 0.0, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    )
    run2.finish()

    # ── Summary run with all comparison plots ────────────────────────────────
    run_summary = wandb.init(project=WANDB_PROJECT, name="label-smoothing-summary", reinit=True)

    fig_loss = plot_loss_curves(hist_smooth, hist_ce)
    wandb.log({"loss_curves": wandb.Image(fig_loss)})
    plt.close(fig_loss)

    fig_conf = plot_confidence_curves(conf_smooth, conf_ce)
    wandb.log({"confidence_curves": wandb.Image(fig_conf,
               caption="Prediction confidence per batch — smoothed vs CE")})
    plt.close(fig_conf)

    fig_hist = plot_confidence_histogram(conf_smooth, conf_ce)
    wandb.log({"confidence_histogram": wandb.Image(fig_hist,
               caption="Distribution of prediction confidence")})
    plt.close(fig_hist)

    fig_summary = plot_perplexity_bleu_comparison({
        "ε=0.1 Smooth": {"perplexity": perp_smooth, "bleu": bleu_smooth},
        "ε=0.0 CE":     {"perplexity": perp_ce,     "bleu": bleu_ce},
    })
    wandb.log({"perplexity_bleu_summary": wandb.Image(fig_summary)})
    plt.close(fig_summary)

    wandb.log({
        "summary/smooth_bleu":       bleu_smooth,
        "summary/ce_bleu":           bleu_ce,
        "summary/smooth_perplexity": perp_smooth,
        "summary/ce_perplexity":     perp_ce,
        "summary/smooth_confidence": sum(conf_smooth) / len(conf_smooth),
        "summary/ce_confidence":     sum(conf_ce)     / len(conf_ce),
    })

    print(f"\n{'='*60}")
    print(f"  {'Metric':<25} {'ε=0.1':>10} {'ε=0.0':>10}")
    print(f"  {'-'*45}")
    print(f"  {'BLEU':<25} {bleu_smooth:>10.2f} {bleu_ce:>10.2f}")
    print(f"  {'Perplexity':<25} {perp_smooth:>10.2f} {perp_ce:>10.2f}")
    print(f"  {'Mean Confidence':<25} {sum(conf_smooth)/len(conf_smooth):>10.3f} {sum(conf_ce)/len(conf_ce):>10.3f}")
    print(f"{'='*60}")

    run_summary.finish()
    print("\nExp 5 complete! Check wandb.ai for all plots.")


if __name__ == "__main__":
    main()