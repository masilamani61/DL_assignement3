"""
exp5_label_smoothing.py
Section 2.5 — Decoder Sensitivity: Label Smoothing
Trains two models:
  1. Label smoothing ε = 0.1  (paper default)
  2. Label smoothing ε = 0.0  (standard cross-entropy)
All plots logged as interactive Plotly — no static images.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import wandb

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


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_prediction_confidence(model, data_loader, tgt_vocab, device, n_batches=20):
    pad_idx = tgt_vocab["<pad>"]
    model.eval()
    confidences = []
    with torch.no_grad():
        for i, (src, tgt) in enumerate(data_loader):
            if i >= n_batches:
                break
            src, tgt   = src.to(device), tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask   = make_tgt_mask(tgt_input, pad_idx=pad_idx)
            logits     = model(src, tgt_input, src_mask, tgt_mask)
            probs      = F.softmax(logits, dim=-1)
            target_probs = probs.gather(2, tgt_output.unsqueeze(-1)).squeeze(-1)
            non_pad    = (tgt_output != pad_idx).float()
            mean_conf  = (target_probs * non_pad).sum() / non_pad.sum().clamp_min(1)
            confidences.append(mean_conf.item())
    return confidences


def compute_perplexity(model, data_loader, loss_fn, device, n_batches=50):
    pad_idx = loss_fn.pad_idx
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for i, (src, tgt) in enumerate(data_loader):
            if i >= n_batches:
                break
            src, tgt   = src.to(device), tgt.to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            src_mask   = make_src_mask(src, pad_idx=pad_idx)
            tgt_mask   = make_tgt_mask(tgt_input, pad_idx=pad_idx)
            logits     = model(src, tgt_input, src_mask, tgt_mask)
            loss       = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss   += loss.item()
            total_batches += 1
    return math.exp(total_loss / max(total_batches, 1))


# ── Plotly figures ────────────────────────────────────────────────────────────
def plot_label_distribution_plotly():
    """Interactive bar chart showing how label smoothing changes target distribution."""
    vocab_size = 20
    true_idx   = 5
    eps        = 0.1

    hard   = [0.0] * vocab_size
    hard[true_idx] = 1.0

    smooth = [eps / (vocab_size - 1)] * vocab_size
    smooth[true_idx] = 1.0 - eps

    x      = list(range(vocab_size))
    labels = [f"tok_{i}" if i != true_idx else f"tok_{i} (correct)" for i in x]
    colors_hard   = ["#F44336" if i == true_idx else "#90CAF9" for i in x]
    colors_smooth = ["#F44336" if i == true_idx else "#90CAF9" for i in x]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "ε=0.0 — One-hot (Hard) Targets",
            "ε=0.1 — Smoothed Targets",
        ],
    )
    fig.add_trace(go.Bar(
        x=labels, y=hard,
        marker_color=colors_hard,
        name="Hard (CE)",
        hovertemplate="Token: %{x}<br>Probability: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=labels, y=smooth,
        marker_color=colors_smooth,
        name="Smoothed",
        hovertemplate="Token: %{x}<br>Probability: %{y:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title="Effect of Label Smoothing on Target Distribution<br>"
              "<sup>Red = correct token, Blue = other vocab tokens</sup>",
        height=450, width=1000,
        showlegend=False,
    )
    fig.update_yaxes(title_text="Target Probability", range=[0, 1.1])
    return fig


def plot_loss_curves_plotly(hist_smooth, hist_ce):
    epochs = list(range(len(hist_smooth["train"])))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Training Loss", "Validation Loss"],
    )

    # Train loss
    fig.add_trace(go.Scatter(
        x=epochs, y=hist_smooth["train"],
        mode="lines+markers", name="ε=0.1 train",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=6),
        hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra>ε=0.1</extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=epochs, y=hist_ce["train"],
        mode="lines+markers", name="ε=0.0 train",
        line=dict(color="#F44336", width=2),
        marker=dict(size=6, symbol="square"),
        hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra>ε=0.0</extra>",
    ), row=1, col=1)

    # Val loss
    fig.add_trace(go.Scatter(
        x=epochs, y=hist_smooth["val"],
        mode="lines+markers", name="ε=0.1 val",
        line=dict(color="#2196F3", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra>ε=0.1</extra>",
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=epochs, y=hist_ce["val"],
        mode="lines+markers", name="ε=0.0 val",
        line=dict(color="#F44336", width=2, dash="dash"),
        marker=dict(size=6, symbol="square"),
        hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra>ε=0.0</extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="Loss")
    fig.update_layout(
        title="Label Smoothing ε=0.1 vs ε=0.0 — Loss Curves",
        height=450, width=1000,
        legend=dict(x=0.75, y=0.95),
    )
    return fig


def plot_confidence_plotly(conf_smooth, conf_ce):
    batches = list(range(len(conf_smooth)))
    mean_s  = float(np.mean(conf_smooth))
    mean_c  = float(np.mean(conf_ce))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=batches, y=conf_smooth,
        mode="lines+markers", name=f"ε=0.1 (mean={mean_s:.3f})",
        line=dict(color="#2196F3", width=2),
        marker=dict(size=5),
        hovertemplate="Batch %{x}<br>Confidence: %{y:.4f}<extra>ε=0.1</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=batches, y=conf_ce,
        mode="lines+markers", name=f"ε=0.0 (mean={mean_c:.3f})",
        line=dict(color="#F44336", width=2),
        marker=dict(size=5, symbol="square"),
        hovertemplate="Batch %{x}<br>Confidence: %{y:.4f}<extra>ε=0.0</extra>",
    ))

    # Mean reference lines
    fig.add_hline(y=mean_s, line_dash="dash", line_color="#2196F3", opacity=0.5,
                  annotation_text=f"Mean ε=0.1: {mean_s:.3f}", annotation_position="right")
    fig.add_hline(y=mean_c, line_dash="dash", line_color="#F44336", opacity=0.5,
                  annotation_text=f"Mean ε=0.0: {mean_c:.3f}", annotation_position="right")

    fig.update_layout(
        title="Prediction Confidence — P(correct token)<br>"
              "<sup>Lower confidence with label smoothing = better calibration</sup>",
        xaxis_title="Batch",
        yaxis_title="Mean Prediction Confidence",
        yaxis_range=[0, 1],
        height=500, width=900,
        legend=dict(x=0.7, y=0.1),
    )
    return fig


def plot_confidence_histogram_plotly(conf_smooth, conf_ce):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"ε=0.1 — Mean: {np.mean(conf_smooth):.3f}",
            f"ε=0.0 — Mean: {np.mean(conf_ce):.3f}",
        ],
    )

    fig.add_trace(go.Histogram(
        x=conf_smooth, nbinsx=15,
        marker_color="#2196F3", opacity=0.8,
        name="ε=0.1",
        hovertemplate="Confidence: %{x:.3f}<br>Count: %{y}<extra>ε=0.1</extra>",
    ), row=1, col=1)
    fig.add_vline(x=float(np.mean(conf_smooth)), line_dash="dash",
                  line_color="red", row=1, col=1)

    fig.add_trace(go.Histogram(
        x=conf_ce, nbinsx=15,
        marker_color="#F44336", opacity=0.8,
        name="ε=0.0",
        hovertemplate="Confidence: %{x:.3f}<br>Count: %{y}<extra>ε=0.0</extra>",
    ), row=1, col=2)
    fig.add_vline(x=float(np.mean(conf_ce)), line_dash="dash",
                  line_color="blue", row=1, col=2)

    fig.update_layout(
        title="Prediction Confidence Distribution<br>"
              "<sup>CE model peaks at higher confidence = overconfident</sup>",
        height=450, width=900,
        showlegend=False,
    )
    fig.update_xaxes(title_text="Prediction Confidence", range=[0, 1])
    fig.update_yaxes(title_text="Frequency")
    return fig


def plot_summary_plotly(bleu_smooth, bleu_ce, perp_smooth, perp_ce,
                        conf_smooth, conf_ce):
    methods      = ["ε=0.1 (Smoothed)", "ε=0.0 (CE)"]
    colors       = ["#2196F3", "#F44336"]
    bleus        = [bleu_smooth, bleu_ce]
    perps        = [perp_smooth, perp_ce]
    confs        = [float(np.mean(conf_smooth)), float(np.mean(conf_ce))]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["BLEU Score ↑", "Perplexity ↓", "Mean Confidence ↓"],
    )

    for col, (vals, fmt) in enumerate([(bleus, ".2f"), (perps, ".2f"), (confs, ".3f")], 1):
        fig.add_trace(go.Bar(
            x=methods, y=vals,
            marker_color=colors,
            text=[f"{v:{fmt}}" for v in vals],
            textposition="outside",
            showlegend=False,
            hovertemplate="%{x}<br>Value: %{y}<extra></extra>",
        ), row=1, col=col)

    fig.update_layout(
        title="Label Smoothing Summary — BLEU, Perplexity, Confidence",
        height=450, width=1000,
    )
    return fig


# ── Summary W&B table ─────────────────────────────────────────────────────────
def log_summary_table(bleu_s, bleu_c, perp_s, perp_c, conf_s, conf_c):
    table = wandb.Table(columns=["Metric", "ε=0.1 (Smoothed)", "ε=0.0 (CE)", "Winner"])
    rows = [
        ("BLEU Score ↑",        f"{bleu_s:.2f}", f"{bleu_c:.2f}",
         "ε=0.1" if bleu_s > bleu_c else "ε=0.0"),
        ("Perplexity ↓",        f"{perp_s:.2f}", f"{perp_c:.2f}",
         "ε=0.1" if perp_s < perp_c else "ε=0.0"),
        ("Mean Confidence ↓",   f"{np.mean(conf_s):.3f}", f"{np.mean(conf_c):.3f}",
         "ε=0.1 (better calibrated)" if np.mean(conf_s) < np.mean(conf_c) else "ε=0.0"),
    ]
    for row in rows:
        table.add_data(*row)
    wandb.log({"summary_table": table})


# ── Training helper ───────────────────────────────────────────────────────────
def train_run(run_name, smoothing, config, device,
              train_loader, val_loader, test_loader, src_vocab, tgt_vocab):

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
        val_loss   = run_epoch(val_loader,   model, loss_fn, None, None,
                               epoch, is_train=False, device=device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=50)
    perp = compute_perplexity(model, val_loader, loss_fn, device)
    conf = compute_prediction_confidence(model, val_loader, tgt_vocab, device)
    print(f"  BLEU={bleu:.2f}  PPL={perp:.2f}  Conf={np.mean(conf):.3f}")
    wandb.log({"final_bleu": bleu, "final_perplexity": perp,
               "mean_confidence": float(np.mean(conf))})
    return model, history, bleu, perp, conf


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")


    print("Loading dataset...")
    train_loader, val_loader, test_loader, assets = get_dataloaders(
        batch_size=BASE_CONFIG["batch_size"]
    )
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]

    # ── Label distribution visualization (no training) ────────────────────────
    run_viz = wandb.init(project=WANDB_PROJECT, name="label-smoothing-visualization", reinit=True)
    wandb.log({"label_distribution": wandb.Plotly(plot_label_distribution_plotly())})
    run_viz.finish()

    # ── Run 1: ε=0.1 ─────────────────────────────────────────────────────────
    run1 = wandb.init(project=WANDB_PROJECT, name="label-smoothing-eps-0.1",
                      config={**BASE_CONFIG, "smoothing": 0.1}, reinit=True)
    _, hist_smooth, bleu_smooth, perp_smooth, conf_smooth = train_run(
        "Label Smoothing ε=0.1", 0.1, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab,
    )
    run1.finish()

    # ── Run 2: ε=0.0 ─────────────────────────────────────────────────────────
    run2 = wandb.init(project=WANDB_PROJECT, name="label-smoothing-eps-0.0",
                      config={**BASE_CONFIG, "smoothing": 0.0}, reinit=True)
    _, hist_ce, bleu_ce, perp_ce, conf_ce = train_run(
        "Standard CE ε=0.0", 0.0, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab,
    )
    run2.finish()

    # ── Summary run — all comparison plots ───────────────────────────────────
    run_summary = wandb.init(project=WANDB_PROJECT, name="label-smoothing-summary", reinit=True)

    wandb.log({"loss_curves":             wandb.Plotly(plot_loss_curves_plotly(hist_smooth, hist_ce))})
    wandb.log({"confidence_over_batches": wandb.Plotly(plot_confidence_plotly(conf_smooth, conf_ce))})
    wandb.log({"confidence_histogram":    wandb.Plotly(plot_confidence_histogram_plotly(conf_smooth, conf_ce))})
    wandb.log({"summary_chart":           wandb.Plotly(plot_summary_plotly(
        bleu_smooth, bleu_ce, perp_smooth, perp_ce, conf_smooth, conf_ce
    ))})

    log_summary_table(bleu_smooth, bleu_ce, perp_smooth, perp_ce, conf_smooth, conf_ce)

    # Scalar summary
    wandb.log({
        "summary/smooth_bleu":       bleu_smooth,
        "summary/ce_bleu":           bleu_ce,
        "summary/smooth_perplexity": perp_smooth,
        "summary/ce_perplexity":     perp_ce,
        "summary/smooth_confidence": float(np.mean(conf_smooth)),
        "summary/ce_confidence":     float(np.mean(conf_ce)),
    })

    print(f"\n{'='*60}")
    print(f"  {'Metric':<25} {'ε=0.1':>12} {'ε=0.0':>12}")
    print(f"  {'-'*50}")
    print(f"  {'BLEU':<25} {bleu_smooth:>12.2f} {bleu_ce:>12.2f}")
    print(f"  {'Perplexity':<25} {perp_smooth:>12.2f} {perp_ce:>12.2f}")
    print(f"  {'Mean Confidence':<25} {np.mean(conf_smooth):>12.3f} {np.mean(conf_ce):>12.3f}")
    print(f"{'='*60}")

    run_summary.finish()
    print("\nExp 5 complete! All plots logged as interactive Plotly.")


if __name__ == "__main__":
    main()