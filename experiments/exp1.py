"""
exp1_noam_vs_fixedlr.py
Section 2.1 — The Necessity of the Noam Scheduler
Trains two models:
  1. Noam scheduler (linear warmup + inverse sqrt decay)
  2. Fixed learning rate (1e-4, no warmup)
Logs train_loss and val_loss to W&B for both runs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import wandb

from dataset import get_dataloaders
from model import Transformer
from train import run_epoch, LabelSmoothingLoss
from lr_scheduler import NoamScheduler


# ── Shared config ────────────────────────────────────────────────────────────
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


# ── Helper ───────────────────────────────────────────────────────────────────
def build_model(src_vocab, tgt_vocab, config, device):
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        checkpoint_path=None,          # always train from scratch
    ).to(device)
    model.src_vocab     = src_vocab
    model.tgt_vocab     = tgt_vocab
    model.device        = device
    return model


def train_run(run_name, use_noam, config, device,
              train_loader, val_loader, src_vocab, tgt_vocab):
    """Train one model and log to W&B."""

    
    run = wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config={**config, "scheduler": "noam" if use_noam else "fixed_lr_1e-4"},
        reinit=True,
    )

    model = build_model(src_vocab, tgt_vocab, config, device)

    if use_noam:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = NoamScheduler(
            optimizer,
            d_model=config["d_model"],
            warmup_steps=config["warmup_steps"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=1e-4,                   # fixed learning rate
            betas=(0.9, 0.98),
            eps=1e-9,
        )
        scheduler = None

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab["<pad>"],
        smoothing=0.1,
    )

    print(f"\n{'='*50}")
    print(f"  Starting run: {run_name}")
    print(f"{'='*50}")

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch, is_train=False, device=device,
        )

        # Log current LR for visualization
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}: train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.6f}")
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

    run.finish()
    print(f"\nRun '{run_name}' complete.\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data once — reused by both runs
    print("Loading dataset...")
    train_loader, val_loader, _, assets = get_dataloaders(
        batch_size=BASE_CONFIG["batch_size"]
    )
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]
    print(f"Vocab sizes — src: {len(src_vocab)}  tgt: {len(tgt_vocab)}")

    # Run 1 — Noam scheduler
    train_run(
        run_name="noam-scheduler",
        use_noam=True,
        config=BASE_CONFIG,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    # Run 2 — Fixed LR
    train_run(
        run_name="fixed-lr-1e-4",
        use_noam=False,
        config=BASE_CONFIG,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
    )

    print("Both runs complete! Go to wandb.ai to compare the curves.")


if __name__ == "__main__":
    main()