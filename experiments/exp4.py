"""
exp4_pe_vs_learned.py  —  Section 2.4: Sinusoidal PE vs Learned PE

EXPERIMENT
  Trains Transformer with sinusoidal PE (fixed) vs learned PE (nn.Embedding).
  Logs val BLEU once at the end (not every epoch — too slow) for comparison.

THEORETICAL CHALLENGE (extrapolation)
  After training, evaluates BLEU at 1x, 1.5x, 2x training length for both
  models. Sinusoidal PE degrades gracefully; learned PE fails beyond seen positions.

ALL W&B logging is native — zero matplotlib, zero saved images:
  train_loss / val_loss  per epoch  →  live scalars → line charts
  PE heatmap (50 pos x 64 dims)    →  wandb.Table + scatter
  PE line across dims               →  wandb.plot.line
  Final BLEU comparison             →  wandb.plot.bar
  Loss curves overlay               →  wandb.plot.line
  Extrapolation BLEU table          →  wandb.Table
  Summary table                     →  wandb.Table
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
import numpy as np
import wandb

from dataset import get_dataloaders
from model import (
    Transformer,
    PositionwiseFeedForward, EncoderLayer, DecoderLayer,
    Encoder, Decoder,
)
from train import LabelSmoothingLoss, run_epoch, evaluate_bleu
from lr_scheduler import NoamScheduler


# ── Config ────────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    "batch_size":   64,
    "num_epochs":   10,
    "d_model":      256,
    "N":            3,
    "num_heads":    8,
    "d_ff":         1024,
    "dropout":      0.1,
    "warmup_steps": 4000,
    "max_train_len": 50,   # used for extrapolation test
}

WANDB_PROJECT = "da6401-a3"
# WANDB_API_KEY = "your_key"  # uncomment if not using .netrc


# ── Learned Positional Encoding ───────────────────────────────────────────────
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout   = nn.Dropout(p=dropout)
        self.embedding = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.dropout(x + self.embedding(positions))


# ── Transformer with learned PE ───────────────────────────────────────────────
class TransformerLearnedPE(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, config):
        super().__init__()
        d_model, N, num_heads = config["d_model"], config["N"], config["num_heads"]
        d_ff, dropout         = config["d_ff"],    config["dropout"]

        self.src_embedding       = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding       = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = LearnedPositionalEncoding(d_model, dropout)
        self.d_model             = d_model

        enc_layer      = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer      = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder   = Encoder(enc_layer, N)
        self.decoder   = Decoder(dec_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.dropout   = nn.Dropout(dropout)

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


# ── W&B PE visualization helpers (all native, no matplotlib) ─────────────────

def _build_pe_matrix(d_model: int, max_len: int = 100) -> np.ndarray:
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.numpy()


def _log_pe_charts(prefix: str, matrix: np.ndarray,
                   n_pos: int = 50, n_dim: int = 64):
    """Log heatmap scatter + line chart for any PE matrix."""
    mat = matrix[:n_pos, :n_dim]

    # 1. Heatmap as flat table + scatter (colour by pe_value in W&B UI)
    hm_table = wandb.Table(columns=["position", "dim", "pe_value"])
    for p in range(n_pos):
        for d in range(n_dim):
            hm_table.add_data(int(p), int(d), float(mat[p, d]))

    wandb.log({
        f"{prefix}/heatmap_table": hm_table,
        f"{prefix}/heatmap_scatter": wandb.plot.scatter(
            hm_table, x="dim", y="position",
            title=f"{prefix} — heatmap (colour=pe_value, first {n_pos} pos x {n_dim} dims)",
        ),
    })

    # 2. Line chart: pe_value across dims for 5 positions
    selected   = [0, 5, 10, 20, min(n_pos - 1, 49)]
    line_table = wandb.Table(columns=["dim", "pe_value", "position"])
    for pos in selected:
        for d in range(n_dim):
            line_table.add_data(int(d), float(mat[pos, d]), f"pos={pos}")

    wandb.log({
        f"{prefix}/line_across_dims": wandb.plot.line(
            line_table, x="dim", y="pe_value", stroke="position",
            title=f"{prefix} — PE values across embedding dimensions",
        ),
    })


def log_sinusoidal_pe_charts(d_model: int):
    pe_matrix = _build_pe_matrix(d_model, max_len=100)
    _log_pe_charts("sinusoidal_pe", pe_matrix)

    # Cosine similarity between adjacent positions — shows smooth structure
    # that enables extrapolation (report evidence for theoretical challenge)
    sim_table = wandb.Table(columns=["position", "cosine_sim_to_next"])
    for i in range(99):
        v1  = pe_matrix[i]
        v2  = pe_matrix[i + 1]
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        sim_table.add_data(i, cos)

    wandb.log({
        "sinusoidal_pe/adjacent_position_similarity": wandb.plot.line(
            sim_table, x="position", y="cosine_sim_to_next",
            title="Sinusoidal PE — cosine similarity between adjacent positions "
                  "(smooth beyond training length = extrapolation)",
        )
    })


def log_learned_pe_charts(model: TransformerLearnedPE, max_len: int = 100):
    weights = (
        model.positional_encoding.embedding.weight[:max_len, :]
        .detach().cpu().numpy()
    )
    _log_pe_charts("learned_pe", weights)


# ── Extrapolation test ────────────────────────────────────────────────────────

def log_extrapolation_table(model_sin, model_learned,
                             test_loader, tgt_vocab, device,
                             train_max_len: int):
    """
    Evaluate both models at 1x, 1.5x, 2x training sequence length.
    Sinusoidal: sin/cos defined for all positions -> graceful degradation.
    Learned:    embedding rows beyond train_max_len were never trained -> fails.
    """
    multipliers  = [1.0, 1.5, 2.0]
    extrap_table = wandb.Table(
        columns=["max_len", "length_multiplier",
                 "sinusoidal_bleu", "learned_bleu", "bleu_drop_learned"]
    )

    print("\n── Extrapolation test ───────────────────────────────────────────")
    for mult in multipliers:
        eval_len = int(train_max_len * mult)
        print(f"  max_len={eval_len} ({mult}x training length)...")

        bleu_sin = evaluate_bleu(model_sin,     test_loader, tgt_vocab,
                                 device=device, max_len=eval_len)
        bleu_lrn = evaluate_bleu(model_learned, test_loader, tgt_vocab,
                                 device=device, max_len=eval_len)
        drop = bleu_sin - bleu_lrn
        print(f"    Sinusoidal={bleu_sin:.2f}  Learned={bleu_lrn:.2f}  "
              f"Drop={drop:+.2f}")
        extrap_table.add_data(eval_len, mult, bleu_sin, bleu_lrn, drop)

    wandb.log({"extrapolation/bleu_vs_length_table": extrap_table})


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

    print(f"\n{'='*55}\n  {run_name}\n{'='*55}")
    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch, is_train=False, device=device,
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:02d} | train={train_loss:.4f}  "
              f"val={val_loss:.4f}  lr={lr:.6f}")

        # Scalars -> W&B renders as live interactive line charts
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         lr,
        })

        if val_loss < best_val:
            best_val = val_loss

    # ── Final BLEU on test set (once, after all epochs) ───────────────────────
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.device    = device
    test_bleu = evaluate_bleu(
        model, test_loader, tgt_vocab,
        device=device, max_len=config["max_train_len"],
    )
    print(f"  -> Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})

    return history, test_bleu


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading dataset...")
    train_loader, val_loader, test_loader, assets = get_dataloaders(
        batch_size=BASE_CONFIG["batch_size"]
    )
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]

    # ═══════════════════════════════════════════════════════════════════════════
    # Run 1 — Sinusoidal PE
    # ═══════════════════════════════════════════════════════════════════════════
    run1 = wandb.init(
        project=WANDB_PROJECT,
        name="sinusoidal-pe",
        config={**BASE_CONFIG, "pe_type": "sinusoidal"},
        reinit=True,
    )

    log_sinusoidal_pe_charts(d_model=BASE_CONFIG["d_model"])

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
    model_sin.device    = device

    history_sin, bleu_sin = train_model(
        "Sinusoidal PE", model_sin, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab,
    )
    run1.finish()

    # ═══════════════════════════════════════════════════════════════════════════
    # Run 2 — Learned PE
    # ═══════════════════════════════════════════════════════════════════════════
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
    model_learned.device    = device

    history_learned, bleu_learned = train_model(
        "Learned PE", model_learned, BASE_CONFIG, device,
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab,
    )

    log_learned_pe_charts(model_learned)

    run2.finish()

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary + Extrapolation run
    # ═══════════════════════════════════════════════════════════════════════════
    run_summary = wandb.init(
        project=WANDB_PROJECT,
        name="pe-comparison-summary",
        reinit=True,
    )

    # BLEU bar chart
    bar_table = wandb.Table(
        columns=["pe_type", "test_bleu"],
        data=[
            ["Sinusoidal PE", bleu_sin],
            ["Learned PE",    bleu_learned],
        ],
    )
    wandb.log({
        "comparison/bleu_bar_chart": wandb.plot.bar(
            bar_table, label="pe_type", value="test_bleu",
            title="Sinusoidal PE vs Learned PE — Test BLEU",
        )
    })

    # Loss curves overlay
    loss_table = wandb.Table(columns=["epoch", "loss", "split", "pe_type"])
    for ep, (tl, vl) in enumerate(zip(history_sin["train"], history_sin["val"])):
        loss_table.add_data(ep, tl, "train", "sinusoidal")
        loss_table.add_data(ep, vl, "val",   "sinusoidal")
    for ep, (tl, vl) in enumerate(zip(history_learned["train"], history_learned["val"])):
        loss_table.add_data(ep, tl, "train", "learned")
        loss_table.add_data(ep, vl, "val",   "learned")

    wandb.log({
        "comparison/loss_curves": wandb.plot.line(
            loss_table, x="epoch", y="loss", stroke="pe_type",
            title="Sinusoidal PE vs Learned PE — Loss Curves",
        )
    })

    # Extrapolation test (evidence for theoretical challenge in report)
    log_extrapolation_table(
        model_sin, model_learned,
        test_loader, tgt_vocab, device,
        train_max_len=BASE_CONFIG["max_train_len"],
    )

    # Summary table
    summary_table = wandb.Table(
        columns=["pe_type", "test_bleu", "best_val_loss", "best_train_loss"],
        data=[
            ["sinusoidal", bleu_sin,
             min(history_sin["val"]),     min(history_sin["train"])],
            ["learned",    bleu_learned,
             min(history_learned["val"]), min(history_learned["train"])],
        ],
    )
    wandb.log({"comparison/summary_table": summary_table})

    wandb.log({
        "summary/sinusoidal_bleu": bleu_sin,
        "summary/learned_bleu":    bleu_learned,
        "summary/bleu_difference": bleu_sin - bleu_learned,
    })

    print(f"\n{'='*55}")
    print(f"  Sinusoidal PE BLEU : {bleu_sin:.2f}")
    print(f"  Learned PE BLEU    : {bleu_learned:.2f}")
    print(f"  Difference         : {bleu_sin - bleu_learned:+.2f}")
    print(f"{'='*55}")

    run_summary.finish()
    print("\nExp 4 complete. All charts live in wandb.ai — zero images saved.")


if __name__ == "__main__":
    main()