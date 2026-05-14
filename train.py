"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import os
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloaders
from lr_scheduler import NoamScheduler
from model import Transformer, make_src_mask, make_tgt_mask

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None


def maybe_login_wandb() -> bool:
    """
    Log into Weights & Biases if an API key is available.

    Expected environment variables:
        WANDB_API_KEY  : W&B API key
        WANDB_DISABLED : set to 'true' to disable W&B entirely
    """
    if wandb is None:
        return False

    if os.getenv("WANDB_DISABLED", "").lower() in {"true", "1", "yes"}:
        return False

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        return False

    wandb.login(key=api_key, relogin=True)
    return True


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0
            pad_mask = target == self.pad_idx
            true_dist[pad_mask] = 0

        loss = -(true_dist * log_probs).sum(dim=1)
        non_pad = (target != self.pad_idx).sum().clamp_min(1)
        return loss.sum() / non_pad


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    model.train(is_train)
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(data_iter, desc=f"{'train' if is_train else 'eval'} epoch {epoch_num}", leave=False)
    for src, tgt in progress:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        if is_train:
            if optimizer is None:
                raise ValueError("optimizer must be provided during training")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(total_batches, 1)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    model.eval()
    with torch.no_grad():
        src = src.to(device)
        src_mask = src_mask.to(device)
        memory = model.encode(src, src_mask)
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys)
            out = model.decode(memory, src_mask, ys, tgt_mask)
            next_token = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if next_token.item() == end_symbol:
                break
        return ys


def _lookup_token(vocab, idx: int) -> str:
    if hasattr(vocab, "lookup_token"):
        return vocab.lookup_token(idx)
    if hasattr(vocab, "itos"):
        return vocab.itos[idx]
    raise AttributeError("Vocabulary must expose lookup_token() or itos.")


def _lookup_index(vocab, token: str) -> int:
    if hasattr(vocab, "stoi"):
        return vocab.stoi[token]
    return vocab[token]


def _modified_precision(references, hypotheses, n: int) -> tuple[int, int]:
    matches = 0
    total = 0
    for ref, hyp in zip(references, hypotheses):
        hyp_ngrams = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        matches += sum(min(count, ref_ngrams[ngram]) for ngram, count in hyp_ngrams.items())
        total += max(len(hyp) - n + 1, 0)
    return matches, total


def _corpus_bleu(references, hypotheses) -> float:
    precisions = []
    for n in range(1, 5):
        matches, total = _modified_precision(references, hypotheses, n)
        precisions.append((matches + 1) / (total + 1))

    ref_len = sum(len(ref) for ref in references)
    hyp_len = sum(len(hyp) for hyp in hypotheses)
    if hyp_len == 0:
        return 0.0

    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1 - (ref_len / hyp_len))
    score = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return score * 100.0


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    sos_idx = _lookup_index(tgt_vocab, "<sos>")
    eos_idx = _lookup_index(tgt_vocab, "<eos>")
    pad_idx = _lookup_index(tgt_vocab, "<pad>")

    references = []
    hypotheses = []
    model.eval()

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="bleu", leave=False):
            src = src.to(device)
            tgt = tgt.to(device)
            src_mask = make_src_mask(src, pad_idx=pad_idx)
            decoded = greedy_decode(model, src, src_mask, max_len, sos_idx, eos_idx, device=device)

            ref_tokens = []
            for idx in tgt.squeeze(0).tolist():
                token = _lookup_token(tgt_vocab, idx)
                if token in {"<sos>", "<pad>"}:
                    continue
                if token == "<eos>":
                    break
                ref_tokens.append(token)

            hyp_tokens = []
            for idx in decoded.squeeze(0).tolist():
                token = _lookup_token(tgt_vocab, idx)
                if token in {"<sos>", "<pad>"}:
                    continue
                if token == "<eos>":
                    break
                hyp_tokens.append(token)

            references.append(ref_tokens)
            hypotheses.append(hyp_tokens)

    return _corpus_bleu(references, hypotheses)


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "model_config": model.config,
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint["epoch"])


def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    config = {
        "batch_size": 32,
        "num_epochs": 5,
        "d_model": 256,
        "N": 4,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "warmup_steps": 4000,
        "learning_rate": 1.0,
        "checkpoint_path": "checkpoint.pt",
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_wandb = maybe_login_wandb()
    run = None
    if wandb is not None and use_wandb:
        run = wandb.init(project="da6401-a3", config=config)

    train_loader, val_loader, test_loader, assets = get_dataloaders(batch_size=config["batch_size"])
    src_vocab = assets["src_vocab"]
    tgt_vocab = assets["tgt_vocab"]

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    ).to(device)
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model.src_tokenizer = assets["src_tokenizer"]
    model.tgt_tokenizer = assets["tgt_tokenizer"]
    model.device = device

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab["<pad>"],
        smoothing=0.1,
    )

    for epoch in range(config["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, True, device)
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, False, device)
        save_checkpoint(model, optimizer, scheduler, epoch, path=config["checkpoint_path"])
        if run is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    if run is not None:
        wandb.log({"test_bleu": bleu})
        run.finish()
    print(f"Final BLEU: {bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()
