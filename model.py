"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import copy
import math
from typing import Optional, Tuple

try:
    import gdown
except ImportError:  # pragma: no cover - optional dependency
    gdown = None

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    dk = Q.size(-1)
    attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))
    attn_weights = F.softmax(attn_logits, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """Build a padding mask for the encoder."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Build padding + causal mask for the decoder."""
    tgt_len = tgt.size(1)
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)
    return pad_mask | causal_mask


class MultiHeadAttention(nn.Module):
    """Multi-head attention implemented from first principles."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.W_q(query))
        k = self._split_heads(self.W_k(key))
        v = self._split_heads(self.W_v(value))
        # CORRECT:
        attended, _ = scaled_dot_product_attention(q, k, v, mask)
        attended = self._combine_heads(attended)
        return self.W_o(attended)
        return self.W_o(attended)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):
    """Single decoder block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_out))
        cross_attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_out))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))
        return x


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(layer) for _ in range(N))
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for machine translation."""

    def __init__(
    self,
    src_vocab_size: int = None,
    tgt_vocab_size: int = None,
    d_model: int = 512,
    N: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    dropout: float = 0.1,
    checkpoint_path: str = "checkpoint.pt",
) -> None:
        super().__init__()

        checkpoint = None
        if checkpoint_path is not None:
            if gdown is None:
                raise ImportError("gdown is required.")
            gdown.download(id="1hYsQhTN-XsTVOb_I22XCwCIEhziWg0to", output=checkpoint_path, quiet=False)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            cfg = checkpoint.get("model_config", {})
            src_vocab_size = src_vocab_size or cfg["src_vocab_size"]
            tgt_vocab_size = tgt_vocab_size or cfg["tgt_vocab_size"]
            d_model   = cfg.get("d_model",   d_model)
            N         = cfg.get("N",         N)
            num_heads = cfg.get("num_heads", num_heads)
            d_ff      = cfg.get("d_ff",      d_ff)
            dropout   = cfg.get("dropout",   dropout)

        self.config = {
            "src_vocab_size": src_vocab_size, "tgt_vocab_size": tgt_vocab_size,
            "d_model": d_model, "N": N, "num_heads": num_heads,
            "d_ff": d_ff, "dropout": dropout,
        }
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, N)
        self.decoder = Decoder(decoder_layer, N)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

        if checkpoint is not None:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.load_state_dict(state_dict)
        if checkpoint is not None:
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.load_state_dict(state_dict)

            # Auto-build inference assets so infer() works without manual setup
            from dataset import get_dataloaders
            import spacy

            def _load_tokenizer(model_name, lang_code):
                try:
                    nlp = spacy.load(model_name)
                except OSError:
                    nlp = spacy.blank(lang_code)
                return lambda text: [tok.text.lower() for tok in nlp(text.strip()) if tok.text.strip()]

            _, _, _, assets = get_dataloaders(batch_size=1)
            self.src_vocab      = assets["src_vocab"]
            self.tgt_vocab      = assets["tgt_vocab"]
            self.src_tokenizer  = assets["src_tokenizer"]
            self.tgt_tokenizer  = assets["tgt_tokenizer"]
            self.device         = "cuda" if torch.cuda.is_available() else "cpu"
            self.to(self.device)

    def _reset_parameters(self) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_embed = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embed = self.positional_encoding(src_embed)
        return self.encoder(src_embed, src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.positional_encoding(tgt_embed)
        decoded = self.decoder(tgt_embed, memory, src_mask, tgt_mask)
        return self.generator(decoded)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence using greedy autoregressive decoding.
        """
        if not all(
            hasattr(self, attr)
            for attr in ("src_vocab", "tgt_vocab", "src_tokenizer", "tgt_tokenizer", "device")
        ):
            raise ValueError(
                "Inference assets missing. Attach src_vocab, tgt_vocab, src_tokenizer, "
                "tgt_tokenizer, and device to the model before calling infer()."
            )

        from train import greedy_decode

        src_tokens = ["<sos>"] + self.src_tokenizer(src_sentence) + ["<eos>"]
        unk_idx = self.src_vocab["<unk>"]
        src_ids = [self.src_vocab[token] if token in self.src_vocab else unk_idx for token in src_tokens]
        src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx=self.src_vocab["<pad>"])

        decoded = greedy_decode(
            self,
            src,
            src_mask,
            max_len=100,
            start_symbol=self.tgt_vocab["<sos>"],
            end_symbol=self.tgt_vocab["<eos>"],
            device=self.device,
        )

        words = []
        for idx in decoded.squeeze(0).tolist():
            token = self.tgt_vocab.lookup_token(idx)
            if token in {"<sos>", "<pad>"}:
                continue
            if token == "<eos>":
                break
            words.append(token)
        return " ".join(words)
