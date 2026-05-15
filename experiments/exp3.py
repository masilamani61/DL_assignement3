"""
exp3_attention_heatmap.py
Section 2.3 — Attention Rollout & Head Specialization
Extracts attention weights from the last encoder layer for a
single German sentence and logs a heatmap for each head to W&B.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import wandb

from model import Transformer, make_src_mask


WANDB_PROJECT = "da6401-a3"
WANDB_API_KEY = "your_api_key_here"   # ← paste your key

# German test sentence
TEST_SENTENCE = "Ein Mann sitzt auf einer Bank und liest eine Zeitung ."


# ── Hook to capture attention weights ────────────────────────────────────────
class AttentionHook:
    """Registers a forward hook on a MultiHeadAttention module to capture weights."""

    def __init__(self):
        self.weights = None   # shape: (batch, heads, tgt_len, src_len)

    def hook_fn(self, module, input, output):
        # Re-run SDPA to capture weights (model already computed output)
        import math
        import torch.nn.functional as F

        query, key, value = input[0], input[1], input[2]
        # These are already projected + split — recompute from module internals
        # Instead, monkey-patch to capture during forward
        pass

    def register(self, mha_module):
        """Monkey-patch the MHA forward to capture attn_weights."""
        original_forward = mha_module.forward

        hook_self = self

        def patched_forward(query, key, value, mask=None):
            import math
            import torch.nn.functional as F

            q = mha_module._split_heads(mha_module.W_q(query))
            k = mha_module._split_heads(mha_module.W_k(key))
            v = mha_module._split_heads(mha_module.W_v(value))

            dk = q.size(-1)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            # Save weights
            hook_self.weights = attn_weights.detach().cpu()

            attended = torch.matmul(attn_weights, v)
            attended = mha_module.dropout(attended)
            attended = mha_module._combine_heads(attended)
            return mha_module.W_o(attended)

        mha_module.forward = patched_forward


# ── Tokenize sentence and get src ids ────────────────────────────────────────
def encode_sentence(sentence, model):
    tokens = model.src_tokenizer(sentence)
    unk_idx = model.src_vocab["<unk>"]
    pad_idx = model.src_vocab["<pad>"]

    ids  = [model.src_vocab["<sos>"]]
    ids += [model.src_vocab[t] if t in model.src_vocab else unk_idx for t in tokens]
    ids += [model.src_vocab["<eos>"]]

    token_labels = ["<sos>"] + tokens + ["<eos>"]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return src, token_labels


# ── Plot one head heatmap ─────────────────────────────────────────────────────
def plot_head_heatmap(attn_matrix, tokens, head_idx, title):
    """
    attn_matrix: numpy (seq_len, seq_len)
    tokens:      list of str
    """
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.6), max(5, len(tokens) * 0.5)))

    im = ax.imshow(attn_matrix, cmap="Blues", vmin=0, vmax=attn_matrix.max())
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tokens, fontsize=9)

    ax.set_xlabel("Key positions (attending to)", fontsize=10)
    ax.set_ylabel("Query positions", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate cells with values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = attn_matrix[i, j]
            color = "white" if val > 0.5 * attn_matrix.max() else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    wandb.login(key=WANDB_API_KEY)
    run = wandb.init(
        project=WANDB_PROJECT,
        name="attention-head-heatmaps",
        reinit=True,
    )

    # Load trained model
    print("Loading model...")
    model = Transformer()
    model.eval()
    model = model.to(device)
    print("Model loaded.")

    # Hook the last encoder layer's self-attention
    last_encoder_layer = model.encoder.layers[-1]
    hook = AttentionHook()
    hook.register(last_encoder_layer.self_attn)

    # Encode sentence
    src, token_labels = encode_sentence(TEST_SENTENCE, model)
    src = src.to(device)
    src_mask = make_src_mask(src, pad_idx=model.src_vocab["<pad>"])

    print(f"Sentence : {TEST_SENTENCE}")
    print(f"Tokens   : {token_labels}")

    # Forward pass through encoder only
    with torch.no_grad():
        _ = model.encode(src, src_mask)

    if hook.weights is None:
        print("ERROR: Attention weights not captured. Check hook registration.")
        return

    # hook.weights shape: (1, num_heads, seq_len, seq_len)
    attn = hook.weights[0]  # (num_heads, seq_len, seq_len)
    num_heads = attn.shape[0]
    print(f"Captured attention: {attn.shape} — {num_heads} heads")

    # Log individual heatmaps + combined summary
    wandb_images = []
    all_figs = []

    for h in range(num_heads):
        head_attn = attn[h].numpy()   # (seq_len, seq_len)
        title = f"Head {h+1} / {num_heads} — Last Encoder Layer"
        fig = plot_head_heatmap(head_attn, token_labels, h, title)
        wandb_images.append(wandb.Image(fig, caption=title))
        all_figs.append(fig)
        plt.close(fig)
        print(f"  Head {h+1}: max_attn={head_attn.max():.3f}  entropy={-(head_attn * np.log(head_attn + 1e-9)).sum(axis=-1).mean():.3f}")

    # Combined grid of all heads
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols
    fig_all, axes = plt.subplots(rows, cols,
                                  figsize=(cols * max(5, len(token_labels) * 0.5),
                                           rows * max(4, len(token_labels) * 0.4)))
    axes = np.array(axes).flatten()

    for h in range(num_heads):
        head_attn = attn[h].numpy()
        ax = axes[h]
        im = ax.imshow(head_attn, cmap="Blues", vmin=0, vmax=head_attn.max())
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(token_labels, fontsize=7)
        ax.set_title(f"Head {h+1}", fontsize=9, fontweight="bold")

    # Hide unused axes
    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    fig_all.suptitle(
        f"All {num_heads} Heads — Last Encoder Layer\n\"{TEST_SENTENCE}\"",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()

    # Log to W&B
    wandb.log({
        "attention_heatmaps": wandb_images,
        "all_heads_grid": wandb.Image(fig_all, caption="All heads combined"),
    })
    plt.close(fig_all)

    # Also log per-head entropy (measures head specialization)
    entropy_data = []
    for h in range(num_heads):
        head_attn = attn[h].numpy()
        entropy = -(head_attn * np.log(head_attn + 1e-9)).sum(axis=-1).mean()
        max_attn = head_attn.max()
        wandb.log({f"head_{h+1}_entropy": entropy, f"head_{h+1}_max_attn": max_attn})
        entropy_data.append({"head": h+1, "entropy": entropy, "max_attn": max_attn})

    # Print analysis summary
    print("\n── Head Analysis ──────────────────────────────────")
    print(f"{'Head':<8} {'Entropy':<12} {'Max Attn':<12} {'Behavior'}")
    print("-" * 50)
    for d in entropy_data:
        behavior = (
            "sharp/focused" if d["entropy"] < 1.0 else
            "local/next-token" if d["max_attn"] > 0.6 else
            "distributed"
        )
        print(f"  {d['head']:<6} {d['entropy']:<12.3f} {d['max_attn']:<12.3f} {behavior}")

    run.finish()
    print("\nExp 3 complete! Check wandb.ai for attention heatmaps.")


if __name__ == "__main__":
    main()