"""
exp3_attention_heatmap.py
Section 2.3 — Attention Rollout & Head Specialization
Extracts attention weights from the last encoder layer and logs
interactive plotly heatmaps for each head to W&B.
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

from model import Transformer, make_src_mask


WANDB_PROJECT = "da6401-a3"
WANDB_API_KEY = "your_api_key_here"   # ← paste your key

TEST_SENTENCE = "Ein Mann sitzt auf einer Bank und liest eine Zeitung ."


# ── Hook to capture attention weights ────────────────────────────────────────
class AttentionHook:
    def __init__(self):
        self.weights = None

    def register(self, mha_module):
        hook_self = self

        def patched_forward(query, key, value, mask=None):
            q = mha_module._split_heads(mha_module.W_q(query))
            k = mha_module._split_heads(mha_module.W_k(key))
            v = mha_module._split_heads(mha_module.W_v(value))

            dk = q.size(-1)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
            if mask is not None:
                attn_logits = attn_logits.masked_fill(mask, float("-inf"))
            attn_weights = F.softmax(attn_logits, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            hook_self.weights = attn_weights.detach().cpu()

            attended = torch.matmul(attn_weights, v)
            attended = mha_module.dropout(attended)
            attended = mha_module._combine_heads(attended)
            return mha_module.W_o(attended)

        mha_module.forward = patched_forward


def encode_sentence(sentence, model):
    tokens  = model.src_tokenizer(sentence)
    unk_idx = model.src_vocab["<unk>"]
    ids     = [model.src_vocab["<sos>"]]
    ids    += [model.src_vocab[t] if t in model.src_vocab else unk_idx for t in tokens]
    ids    += [model.src_vocab["<eos>"]]
    labels  = ["<sos>"] + tokens + ["<eos>"]
    src     = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    return src, labels


def plot_head_heatmap_plotly(attn_matrix, tokens, head_idx, num_heads):
    # Round to 2 decimals, only show values above threshold
    threshold = 0.05
    text_vals = [
        [f"{v:.2f}" if v > threshold else "" for v in row]
        for row in attn_matrix.tolist()
    ]

    fig = go.Figure(data=go.Heatmap(
        z=attn_matrix.tolist(),
        x=tokens,
        y=tokens,
        colorscale="Blues",
        zmin=0,
        zmax=float(attn_matrix.max()),
        text=text_vals,
        texttemplate="%{text}",   # shows empty string for low values
        textfont={"size": 9},
    ))
    fig.update_layout(
        title=f"Head {head_idx+1} / {num_heads} — Last Encoder Layer Self-Attention",
        xaxis=dict(title="Key (attending to)", tickangle=45),
        yaxis=dict(title="Query", autorange="reversed"),
        width=600, height=600,
        margin=dict(l=100, r=40, t=60, b=120),
    )
    return fig


def plot_all_heads_grid_plotly(attn, tokens, num_heads):
    cols = min(4, num_heads)
    rows = math.ceil(num_heads / cols)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Head {h+1}" for h in range(num_heads)],
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
    )
    for h in range(num_heads):
        row = h // cols + 1
        col = h % cols + 1
        fig.add_trace(
            go.Heatmap(
                z=attn[h].numpy().tolist(),
                x=tokens, y=tokens,
                colorscale="Blues",
                showscale=(h == 0),
                name=f"Head {h+1}",
            ),
            row=row, col=col,
        )
    fig.update_layout(
        title=f"All {num_heads} Heads — Last Encoder Layer<br><sup>\"{TEST_SENTENCE}\"</sup>",
        height=400 * rows,
        width=350 * cols,
        showlegend=False,
    )
    for h in range(num_heads):
        idx = "" if h == 0 else str(h + 1)
        fig.update_layout(**{f"yaxis{idx}": dict(autorange="reversed")})
    return fig


def plot_entropy_bar_plotly(entropy_data):
    heads     = [f"Head {d['head']}" for d in entropy_data]
    entropy   = [d["entropy"]        for d in entropy_data]
    max_attn  = [d["max_attn"]       for d in entropy_data]
    behaviors = [d["behavior"]       for d in entropy_data]

    color_map = {
        "Sharp/Focused":       "#1565C0",
        "Next-token":          "#2E7D32",
        "Long-range":          "#6A1B9A",
        "Distributed":         "#F57F17",
    }
    colors = [color_map.get(b, "#607D8B") for b in behaviors]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Entropy per Head (lower = more focused)",
            "Max Attention Weight per Head",
        ],
    )
    fig.add_trace(go.Bar(
        x=heads, y=entropy,
        marker_color=colors,
        text=[f"{e:.3f}" for e in entropy],
        textposition="outside",
        hovertext=behaviors,
        name="Entropy",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=heads, y=max_attn,
        marker_color=colors,
        text=[f"{m:.3f}" for m in max_attn],
        textposition="outside",
        hovertext=behaviors,
        name="Max Attn",
    ), row=1, col=2)

    fig.update_layout(
        title="Head Specialization — Entropy & Max Attention Weight",
        height=450, width=950,
        showlegend=False,
    )
    return fig


def plot_rollout_plotly(rollout, tokens):
    fig = go.Figure(data=go.Heatmap(
        z=rollout.tolist(),
        x=tokens, y=tokens,
        colorscale="Viridis",
        text=np.round(rollout, 3).tolist(),
        
        textfont={"size": 8},
    ))
    fig.update_layout(
        title="Attention Rollout — Aggregated Across All Encoder Layers",
        xaxis=dict(title="Token (source)", tickangle=45),
        yaxis=dict(title="Token (receiver)", autorange="reversed"),
        width=650, height=600,
    )
    return fig


def compute_attention_rollout(all_layer_weights):
    rollout = None
    for layer_weights in all_layer_weights:
        avg = layer_weights.mean(dim=0).numpy()
        avg = avg + np.eye(avg.shape[0])
        avg = avg / avg.sum(axis=-1, keepdims=True)
        rollout = avg if rollout is None else avg @ rollout
    return rollout


def register_all_layer_hooks(model):
    hooks = []
    for layer in model.encoder.layers:
        hook = AttentionHook()
        hook.register(layer.self_attn)
        hooks.append(hook)
    return hooks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run = wandb.init(project=WANDB_PROJECT, name="attention-head-heatmaps", reinit=True)

    print("Loading model...")
    model = Transformer()
    model.eval()
    model = model.to(device)
    print("Model loaded.")

    # Register hooks on all layers
    all_hooks = register_all_layer_hooks(model)
    last_hook  = all_hooks[-1]

    src, token_labels = encode_sentence(TEST_SENTENCE, model)
    src      = src.to(device)
    src_mask = make_src_mask(src, pad_idx=model.src_vocab["<pad>"])

    print(f"Sentence : {TEST_SENTENCE}")
    print(f"Tokens   : {token_labels}")

    with torch.no_grad():
        _ = model.encode(src, src_mask)

    if last_hook.weights is None:
        print("ERROR: weights not captured.")
        return

    attn      = last_hook.weights[0]   # (num_heads, seq, seq)
    num_heads = attn.shape[0]
    print(f"Captured: {attn.shape} — {num_heads} heads")

    # ── 1. Individual head heatmaps ───────────────────────────────────────────
    print("Logging individual head heatmaps...")
    for h in range(num_heads):
        fig = plot_head_heatmap_plotly(attn[h].numpy(), token_labels, h, num_heads)
        wandb.log({f"head_{h+1}_heatmap": wandb.Plotly(fig)})
        print(f"  Head {h+1} logged")

    # ── 2. All heads grid ─────────────────────────────────────────────────────
    fig_grid = plot_all_heads_grid_plotly(attn, token_labels, num_heads)
    wandb.log({"all_heads_grid": wandb.Plotly(fig_grid)})

    # ── 3. Entropy & specialization ───────────────────────────────────────────
    entropy_data = []
    for h in range(num_heads):
        head_attn = attn[h].numpy()
        entropy   = float(-(head_attn * np.log(head_attn + 1e-9)).sum(axis=-1).mean())
        max_attn  = float(head_attn.max())
        subdiag   = float(np.diag(head_attn[1:, :-1]).mean()) if head_attn.shape[0] > 1 else 0.0
        behavior  = (
            "Sharp/Focused" if entropy  < 1.0 else
            "Next-token"    if subdiag  > 0.3 else
            "Long-range"    if max_attn < 0.3 else
            "Distributed"
        )
        entropy_data.append({
            "head": h+1, "entropy": entropy,
            "max_attn": max_attn, "subdiag": subdiag,
            "behavior": behavior,
        })

    fig_entropy = plot_entropy_bar_plotly(entropy_data)
    wandb.log({"head_specialization": wandb.Plotly(fig_entropy)})

    # ── 4. Attention rollout ──────────────────────────────────────────────────
    all_layer_attn = [h.weights[0] for h in all_hooks if h.weights is not None]
    if all_layer_attn:
        rollout     = compute_attention_rollout(all_layer_attn)
        fig_rollout = plot_rollout_plotly(rollout, token_labels)
        wandb.log({"attention_rollout": wandb.Plotly(fig_rollout)})

    # ── 5. Summary table ──────────────────────────────────────────────────────
    table = wandb.Table(columns=["Head", "Entropy", "Max Attn", "Next-token Score", "Behavior"])
    for d in entropy_data:
        table.add_data(
            f"Head {d['head']}",
            round(d["entropy"],  4),
            round(d["max_attn"], 4),
            round(d["subdiag"],  4),
            d["behavior"],
        )
    wandb.log({"head_analysis_table": table})

    # Print summary
    print(f"\n── Head Analysis ───────────────────────────────────────")
    print(f"{'Head':<8} {'Entropy':<12} {'MaxAttn':<12} {'NextTok':<12} Behavior")
    print("-" * 62)
    for d in entropy_data:
        print(f"  {d['head']:<6} {d['entropy']:<12.3f} {d['max_attn']:<12.3f} {d['subdiag']:<12.3f} {d['behavior']}")

    run.finish()
    print("\nExp 3 complete! Check wandb.ai for interactive heatmaps.")


if __name__ == "__main__":
    main()