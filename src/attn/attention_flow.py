"""Attention flow computation for Phase 2 experiments.

Computes per-layer text-to-visual attention metrics from a model's
output_attentions tensors. Memory-efficient: processes layer by layer
and never accumulates attention tensors.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


def compute_layer_attention_flow(
    attn_weights: torch.Tensor,
    visual_token_range: Tuple[int, int],
) -> dict:
    """Compute attention flow metrics for a single layer.

    Args:
        attn_weights: Attention tensor of shape (1, num_heads, seq_len, seq_len).
            Row i contains the attention weights from token i to all tokens.
        visual_token_range: (vstart, vend) exclusive end index of visual tokens.

    Returns:
        Dict with keys:
            - text_to_visual: scalar float, mean attention from text tokens to
              visual tokens (averaged over heads and text token positions).
            - visual_to_visual: scalar float, mean attention from visual tokens
              to other visual tokens (sanity-check metric).
            - t2v_per_head: np.ndarray of shape (num_heads,), per-head t2v.
    """
    vstart, vend = visual_token_range
    # attn_weights: (1, H, S, S) → squeeze batch dim → (H, S, S)
    attn = attn_weights[0]  # (H, S, S)
    seq_len = attn.shape[-1]

    # Text token indices: everything after vend
    # (system tokens before vstart are excluded per design)
    text_start = vend
    if text_start >= seq_len:
        # No text tokens after visual block (degenerate case)
        return {
            "text_to_visual": 0.0,
            "visual_to_visual": 0.0,
            "t2v_per_head": np.zeros(attn.shape[0]),
        }

    # Slice: attention from text tokens → to visual tokens
    # attn[:, text_start:, vstart:vend] has shape (H, n_text, n_visual)
    t2v = attn[:, text_start:, vstart:vend]  # (H, n_text, n_visual)

    # Sum over visual token dim → total attention budget to visual tokens
    # Then average over text positions and heads
    t2v_sum_visual = t2v.sum(dim=-1)  # (H, n_text)
    t2v_per_head = t2v_sum_visual.mean(dim=-1)  # (H,)
    text_to_visual = float(t2v_per_head.mean().item())

    # Visual-to-visual: attention from visual tokens → visual tokens (excl. self)
    v2v = attn[:, vstart:vend, vstart:vend]  # (H, n_vis, n_vis)
    visual_to_visual = float(v2v.mean().item())

    return {
        "text_to_visual": text_to_visual,
        "visual_to_visual": visual_to_visual,
        "t2v_per_head": t2v_per_head.float().cpu().numpy(),
    }


def compute_uniform_baseline(
    visual_token_range: Tuple[int, int],
    seq_len: int,
) -> float:
    """Compute the expected text-to-visual attention under a uniform distribution.

    Under uniform attention, each query token distributes 1/seq_len weight to
    every key token. The fraction allocated to visual tokens is simply:
        n_visual / seq_len

    Note: For causal attention the denominator should be the number of tokens
    the query can attend to (i.e., its own position + 1 for causal mask).
    We approximate with a mean over text token positions.

    Args:
        visual_token_range: (vstart, vend).
        seq_len: Total sequence length.

    Returns:
        Scalar float representing the uniform baseline fraction.
    """
    vstart, vend = visual_token_range
    n_visual = vend - vstart
    text_positions = list(range(vend, seq_len))
    if not text_positions:
        return 0.0
    # For causal attention, token at position i can attend to tokens 0..i
    # The fraction of its attend-able tokens that are visual is n_visual / (i+1)
    fractions = [n_visual / (pos + 1) for pos in text_positions]
    return float(np.mean(fractions))
