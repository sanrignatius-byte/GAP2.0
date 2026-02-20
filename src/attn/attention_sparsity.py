"""Attention sparsity and entropy analysis for visual re-grounding hypothesis.

Extends the basic attention flow analysis with metrics that distinguish
genuine re-grounding (sparse, semantically targeted) from attention sinks
(diffuse, structural artifacts).

Metrics:
    - Entropy: Shannon entropy of text-to-visual attention distribution
    - N_eff: Effective number of attended visual tokens (exp of entropy)
    - Top-k concentration: Fraction of attention on top-k visual tokens
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Tuple


def compute_attention_sparsity(
    attn_weights: torch.Tensor,
    visual_token_range: Tuple[int, int],
    top_k_values: list[int] | None = None,
) -> dict:
    """Compute sparsity metrics for text-to-visual attention at a single layer.

    Args:
        attn_weights: Attention tensor of shape (1, num_heads, seq_len, seq_len).
        visual_token_range: (vstart, vend) of visual tokens.
        top_k_values: List of k values for top-k concentration. Default [5, 10, 50].

    Returns:
        Dict with keys:
            - entropy_mean: Mean Shannon entropy of t2v attention (over heads & text positions)
            - entropy_per_head: (H,) array of per-head mean entropy
            - n_eff_mean: Mean effective token count (exp of entropy)
            - n_eff_per_head: (H,) per-head N_eff
            - top_k_concentration: Dict mapping k -> mean fraction of attention on top-k visual tokens
            - t2v_mass_mean: Mean total attention mass from text to visual
            - t2v_mass_per_head: (H,) per-head total t2v mass
            - max_attention_mean: Mean of max attention weight to any visual token
    """
    if top_k_values is None:
        top_k_values = [5, 10, 50]

    vstart, vend = visual_token_range
    attn = attn_weights[0]  # (H, S, S)
    seq_len = attn.shape[-1]
    text_start = vend
    n_visual = vend - vstart

    if text_start >= seq_len or n_visual == 0:
        n_heads = attn.shape[0]
        return {
            "entropy_mean": 0.0,
            "entropy_per_head": np.zeros(n_heads),
            "n_eff_mean": 1.0,
            "n_eff_per_head": np.ones(n_heads),
            "top_k_concentration": {k: 0.0 for k in top_k_values},
            "t2v_mass_mean": 0.0,
            "t2v_mass_per_head": np.zeros(n_heads),
            "max_attention_mean": 0.0,
        }

    # t2v: (H, n_text, n_visual) — attention from text tokens to visual tokens
    t2v = attn[:, text_start:, vstart:vend]

    # Normalize t2v to a probability distribution over visual tokens
    # (attention weights are already softmaxed over all keys; we renormalize
    # to get the distribution conditional on attending to visual tokens)
    t2v_sum = t2v.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    t2v_prob = t2v / t2v_sum  # (H, n_text, n_visual)

    # Shannon entropy: -sum(p * log(p))
    log_prob = torch.log(t2v_prob.clamp(min=1e-12))
    entropy = -(t2v_prob * log_prob).sum(dim=-1)  # (H, n_text)

    entropy_per_head = entropy.mean(dim=-1).float().cpu().numpy()  # (H,)
    entropy_mean = float(entropy.mean().item())

    # N_eff = exp(entropy)
    n_eff = torch.exp(entropy)  # (H, n_text)
    n_eff_per_head = n_eff.mean(dim=-1).float().cpu().numpy()
    n_eff_mean = float(n_eff.mean().item())

    # Top-k concentration
    top_k_conc = {}
    for k in top_k_values:
        if k >= n_visual:
            top_k_conc[k] = 1.0
            continue
        topk_vals, _ = t2v_prob.topk(k, dim=-1)  # (H, n_text, k)
        conc = topk_vals.sum(dim=-1).mean()  # scalar
        top_k_conc[k] = float(conc.item())

    # Total t2v mass (unnormalized — how much attention budget goes to visual)
    t2v_mass = t2v.sum(dim=-1)  # (H, n_text)
    t2v_mass_per_head = t2v_mass.mean(dim=-1).float().cpu().numpy()
    t2v_mass_mean = float(t2v_mass.mean().item())

    # Max attention to any single visual token
    max_attn = t2v_prob.max(dim=-1).values  # (H, n_text)
    max_attention_mean = float(max_attn.mean().item())

    return {
        "entropy_mean": entropy_mean,
        "entropy_per_head": entropy_per_head,
        "n_eff_mean": n_eff_mean,
        "n_eff_per_head": n_eff_per_head,
        "top_k_concentration": top_k_conc,
        "t2v_mass_mean": t2v_mass_mean,
        "t2v_mass_per_head": t2v_mass_per_head,
        "max_attention_mean": max_attention_mean,
    }
