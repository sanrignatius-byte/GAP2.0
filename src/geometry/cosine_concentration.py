"""Inter-token Cosine Concentration (ICC).

Measures the mean pairwise cosine similarity among tokens within a modality.
High ICC (tokens becoming more similar) indicates over-smoothing.

Visual tokens approaching ICC = 1.0 means they are collapsing to a single
direction in representation space — a geometric signature of information loss.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def compute_cosine_concentration(
    H: torch.Tensor | np.ndarray,
    subsample: Optional[int] = None,
) -> float:
    """Compute mean pairwise cosine similarity among token representations.

    Args:
        H: (N, d) representation matrix where N is num tokens.
        subsample: If set, randomly subsample this many token pairs
            for efficiency (useful when N is large).

    Returns:
        Mean cosine similarity (float in [-1, 1], typically [0, 1]).
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float()
    else:
        H = H.float()

    if H.shape[0] < 2:
        return 1.0  # Single token — trivially concentrated

    # Normalize rows to unit vectors
    norms = H.norm(dim=1, keepdim=True).clamp(min=1e-8)
    H_norm = H / norms

    # Pairwise cosine similarity matrix
    sim_matrix = H_norm @ H_norm.T  # (N, N)

    N = H.shape[0]

    if subsample is not None and N * (N - 1) // 2 > subsample:
        # Random sampling of pairs
        indices = torch.triu_indices(N, N, offset=1)
        num_pairs = indices.shape[1]
        selected = torch.randperm(num_pairs)[:subsample]
        values = sim_matrix[indices[0][selected], indices[1][selected]]
        return float(values.mean().item())
    else:
        # Use all upper-triangular pairs (exclude diagonal)
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        values = sim_matrix[mask]
        return float(values.mean().item())


def compute_cosine_concentration_per_layer(
    activations: dict[int, torch.Tensor],
    subsample: Optional[int] = None,
) -> dict[int, float]:
    """Compute ICC for each layer.

    Args:
        activations: Dict mapping layer_idx -> hidden states (N, d).
        subsample: Optional pair subsampling count.

    Returns:
        Dict mapping layer_idx -> ICC value.
    """
    return {
        layer_idx: compute_cosine_concentration(H, subsample)
        for layer_idx, H in activations.items()
    }


def compute_cosine_histogram(
    H: torch.Tensor | np.ndarray,
    bins: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute histogram of pairwise cosine similarities.

    Useful for visualizing the distribution shape: uniform vs. peaked.

    Returns:
        (bin_edges, counts) for plotting.
    """
    if isinstance(H, np.ndarray):
        H = torch.from_numpy(H).float()
    else:
        H = H.float()

    norms = H.norm(dim=1, keepdim=True).clamp(min=1e-8)
    H_norm = H / norms
    sim_matrix = H_norm @ H_norm.T

    N = H.shape[0]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    values = sim_matrix[mask].cpu().numpy()

    counts, bin_edges = np.histogram(values, bins=bins, range=(-1, 1))
    return bin_edges, counts
