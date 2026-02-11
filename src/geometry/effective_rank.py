"""Effective Rank computation via SVD.

Effective Rank (ER) measures the "spread" of the singular value spectrum
of a representation matrix. A low ER indicates rank collapse (degenerate
representations), while a high ER indicates rich, diverse representations.

ER(H) = exp(-sum_i p_i * log(p_i))

where p_i = sigma_i / sum_j sigma_j and sigma_i are the singular values of H.
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def compute_effective_rank(
    H: torch.Tensor | np.ndarray,
    normalize: bool = True,
) -> float:
    """Compute effective rank of a representation matrix.

    Args:
        H: Representation matrix of shape (N, d) where N is the number
           of tokens and d is the hidden dimension.
        normalize: Whether to center the matrix (subtract mean) before SVD.

    Returns:
        Effective rank (float, between 1 and min(N, d)).
    """
    if isinstance(H, torch.Tensor):
        H = H.float().cpu().numpy()

    if H.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {H.shape}")

    if normalize:
        H = H - H.mean(axis=0, keepdims=True)

    # SVD — we only need singular values
    singular_values = np.linalg.svd(H, compute_uv=False)

    # Filter out near-zero singular values for numerical stability
    singular_values = singular_values[singular_values > 1e-10]

    if len(singular_values) == 0:
        return 0.0

    # Normalize to probability distribution
    p = singular_values / singular_values.sum()

    # Shannon entropy of the singular value distribution
    entropy = -np.sum(p * np.log(p))

    # Effective rank = exp(entropy)
    return float(np.exp(entropy))


def compute_effective_rank_per_layer(
    activations: dict[int, torch.Tensor],
    normalize: bool = True,
) -> dict[int, float]:
    """Compute effective rank for each layer's token representations.

    Args:
        activations: Dict mapping layer_idx -> hidden states tensor (N, d).
        normalize: Whether to center before SVD.

    Returns:
        Dict mapping layer_idx -> effective rank value.
    """
    return {
        layer_idx: compute_effective_rank(H, normalize)
        for layer_idx, H in activations.items()
    }


def compute_singular_spectrum(
    H: torch.Tensor | np.ndarray,
    top_k: Optional[int] = None,
    normalize_values: bool = True,
) -> np.ndarray:
    """Compute the singular value spectrum of a representation matrix.

    Useful for detailed visualization of the spectral structure.

    Args:
        H: (N, d) representation matrix.
        top_k: Return only top-k singular values. None = all.
        normalize_values: Normalize so singular values sum to 1.

    Returns:
        Array of singular values (descending order).
    """
    if isinstance(H, torch.Tensor):
        H = H.float().cpu().numpy()

    H = H - H.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(H, compute_uv=False)

    if normalize_values:
        total = sv.sum()
        if total > 0:
            sv = sv / total

    if top_k is not None:
        sv = sv[:top_k]

    return sv
