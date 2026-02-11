"""Cross-modal Centered Kernel Alignment (CKA).

CKA measures representational similarity between two sets of representations.
In GAP 2.0, we use it to measure whether the visual token subspace is being
"absorbed" by the text token subspace across layers.

Rising CKA(H_v, H_t) through depth = visual representations becoming
increasingly similar to text representations = loss of visual distinctiveness.

Linear CKA:
    CKA(H_v, H_t) = ||H_t^T H_v||_F^2 / (||H_v^T H_v||_F * ||H_t^T H_t||_F)
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional


def compute_linear_cka(
    H_v: torch.Tensor | np.ndarray,
    H_t: torch.Tensor | np.ndarray,
    debiased: bool = True,
) -> float:
    """Compute linear CKA between visual and text token representations.

    Args:
        H_v: Visual token representations (N_v, d).
        H_t: Text token representations (N_t, d).
        debiased: Use debiased HSIC estimator (recommended for small N).

    Returns:
        CKA value in [0, 1]. Higher = more similar representations.
    """
    if isinstance(H_v, torch.Tensor):
        H_v = H_v.float().cpu().numpy()
    if isinstance(H_t, torch.Tensor):
        H_t = H_t.float().cpu().numpy()

    # Center the representations
    H_v = H_v - H_v.mean(axis=0, keepdims=True)
    H_t = H_t - H_t.mean(axis=0, keepdims=True)

    # Compute Gram matrices (linear kernel)
    K = H_v @ H_v.T  # (N_v, N_v)
    L = H_t @ H_t.T  # (N_t, N_t)

    if debiased:
        return _debiased_cka(K, L)
    else:
        return _standard_cka(H_v, H_t)


def _standard_cka(H_v: np.ndarray, H_t: np.ndarray) -> float:
    """Standard linear CKA using the cross-covariance formulation.

    CKA = ||H_t^T H_v||_F^2 / (||H_v^T H_v||_F * ||H_t^T H_t||_F)
    """
    cross = H_t.T @ H_v  # (d, d) — but we need cross-set term

    # Actually, for differently-sized sets, we use the HSIC formulation:
    # HSIC(K, L) = trace(KHLH) / (n-1)^2
    # where H is the centering matrix

    # For simplicity and correctness with different-sized sets,
    # we use the RBF-free CKA via singular values

    # Compute cross-covariance
    # H_v: (N_v, d), H_t: (N_t, d)
    # We need to work with the Gram matrices in a compatible way

    # Use the feature-space formulation:
    cross_cov = H_v.T @ H_v  # (d, d)
    cross_cov2 = H_t.T @ H_t  # (d, d)
    cross_term = H_v.T @ H_t   # Only valid if N_v == N_t

    # If dimensions match, use direct formulation
    if H_v.shape[0] == H_t.shape[0]:
        YtX = H_t.T @ H_v
        numerator = np.linalg.norm(YtX, 'fro') ** 2
        denom1 = np.linalg.norm(H_v.T @ H_v, 'fro')
        denom2 = np.linalg.norm(H_t.T @ H_t, 'fro')
        if denom1 * denom2 < 1e-10:
            return 0.0
        return float(numerator / (denom1 * denom2))
    else:
        # For differently-sized token sets, use the minibatch CKA approach
        return _minibatch_cka(H_v, H_t)


def _debiased_cka(K: np.ndarray, L: np.ndarray) -> float:
    """Debiased CKA using the HSIC_1 estimator from Nguyen et al. (2020).

    This handles the bias in HSIC estimation for small sample sizes.
    Requires same-sized Gram matrices, so we subsample the larger set.
    """
    # Make sure K and L have the same size
    n_k, n_l = K.shape[0], L.shape[0]

    if n_k != n_l:
        # Subsample the larger set to match sizes
        n = min(n_k, n_l)
        if n_k > n:
            idx = np.random.choice(n_k, n, replace=False)
            K = K[np.ix_(idx, idx)]
        if n_l > n:
            idx = np.random.choice(n_l, n, replace=False)
            L = L[np.ix_(idx, idx)]
    else:
        n = n_k

    if n < 4:
        # Too few samples for debiased estimator
        return 0.0

    # Center the Gram matrices
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)

    ones = np.ones(n)

    # Debiased HSIC estimator
    term1 = np.trace(K @ L)
    term2 = (ones @ K @ ones) * (ones @ L @ ones) / ((n - 1) * (n - 2))
    term3 = 2 * (ones @ K @ L @ ones) / (n - 2)

    hsic_kl = (term1 + term2 - term3) / (n * (n - 3))
    hsic_kk = (np.trace(K @ K) + (ones @ K @ ones) ** 2 / ((n - 1) * (n - 2))
               - 2 * (ones @ K @ K @ ones) / (n - 2)) / (n * (n - 3))
    hsic_ll = (np.trace(L @ L) + (ones @ L @ ones) ** 2 / ((n - 1) * (n - 2))
               - 2 * (ones @ L @ L @ ones) / (n - 2)) / (n * (n - 3))

    denom = np.sqrt(max(hsic_kk, 0)) * np.sqrt(max(hsic_ll, 0))
    if denom < 1e-10:
        return 0.0

    return float(np.clip(hsic_kl / denom, 0.0, 1.0))


def _minibatch_cka(
    H_v: np.ndarray,
    H_t: np.ndarray,
    batch_size: int = 256,
    num_batches: int = 10,
) -> float:
    """Minibatch CKA for differently-sized token sets.

    Randomly samples matching-size subsets and averages CKA across batches.
    """
    n = min(H_v.shape[0], H_t.shape[0], batch_size)
    cka_values = []

    for _ in range(num_batches):
        idx_v = np.random.choice(H_v.shape[0], n, replace=False)
        idx_t = np.random.choice(H_t.shape[0], n, replace=False)

        Hv_sub = H_v[idx_v]
        Ht_sub = H_t[idx_t]

        YtX = Ht_sub.T @ Hv_sub
        numerator = np.linalg.norm(YtX, 'fro') ** 2
        denom1 = np.linalg.norm(Hv_sub.T @ Hv_sub, 'fro')
        denom2 = np.linalg.norm(Ht_sub.T @ Ht_sub, 'fro')

        if denom1 * denom2 > 1e-10:
            cka_values.append(numerator / (denom1 * denom2))

    if not cka_values:
        return 0.0
    return float(np.mean(cka_values))


def compute_cka_per_layer(
    visual_activations: dict[int, torch.Tensor],
    text_activations: dict[int, torch.Tensor],
    debiased: bool = True,
) -> dict[int, float]:
    """Compute CKA between visual and text tokens at each layer.

    Args:
        visual_activations: {layer_idx: (N_v, d)} visual hidden states.
        text_activations: {layer_idx: (N_t, d)} text hidden states.
        debiased: Use debiased HSIC estimator.

    Returns:
        {layer_idx: CKA value}
    """
    common_layers = set(visual_activations.keys()) & set(text_activations.keys())
    return {
        layer_idx: compute_linear_cka(
            visual_activations[layer_idx],
            text_activations[layer_idx],
            debiased=debiased,
        )
        for layer_idx in sorted(common_layers)
    }
