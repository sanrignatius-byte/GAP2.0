"""Effective Visual Depth (EVD) computation and analysis.

EVD = max{l : Delta^(l) >= tau}

where Delta^(l) is the average causal effect of corrupting visual tokens at layer l.
"""

from __future__ import annotations

import numpy as np
from typing import Optional
from loguru import logger

from src.causal.patching import CausalTraceResult


def compute_evd(
    trace_result: CausalTraceResult,
    threshold: float = 0.01,
) -> int:
    """Compute Effective Visual Depth from a single causal trace.

    Args:
        trace_result: Causal trace result for a single sample.
        threshold: tau — minimum Delta value for a layer to count.

    Returns:
        EVD (layer index).
    """
    evd = 0
    for effect in sorted(trace_result.effects, key=lambda e: e.layer_idx):
        if effect.delta >= threshold:
            evd = effect.layer_idx
    return evd


def compute_evd_batch(
    trace_results: list[CausalTraceResult],
    threshold: float = 0.01,
) -> np.ndarray:
    """Compute EVD for a batch of causal traces.

    Returns:
        Array of EVD values, one per sample.
    """
    return np.array([compute_evd(r, threshold) for r in trace_results])


def compute_mean_delta_curve(
    trace_results: list[CausalTraceResult],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean and std of Delta curves across samples.

    Returns:
        (layer_indices, mean_deltas, std_deltas)
    """
    if not trace_results:
        return np.array([]), np.array([]), np.array([])

    all_deltas = np.stack([r.delta_curve for r in trace_results])
    layers = trace_results[0].layer_indices

    return layers, all_deltas.mean(axis=0), all_deltas.std(axis=0)


def estimate_cliff_boundary(
    trace_results: list[CausalTraceResult],
    threshold: float = 0.01,
    method: str = "median_evd",
) -> int:
    """Estimate the Modality Cliff boundary layer.

    Args:
        trace_results: List of causal trace results.
        threshold: EVD threshold.
        method: Estimation method:
            - "median_evd": Median EVD across samples.
            - "gradient": Layer with steepest Delta decline.
            - "threshold_crossing": First layer where mean Delta drops below threshold.

    Returns:
        Estimated cliff boundary layer index.
    """
    if method == "median_evd":
        evds = compute_evd_batch(trace_results, threshold)
        return int(np.median(evds))

    elif method == "gradient":
        layers, mean_deltas, _ = compute_mean_delta_curve(trace_results)
        if len(mean_deltas) < 2:
            return 0
        gradients = np.diff(mean_deltas)
        # Steepest decline = most negative gradient
        return int(layers[np.argmin(gradients)])

    elif method == "threshold_crossing":
        layers, mean_deltas, _ = compute_mean_delta_curve(trace_results)
        crossing_indices = np.where(mean_deltas < threshold)[0]
        if len(crossing_indices) == 0:
            return int(layers[-1])
        return int(layers[crossing_indices[0]])

    else:
        raise ValueError(f"Unknown method: {method}")


def evd_performance_correlation(
    evd_values: np.ndarray,
    accuracy_values: np.ndarray,
) -> dict:
    """Compute correlation between EVD and downstream accuracy.

    Used for Checkpoint 2: Does higher EVD correlate with better performance
    on visually demanding tasks?

    Returns:
        Dict with 'spearman_rho', 'spearman_p', 'pearson_r', 'pearson_p'.
    """
    from scipy.stats import spearmanr, pearsonr

    spearman_rho, spearman_p = spearmanr(evd_values, accuracy_values)
    pearson_r, pearson_p = pearsonr(evd_values, accuracy_values)

    result = {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }

    logger.info(
        f"EVD-Performance correlation: "
        f"Spearman rho={spearman_rho:.4f} (p={spearman_p:.4f}), "
        f"Pearson r={pearson_r:.4f} (p={pearson_p:.4f})"
    )

    return result
