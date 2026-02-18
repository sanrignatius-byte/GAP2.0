"""Visualization pipeline for GAP 2.0 experiments.

Generates all key plots:
  - Layer-wise causal effect curves (Delta^(l) vs l)
  - Effective rank decay curves (visual vs text tokens)
  - Cosine concentration curves
  - Cross-modal CKA curves
  - EVD vs accuracy scatter plots
  - Truncation accuracy curves (hard vs easy tasks)
  - Singular value spectra
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional
from loguru import logger

# Use non-interactive backend for server environments
matplotlib.use("Agg")


class GAPVisualizer:
    """Generates all GAP 2.0 experiment visualizations."""

    def __init__(
        self,
        output_dir: str = "./results/plots",
        figsize: tuple = (12, 8),
        dpi: int = 150,
        style: str = "seaborn-v0_8-paper",
        font_size: int = 12,
    ):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        self.style = style
        self.font_size = font_size

        os.makedirs(output_dir, exist_ok=True)

        plt.rcParams.update({
            "font.size": font_size,
            "axes.labelsize": font_size + 2,
            "axes.titlesize": font_size + 4,
            "legend.fontsize": font_size - 1,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
        })

    def plot_causal_effect_curve(
        self,
        layers: np.ndarray,
        mean_deltas: np.ndarray,
        std_deltas: np.ndarray,
        cliff_boundary: Optional[int] = None,
        title: str = "Causal Effect of Visual Token Corruption",
        filename: str = "causal_effect_curve.pdf",
        task_label: Optional[str] = None,
    ) -> str:
        """Plot Delta^(l) vs layer depth with confidence band.

        Args:
            layers: Array of layer indices.
            mean_deltas: Mean Delta values per layer.
            std_deltas: Std of Delta values per layer.
            cliff_boundary: Estimated cliff layer (drawn as vertical line).
            title: Plot title.
            filename: Output filename.
            task_label: Label for legend.

        Returns:
            Path to saved plot.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        label = task_label or "Mean $\\Delta^{(l)}$"
        ax.plot(layers, mean_deltas, "b-o", linewidth=2, markersize=4, label=label)
        ax.fill_between(
            layers,
            mean_deltas - std_deltas,
            mean_deltas + std_deltas,
            alpha=0.2,
            color="blue",
        )

        if cliff_boundary is not None:
            ax.axvline(
                x=cliff_boundary,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Cliff boundary (layer {cliff_boundary})",
            )

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("$\\Delta^{(l)}$ (Causal Effect)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_causal_effect_comparison(
        self,
        layers: np.ndarray,
        hard_deltas: np.ndarray,
        easy_deltas: np.ndarray,
        hard_std: np.ndarray,
        easy_std: np.ndarray,
        cliff_boundary: Optional[int] = None,
        filename: str = "causal_effect_comparison.pdf",
    ) -> str:
        """Plot causal effect curves for hard vs easy tasks (Checkpoint 1)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(layers, hard_deltas, "r-o", linewidth=2, markersize=4, label="Hard tasks")
        ax.fill_between(layers, hard_deltas - hard_std, hard_deltas + hard_std, alpha=0.15, color="red")

        ax.plot(layers, easy_deltas, "b-s", linewidth=2, markersize=4, label="Easy tasks")
        ax.fill_between(layers, easy_deltas - easy_std, easy_deltas + easy_std, alpha=0.15, color="blue")

        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", linewidth=2,
                       label=f"Cliff boundary (layer {cliff_boundary})")

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("$\\Delta^{(l)}$ (Causal Effect)")
        ax.set_title("Causal Effect: Hard vs Easy Visual Tasks")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_effective_rank(
        self,
        layers: np.ndarray,
        visual_er: np.ndarray,
        text_er: np.ndarray,
        cliff_boundary: Optional[int] = None,
        filename: str = "effective_rank.pdf",
    ) -> str:
        """Plot effective rank decay: visual vs text tokens (Checkpoint 3)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(layers, visual_er, "r-o", linewidth=2, markersize=5, label="Visual tokens")
        ax.plot(layers, text_er, "b-s", linewidth=2, markersize=5, label="Text tokens")

        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", linewidth=2,
                       label=f"Cliff boundary (layer {cliff_boundary})")

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("Effective Rank")
        ax.set_title("Effective Rank Across Layers")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_cosine_concentration(
        self,
        layers: np.ndarray,
        visual_icc: np.ndarray,
        text_icc: np.ndarray,
        cliff_boundary: Optional[int] = None,
        filename: str = "cosine_concentration.pdf",
    ) -> str:
        """Plot inter-token cosine concentration: visual vs text (Checkpoint 3)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(layers, visual_icc, "r-o", linewidth=2, markersize=5, label="Visual tokens")
        ax.plot(layers, text_icc, "b-s", linewidth=2, markersize=5, label="Text tokens")

        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", linewidth=2,
                       label=f"Cliff boundary (layer {cliff_boundary})")

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("Mean Pairwise Cosine Similarity")
        ax.set_title("Inter-token Cosine Concentration Across Layers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_cka_curve(
        self,
        layers: np.ndarray,
        cka_values: np.ndarray,
        cliff_boundary: Optional[int] = None,
        filename: str = "cross_modal_cka.pdf",
    ) -> str:
        """Plot cross-modal CKA across layers (Checkpoint 3)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(layers, cka_values, "g-D", linewidth=2, markersize=5,
                label="CKA(visual, text)")

        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", linewidth=2,
                       label=f"Cliff boundary (layer {cliff_boundary})")

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("CKA")
        ax.set_title("Cross-modal CKA (Visual ↔ Text) Across Layers")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_geometric_dashboard(
        self,
        layers: np.ndarray,
        visual_er: np.ndarray,
        text_er: np.ndarray,
        visual_icc: np.ndarray,
        text_icc: np.ndarray,
        cka_values: np.ndarray,
        cliff_boundary: Optional[int] = None,
        filename: str = "geometric_dashboard.pdf",
    ) -> str:
        """Combined 3-panel dashboard of all geometric metrics (Checkpoint 3)."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: Effective Rank
        ax = axes[0]
        ax.plot(layers, visual_er, "r-o", linewidth=2, markersize=4, label="Visual")
        ax.plot(layers, text_er, "b-s", linewidth=2, markersize=4, label="Text")
        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Effective Rank")
        ax.set_title("(a) Effective Rank")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 2: Cosine Concentration
        ax = axes[1]
        ax.plot(layers, visual_icc, "r-o", linewidth=2, markersize=4, label="Visual")
        ax.plot(layers, text_icc, "b-s", linewidth=2, markersize=4, label="Text")
        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Cosine Similarity")
        ax.set_title("(b) Cosine Concentration")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Panel 3: CKA
        ax = axes[2]
        ax.plot(layers, cka_values, "g-D", linewidth=2, markersize=4, label="CKA(V, T)")
        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_ylabel("CKA")
        ax.set_title("(c) Cross-modal CKA")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("Geometric Collapse Indicators Across Layers", fontsize=16, y=1.02)
        fig.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_truncation_curves(
        self,
        layers: np.ndarray,
        hard_accuracy: np.ndarray,
        easy_accuracy: np.ndarray,
        baseline_hard: float,
        baseline_easy: float,
        cliff_boundary: Optional[int] = None,
        filename: str = "truncation_curves.pdf",
    ) -> str:
        """Plot truncation accuracy curves for hard vs easy tasks (Checkpoint 1)."""
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(layers, hard_accuracy, "r-o", linewidth=2, markersize=4, label="Hard tasks")
        ax.plot(layers, easy_accuracy, "b-s", linewidth=2, markersize=4, label="Easy tasks")

        ax.axhline(y=baseline_hard, color="red", linestyle=":", alpha=0.5, label=f"Hard baseline ({baseline_hard:.2%})")
        ax.axhline(y=baseline_easy, color="blue", linestyle=":", alpha=0.5, label=f"Easy baseline ({baseline_easy:.2%})")

        if cliff_boundary is not None:
            ax.axvline(x=cliff_boundary, color="gray", linestyle="--", linewidth=2,
                       label=f"Cliff boundary (layer {cliff_boundary})")

        ax.set_xlabel("Truncation Layer $l$ (visual tokens removed at layers > $l$)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Visual Token Truncation: Hard vs Easy Tasks")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_evd_vs_accuracy(
        self,
        evd_values: np.ndarray,
        accuracy_values: np.ndarray,
        correlation: Optional[dict] = None,
        filename: str = "evd_vs_accuracy.pdf",
    ) -> str:
        """Scatter plot of EVD vs downstream accuracy (Checkpoint 2)."""
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(evd_values, accuracy_values, alpha=0.6, s=40, c="steelblue", edgecolors="navy")

        # Trend line
        if len(evd_values) > 2:
            z = np.polyfit(evd_values, accuracy_values, 1)
            p = np.poly1d(z)
            x_range = np.linspace(evd_values.min(), evd_values.max(), 100)
            ax.plot(x_range, p(x_range), "r--", linewidth=2, alpha=0.7, label="Linear trend")

        if correlation:
            rho = correlation.get("spearman_rho", 0)
            p_val = correlation.get("spearman_p", 1)
            ax.text(
                0.05, 0.95,
                f"Spearman $\\rho$ = {rho:.3f}\n$p$ = {p_val:.4f}",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                fontsize=self.font_size,
            )

        ax.set_xlabel("Effective Visual Depth (EVD)")
        ax.set_ylabel("Accuracy")
        ax.set_title("EVD vs Downstream Performance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_attention_flow(
        self,
        layers: np.ndarray,
        t2v_mean: np.ndarray,
        t2v_std: np.ndarray,
        v2v_mean: np.ndarray,
        uniform_baseline: float,
        readout_window: Optional[tuple] = None,
        filename: str = "attention_flow.pdf",
    ) -> str:
        """Plot text-to-visual attention flow across layers (Phase 2).

        Args:
            layers: Array of layer indices.
            t2v_mean: Mean text-to-visual attention per layer.
            t2v_std: Std of text-to-visual attention per layer.
            v2v_mean: Mean visual-to-visual attention per layer (sanity check).
            uniform_baseline: Expected attention fraction under uniform distribution.
            readout_window: Optional (start, end) layer range to highlight
                (e.g., the Late-Stage Readout window from Phase 1).
            filename: Output filename.

        Returns:
            Path to saved plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Panel 1: Raw t2v with confidence band + uniform baseline
        ax = axes[0]
        ax.plot(layers, t2v_mean, "b-o", linewidth=2, markersize=4,
                label="Text→Visual attention")
        ax.fill_between(
            layers,
            t2v_mean - t2v_std,
            t2v_mean + t2v_std,
            alpha=0.2,
            color="blue",
        )
        ax.axhline(
            y=uniform_baseline,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label=f"Uniform baseline ({uniform_baseline:.3f})",
        )
        ax.plot(layers, v2v_mean, "g--s", linewidth=1.5, markersize=3,
                alpha=0.6, label="Visual→Visual attention")

        if readout_window is not None:
            rw_start, rw_end = readout_window
            ax.axvspan(rw_start, rw_end, alpha=0.12, color="red",
                       label=f"Readout window ({rw_start}-{rw_end})")

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("Mean Attention Weight (to Visual Tokens)")
        ax.set_title("(a) Text→Visual Attention per Layer")
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3)

        # Panel 2: Normalized t2v (ratio to uniform baseline)
        ax = axes[1]
        t2v_norm = t2v_mean / uniform_baseline if uniform_baseline > 0 else t2v_mean
        t2v_norm_std = t2v_std / uniform_baseline if uniform_baseline > 0 else t2v_std

        ax.plot(layers, t2v_norm, "r-o", linewidth=2, markersize=4,
                label="T2V / uniform baseline")
        ax.fill_between(
            layers,
            t2v_norm - t2v_norm_std,
            t2v_norm + t2v_norm_std,
            alpha=0.2,
            color="red",
        )
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5,
                   label="Uniform (ratio=1)")

        if readout_window is not None:
            rw_start, rw_end = readout_window
            ax.axvspan(rw_start, rw_end, alpha=0.12, color="red",
                       label=f"Readout window ({rw_start}-{rw_end})")

        # Annotate peak
        peak_idx = int(np.argmax(t2v_norm))
        ax.annotate(
            f"Peak: layer {layers[peak_idx]}",
            xy=(layers[peak_idx], t2v_norm[peak_idx]),
            xytext=(layers[peak_idx] + 2, t2v_norm[peak_idx] + 0.05),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=self.font_size - 1,
        )

        ax.set_xlabel("Layer $l$")
        ax.set_ylabel("Normalized T2V Attention (ratio to uniform)")
        ax.set_title("(b) Normalized Text→Visual Attention")
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Attention Flow: Text Tokens → Visual Tokens", fontsize=16, y=1.02)
        fig.tight_layout()

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path

    def plot_singular_spectrum(
        self,
        spectra: dict[str, np.ndarray],
        top_k: int = 50,
        filename: str = "singular_spectrum.pdf",
    ) -> str:
        """Plot singular value spectra for different layers.

        Args:
            spectra: Dict mapping "Layer X (visual/text)" -> singular values array.
            top_k: Show top-k singular values.
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra)))
        for (label, sv), color in zip(spectra.items(), colors):
            ax.plot(range(min(len(sv), top_k)), sv[:top_k], linewidth=2,
                    label=label, color=color)

        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Normalized Singular Value")
        ax.set_title("Singular Value Spectra")
        ax.legend(fontsize=self.font_size - 2)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")
        return path
