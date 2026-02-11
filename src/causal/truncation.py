"""Visual token truncation experiment.

At each layer l, remove visual tokens entirely from the residual stream
for layers > l and measure downstream task accuracy.

This validates Checkpoint 1: Does truncation at the cliff boundary
significantly hurt performance on hard visual reasoning tasks?
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from loguru import logger

from src.models.hooks import TruncationHook, ActivationExtractor


@dataclass
class TruncationResult:
    """Result of truncating visual tokens at a specific layer."""

    truncation_layer: int
    accuracy: float
    num_correct: int
    num_total: int
    per_sample_correct: list[bool]


class TruncationExperiment:
    """Run progressive visual token truncation at different layers.

    For each truncation point l:
      - Register hooks that zero out visual tokens at layers > l
      - Run model on evaluation set
      - Measure accuracy

    Compare truncation curves for 'hard' vs 'easy' task subsets.
    """

    def __init__(
        self,
        model: nn.Module,
        visual_token_range: tuple[int, int],
    ):
        self.model = model
        self.visual_token_range = visual_token_range

        temp = ActivationExtractor(model, visual_token_range)
        self._layers = temp._transformer_layers

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    @torch.no_grad()
    def run_truncation_sweep(
        self,
        samples: list[dict],
        prepare_fn: callable,
        evaluate_fn: callable,
        truncation_layers: Optional[list[int]] = None,
        show_progress: bool = True,
    ) -> list[TruncationResult]:
        """Run truncation at each layer and measure accuracy.

        Args:
            samples: List of sample dicts.
            prepare_fn: Function(sample) -> model_inputs dict.
            evaluate_fn: Function(model, model_inputs, sample) -> bool (correct/incorrect).
            truncation_layers: Layers to truncate at. None = all layers.
            show_progress: Show progress bar.

        Returns:
            List of TruncationResult, one per truncation layer.
        """
        if truncation_layers is None:
            truncation_layers = list(range(self.num_layers))

        results = []

        # Also run baseline (no truncation)
        logger.info("Running baseline (no truncation)")
        baseline_correct = []
        for sample in tqdm(samples, desc="Baseline", disable=not show_progress):
            model_inputs = prepare_fn(sample)
            correct = evaluate_fn(self.model, model_inputs, sample)
            baseline_correct.append(correct)

        baseline_acc = sum(baseline_correct) / len(baseline_correct)
        results.append(TruncationResult(
            truncation_layer=-1,  # -1 means no truncation
            accuracy=baseline_acc,
            num_correct=sum(baseline_correct),
            num_total=len(baseline_correct),
            per_sample_correct=baseline_correct,
        ))
        logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Run truncation at each layer
        for trunc_layer in tqdm(
            truncation_layers,
            desc="Truncation sweep",
            disable=not show_progress,
        ):
            per_sample = []

            trunc_hook = TruncationHook(
                model=self.model,
                truncation_layer=trunc_layer,
                visual_token_range=self.visual_token_range,
                transformer_layers=self._layers,
            )

            with trunc_hook:
                for sample in samples:
                    model_inputs = prepare_fn(sample)
                    correct = evaluate_fn(self.model, model_inputs, sample)
                    per_sample.append(correct)

            acc = sum(per_sample) / len(per_sample)
            results.append(TruncationResult(
                truncation_layer=trunc_layer,
                accuracy=acc,
                num_correct=sum(per_sample),
                num_total=len(per_sample),
                per_sample_correct=per_sample,
            ))
            logger.info(f"Layer {trunc_layer}: accuracy={acc:.4f}")

        return results

    @staticmethod
    def compute_accuracy_drop(
        results: list[TruncationResult],
    ) -> dict[int, float]:
        """Compute accuracy drop relative to baseline for each truncation layer.

        Returns:
            Dict mapping truncation_layer -> accuracy_drop (positive = worse).
        """
        baseline = next(r for r in results if r.truncation_layer == -1)
        drops = {}
        for r in results:
            if r.truncation_layer >= 0:
                drops[r.truncation_layer] = baseline.accuracy - r.accuracy
        return drops
