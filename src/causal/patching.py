"""Causal patching for measuring visual token contribution at each layer.

Implements three corruption methods:
  - Zero-out: Replace visual token states with zeros
  - Gaussian noise: Add noise with sigma = std of clean activations
  - Mean substitution: Replace each visual token with the mean of all visual tokens

For each layer l, measures:
  Delta^(l) = E[s(y|x_v, x_t) - s(y|patch(h_v^(l)), x_t)]

where s(.) is the log-probability of the correct answer token.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
from loguru import logger

from src.models.hooks import ActivationExtractor, PatchingHook


@dataclass
class CausalEffect:
    """Causal effect of corrupting visual tokens at a specific layer."""

    layer_idx: int
    delta: float               # s_clean - s_patched
    s_clean: float             # Log-prob of correct answer (clean)
    s_patched: float           # Log-prob of correct answer (patched)
    corruption_method: str


@dataclass
class CausalTraceResult:
    """Full causal trace across all layers for a single sample."""

    sample_id: str
    effects: list[CausalEffect]  # One per layer
    evd: Optional[int] = None    # Effective Visual Depth

    @property
    def delta_curve(self) -> np.ndarray:
        """Return array of Delta values indexed by layer."""
        return np.array([e.delta for e in sorted(self.effects, key=lambda x: x.layer_idx)])

    @property
    def layer_indices(self) -> np.ndarray:
        return np.array([e.layer_idx for e in sorted(self.effects, key=lambda x: x.layer_idx)])


class CausalPatcher:
    """Runs causal patching experiments on an MLLM.

    For each layer, corrupts visual token hidden states and measures the
    change in log-probability of the correct answer.
    """

    CORRUPTION_METHODS = {"zero", "gaussian", "mean_sub"}

    def __init__(
        self,
        model: nn.Module,
        visual_token_range: tuple[int, int],
        corruption_method: str = "gaussian",
        evd_threshold: float = 0.01,
    ):
        """
        Args:
            model: The MLLM in eval mode.
            visual_token_range: (start, end) of visual tokens.
            corruption_method: One of "zero", "gaussian", "mean_sub".
            evd_threshold: Threshold tau for EVD computation.
        """
        if corruption_method not in self.CORRUPTION_METHODS:
            raise ValueError(
                f"Unknown corruption method: {corruption_method}. "
                f"Choose from {self.CORRUPTION_METHODS}"
            )

        self.model = model
        self.visual_token_range = visual_token_range
        self.corruption_method = corruption_method
        self.evd_threshold = evd_threshold

        # Discover transformer layers
        self._extractor = ActivationExtractor(model, visual_token_range)
        self._layers = self._extractor._transformer_layers

    def _make_corruption_fn(
        self,
        method: str,
        clean_visual_states: Optional[torch.Tensor] = None,
    ) -> callable:
        """Create a corruption function for visual token states.

        Args:
            method: Corruption method name.
            clean_visual_states: Clean visual states for computing
                statistics (needed for gaussian and mean_sub).
        """
        if method == "zero":
            return lambda x: torch.zeros_like(x)

        elif method == "gaussian":
            if clean_visual_states is None:
                raise ValueError("Gaussian corruption requires clean_visual_states")
            sigma = clean_visual_states.std().item()

            def gaussian_corrupt(x):
                noise = torch.randn_like(x) * sigma
                return x + noise

            return gaussian_corrupt

        elif method == "mean_sub":
            if clean_visual_states is None:
                raise ValueError("Mean substitution requires clean_visual_states")
            # Mean across all visual tokens -> (1, 1, d)
            mean_state = clean_visual_states.mean(dim=1, keepdim=True)

            def mean_corrupt(x):
                return mean_state.expand_as(x)

            return mean_corrupt

        else:
            raise ValueError(f"Unknown method: {method}")

    @torch.no_grad()
    def _get_answer_logprob(
        self,
        model_inputs: dict,
        answer_token_ids: torch.Tensor,
    ) -> float:
        """Run forward pass and get log-probability of answer tokens.

        Args:
            model_inputs: Dict of model inputs (input_ids, attention_mask,
                pixel_values, etc.)
            answer_token_ids: Token IDs of the correct answer.

        Returns:
            Sum of log-probabilities of the answer tokens.
        """
        outputs = self.model(**model_inputs)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

        # Get log-probs at the positions where answer tokens should appear
        log_probs = torch.log_softmax(logits[0, -len(answer_token_ids) - 1:-1, :], dim=-1)

        answer_logprob = 0.0
        for i, token_id in enumerate(answer_token_ids):
            answer_logprob += log_probs[i, token_id].item()

        return answer_logprob

    @torch.no_grad()
    def trace_single_sample(
        self,
        model_inputs: dict,
        answer_token_ids: torch.Tensor,
        sample_id: str = "",
        layers: Optional[list[int]] = None,
    ) -> CausalTraceResult:
        """Run causal tracing for a single sample across all layers.

        Args:
            model_inputs: Model inputs dict.
            answer_token_ids: Correct answer token IDs.
            sample_id: Identifier for this sample.
            layers: Specific layers to trace. None = all layers.

        Returns:
            CausalTraceResult with per-layer effects and computed EVD.
        """
        target_layers = layers if layers is not None else list(range(len(self._layers)))

        # Step 1: Clean forward pass
        s_clean = self._get_answer_logprob(model_inputs, answer_token_ids)

        # Step 2: Get clean activations for computing corruption statistics
        with self._extractor:
            self.model(**model_inputs)
            clean_acts = self._extractor.get_visual_activations()

        # Step 3: Patch each layer and measure effect
        effects = []
        for layer_idx in target_layers:
            # Get clean visual states at this layer for corruption fn
            clean_visual = clean_acts.get(layer_idx)
            if clean_visual is not None:
                clean_visual = clean_visual.unsqueeze(0).to(self.model.device)

            corruption_fn = self._make_corruption_fn(
                self.corruption_method, clean_visual
            )

            patching_hook = PatchingHook(
                model=self.model,
                target_layer_idx=layer_idx,
                corruption_fn=corruption_fn,
                visual_token_range=self.visual_token_range,
                transformer_layers=self._layers,
            )

            with patching_hook:
                s_patched = self._get_answer_logprob(model_inputs, answer_token_ids)

            delta = s_clean - s_patched

            effects.append(CausalEffect(
                layer_idx=layer_idx,
                delta=delta,
                s_clean=s_clean,
                s_patched=s_patched,
                corruption_method=self.corruption_method,
            ))

        result = CausalTraceResult(sample_id=sample_id, effects=effects)
        result.evd = self._compute_evd(result)

        return result

    def trace_dataset(
        self,
        samples: list[dict],
        prepare_fn: callable,
        layers: Optional[list[int]] = None,
        show_progress: bool = True,
    ) -> list[CausalTraceResult]:
        """Run causal tracing across a dataset.

        Args:
            samples: List of sample dicts (each must have 'id', 'image',
                'question', 'answer' fields).
            prepare_fn: Function(sample) -> (model_inputs, answer_token_ids).
            layers: Layers to trace. None = all.
            show_progress: Show tqdm progress bar.

        Returns:
            List of CausalTraceResult, one per sample.
        """
        results = []
        iterator = tqdm(samples, desc="Causal tracing") if show_progress else samples

        for sample in iterator:
            model_inputs, answer_token_ids = prepare_fn(sample)
            result = self.trace_single_sample(
                model_inputs=model_inputs,
                answer_token_ids=answer_token_ids,
                sample_id=sample.get("id", ""),
                layers=layers,
            )
            results.append(result)

        return results

    def _compute_evd(self, result: CausalTraceResult) -> int:
        """Compute Effective Visual Depth from causal trace result.

        EVD = max{l : Delta^(l) >= tau}
        """
        evd = 0
        for effect in sorted(result.effects, key=lambda e: e.layer_idx):
            if effect.delta >= self.evd_threshold:
                evd = effect.layer_idx
        return evd
