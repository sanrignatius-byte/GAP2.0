"""Linear probing utilities for Phase 2 assimilation experiments.

This module measures whether visual information becomes decodable from
selected token states (e.g., the last instruction token) as depth increases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.models.hooks import ActivationExtractor


YES_NO_MAP = {
    "yes": 1,
    "no": 0,
}

COMMON_COLORS = {
    "red",
    "blue",
    "green",
    "yellow",
    "black",
    "white",
    "orange",
    "purple",
    "pink",
    "brown",
    "gray",
    "grey",
}


@dataclass
class ProbeMetrics:
    """Probe accuracy summary for one layer."""

    layer_idx: int
    mean_accuracy: float
    std_accuracy: float
    num_samples: int
    num_folds: int


@dataclass
class ProbeRunResult:
    """Collection of probe metrics across layers."""

    target_name: str
    metrics: list[ProbeMetrics]

    def to_dict(self) -> dict:
        return {
            "target_name": self.target_name,
            "layers": [m.layer_idx for m in self.metrics],
            "mean_accuracy": [m.mean_accuracy for m in self.metrics],
            "std_accuracy": [m.std_accuracy for m in self.metrics],
            "num_samples": self.metrics[0].num_samples if self.metrics else 0,
            "num_folds": self.metrics[0].num_folds if self.metrics else 0,
        }


def infer_probe_label(sample: dict, task: str = "yes_no") -> Optional[int]:
    """Infer a small closed-set label from sample answer text.

    Args:
        sample: Normalized dataset sample with an ``answer`` field.
        task: One of ``yes_no`` or ``color``.

    Returns:
        Integer class label, or None when sample cannot be mapped.
    """
    answer = str(sample.get("answer", "")).strip().lower()
    if not answer:
        return None

    if task == "yes_no":
        return YES_NO_MAP.get(answer)

    if task == "color":
        if answer in COMMON_COLORS:
            color_list = sorted(list(COMMON_COLORS))
            return color_list.index(answer)
        return None

    raise ValueError(f"Unsupported probe task: {task}")


def default_probe_indices(
    input_ids: torch.Tensor,
    visual_token_range: tuple[int, int],
    probe_targets: list[str],
) -> dict[str, int]:
    """Select token indices for requested probe targets.

    Conventions:
      - visual_token: midpoint index inside visual token range
      - text_instruction_token: last text token in prompt (outside visual range)
      - answer_token: fallback to final prompt token (no teacher-forced answer token)
    """
    seq_len = int(input_ids.shape[-1])
    vstart, vend = visual_token_range

    valid_text = [idx for idx in range(seq_len) if not (vstart <= idx < vend)]
    if len(valid_text) == 0:
        raise ValueError("No text token positions available outside visual range")

    target_to_idx: dict[str, int] = {}
    for target in probe_targets:
        if target == "visual_token":
            target_to_idx[target] = int((vstart + vend - 1) // 2)
        elif target == "text_instruction_token":
            target_to_idx[target] = int(valid_text[-1])
        elif target == "answer_token":
            target_to_idx[target] = seq_len - 1
        else:
            raise ValueError(f"Unsupported probe target: {target}")

    return target_to_idx


class TextTokenProbeExperiment:
    """Extract per-layer features and run linear probes."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._extractor = ActivationExtractor(
            model=model,
            visual_token_range=None,
            detach=True,
            store_on_cpu=True,
        )

    @torch.no_grad()
    def extract_features(
        self,
        samples: list[dict],
        prepare_inputs_fn: Callable[[dict], dict],
        visual_range_fn: Callable[[dict, dict], tuple[int, int]],
        label_fn: Callable[[dict], Optional[int]],
        probe_targets: list[str],
        token_index_fn: Callable[[torch.Tensor, tuple[int, int], list[str]], dict[str, int]] = default_probe_indices,
        show_progress: bool = True,
    ) -> tuple[dict[str, dict[int, np.ndarray]], np.ndarray, list[str]]:
        """Extract hidden-state features for probe targets across layers.

        Returns:
            features[target][layer] -> (N, d) numpy array
            labels -> (N,) numpy array
            sample_ids -> list[str]
        """
        if len(samples) == 0:
            raise ValueError("No samples provided for probing")

        layers = list(range(self._extractor.num_layers))
        buffers: dict[str, dict[int, list[np.ndarray]]] = {
            t: {l: [] for l in layers} for t in probe_targets
        }

        labels: list[int] = []
        sample_ids: list[str] = []

        iterator = tqdm(samples, desc="Probe feature extraction", disable=not show_progress)
        skipped = 0

        for sample in iterator:
            label = label_fn(sample)
            if label is None:
                skipped += 1
                continue

            model_inputs = prepare_inputs_fn(sample)
            visual_range = visual_range_fn(sample, model_inputs)
            target_indices = token_index_fn(
                model_inputs["input_ids"],
                visual_range,
                probe_targets,
            )

            with self._extractor:
                self.model(**model_inputs)
                acts = self._extractor.get_activations()

            for layer_idx, layer_act in acts.items():
                h = layer_act.full  # (seq_len, hidden_dim) on CPU
                for target in probe_targets:
                    idx = target_indices[target]
                    if idx < 0 or idx >= h.shape[0]:
                        raise IndexError(
                            f"Target index out of bounds: target={target}, idx={idx}, seq_len={h.shape[0]}"
                        )
                    buffers[target][layer_idx].append(h[idx].numpy())

            labels.append(int(label))
            sample_ids.append(sample.get("id", f"sample_{len(sample_ids)}"))

        logger.info(
            "Probe extraction complete: kept {} / {} samples (skipped={})",
            len(labels),
            len(samples),
            skipped,
        )

        if len(labels) == 0:
            raise RuntimeError("No valid labeled samples left after filtering")

        features: dict[str, dict[int, np.ndarray]] = {t: {} for t in probe_targets}
        for target in probe_targets:
            for layer_idx in layers:
                vecs = buffers[target][layer_idx]
                if len(vecs) != len(labels):
                    raise RuntimeError(
                        f"Feature/label length mismatch for target={target}, layer={layer_idx}: "
                        f"features={len(vecs)}, labels={len(labels)}"
                    )
                features[target][layer_idx] = np.stack(vecs, axis=0)

        return features, np.asarray(labels, dtype=np.int64), sample_ids

    def run_linear_probe(
        self,
        features_per_layer: dict[int, np.ndarray],
        labels: np.ndarray,
        target_name: str,
        cv_folds: int = 5,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> ProbeRunResult:
        """Run layer-wise logistic-regression probes with stratified CV."""
        unique_classes = np.unique(labels)
        if unique_classes.shape[0] < 2:
            raise ValueError("Need at least two classes for linear probing")

        class_counts = [int((labels == c).sum()) for c in unique_classes]
        max_allowed_folds = min(class_counts)
        if max_allowed_folds < 2:
            raise ValueError("Insufficient class balance for stratified cross-validation")

        folds = min(cv_folds, max_allowed_folds)
        if folds < cv_folds:
            logger.warning(
                "Reducing cv_folds from {} to {} due to class counts {}",
                cv_folds,
                folds,
                class_counts,
            )

        metrics: list[ProbeMetrics] = []

        for layer_idx in sorted(features_per_layer.keys()):
            X = features_per_layer[layer_idx]
            y = labels

            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
            scores = []

            for train_idx, test_idx in skf.split(X, y):
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=max_iter,
                            random_state=random_state,
                            class_weight="balanced",
                        ),
                    ),
                ])
                clf.fit(X[train_idx], y[train_idx])
                scores.append(float(clf.score(X[test_idx], y[test_idx])))

            metrics.append(
                ProbeMetrics(
                    layer_idx=layer_idx,
                    mean_accuracy=float(np.mean(scores)),
                    std_accuracy=float(np.std(scores)),
                    num_samples=int(len(y)),
                    num_folds=int(folds),
                )
            )

        return ProbeRunResult(target_name=target_name, metrics=metrics)
