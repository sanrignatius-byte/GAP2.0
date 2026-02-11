"""Subset sampling for hard/easy task categorization.

Samples subsets of evaluation data stratified by expected visual reasoning
difficulty. This is critical for Checkpoint 1: proving that the Modality Cliff
causes real performance failures on hard (but not easy) tasks.

Hard tasks: Require spatial reasoning, cross-region comparison, fine-grained detail.
Easy tasks: Object presence, simple attribute identification, basic color/shape.
"""

from __future__ import annotations

import random
from typing import Optional
from loguru import logger


# Keywords/patterns that indicate hard visual reasoning
HARD_INDICATORS = {
    "chartqa": [
        "compare", "difference", "between", "trend", "increase", "decrease",
        "higher", "lower", "which", "how many", "relationship", "correlation",
        "maximum", "minimum", "average", "total", "percentage",
    ],
    "docvqa": [
        "table", "which column", "which row", "address", "date", "between",
        "compare", "total", "sum", "layout", "header", "footer",
        "section", "paragraph", "figure",
    ],
    "general": [
        "spatial", "left", "right", "above", "below", "between", "behind",
        "in front of", "next to", "how many", "count", "compare",
        "relationship", "position", "location", "arrangement",
    ],
}

# Keywords/patterns that indicate easy visual reasoning
EASY_INDICATORS = [
    "what color", "what is this", "is there", "yes or no",
    "what type", "what kind", "what animal", "what object",
]


def sample_hard_easy_subsets(
    samples: list[dict],
    n_hard: int = 100,
    n_easy: int = 100,
    seed: int = 42,
    dataset_name: Optional[str] = None,
) -> tuple[list[dict], list[dict]]:
    """Sample hard and easy subsets from a dataset.

    Uses question text heuristics to classify difficulty.
    Falls back to random sampling if not enough samples match.

    Args:
        samples: Full list of sample dicts.
        n_hard: Number of hard samples to select.
        n_easy: Number of easy samples to select.
        seed: Random seed for reproducibility.
        dataset_name: Dataset name (for dataset-specific indicators).

    Returns:
        (hard_samples, easy_samples)
    """
    rng = random.Random(seed)

    hard_candidates = []
    easy_candidates = []
    ambiguous = []

    # Get dataset-specific hard indicators
    ds_indicators = HARD_INDICATORS.get(
        dataset_name.lower() if dataset_name else "", []
    )
    all_hard_indicators = ds_indicators + HARD_INDICATORS.get("general", [])

    for sample in samples:
        q = sample["question"].lower()
        score = _compute_difficulty_score(q, all_hard_indicators, EASY_INDICATORS)

        if score > 0:
            hard_candidates.append(sample)
        elif score < 0:
            easy_candidates.append(sample)
        else:
            ambiguous.append(sample)

    logger.info(
        f"Difficulty classification: {len(hard_candidates)} hard, "
        f"{len(easy_candidates)} easy, {len(ambiguous)} ambiguous"
    )

    # Sample from candidates, filling from ambiguous if needed
    hard_subset = _sample_with_fallback(
        hard_candidates, ambiguous, n_hard, rng
    )
    easy_subset = _sample_with_fallback(
        easy_candidates, ambiguous, n_easy, rng
    )

    # Tag samples with difficulty labels
    for s in hard_subset:
        s["metadata"]["assigned_difficulty"] = "hard"
    for s in easy_subset:
        s["metadata"]["assigned_difficulty"] = "easy"

    logger.info(f"Sampled {len(hard_subset)} hard, {len(easy_subset)} easy")
    return hard_subset, easy_subset


def _compute_difficulty_score(
    question: str,
    hard_keywords: list[str],
    easy_keywords: list[str],
) -> int:
    """Score a question's visual reasoning difficulty.

    Returns positive for hard, negative for easy, 0 for ambiguous.
    """
    hard_count = sum(1 for kw in hard_keywords if kw in question)
    easy_count = sum(1 for kw in easy_keywords if kw in question)

    # Also consider question length as a weak signal
    # (longer questions tend to require more reasoning)
    length_bonus = 1 if len(question.split()) > 15 else 0

    return (hard_count + length_bonus) - easy_count * 2


def _sample_with_fallback(
    primary: list[dict],
    fallback: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n items from primary, using fallback if needed."""
    if len(primary) >= n:
        return rng.sample(primary, n)
    else:
        result = list(primary)
        remaining = n - len(result)
        if len(fallback) >= remaining:
            result.extend(rng.sample(fallback, remaining))
        else:
            result.extend(fallback)
        return result
