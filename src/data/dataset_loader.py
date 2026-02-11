"""Dataset loading utilities for GAP 2.0 evaluation.

Loads and prepares datasets for the truncation and causal tracing experiments.
Supports: ChartQA, DocVQA, TextVQA, VQAv2, ScienceQA.
"""

from __future__ import annotations

import os
from typing import Optional
from loguru import logger


def load_dataset_for_eval(
    dataset_name: str,
    split: str = "validation",
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> list[dict]:
    """Load a dataset and return normalized sample dicts.

    Each returned sample has the schema:
        {
            "id": str,
            "image": PIL.Image,
            "question": str,
            "answer": str,
            "metadata": dict  (dataset-specific fields)
        }

    Args:
        dataset_name: One of "chartqa", "docvqa", "textvqa", "vqav2", "scienceqa".
        split: Dataset split.
        cache_dir: HuggingFace cache directory.
        max_samples: Limit number of samples loaded.

    Returns:
        List of normalized sample dicts.
    """
    loaders = {
        "chartqa": _load_chartqa,
        "docvqa": _load_docvqa,
        "textvqa": _load_textvqa,
        "vqav2": _load_vqav2,
        "scienceqa": _load_scienceqa,
    }

    name = dataset_name.lower().strip()
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(loaders.keys())}")

    logger.info(f"Loading {name} ({split}), max_samples={max_samples}")
    samples = loaders[name](split, cache_dir)

    if max_samples is not None:
        samples = samples[:max_samples]

    logger.info(f"Loaded {len(samples)} samples from {name}")
    return samples


def _load_chartqa(split: str, cache_dir: Optional[str]) -> list[dict]:
    """Load ChartQA dataset."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/ChartQA", split=split, cache_dir=cache_dir)

    samples = []
    for i, item in enumerate(ds):
        samples.append({
            "id": f"chartqa_{i}",
            "image": item["image"],
            "question": item["query"],
            "answer": item["label"][0] if isinstance(item["label"], list) else str(item["label"]),
            "metadata": {
                "dataset": "chartqa",
                "difficulty": "hard",  # Chart reasoning requires deep visual info
            },
        })
    return samples


def _load_docvqa(split: str, cache_dir: Optional[str]) -> list[dict]:
    """Load DocVQA dataset."""
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/DocVQA", split=split, cache_dir=cache_dir)

    samples = []
    for i, item in enumerate(ds):
        answers = item.get("answers", [item.get("answer", "")])
        if isinstance(answers, list):
            answer = answers[0] if answers else ""
        else:
            answer = str(answers)

        samples.append({
            "id": f"docvqa_{i}",
            "image": item["image"],
            "question": item["question"],
            "answer": answer,
            "metadata": {
                "dataset": "docvqa",
                "difficulty": "hard",
            },
        })
    return samples


def _load_textvqa(split: str, cache_dir: Optional[str]) -> list[dict]:
    """Load TextVQA dataset."""
    from datasets import load_dataset

    ds = load_dataset("facebook/textvqa", split=split, cache_dir=cache_dir)

    samples = []
    for i, item in enumerate(ds):
        answers = item.get("answers", [])
        answer = answers[0] if answers else ""

        samples.append({
            "id": f"textvqa_{i}",
            "image": item["image"],
            "question": item["question"],
            "answer": answer,
            "metadata": {
                "dataset": "textvqa",
                "difficulty": "medium",
            },
        })
    return samples


def _load_vqav2(split: str, cache_dir: Optional[str]) -> list[dict]:
    """Load VQAv2 dataset (simple subset for control group)."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/VQAv2", split=split, cache_dir=cache_dir)

    samples = []
    for i, item in enumerate(ds):
        # VQAv2 has multiple answers; take the most common
        answers = item.get("answers", [])
        if isinstance(answers, list) and len(answers) > 0:
            if isinstance(answers[0], dict):
                answer = answers[0].get("answer", "")
            else:
                answer = str(answers[0])
        else:
            answer = str(item.get("multiple_choice_answer", ""))

        samples.append({
            "id": f"vqav2_{i}",
            "image": item["image"],
            "question": item["question"],
            "answer": answer,
            "metadata": {
                "dataset": "vqav2",
                "difficulty": "easy",
            },
        })
    return samples


def _load_scienceqa(split: str, cache_dir: Optional[str]) -> list[dict]:
    """Load ScienceQA (image subset only)."""
    from datasets import load_dataset

    ds = load_dataset("derek-thomas/ScienceQA", split=split, cache_dir=cache_dir)

    samples = []
    for i, item in enumerate(ds):
        # Only keep samples with images
        if item.get("image") is None:
            continue

        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)
        answer = choices[answer_idx] if answer_idx < len(choices) else ""

        samples.append({
            "id": f"scienceqa_{i}",
            "image": item["image"],
            "question": item["question"],
            "answer": answer,
            "metadata": {
                "dataset": "scienceqa",
                "difficulty": "medium-hard",
                "subject": item.get("subject", ""),
                "topic": item.get("topic", ""),
            },
        })
    return samples
