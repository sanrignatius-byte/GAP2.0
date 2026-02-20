"""Deep-Layer Targeted Visual Masking Experiment (P0/P1 — re-grounding test).

Tests whether deep layers functionally need visual tokens for re-grounding.
Unlike full truncation (which removes visual tokens from ALL layers past a
point), this experiment keeps visual tokens intact in early layers and masks
them ONLY in a specified deep-layer window.

Key hypothesis: If U-shape attention is functional re-grounding (not sink),
then masking visual tokens only in deep layers should hurt accuracy even
though early/mid layers had full visual access.

Experiment conditions:
    1. Baseline: no intervention
    2. Early mask (L0→L_mid):  mask visual in early layers, keep in deep
    3. Deep mask (L_mid→L_end): mask visual in deep layers, keep in early
    4. Full mask (L0→L_end):   mask visual everywhere (= hard truncation at L0)

Usage:
    python scripts/run_deep_layer_masking.py \
      --config configs/default.yaml \
      --model_config configs/qwen_vl_full.yaml \
      --dataset chartqa \
      --boundary_layer 24 \
      --max_samples 200 \
      --output_dir results/deep_layer_masking_chartqa
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.models.hooks import (
    DeepLayerVisualMaskHook,
    ActivationExtractor,
)
from src.data.dataset_loader import load_dataset_for_eval


def _normalize_yes_no(text: str) -> str | None:
    s = re.sub(r"\s+", " ", str(text).strip().lower())
    if not s:
        return None
    if s.startswith("yes"):
        return "yes"
    if s.startswith("no"):
        return "no"
    m = re.search(r"\b(yes|no)\b", s)
    if m:
        return m.group(1)
    return None


def _decode_generation(model, tokenizer, model_inputs: dict, max_new_tokens: int) -> str:
    with torch.no_grad():
        out_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    prompt_len = int(model_inputs["input_ids"].shape[1])
    return tokenizer.decode(out_ids[0][prompt_len:], skip_special_tokens=True).strip()


def _evaluate_condition(
    samples, bundle, device, max_new_tokens, hook=None,
):
    """Evaluate model accuracy under a given hook condition."""
    records = []
    ctx = hook if hook is not None else _nullcontext()
    with ctx:
        for sample in samples:
            gold = _normalize_yes_no(sample["answer"])
            model_inputs, _ = prepare_model_input(
                sample, bundle, device,
                include_answer_in_prompt=False, return_metadata=False,
            )
            generated = _decode_generation(
                bundle.model, bundle.tokenizer, model_inputs, max_new_tokens,
            )
            pred = _normalize_yes_no(generated)
            records.append({
                "id": sample.get("id"),
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "generated": generated,
            })
    acc = sum(r["correct"] for r in records) / len(records) if records else 0.0
    return acc, records


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Deep-Layer Targeted Masking")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="chartqa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument(
        "--boundary_layer", type=int, default=24,
        help="Layer that separates 'early' from 'deep' (default 24).",
    )
    parser.add_argument(
        "--extra_boundaries",
        nargs="*",
        type=int,
        default=None,
        help="Additional boundary layers to sweep (e.g. 12 18 30 36).",
    )
    parser.add_argument(
        "--mask_strategies",
        nargs="+",
        default=["zero", "norm_matched_noise"],
        choices=["zero", "norm_matched_noise", "dataset_mean"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.output_dir:
        cfg.output.results_dir = args.output_dir

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.results_dir, "run_deep_layer_masking.log")
    logger.add(log_path, rotation="10 MB")

    # Load model
    logger.info("Loading model: {}", cfg.model.name)
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )
    num_layers = bundle.num_layers

    # Boundary layers
    boundaries = [args.boundary_layer]
    if args.extra_boundaries:
        boundaries.extend(args.extra_boundaries)
    boundaries = sorted(set(boundaries))
    logger.info("Boundary layers: {}", boundaries)

    # Load + filter yes/no
    logger.info("Loading dataset {} split={}", args.dataset, args.split)
    all_samples = load_dataset_for_eval(
        args.dataset, split=args.split,
        cache_dir=cfg.data.cache_dir, max_samples=args.max_samples,
    )
    yn_samples = [
        s for s in all_samples
        if _normalize_yes_no(s.get("answer", "")) is not None
    ]
    random.seed(args.seed)
    if len(yn_samples) > args.max_samples:
        yn_samples = random.sample(yn_samples, args.max_samples)
    logger.info("Yes/No samples: {}/{}", len(yn_samples), len(all_samples))

    if not yn_samples:
        raise RuntimeError("No valid yes/no samples found")

    # Visual range
    first_inputs, _ = prepare_model_input(yn_samples[0], bundle, cfg.model.device)
    visual_range = find_visual_token_positions(bundle, first_inputs["input_ids"])
    logger.info("Visual token range: {}", visual_range)
    del first_inputs

    temp_ext = ActivationExtractor(bundle.model, visual_range)
    transformer_layers = temp_ext._transformer_layers

    # === Baseline ===
    logger.info("Running baseline...")
    baseline_acc, baseline_records = _evaluate_condition(
        yn_samples, bundle, cfg.model.device, args.max_new_tokens,
    )
    logger.info("Baseline accuracy: {:.4f}", baseline_acc)

    results = {
        "model": cfg.model.name,
        "dataset": args.dataset,
        "num_layers": num_layers,
        "num_samples": len(yn_samples),
        "sample_ids": [s.get("id") for s in yn_samples],
        "visual_token_range": list(visual_range),
        "boundary_layers": boundaries,
        "mask_strategies": args.mask_strategies,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "baseline_accuracy": baseline_acc,
        "conditions": {},
    }

    # === Run conditions for each boundary × strategy ===
    for boundary in boundaries:
        for strategy in args.mask_strategies:
            conditions = {
                f"early_mask_L0-{boundary}": (0, boundary),
                f"deep_mask_L{boundary}-{num_layers}": (boundary, num_layers),
                f"full_mask_L0-{num_layers}": (0, num_layers),
            }

            for cond_name, (mask_start, mask_end) in conditions.items():
                full_name = f"{cond_name}__{strategy}"
                logger.info("Condition: {} (layers {}-{}, strategy={})",
                            cond_name, mask_start, mask_end, strategy)

                hook = DeepLayerVisualMaskHook(
                    model=bundle.model,
                    mask_start_layer=mask_start,
                    mask_end_layer=mask_end,
                    visual_token_range=visual_range,
                    strategy=strategy,
                    seed=args.seed,
                    transformer_layers=transformer_layers,
                )

                acc, records = _evaluate_condition(
                    yn_samples, bundle, cfg.model.device,
                    args.max_new_tokens, hook=hook,
                )
                drop = baseline_acc - acc
                logger.info("  accuracy={:.4f}, drop from baseline={:+.4f}", acc, -drop)

                results["conditions"][full_name] = {
                    "mask_start_layer": mask_start,
                    "mask_end_layer": mask_end,
                    "strategy": strategy,
                    "boundary": boundary,
                    "accuracy": acc,
                    "accuracy_drop": drop,
                    "num_correct": sum(r["correct"] for r in records),
                    "num_total": len(records),
                    "records": records,
                }

    # === Summary ===
    logger.info("=" * 60)
    logger.info("SUMMARY: Deep-Layer Masking Results")
    logger.info("=" * 60)
    logger.info("Baseline: {:.4f}", baseline_acc)
    for cond_name, data in results["conditions"].items():
        logger.info("  {}: acc={:.4f}, drop={:+.4f}",
                     cond_name, data["accuracy"], -data["accuracy_drop"])

    # Re-grounding diagnosis
    for boundary in boundaries:
        for strategy in args.mask_strategies:
            early_key = f"early_mask_L0-{boundary}__{strategy}"
            deep_key = f"deep_mask_L{boundary}-{num_layers}__{strategy}"

            if early_key in results["conditions"] and deep_key in results["conditions"]:
                early_drop = results["conditions"][early_key]["accuracy_drop"]
                deep_drop = results["conditions"][deep_key]["accuracy_drop"]

                logger.info("")
                logger.info("Boundary={}, strategy={}:", boundary, strategy)
                logger.info("  Early mask (L0-{}): drop={:+.4f}", boundary, -early_drop)
                logger.info("  Deep mask (L{}-{}): drop={:+.4f}", boundary, num_layers, -deep_drop)

                if deep_drop > 0.03:
                    logger.info(
                        "  → RE-GROUNDING SUPPORTED: Deep layers need visual tokens "
                        "even after early layers had full access."
                    )
                elif deep_drop < 0.01 and early_drop > 0.05:
                    logger.info(
                        "  → EARLY ASSIMILATION: Visual info fully absorbed by layer {}; "
                        "deep layers don't need visual tokens.",
                        boundary,
                    )
                else:
                    logger.info("  → INCONCLUSIVE at this boundary.")

    # Save
    out_path = os.path.join(cfg.output.results_dir, "deep_layer_masking_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to {}", out_path)


if __name__ == "__main__":
    main()
