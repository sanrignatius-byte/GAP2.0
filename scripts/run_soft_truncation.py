"""Norm-matched Soft Truncation Experiment (P0 — artifact ruling).

Distinguishes "model depends on visual content" from "hard truncation
causes distributional shock". At each truncation layer, replaces visual
tokens with norm-matched noise (or other controlled substitutes) instead
of hard zeros, then measures task accuracy.

Key comparison:
    hard_zero vs norm_matched_noise vs pca_matched_noise vs shuffle

If hard zero ≫ norm_matched_noise in accuracy drop:
    → Hard truncation artifact dominates; previous results confounded.
If all strategies drop similarly:
    → Deep layers genuinely depend on visual content.

Usage:
    python scripts/run_soft_truncation.py \
      --config configs/default.yaml \
      --model_config configs/qwen_vl_full.yaml \
      --dataset chartqa \
      --strategies zero norm_matched_noise pca_matched_noise shuffle \
      --truncation_layers 0 6 12 18 24 30 36 42 47 \
      --max_samples 200 \
      --output_dir results/soft_truncation_chartqa
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.models.hooks import (
    TruncationHook,
    SoftTruncationHook,
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


def main():
    parser = argparse.ArgumentParser(description="Soft Truncation Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="chartqa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["zero", "norm_matched_noise", "pca_matched_noise", "shuffle"],
    )
    parser.add_argument(
        "--truncation_layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to truncate at. Default: every 6th layer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--pca_rank", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.output_dir:
        cfg.output.results_dir = args.output_dir

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.results_dir, "run_soft_truncation.log")
    logger.add(log_path, rotation="10 MB")

    # Load model
    logger.info("Loading model: {}", cfg.model.name)
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )
    num_layers = bundle.num_layers

    # Determine truncation layers
    if args.truncation_layers is not None:
        truncation_layers = args.truncation_layers
    else:
        step = max(1, num_layers // 8)
        truncation_layers = list(range(0, num_layers, step))
        if (num_layers - 1) not in truncation_layers:
            truncation_layers.append(num_layers - 1)
    logger.info("Truncation layers: {}", truncation_layers)

    # Load dataset — filter for yes/no questions
    logger.info("Loading dataset {} split={}", args.dataset, args.split)
    all_samples = load_dataset_for_eval(
        args.dataset,
        split=args.split,
        cache_dir=cfg.data.cache_dir,
        max_samples=args.max_samples,
    )

    # Filter yes/no
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

    # Find visual token range from first sample
    first_inputs, _ = prepare_model_input(yn_samples[0], bundle, cfg.model.device)
    visual_range = find_visual_token_positions(bundle, first_inputs["input_ids"])
    logger.info("Visual token range: {}", visual_range)
    del first_inputs

    # Pre-discover transformer layers (shared across hooks)
    temp_ext = ActivationExtractor(bundle.model, visual_range)
    transformer_layers = temp_ext._transformer_layers

    # === Run baseline (no intervention) ===
    logger.info("Running baseline (no intervention)...")
    baseline_records = []
    for sample in yn_samples:
        gold = _normalize_yes_no(sample["answer"])
        model_inputs, _ = prepare_model_input(
            sample, bundle, cfg.model.device,
            include_answer_in_prompt=False, return_metadata=False,
        )
        generated = _decode_generation(
            bundle.model, bundle.tokenizer, model_inputs, args.max_new_tokens,
        )
        pred = _normalize_yes_no(generated)
        baseline_records.append({
            "id": sample.get("id"),
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
        })
    baseline_acc = sum(r["correct"] for r in baseline_records) / len(baseline_records)
    logger.info("Baseline accuracy: {:.4f} ({}/{})",
                baseline_acc, sum(r["correct"] for r in baseline_records), len(baseline_records))

    # === Run each strategy × truncation layer ===
    results = {
        "model": cfg.model.name,
        "dataset": args.dataset,
        "num_samples": len(yn_samples),
        "sample_ids": [s.get("id") for s in yn_samples],
        "visual_token_range": list(visual_range),
        "truncation_layers": truncation_layers,
        "strategies": args.strategies,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "baseline_accuracy": baseline_acc,
        "sweep": {},
    }

    for strategy in args.strategies:
        logger.info("=" * 60)
        logger.info("Strategy: {}", strategy)
        logger.info("=" * 60)

        strategy_results = {}
        for trunc_layer in truncation_layers:
            logger.info("  Truncation layer: {}", trunc_layer)

            # Create the appropriate hook
            if strategy == "zero":
                hook = TruncationHook(
                    model=bundle.model,
                    truncation_layer=trunc_layer,
                    visual_token_range=visual_range,
                    transformer_layers=transformer_layers,
                )
            else:
                hook = SoftTruncationHook(
                    model=bundle.model,
                    truncation_layer=trunc_layer,
                    visual_token_range=visual_range,
                    strategy=strategy,
                    pca_rank=args.pca_rank,
                    seed=args.seed,
                    transformer_layers=transformer_layers,
                )

            records = []
            with hook:
                for sample in yn_samples:
                    gold = _normalize_yes_no(sample["answer"])
                    model_inputs, _ = prepare_model_input(
                        sample, bundle, cfg.model.device,
                        include_answer_in_prompt=False, return_metadata=False,
                    )
                    generated = _decode_generation(
                        bundle.model, bundle.tokenizer, model_inputs, args.max_new_tokens,
                    )
                    pred = _normalize_yes_no(generated)
                    records.append({
                        "id": sample.get("id"),
                        "gold": gold,
                        "pred": pred,
                        "correct": pred == gold,
                    })

            acc = sum(r["correct"] for r in records) / len(records)
            drop = baseline_acc - acc
            logger.info("    accuracy={:.4f}, drop={:.4f}", acc, drop)

            strategy_results[str(trunc_layer)] = {
                "accuracy": acc,
                "accuracy_drop": drop,
                "num_correct": sum(r["correct"] for r in records),
                "num_total": len(records),
                "records": records,
            }

        results["sweep"][strategy] = strategy_results

    # === Summary comparison ===
    logger.info("=" * 60)
    logger.info("SUMMARY: Hard Zero vs Soft Truncation")
    logger.info("=" * 60)

    for trunc_layer in truncation_layers:
        drops = {}
        for strategy in args.strategies:
            entry = results["sweep"][strategy].get(str(trunc_layer))
            if entry:
                drops[strategy] = entry["accuracy_drop"]
        if drops:
            drop_str = " | ".join(f"{s}={d:+.4f}" for s, d in drops.items())
            logger.info("Layer {:3d}: {}", trunc_layer, drop_str)

    # Artifact diagnosis
    if "zero" in results["sweep"] and "norm_matched_noise" in results["sweep"]:
        zero_drops = [
            results["sweep"]["zero"][str(l)]["accuracy_drop"]
            for l in truncation_layers
            if str(l) in results["sweep"]["zero"]
        ]
        noise_drops = [
            results["sweep"]["norm_matched_noise"][str(l)]["accuracy_drop"]
            for l in truncation_layers
            if str(l) in results["sweep"]["norm_matched_noise"]
        ]
        mean_zero = np.mean(zero_drops)
        mean_noise = np.mean(noise_drops)
        logger.info("")
        logger.info("Mean accuracy drop — zero: {:.4f}, norm_matched_noise: {:.4f}",
                     mean_zero, mean_noise)
        if mean_zero > mean_noise * 1.5 and mean_zero > 0.05:
            logger.warning(
                "⚠ ARTIFACT WARNING: Hard zero drops {:.1f}x more than norm-matched noise. "
                "Previous hard-truncation results may be confounded by distributional shock.",
                mean_zero / max(mean_noise, 1e-6),
            )
        else:
            logger.info(
                "✓ Consistent: Both strategies show similar drops → "
                "deep layers genuinely depend on visual content."
            )

    # Save
    out_path = os.path.join(cfg.output.results_dir, "soft_truncation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to {}", out_path)


if __name__ == "__main__":
    main()
