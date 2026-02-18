"""Phase 2 companion: model-generation accuracy under image controls.

Evaluates the model's own answer accuracy on the same sample subset used by
text probing (e.g., original vs blind_randomimage).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

from PIL import Image, ImageStat
from loguru import logger
from omegaconf import OmegaConf

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_loader import load_dataset_for_eval
from src.models.input_preparation import prepare_model_input
from src.models.model_loader import load_model


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
    parser = argparse.ArgumentParser(description="Phase 2: model generation accuracy")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="chartqa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=300)
    parser.add_argument(
        "--image_modes",
        nargs="+",
        default=["original", "blind_randomimage"],
        choices=["original", "blind_black", "blind_mean", "blind_randomimage"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--sample_ids_json",
        type=str,
        default=None,
        help="Optional probing result json containing sample_ids to evaluate on.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.model_name:
        cfg.model.name = args.model_name
    if args.output_dir:
        cfg.output.results_dir = args.output_dir

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.results_dir, "run_phase2_model_accuracy.log")
    logger.add(log_path, rotation="10 MB")

    logger.info("Loading model: {}", cfg.model.name)
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )

    logger.info(
        "Loading dataset {} split={} max_samples={}",
        args.dataset,
        args.split,
        args.max_samples,
    )
    all_samples = load_dataset_for_eval(
        args.dataset,
        split=args.split,
        cache_dir=cfg.data.cache_dir,
        max_samples=args.max_samples,
    )

    sample_ids_filter: set[str] | None = None
    if args.sample_ids_json:
        with open(args.sample_ids_json) as f:
            sample_ids_filter = set(json.load(f).get("sample_ids", []))
        logger.info(
            "Loaded sample-id filter from {} ({} ids)",
            args.sample_ids_json,
            len(sample_ids_filter),
        )

    samples = all_samples
    if sample_ids_filter is not None:
        samples = [s for s in all_samples if s.get("id") in sample_ids_filter]
    logger.info("Evaluation subset size: {}", len(samples))
    if not samples:
        raise RuntimeError("No samples to evaluate after filtering")

    # Build control images once for deterministic comparisons.
    black_cache: dict[tuple[str, tuple[int, int]], Image.Image] = {}
    blind_mean_rgb: tuple[int, int, int] | None = None
    random_image_map: dict[str, Image.Image] = {}

    if "blind_mean" in args.image_modes:
        rgb_means = []
        for sample in all_samples:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL.Image, got {type(image)}")
            rgb_means.append(ImageStat.Stat(image.convert("RGB")).mean)
        blind_mean_rgb = tuple(
            int(round(sum(ch[i] for ch in rgb_means) / len(rgb_means))) for i in range(3)
        )
        logger.info("blind_mean RGB color={}", blind_mean_rgb)

    if "blind_randomimage" in args.image_modes:
        rng = random.Random(args.seed)
        indices = list(range(len(all_samples)))
        if len(indices) > 1:
            for _ in range(100):
                perm = indices[:]
                rng.shuffle(perm)
                if all(i != p for i, p in zip(indices, perm)):
                    break
            else:
                perm = indices[1:] + indices[:1]
        else:
            perm = indices[:]
        for idx, pidx in zip(indices, perm):
            sid = all_samples[idx].get("id", f"sample_{idx}")
            random_image_map[sid] = all_samples[pidx]["image"]
        logger.info("blind_randomimage map ready: {} samples, seed={}", len(random_image_map), args.seed)

    def make_image_control(sample: dict, mode: str) -> Image.Image:
        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image in sample, got {type(image)}")
        if mode == "original":
            return image
        if mode == "blind_black":
            key = (image.mode, image.size)
            if key not in black_cache:
                black_cache[key] = Image.new(image.mode, image.size, 0)
            return black_cache[key]
        if mode == "blind_mean":
            assert blind_mean_rgb is not None
            mean_rgb_img = Image.new("RGB", image.size, blind_mean_rgb)
            return mean_rgb_img.convert(image.mode)
        if mode == "blind_randomimage":
            sid = sample.get("id")
            if sid not in random_image_map:
                raise KeyError(f"Missing random-image mapping for sample id={sid}")
            return random_image_map[sid]
        raise ValueError(f"Unsupported image mode: {mode}")

    out: dict = {
        "model": cfg.model.name,
        "dataset": args.dataset,
        "split": args.split,
        "max_samples_loaded": len(all_samples),
        "num_eval_samples": len(samples),
        "sample_ids": [s.get("id", "") for s in samples],
        "image_modes": args.image_modes,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "results": {},
    }

    for mode in args.image_modes:
        logger.info("Evaluating mode={}", mode)
        records = []
        correct = 0
        valid = 0
        skipped = 0
        gold_counts = Counter()
        pred_counts = Counter()

        for sample in samples:
            gold = _normalize_yes_no(sample.get("answer", ""))
            if gold is None:
                skipped += 1
                continue

            sample_in = sample
            if mode != "original":
                sample_in = {**sample, "image": make_image_control(sample, mode)}

            model_inputs, _ = prepare_model_input(
                sample_in,
                bundle,
                cfg.model.device,
                include_answer_in_prompt=False,
                return_metadata=False,
            )
            generated = _decode_generation(
                bundle.model,
                bundle.tokenizer,
                model_inputs,
                max_new_tokens=args.max_new_tokens,
            )
            pred = _normalize_yes_no(generated)

            is_correct = int(pred == gold)
            correct += is_correct
            valid += 1
            gold_counts[gold] += 1
            if pred is not None:
                pred_counts[pred] += 1

            records.append({
                "id": sample.get("id"),
                "gold": gold,
                "pred": pred,
                "correct": bool(is_correct),
                "generated": generated,
            })

        acc = (correct / valid) if valid else 0.0
        logger.info(
            "mode={} accuracy={:.4f} valid={} skipped={}",
            mode,
            acc,
            valid,
            skipped,
        )
        out["results"][mode] = {
            "accuracy": acc,
            "num_valid": valid,
            "num_skipped": skipped,
            "num_correct": correct,
            "gold_counts": dict(gold_counts),
            "pred_counts": dict(pred_counts),
            "records": records,
        }

    out_path = os.path.join(cfg.output.results_dir, "phase2_model_accuracy_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info("Saved model-accuracy results to {}", out_path)


if __name__ == "__main__":
    main()

