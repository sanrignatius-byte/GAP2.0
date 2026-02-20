"""Attention Sparsity & Entropy Analysis (P1 — U-shape arbitration).

Extends basic attention flow analysis with entropy and sparsity metrics
that arbitrate between two explanations for the U-shape attention pattern:

    1. Re-grounding: Deep layers selectively attend to specific visual patches
       relevant to the question → LOW entropy, HIGH top-k concentration
    2. Attention sink: Deep layers spread attention uniformly or concentrate
       on structural tokens → HIGH entropy, LOW top-k concentration

Usage:
    python scripts/run_attention_sparsity.py \
      --config configs/default.yaml \
      --model_config configs/qwen_vl_full.yaml \
      --num_samples 30 \
      --output_dir results/attention_sparsity
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Optional

import torch
import numpy as np
from loguru import logger
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.data.dataset_loader import load_dataset_for_eval
from src.attn.attention_flow import compute_layer_attention_flow, compute_uniform_baseline
from src.attn.attention_sparsity import compute_attention_sparsity


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()


def _parse_pixel_schedule(max_pixels: Optional[int], fallback_csv: str) -> list[Optional[int]]:
    schedule: list[Optional[int]] = [max_pixels]
    for item in fallback_csv.split(","):
        item = item.strip()
        if item:
            schedule.append(int(item))
    return list(dict.fromkeys(schedule))  # dedup preserving order


def _prepare_with_processor(sample, bundle, device, processor):
    original_processor = bundle.processor
    original_tokenizer = bundle.tokenizer
    bundle.processor = processor
    bundle.tokenizer = processor.tokenizer
    try:
        return prepare_model_input(sample, bundle, device)
    finally:
        bundle.processor = original_processor
        bundle.tokenizer = original_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Attention Sparsity Analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--fallback_max_pixels", type=str, default="200704,100352,62720")
    parser.add_argument(
        "--top_k_values", nargs="+", type=int, default=[5, 10, 50],
        help="k values for top-k concentration metric.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_success_samples", type=int, default=5)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.output_dir:
        cfg.output.results_dir = args.output_dir
        cfg.output.plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)
    log_path = os.path.join(cfg.output.results_dir, "run_attention_sparsity.log")
    logger.add(log_path, rotation="10 MB")

    pixel_schedule = _parse_pixel_schedule(args.max_pixels, args.fallback_max_pixels)
    logger.info("attn_implementation={!r}", args.attn_implementation)
    logger.info("pixel schedule={}", pixel_schedule)

    # Load model
    logger.info("Loading model...")
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        attn_implementation=args.attn_implementation,
        processor_max_pixels=pixel_schedule[0],
    )
    num_layers = bundle.num_layers

    # Load data
    logger.info("Loading samples...")
    all_samples = []
    for ds_cfg in cfg.data.datasets:
        samples = load_dataset_for_eval(
            ds_cfg.name, split=ds_cfg.split,
            cache_dir=cfg.data.cache_dir, max_samples=args.num_samples,
        )
        all_samples.extend(samples)

    if len(all_samples) > args.num_samples:
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.num_samples)
    logger.info("Using {} samples", len(all_samples))

    is_qwen = "qwen" in cfg.model.name.lower()
    processor_cache = {pixel_schedule[0]: bundle.processor}

    def get_processor(max_pixels):
        if max_pixels in processor_cache:
            return processor_cache[max_pixels]
        from transformers import AutoProcessor
        kwargs = {"trust_remote_code": True}
        if max_pixels is not None:
            kwargs["max_pixels"] = int(max_pixels)
        proc = AutoProcessor.from_pretrained(cfg.model.name, **kwargs)
        processor_cache[max_pixels] = proc
        return proc

    # Accumulators
    entropy_accum = {l: [] for l in range(num_layers)}
    n_eff_accum = {l: [] for l in range(num_layers)}
    t2v_mass_accum = {l: [] for l in range(num_layers)}
    max_attn_accum = {l: [] for l in range(num_layers)}
    top_k_accum = {l: {k: [] for k in args.top_k_values} for l in range(num_layers)}
    # Also keep basic t2v for comparison
    t2v_accum = {l: [] for l in range(num_layers)}

    processed_ids = []
    skipped_oom = 0

    for i, sample in enumerate(all_samples):
        logger.info("Sample {}/{}: {}", i + 1, len(all_samples), sample["id"])
        sample_done = False

        schedules = pixel_schedule if is_qwen else [None]
        for max_pixels in schedules:
            outputs = None
            try:
                if is_qwen:
                    processor = get_processor(max_pixels)
                    inputs, _ = _prepare_with_processor(sample, bundle, cfg.model.device, processor)
                else:
                    inputs, _ = prepare_model_input(sample, bundle, cfg.model.device)

                visual_range = find_visual_token_positions(bundle, inputs["input_ids"])
                seq_len = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = bundle.model(**inputs, output_attentions=True)

                attentions = outputs.attentions
                if attentions is None:
                    raise RuntimeError("No attention weights returned")

                for layer_idx, attn_tensor in enumerate(attentions):
                    # Basic flow
                    flow = compute_layer_attention_flow(attn_tensor, visual_range)
                    t2v_accum[layer_idx].append(flow["text_to_visual"])

                    # Sparsity metrics
                    sparsity = compute_attention_sparsity(
                        attn_tensor, visual_range,
                        top_k_values=args.top_k_values,
                    )
                    entropy_accum[layer_idx].append(sparsity["entropy_mean"])
                    n_eff_accum[layer_idx].append(sparsity["n_eff_mean"])
                    t2v_mass_accum[layer_idx].append(sparsity["t2v_mass_mean"])
                    max_attn_accum[layer_idx].append(sparsity["max_attention_mean"])
                    for k in args.top_k_values:
                        top_k_accum[layer_idx][k].append(
                            sparsity["top_k_concentration"][k]
                        )

                    del attn_tensor

                processed_ids.append(sample["id"])
                sample_done = True
                break

            except Exception as exc:
                if _is_oom_error(exc):
                    logger.warning("OOM on sample={} max_pixels={}", sample["id"], max_pixels)
                    torch.cuda.empty_cache()
                    continue
                raise
            finally:
                if outputs is not None:
                    del outputs
                torch.cuda.empty_cache()

        if not sample_done:
            skipped_oom += 1
            logger.warning("Skipping sample={}", sample["id"])

    success_count = len(processed_ids)
    if success_count < args.min_success_samples:
        raise RuntimeError(f"Only {success_count} samples succeeded")

    # === Aggregate ===
    layers = np.arange(num_layers)

    def _agg(accum):
        return (
            np.array([np.mean(accum[l]) for l in layers]),
            np.array([np.std(accum[l]) for l in layers]),
        )

    entropy_mean, entropy_std = _agg(entropy_accum)
    n_eff_mean, n_eff_std = _agg(n_eff_accum)
    t2v_mass_mean, t2v_mass_std = _agg(t2v_mass_accum)
    max_attn_mean, max_attn_std = _agg(max_attn_accum)
    t2v_mean, t2v_std = _agg(t2v_accum)

    top_k_results = {}
    for k in args.top_k_values:
        k_vals = np.array([np.mean(top_k_accum[l][k]) for l in layers])
        k_stds = np.array([np.std(top_k_accum[l][k]) for l in layers])
        top_k_results[str(k)] = {
            "mean": k_vals.tolist(),
            "std": k_stds.tolist(),
        }

    # === Diagnosis ===
    logger.info("=" * 60)
    logger.info("Attention Sparsity Summary ({} samples)", success_count)
    logger.info("=" * 60)

    # Find early peak and late peak of t2v
    early_range = layers[:num_layers // 2]
    late_range = layers[num_layers // 2:]
    early_peak_layer = int(early_range[np.argmax(t2v_mean[early_range])])
    late_peak_layer = int(late_range[np.argmax(t2v_mean[late_range])])

    logger.info("t2v early peak: layer {} (mass={:.4f})", early_peak_layer, t2v_mean[early_peak_layer])
    logger.info("t2v late peak:  layer {} (mass={:.4f})", late_peak_layer, t2v_mean[late_peak_layer])

    # Compare entropy at early vs late peak
    logger.info("")
    logger.info("Entropy (lower = more sparse/targeted):")
    logger.info("  Early peak (L{}): entropy={:.3f}, N_eff={:.1f}",
                early_peak_layer, entropy_mean[early_peak_layer], n_eff_mean[early_peak_layer])
    logger.info("  Late peak  (L{}): entropy={:.3f}, N_eff={:.1f}",
                late_peak_layer, entropy_mean[late_peak_layer], n_eff_mean[late_peak_layer])

    logger.info("")
    logger.info("Top-k concentration at late peak (L{}):", late_peak_layer)
    for k in args.top_k_values:
        val = float(np.mean(top_k_accum[late_peak_layer][k]))
        logger.info("  Top-{}: {:.3f}", k, val)

    # Diagnosis
    late_entropy = entropy_mean[late_peak_layer]
    early_entropy = entropy_mean[early_peak_layer]

    logger.info("")
    if late_entropy < early_entropy * 0.9:
        logger.info(
            "→ RE-GROUNDING SUPPORTED: Deep-layer attention is MORE sparse "
            "than early layers (entropy {:.3f} < {:.3f}). "
            "Consistent with selective visual re-consultation.",
            late_entropy, early_entropy,
        )
    elif late_entropy > early_entropy * 1.1:
        logger.info(
            "→ ATTENTION SINK LIKELY: Deep-layer attention is MORE diffuse "
            "than early layers (entropy {:.3f} > {:.3f}). "
            "U-shape may be an attention distribution artifact.",
            late_entropy, early_entropy,
        )
    else:
        logger.info(
            "→ INCONCLUSIVE: Early and late entropy are similar "
            "({:.3f} vs {:.3f}). Need per-head or per-sample analysis.",
            late_entropy, early_entropy,
        )

    # === Save ===
    output = {
        "num_samples_succeeded": success_count,
        "num_samples_skipped": skipped_oom,
        "sample_ids": processed_ids,
        "num_layers": num_layers,
        "top_k_values": args.top_k_values,
        "layers": layers.tolist(),
        "text_to_visual_mean": t2v_mean.tolist(),
        "text_to_visual_std": t2v_std.tolist(),
        "entropy_mean": entropy_mean.tolist(),
        "entropy_std": entropy_std.tolist(),
        "n_eff_mean": n_eff_mean.tolist(),
        "n_eff_std": n_eff_std.tolist(),
        "t2v_mass_mean": t2v_mass_mean.tolist(),
        "t2v_mass_std": t2v_mass_std.tolist(),
        "max_attention_mean": max_attn_mean.tolist(),
        "max_attention_std": max_attn_std.tolist(),
        "top_k_concentration": top_k_results,
        "diagnosis": {
            "early_peak_layer": early_peak_layer,
            "late_peak_layer": late_peak_layer,
            "early_peak_entropy": float(early_entropy),
            "late_peak_entropy": float(late_entropy),
            "early_peak_n_eff": float(n_eff_mean[early_peak_layer]),
            "late_peak_n_eff": float(n_eff_mean[late_peak_layer]),
        },
    }

    out_path = os.path.join(cfg.output.results_dir, "attention_sparsity_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to {}", out_path)


if __name__ == "__main__":
    main()
