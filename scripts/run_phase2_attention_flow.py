"""Phase 2: Attention Flow Experiment.

Measures per-layer text-to-visual attention to distinguish between:
  - Late-Stage Readout hypothesis: t2v peaks at layers 37-40
  - Early Assimilation hypothesis: t2v peaks at layers 20-25

Key design decisions:
  - Uses attn_implementation="eager" (SDPA/flash_attn don't return weights)
  - output_attentions=True during a single forward pass per sample
  - Memory-efficient: attention tensors released immediately per layer
  - Computes uniform_baseline to normalise for sequence composition effects

Usage:
    python scripts/run_phase2_attention_flow.py \\
      --config configs/default.yaml \\
      --model_config configs/qwen_vl_full.yaml \\
      --num_samples 30 \\
      --output_dir results/attention_flow
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import random
from typing import Optional

import torch
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.data.dataset_loader import load_dataset_for_eval
from src.attn.attention_flow import compute_layer_attention_flow, compute_uniform_baseline
from src.visualization.plots import GAPVisualizer


def _is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _parse_pixel_schedule(max_pixels: Optional[int], fallback_csv: str) -> list[Optional[int]]:
    schedule: list[Optional[int]] = [max_pixels]
    for item in fallback_csv.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            continue
        schedule.append(value)

    # Deduplicate while preserving order.
    deduped: list[Optional[int]] = []
    seen = set()
    for value in schedule:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _prepare_with_processor(sample: dict, bundle, device: str, processor):
    """Prepare inputs using a temporary processor override."""
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
    parser = argparse.ArgumentParser(description="Phase 2: Attention Flow")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of samples to average over")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        help="Attention implementation (must be 'eager' to get weights)")
    parser.add_argument("--max_pixels", type=int, default=401408,
                        help="Initial Qwen processor max_pixels to reduce visual tokens.")
    parser.add_argument("--fallback_max_pixels", type=str, default="200704,100352,62720",
                        help="Comma-separated fallback max_pixels values used after OOM.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_success_samples", type=int, default=5,
                        help="Minimum successful samples required to accept run.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.output_dir:
        cfg.output.results_dir = args.output_dir
        cfg.output.plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)
    logger.add(
        os.path.join(cfg.output.results_dir, "run_phase2_attention_flow.log"),
        rotation="10 MB",
    )

    pixel_schedule = _parse_pixel_schedule(args.max_pixels, args.fallback_max_pixels)
    logger.info(f"attn_implementation={args.attn_implementation!r}")
    logger.info(f"pixel schedule={pixel_schedule}")

    # Load model with eager attention so weights are returned
    logger.info("Loading model...")
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        attn_implementation=args.attn_implementation,
        processor_max_pixels=pixel_schedule[0],
    )
    num_layers = bundle.num_layers
    logger.info(f"Model loaded: {num_layers} layers")

    # Load data
    logger.info("Loading evaluation samples...")
    all_samples = []
    for ds_cfg in cfg.data.datasets:
        samples = load_dataset_for_eval(
            ds_cfg.name,
            split=ds_cfg.split,
            cache_dir=cfg.data.cache_dir,
            max_samples=args.num_samples,
        )
        all_samples.extend(samples)

    if len(all_samples) > args.num_samples:
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.num_samples)

    logger.info(f"Using {len(all_samples)} samples for attention flow analysis")

    # For Qwen, keep a lazy cache of processors at different max_pixels values.
    is_qwen = "qwen" in cfg.model.name.lower()
    processor_cache: dict[Optional[int], object] = {pixel_schedule[0]: bundle.processor}

    def get_processor(max_pixels: Optional[int]):
        if max_pixels in processor_cache:
            return processor_cache[max_pixels]
        from transformers import AutoProcessor

        kwargs = {"trust_remote_code": True}
        if max_pixels is not None:
            kwargs["max_pixels"] = int(max_pixels)
        proc = AutoProcessor.from_pretrained(cfg.model.name, **kwargs)
        processor_cache[max_pixels] = proc
        return proc

    # ---- Per-layer accumulators ----
    t2v_accum: dict[int, list[float]] = {l: [] for l in range(num_layers)}
    v2v_accum: dict[int, list[float]] = {l: [] for l in range(num_layers)}

    uniform_baselines: list[float] = []
    n_visual_list: list[int] = []
    n_text_list: list[int] = []
    processed_sample_ids: list[str] = []
    per_sample_max_pixels: dict[str, Optional[int]] = {}
    skipped_oom = 0

    for i, sample in enumerate(all_samples):
        logger.info(f"Sample {i+1}/{len(all_samples)}: {sample['id']}")
        sample_done = False

        schedules = pixel_schedule if is_qwen else [None]
        for max_pixels in schedules:
            outputs = None
            attentions = None
            inputs = None
            try:
                if is_qwen:
                    processor = get_processor(max_pixels)
                    inputs, _ = _prepare_with_processor(sample, bundle, cfg.model.device, processor)
                else:
                    inputs, _ = prepare_model_input(sample, bundle, cfg.model.device)

                visual_range = find_visual_token_positions(bundle, inputs["input_ids"])
                vstart, vend = visual_range
                seq_len = inputs["input_ids"].shape[1]

                n_visual = vend - vstart
                n_text = seq_len - vend

                ub = compute_uniform_baseline(visual_range, seq_len)

                logger.debug(
                    f"  max_pixels={max_pixels}, visual_range=({vstart},{vend}), "
                    f"n_visual={n_visual}, n_text={n_text}, uniform_baseline={ub:.4f}"
                )

                # Single forward pass with output_attentions=True
                with torch.no_grad():
                    outputs = bundle.model(**inputs, output_attentions=True)

                # outputs.attentions: tuple of (1, H, S, S) tensors, one per layer
                attentions = outputs.attentions
                if attentions is None:
                    raise RuntimeError(
                        "Model returned no attention weights. "
                        "Ensure attn_implementation='eager' is set."
                    )

                for layer_idx, attn_tensor in enumerate(attentions):
                    metrics = compute_layer_attention_flow(attn_tensor, visual_range)
                    t2v_accum[layer_idx].append(metrics["text_to_visual"])
                    v2v_accum[layer_idx].append(metrics["visual_to_visual"])
                    # Release tensor immediately to avoid memory build-up
                    del attn_tensor

                n_visual_list.append(n_visual)
                n_text_list.append(n_text)
                uniform_baselines.append(ub)
                processed_sample_ids.append(sample["id"])
                per_sample_max_pixels[sample["id"]] = max_pixels
                sample_done = True
                break
            except Exception as exc:  # noqa: BLE001
                if _is_oom_error(exc):
                    logger.warning(
                        "OOM on sample={} with max_pixels={}; trying smaller setting",
                        sample["id"],
                        max_pixels,
                    )
                    torch.cuda.empty_cache()
                    continue
                raise
            finally:
                if outputs is not None:
                    del outputs
                if attentions is not None:
                    del attentions
                if inputs is not None:
                    del inputs
                torch.cuda.empty_cache()

        if not sample_done:
            skipped_oom += 1
            logger.warning("Skipping sample={} after exhausting pixel schedule", sample["id"])

    success_count = len(processed_sample_ids)
    if success_count == 0:
        raise RuntimeError("Attention flow failed: no sample completed successfully.")
    if success_count < args.min_success_samples:
        raise RuntimeError(
            f"Attention flow aborted: only {success_count} samples succeeded; "
            f"min_success_samples={args.min_success_samples}."
        )

    # ---- Aggregate ----
    layers = np.arange(num_layers)

    t2v_mean = np.array([np.mean(t2v_accum[l]) for l in layers])
    t2v_std = np.array([np.std(t2v_accum[l]) for l in layers])
    v2v_mean = np.array([np.mean(v2v_accum[l]) for l in layers])

    uniform_baseline = float(np.mean(uniform_baselines))
    # Normalized: how much higher than baseline is the text-to-visual attention?
    t2v_normalized = t2v_mean / uniform_baseline if uniform_baseline > 0 else t2v_mean

    logger.info("=" * 60)
    logger.info("Attention Flow Summary")
    logger.info("=" * 60)
    logger.info(f"succeeded samples:       {success_count}/{len(all_samples)}")
    logger.info(f"skipped due to OOM:      {skipped_oom}")
    logger.info(f"n_visual_tokens (mean): {np.mean(n_visual_list):.0f}")
    logger.info(f"n_text_tokens (mean):   {np.mean(n_text_list):.0f}")
    logger.info(f"uniform_baseline:       {uniform_baseline:.4f}")
    logger.info(f"t2v_mean range:         [{t2v_mean.min():.4f}, {t2v_mean.max():.4f}]")

    # Hypothesis evaluation
    peak_layer = int(layers[np.argmax(t2v_normalized)])
    logger.info(f"Peak t2v_normalized layer: {peak_layer}")
    if 37 <= peak_layer <= 40:
        logger.info("RESULT: Late-Stage Readout hypothesis SUPPORTED (peak in 37-40)")
    elif 20 <= peak_layer <= 25:
        logger.info("RESULT: Early Assimilation hypothesis SUPPORTED (peak in 20-25)")
    else:
        logger.info(f"RESULT: Neither hypothesis clearly supported (peak at layer {peak_layer})")

    # ---- Visualize ----
    viz = GAPVisualizer(output_dir=cfg.output.plots_dir)
    viz.plot_attention_flow(
        layers=layers,
        t2v_mean=t2v_mean,
        t2v_std=t2v_std,
        v2v_mean=v2v_mean,
        uniform_baseline=uniform_baseline,
        readout_window=(37, 40),
    )

    # ---- Save results ----
    results = {
        "layers": layers.tolist(),
        "text_to_visual_mean": t2v_mean.tolist(),
        "text_to_visual_std": t2v_std.tolist(),
        "visual_to_visual_mean": v2v_mean.tolist(),
        "t2v_normalized_mean": t2v_normalized.tolist(),
        "uniform_baseline": uniform_baseline,
        "num_samples_requested": len(all_samples),
        "num_samples_succeeded": success_count,
        "num_samples_skipped_oom": skipped_oom,
        "sample_ids_succeeded": processed_sample_ids,
        "max_pixels_schedule": pixel_schedule,
        "max_pixels_per_sample": per_sample_max_pixels,
        "n_visual_tokens": float(np.mean(n_visual_list)),
        "n_text_tokens": float(np.mean(n_text_list)),
        "peak_layer": peak_layer,
    }

    results_path = os.path.join(cfg.output.results_dir, "phase2_attention_flow_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
