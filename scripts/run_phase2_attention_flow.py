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


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Attention Flow")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=30,
                        help="Number of samples to average over")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        help="Attention implementation (must be 'eager' to get weights)")
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

    logger.info(f"attn_implementation={args.attn_implementation!r}")

    # Load model with eager attention so weights are returned
    logger.info("Loading model...")
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
        attn_implementation=args.attn_implementation,
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
        random.seed(42)
        all_samples = random.sample(all_samples, args.num_samples)

    logger.info(f"Using {len(all_samples)} samples for attention flow analysis")

    # ---- Per-layer accumulators ----
    t2v_accum: dict[int, list[float]] = {l: [] for l in range(num_layers)}
    v2v_accum: dict[int, list[float]] = {l: [] for l in range(num_layers)}

    uniform_baselines: list[float] = []
    n_visual_list: list[int] = []
    n_text_list: list[int] = []

    for i, sample in enumerate(all_samples):
        logger.info(f"Sample {i+1}/{len(all_samples)}: {sample['id']}")

        inputs, _ = prepare_model_input(sample, bundle, cfg.model.device)
        visual_range = find_visual_token_positions(bundle, inputs["input_ids"])
        vstart, vend = visual_range
        seq_len = inputs["input_ids"].shape[1]

        n_visual = vend - vstart
        n_text = seq_len - vend
        n_visual_list.append(n_visual)
        n_text_list.append(n_text)

        ub = compute_uniform_baseline(visual_range, seq_len)
        uniform_baselines.append(ub)

        logger.debug(
            f"  visual_range=({vstart},{vend}), n_visual={n_visual}, "
            f"n_text={n_text}, uniform_baseline={ub:.4f}"
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

        del outputs, attentions
        torch.cuda.empty_cache()

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
        "num_samples": len(all_samples),
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
