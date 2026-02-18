"""Phase 1 Week 2: Geometric Analysis.

Extracts per-layer hidden states and computes:
  - Effective Rank (ER) for visual and text tokens
  - Inter-token Cosine Concentration (ICC) for visual and text tokens
  - Cross-modal CKA between visual and text tokens

Validates Checkpoint 3: Geometric collapse is observable and distinctive
to visual tokens.

Usage:
    python scripts/run_phase1_geometry.py --config configs/default.yaml
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import torch
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.models.hooks import ActivationExtractor
from src.geometry.effective_rank import (
    compute_effective_rank_per_layer,
    compute_singular_spectrum,
)
from src.geometry.cosine_concentration import compute_cosine_concentration_per_layer
from src.geometry.cka import compute_cka_per_layer
from src.data.dataset_loader import load_dataset_for_eval
from src.visualization.plots import GAPVisualizer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Geometric Analysis")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name/path in config")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of samples to average geometric metrics over")
    parser.add_argument("--probe_layers", type=str, default=None,
                        help="Comma-separated layer indices to probe (default: all)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))
    if args.model_name:
        cfg.model.name = args.model_name
    if args.output_dir:
        cfg.output.results_dir = args.output_dir
        cfg.output.plots_dir = os.path.join(args.output_dir, "plots")

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)
    logger.add(os.path.join(cfg.output.results_dir, "run_phase1_geometry.log"), rotation="10 MB")

    # Parse probe layers
    probe_layers = None
    if args.probe_layers:
        probe_layers = [int(x) for x in args.probe_layers.split(",")]

    # Load model
    logger.info("Loading model...")
    bundle = load_model(cfg.model.name, device=cfg.model.device, dtype=cfg.model.dtype)

    # Load data
    logger.info("Loading evaluation samples...")
    all_samples = []
    for ds_cfg in cfg.data.datasets:
        samples = load_dataset_for_eval(
            ds_cfg.name, split=ds_cfg.split, cache_dir=cfg.data.cache_dir,
            max_samples=args.num_samples,
        )
        all_samples.extend(samples)

    # Limit total samples
    if len(all_samples) > args.num_samples:
        import random
        random.seed(42)
        all_samples = random.sample(all_samples, args.num_samples)

    logger.info(f"Using {len(all_samples)} samples for geometric analysis")

    # ---- Collect activations and compute metrics ----

    # Accumulators for per-layer metrics across samples
    visual_er_accum = {}
    text_er_accum = {}
    visual_icc_accum = {}
    text_icc_accum = {}
    cka_accum = {}

    # For singular spectrum visualization
    sample_spectra = {}

    for i, sample in enumerate(all_samples):
        logger.info(f"Processing sample {i+1}/{len(all_samples)}: {sample['id']}")

        inputs, _ = prepare_model_input(sample, bundle, cfg.model.device)
        visual_range = find_visual_token_positions(bundle, inputs["input_ids"])

        extractor = ActivationExtractor(
            model=bundle.model,
            visual_token_range=visual_range,
            layers=probe_layers,
            detach=True,
            store_on_cpu=True,
        )

        with extractor:
            with torch.no_grad():
                bundle.model(**inputs)

        visual_acts = extractor.get_visual_activations()
        text_acts = extractor.get_text_activations()

        # Effective Rank
        v_er = compute_effective_rank_per_layer(visual_acts)
        t_er = compute_effective_rank_per_layer(text_acts)

        for layer_idx, val in v_er.items():
            visual_er_accum.setdefault(layer_idx, []).append(val)
        for layer_idx, val in t_er.items():
            text_er_accum.setdefault(layer_idx, []).append(val)

        # Cosine Concentration
        v_icc = compute_cosine_concentration_per_layer(visual_acts)
        t_icc = compute_cosine_concentration_per_layer(text_acts)

        for layer_idx, val in v_icc.items():
            visual_icc_accum.setdefault(layer_idx, []).append(val)
        for layer_idx, val in t_icc.items():
            text_icc_accum.setdefault(layer_idx, []).append(val)

        # CKA
        layer_cka = compute_cka_per_layer(visual_acts, text_acts, debiased=cfg.geometry.cka.debiased)
        for layer_idx, val in layer_cka.items():
            cka_accum.setdefault(layer_idx, []).append(val)

        # Save singular spectra for the first sample at key layers
        if i == 0:
            for layer_idx in sorted(visual_acts.keys()):
                if layer_idx % 8 == 0 or layer_idx == len(visual_acts) - 1:
                    sv = compute_singular_spectrum(visual_acts[layer_idx], top_k=50)
                    sample_spectra[f"Layer {layer_idx} (visual)"] = sv

    # ---- Aggregate metrics ----
    layers = np.array(sorted(visual_er_accum.keys()))

    visual_er_mean = np.array([np.mean(visual_er_accum[l]) for l in layers])
    text_er_mean = np.array([np.mean(text_er_accum[l]) for l in layers])
    visual_icc_mean = np.array([np.mean(visual_icc_accum[l]) for l in layers])
    text_icc_mean = np.array([np.mean(text_icc_accum[l]) for l in layers])
    cka_mean = np.array([np.mean(cka_accum[l]) for l in layers])

    # Load cliff boundary if available
    cliff = None
    causal_path = os.path.join(cfg.output.results_dir, "phase1_causal_results.json")
    if os.path.exists(causal_path):
        with open(causal_path) as f:
            cliff = json.load(f).get("cliff_boundary")

    # ---- Visualize ----
    viz = GAPVisualizer(output_dir=cfg.output.plots_dir)

    viz.plot_effective_rank(layers, visual_er_mean, text_er_mean, cliff)
    viz.plot_cosine_concentration(layers, visual_icc_mean, text_icc_mean, cliff)
    viz.plot_cka_curve(layers, cka_mean, cliff)

    # Combined dashboard
    viz.plot_geometric_dashboard(
        layers, visual_er_mean, text_er_mean,
        visual_icc_mean, text_icc_mean,
        cka_mean, cliff,
    )

    # Singular spectrum plot
    if sample_spectra:
        viz.plot_singular_spectrum(sample_spectra)

    # ---- Save results ----
    results = {
        "layers": layers.tolist(),
        "visual_effective_rank": visual_er_mean.tolist(),
        "text_effective_rank": text_er_mean.tolist(),
        "visual_cosine_concentration": visual_icc_mean.tolist(),
        "text_cosine_concentration": text_icc_mean.tolist(),
        "cross_modal_cka": cka_mean.tolist(),
        "cliff_boundary": cliff,
        "num_samples": len(all_samples),
    }

    results_path = os.path.join(cfg.output.results_dir, "phase1_geometry_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # ---- Checkpoint 3 Evaluation ----
    logger.info("=" * 60)
    logger.info("CHECKPOINT 3: Geometric Collapse Assessment")
    logger.info("=" * 60)

    if cliff is not None:
        cliff_idx = np.argmin(np.abs(layers - cliff))

        # Check 1: Visual ER drops around cliff, text ER stable
        er_before = visual_er_mean[:cliff_idx].mean() if cliff_idx > 0 else visual_er_mean[0]
        er_after = visual_er_mean[cliff_idx:].mean()
        er_drop_ratio = er_after / er_before if er_before > 0 else 1.0

        text_er_before = text_er_mean[:cliff_idx].mean() if cliff_idx > 0 else text_er_mean[0]
        text_er_after = text_er_mean[cliff_idx:].mean()
        text_er_ratio = text_er_after / text_er_before if text_er_before > 0 else 1.0

        logger.info(f"Visual ER ratio (post/pre cliff): {er_drop_ratio:.3f}")
        logger.info(f"Text ER ratio (post/pre cliff): {text_er_ratio:.3f}")

        # Check 2: Visual ICC increases
        icc_before = visual_icc_mean[:cliff_idx].mean() if cliff_idx > 0 else visual_icc_mean[0]
        icc_after = visual_icc_mean[cliff_idx:].mean()
        logger.info(f"Visual ICC: before cliff={icc_before:.3f}, after cliff={icc_after:.3f}")

        # Check 3: CKA increases
        cka_before = cka_mean[:cliff_idx].mean() if cliff_idx > 0 else cka_mean[0]
        cka_after = cka_mean[cliff_idx:].mean()
        logger.info(f"CKA: before cliff={cka_before:.3f}, after cliff={cka_after:.3f}")

        go_signals = 0
        if er_drop_ratio < 0.7:  # Visual ER drops significantly
            logger.info("✓ Visual effective rank drops sharply around cliff")
            go_signals += 1
        if text_er_ratio > 0.85:  # Text ER stays stable
            logger.info("✓ Text effective rank remains stable")
            go_signals += 1
        if icc_after > icc_before + 0.05:  # Visual ICC increases
            logger.info("✓ Visual cosine concentration increases (over-smoothing)")
            go_signals += 1
        if cka_after > cka_before + 0.05:  # CKA increases
            logger.info("✓ Cross-modal CKA increases (visual absorbed by text)")
            go_signals += 1

        if go_signals >= 3:
            logger.info(f"GO ({go_signals}/4): Strong evidence for geometric collapse")
        elif go_signals >= 2:
            logger.info(f"MARGINAL ({go_signals}/4): Some evidence, needs more investigation")
        else:
            logger.warning(f"NO-GO ({go_signals}/4): Insufficient evidence for geometric collapse")
    else:
        logger.warning("No cliff boundary available; run causal tracing first")


if __name__ == "__main__":
    main()
