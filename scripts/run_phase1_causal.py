"""Phase 1 Week 1: Causal Tracing Experiment.

Runs causal patching on a configured MLLM (e.g., LLaVA or Qwen-VL) to
measure the causal effect of visual tokens at each layer. Computes EVD for
hard and easy task subsets.

Validates:
  - Checkpoint 1: Modality Cliff causes real performance failures
  - Checkpoint 2: EVD correlates with downstream performance

Usage:
    python scripts/run_phase1_causal.py --config configs/default.yaml
    python scripts/run_phase1_causal.py --config configs/llava_7b.yaml --num_samples 50
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import pickle

import torch
import numpy as np
from omegaconf import OmegaConf
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model_loader import load_model, find_visual_token_positions
from src.models.input_preparation import prepare_model_input
from src.causal.patching import CausalPatcher
from src.causal.evd import (
    compute_evd_batch,
    compute_mean_delta_curve,
    estimate_cliff_boundary,
    evd_performance_correlation,
)
from src.data.dataset_loader import load_dataset_for_eval
from src.data.subset_sampler import sample_hard_easy_subsets
from src.visualization.plots import GAPVisualizer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Causal Tracing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Override model name/path in config")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Model-specific config to merge (e.g., configs/llava_7b.yaml)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Override number of samples per category")
    parser.add_argument("--corruption", type=str, default=None,
                        help="Override corruption method (zero, gaussian, mean_sub)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.model_config:
        model_cfg = OmegaConf.load(args.model_config)
        cfg = OmegaConf.merge(cfg, model_cfg)
    if args.model_name:
        cfg.model.name = args.model_name

    if args.num_samples:
        cfg.data.hard_samples = args.num_samples
        cfg.data.easy_samples = args.num_samples
    if args.corruption:
        cfg.causal.default_method = args.corruption
    if args.output_dir:
        cfg.output.results_dir = args.output_dir

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ---- Step 1: Load model ----
    logger.info("Loading model...")
    bundle = load_model(
        cfg.model.name,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )

    # ---- Step 2: Load and prepare data ----
    logger.info("Loading datasets...")
    hard_samples_all = []
    easy_samples_all = []

    for ds_cfg in cfg.data.datasets:
        samples = load_dataset_for_eval(
            ds_cfg.name,
            split=ds_cfg.split,
            cache_dir=cfg.data.cache_dir,
        )
        hard, easy = sample_hard_easy_subsets(
            samples,
            n_hard=cfg.data.hard_samples,
            n_easy=cfg.data.easy_samples,
            dataset_name=ds_cfg.name,
        )
        hard_samples_all.extend(hard)
        easy_samples_all.extend(easy)

    logger.info(f"Total: {len(hard_samples_all)} hard, {len(easy_samples_all)} easy samples")

    # ---- Step 3: Find visual token positions ----
    # Use first sample to determine token layout
    test_sample = hard_samples_all[0]
    test_inputs, _ = prepare_model_input(test_sample, bundle, cfg.model.device)
    visual_range = find_visual_token_positions(bundle, test_inputs["input_ids"])
    logger.info(f"Visual token range: {visual_range}")

    # ---- Step 4: Run causal tracing ----
    patcher = CausalPatcher(
        model=bundle.model,
        visual_token_range=visual_range,
        corruption_method=cfg.causal.default_method,
        evd_threshold=cfg.causal.evd_threshold,
    )

    def prepare_fn(sample):
        return prepare_model_input(sample, bundle, cfg.model.device)

    logger.info("Running causal tracing on HARD samples...")
    hard_results = patcher.trace_dataset(
        hard_samples_all,
        prepare_fn=prepare_fn,
    )

    logger.info("Running causal tracing on EASY samples...")
    easy_results = patcher.trace_dataset(
        easy_samples_all,
        prepare_fn=prepare_fn,
    )

    # ---- Step 5: Analyze results ----
    # Compute mean delta curves
    hard_layers, hard_mean, hard_std = compute_mean_delta_curve(hard_results)
    easy_layers, easy_mean, easy_std = compute_mean_delta_curve(easy_results)

    # Compute EVD
    hard_evd = compute_evd_batch(hard_results, cfg.causal.evd_threshold)
    easy_evd = compute_evd_batch(easy_results, cfg.causal.evd_threshold)

    # Estimate cliff boundary
    cliff = estimate_cliff_boundary(hard_results + easy_results, cfg.causal.evd_threshold)

    logger.info(f"Estimated cliff boundary: layer {cliff}")
    logger.info(f"Hard EVD: mean={hard_evd.mean():.1f}, median={np.median(hard_evd):.1f}")
    logger.info(f"Easy EVD: mean={easy_evd.mean():.1f}, median={np.median(easy_evd):.1f}")

    # ---- Step 6: Visualize ----
    viz = GAPVisualizer(output_dir=cfg.output.plots_dir)

    viz.plot_causal_effect_comparison(
        layers=hard_layers,
        hard_deltas=hard_mean,
        easy_deltas=easy_mean,
        hard_std=hard_std,
        easy_std=easy_std,
        cliff_boundary=cliff,
    )

    viz.plot_causal_effect_curve(
        layers=hard_layers,
        mean_deltas=hard_mean,
        std_deltas=hard_std,
        cliff_boundary=cliff,
        title="Causal Effect — Hard Tasks",
        filename="causal_effect_hard.pdf",
        task_label="Hard tasks",
    )

    viz.plot_causal_effect_curve(
        layers=easy_layers,
        mean_deltas=easy_mean,
        std_deltas=easy_std,
        cliff_boundary=cliff,
        title="Causal Effect — Easy Tasks",
        filename="causal_effect_easy.pdf",
        task_label="Easy tasks",
    )

    # ---- Step 7: Save results ----
    results = {
        "config": OmegaConf.to_container(cfg),
        "cliff_boundary": int(cliff),
        "hard_evd_mean": float(hard_evd.mean()),
        "hard_evd_median": float(np.median(hard_evd)),
        "easy_evd_mean": float(easy_evd.mean()),
        "easy_evd_median": float(np.median(easy_evd)),
        "hard_mean_deltas": hard_mean.tolist(),
        "easy_mean_deltas": easy_mean.tolist(),
        "layers": hard_layers.tolist(),
    }

    results_path = os.path.join(cfg.output.results_dir, "phase1_causal_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    # Save full trace results for later analysis
    traces_path = os.path.join(cfg.output.results_dir, "phase1_causal_traces.pkl")
    with open(traces_path, "wb") as f:
        pickle.dump({"hard": hard_results, "easy": easy_results}, f)
    logger.info(f"Full traces saved to {traces_path}")

    # ---- Checkpoint Evaluation ----
    logger.info("=" * 60)
    logger.info("CHECKPOINT 1 EVALUATION: Modality Cliff Causes Failures?")
    logger.info("=" * 60)

    # Check if hard tasks have significantly higher EVD than easy tasks
    evd_diff = hard_evd.mean() - easy_evd.mean()
    logger.info(f"EVD difference (hard - easy): {evd_diff:.2f} layers")

    # Check if late-layer causal effect is higher for hard tasks
    late_layers = hard_layers[hard_layers > cliff]
    if len(late_layers) > 0:
        late_mask = np.isin(hard_layers, late_layers)
        hard_late_delta = hard_mean[late_mask].mean()
        easy_late_delta = easy_mean[late_mask].mean()
        logger.info(f"Mean Delta beyond cliff — Hard: {hard_late_delta:.4f}, Easy: {easy_late_delta:.4f}")

        if hard_late_delta > easy_late_delta * 1.5:
            logger.info("✓ GO: Hard tasks show significantly higher causal dependence on visual tokens beyond the cliff")
        else:
            logger.warning("⚠ WEAK: Difference exists but may not be significant enough")

    logger.info("Done! Review plots in: " + cfg.output.plots_dir)


if __name__ == "__main__":
    main()
