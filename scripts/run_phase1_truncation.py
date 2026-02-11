"""Phase 1 Week 1: Visual Token Truncation Experiment.

At each layer l, removes visual tokens entirely from the residual stream
for layers > l, then measures downstream accuracy.

Validates Checkpoint 1: Is there a task category where truncation at the
cliff boundary significantly hurts performance?

Go criterion: >5% accuracy drop on hard cases vs <1% on easy cases.

Usage:
    python scripts/run_phase1_truncation.py --config configs/default.yaml
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
from src.causal.truncation import TruncationExperiment
from src.data.dataset_loader import load_dataset_for_eval
from src.data.subset_sampler import sample_hard_easy_subsets
from src.visualization.plots import GAPVisualizer


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Truncation Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--cliff_boundary", type=int, default=None,
                        help="Pre-computed cliff boundary from causal tracing")
    parser.add_argument("--step", type=int, default=2,
                        help="Step size for truncation layer sweep")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.model_config:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.model_config))

    os.makedirs(cfg.output.results_dir, exist_ok=True)
    os.makedirs(cfg.output.plots_dir, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    bundle = load_model(cfg.model.name, device=cfg.model.device, dtype=cfg.model.dtype)

    # Load data
    logger.info("Loading datasets...")
    hard_samples = []
    easy_samples = []
    for ds_cfg in cfg.data.datasets:
        samples = load_dataset_for_eval(ds_cfg.name, split=ds_cfg.split, cache_dir=cfg.data.cache_dir)
        h, e = sample_hard_easy_subsets(
            samples, n_hard=cfg.data.hard_samples, n_easy=cfg.data.easy_samples,
            dataset_name=ds_cfg.name,
        )
        hard_samples.extend(h)
        easy_samples.extend(e)

    # Find visual token positions
    from scripts.run_phase1_causal import prepare_llava_input
    test_inputs, _ = prepare_llava_input(
        hard_samples[0], bundle.processor, bundle.model, cfg.model.device
    )
    visual_range = find_visual_token_positions(bundle, test_inputs["input_ids"])

    # Setup truncation experiment
    trunc_exp = TruncationExperiment(
        model=bundle.model,
        visual_token_range=visual_range,
    )

    num_layers = trunc_exp.num_layers
    truncation_layers = list(range(0, num_layers, args.step))

    def prepare_fn(sample):
        inputs, _ = prepare_llava_input(
            sample, bundle.processor, bundle.model, cfg.model.device
        )
        return inputs

    def evaluate_fn(model, model_inputs, sample):
        with torch.no_grad():
            output_ids = model.generate(**model_inputs, max_new_tokens=128, do_sample=False)
        input_len = model_inputs["input_ids"].shape[1]
        generated = bundle.processor.tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip().lower()
        return sample["answer"].strip().lower() in generated

    # Run truncation on hard samples
    logger.info("Running truncation sweep on HARD samples...")
    hard_results = trunc_exp.run_truncation_sweep(
        hard_samples, prepare_fn, evaluate_fn, truncation_layers
    )

    # Run truncation on easy samples
    logger.info("Running truncation sweep on EASY samples...")
    easy_results = trunc_exp.run_truncation_sweep(
        easy_samples, prepare_fn, evaluate_fn, truncation_layers
    )

    # Extract curves
    hard_baseline = next(r for r in hard_results if r.truncation_layer == -1)
    easy_baseline = next(r for r in easy_results if r.truncation_layer == -1)

    hard_trunc = [r for r in hard_results if r.truncation_layer >= 0]
    easy_trunc = [r for r in easy_results if r.truncation_layer >= 0]

    layers_arr = np.array([r.truncation_layer for r in hard_trunc])
    hard_acc = np.array([r.accuracy for r in hard_trunc])
    easy_acc = np.array([r.accuracy for r in easy_trunc])

    # Compute accuracy drops
    hard_drops = TruncationExperiment.compute_accuracy_drop(hard_results)
    easy_drops = TruncationExperiment.compute_accuracy_drop(easy_results)

    # Determine cliff boundary
    cliff = args.cliff_boundary
    if cliff is None:
        # Load from causal tracing results if available
        causal_results_path = os.path.join(cfg.output.results_dir, "phase1_causal_results.json")
        if os.path.exists(causal_results_path):
            with open(causal_results_path) as f:
                causal_data = json.load(f)
                cliff = causal_data.get("cliff_boundary")

    # Visualize
    viz = GAPVisualizer(output_dir=cfg.output.plots_dir)
    viz.plot_truncation_curves(
        layers=layers_arr,
        hard_accuracy=hard_acc,
        easy_accuracy=easy_acc,
        baseline_hard=hard_baseline.accuracy,
        baseline_easy=easy_baseline.accuracy,
        cliff_boundary=cliff,
    )

    # Save results
    results = {
        "hard_baseline_accuracy": hard_baseline.accuracy,
        "easy_baseline_accuracy": easy_baseline.accuracy,
        "truncation_layers": layers_arr.tolist(),
        "hard_accuracy": hard_acc.tolist(),
        "easy_accuracy": easy_acc.tolist(),
        "hard_accuracy_drops": hard_drops,
        "easy_accuracy_drops": easy_drops,
        "cliff_boundary": cliff,
    }

    results_path = os.path.join(cfg.output.results_dir, "phase1_truncation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # ---- Checkpoint 1 Evaluation ----
    logger.info("=" * 60)
    logger.info("CHECKPOINT 1: Truncation Impact Assessment")
    logger.info("=" * 60)

    if cliff is not None and cliff in hard_drops:
        hard_drop_at_cliff = hard_drops[cliff]
        easy_drop_at_cliff = easy_drops.get(cliff, 0)

        logger.info(f"At cliff boundary (layer {cliff}):")
        logger.info(f"  Hard task accuracy drop: {hard_drop_at_cliff:.2%}")
        logger.info(f"  Easy task accuracy drop: {easy_drop_at_cliff:.2%}")

        if hard_drop_at_cliff > 0.05 and easy_drop_at_cliff < 0.01:
            logger.info("✓ GO: Clear evidence that cliff causes failures on hard tasks")
        elif hard_drop_at_cliff > 0.03:
            logger.info("⚠ MARGINAL: Some evidence, consider increasing sample size")
        else:
            logger.warning("✗ NO-GO: Truncation does not differentially affect hard tasks")


if __name__ == "__main__":
    main()
