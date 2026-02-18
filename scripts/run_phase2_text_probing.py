"""Phase 2: Text-token linear probing for assimilation analysis.

This script measures whether visual information becomes decodable from
selected token states (e.g., last instruction token) across layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.probing import TextTokenProbeExperiment, infer_probe_label
from src.data.dataset_loader import load_dataset_for_eval
from src.models.input_preparation import prepare_model_input
from src.models.model_loader import find_visual_token_positions, load_model


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Text Token Probing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="vqav2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--probe_task", type=str, default="yes_no", choices=["yes_no", "color"])
    parser.add_argument("--cv_folds", type=int, default=5)
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
    log_path = os.path.join(cfg.output.results_dir, "run_phase2_text_probing.log")
    logger.add(log_path, rotation="10 MB")

    probe_targets = list(getattr(cfg.analysis, "probe_targets", ["text_instruction_token"]))

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
    samples = load_dataset_for_eval(
        args.dataset,
        split=args.split,
        cache_dir=cfg.data.cache_dir,
        max_samples=args.max_samples,
    )

    label_fn = lambda s: infer_probe_label(s, task=args.probe_task)

    def prepare_inputs_fn(sample: dict) -> dict:
        model_inputs, _ = prepare_model_input(sample, bundle, cfg.model.device)
        return model_inputs

    def visual_range_fn(sample: dict, model_inputs: dict) -> tuple[int, int]:
        return find_visual_token_positions(bundle, model_inputs["input_ids"])

    experiment = TextTokenProbeExperiment(bundle.model)
    features, labels, sample_ids = experiment.extract_features(
        samples=samples,
        prepare_inputs_fn=prepare_inputs_fn,
        visual_range_fn=visual_range_fn,
        label_fn=label_fn,
        probe_targets=probe_targets,
        show_progress=True,
    )

    target_results = {}
    for target in probe_targets:
        result = experiment.run_linear_probe(
            features_per_layer=features[target],
            labels=labels,
            target_name=target,
            cv_folds=args.cv_folds,
        )
        target_results[target] = result.to_dict()

    output = {
        "model": cfg.model.name,
        "dataset": args.dataset,
        "split": args.split,
        "probe_task": args.probe_task,
        "probe_targets": probe_targets,
        "num_valid_samples": int(labels.shape[0]),
        "sample_ids": sample_ids,
        "results": target_results,
    }

    output_path = os.path.join(cfg.output.results_dir, "phase2_text_probing_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Saved probing results to {}", output_path)


if __name__ == "__main__":
    main()
