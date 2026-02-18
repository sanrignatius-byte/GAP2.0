"""Phase 2: Text-token linear probing for assimilation analysis.

This script measures whether visual information becomes decodable from
selected token states (e.g., last instruction token) across layers.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Callable

from PIL import Image, ImageStat

from loguru import logger
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal.probing import TextTokenProbeExperiment, infer_probe_label
from src.data.dataset_loader import load_dataset_for_eval
from src.models.input_preparation import prepare_model_input
from src.models.model_loader import find_visual_token_positions, load_model


def _build_probe_index_fn(bundle) -> Callable[[object, tuple[int, int], list[str], dict | None], dict[str, int]]:
    """Build a tokenizer-aware probe index selector.

    This keeps:
      - text_instruction_token at the end of user content
      - answer_token at the first teacher-forced answer token when available
        (fallback to assistant-side context token only if metadata is missing)
    """
    tokenizer = bundle.tokenizer
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    assistant_first_id = assistant_ids[0] if assistant_ids else None

    llava_assistant_ids = tokenizer.encode("ASSISTANT", add_special_tokens=False)
    llava_assistant_first_id = llava_assistant_ids[0] if llava_assistant_ids else None

    def token_index_fn(input_ids, visual_token_range, probe_targets, aux=None):
        seq = input_ids[0].tolist()
        seq_len = len(seq)
        vstart, vend = visual_token_range

        def is_visual(idx: int) -> bool:
            return vstart <= idx < vend

        valid_text = [idx for idx in range(seq_len) if not is_visual(idx)]
        if len(valid_text) == 0:
            raise ValueError("No text token positions available outside visual range")

        def find_prev_content_idx(start_idx: int) -> int:
            idx = start_idx
            while idx >= 0:
                token_id = seq[idx]
                if not is_visual(idx) and token_id not in special_ids:
                    return idx
                idx -= 1
            return -1

        assistant_start = None
        if im_start_id is not None and im_start_id >= 0:
            im_start_positions = [
                idx for idx, token_id in enumerate(seq)
                if token_id == im_start_id and not is_visual(idx)
            ]
            if im_start_positions:
                for pos in reversed(im_start_positions):
                    if (
                        assistant_first_id is not None
                        and pos + 1 < seq_len
                        and seq[pos + 1] == assistant_first_id
                    ):
                        assistant_start = pos
                        break
                if assistant_start is None:
                    assistant_start = im_start_positions[-1]

        if assistant_start is None and llava_assistant_first_id is not None:
            llava_positions = [
                idx for idx, token_id in enumerate(seq)
                if token_id == llava_assistant_first_id and not is_visual(idx)
            ]
            if llava_positions:
                assistant_start = llava_positions[-1]

        # Defaults: fall back to the final non-visual prompt token.
        text_instruction_idx = valid_text[-1]
        answer_idx = valid_text[-1]

        if aux is not None and aux.get("answer_token_start_idx") is not None:
            answer_idx = int(aux["answer_token_start_idx"])

        if assistant_start is not None:
            if aux is None or aux.get("answer_token_start_idx") is None:
                answer_candidates = [idx for idx in range(assistant_start, seq_len) if not is_visual(idx)]
                if answer_candidates:
                    answer_idx = answer_candidates[-1]

            # Qwen-style primary choice: token before user-turn <|im_end|>.
            if im_end_id is not None and im_end_id >= 0:
                im_end_positions = [
                    idx for idx in range(assistant_start - 1, -1, -1)
                    if seq[idx] == im_end_id and not is_visual(idx)
                ]
                if im_end_positions:
                    cand = find_prev_content_idx(im_end_positions[0] - 1)
                    if cand >= 0:
                        text_instruction_idx = cand

            # Fallback: token right before assistant turn starts.
            if text_instruction_idx == valid_text[-1]:
                cand = find_prev_content_idx(assistant_start - 1)
                if cand >= 0:
                    text_instruction_idx = cand

        # Final safeguard: keep instruction on user side when answer span metadata exists.
        if (
            aux is not None
            and aux.get("answer_token_start_idx") is not None
            and text_instruction_idx >= int(aux["answer_token_start_idx"])
        ):
            cand = find_prev_content_idx(int(aux["answer_token_start_idx"]) - 1)
            if cand >= 0:
                text_instruction_idx = cand

        # Final safeguard: make sure instruction/answer probes are not identical.
        if text_instruction_idx == answer_idx:
            cand = find_prev_content_idx(answer_idx - 1)
            if cand >= 0:
                text_instruction_idx = cand

        target_to_idx: dict[str, int] = {}
        for target in probe_targets:
            if target == "visual_token":
                target_to_idx[target] = int((vstart + vend - 1) // 2)
            elif target == "text_instruction_token":
                target_to_idx[target] = int(text_instruction_idx)
            elif target == "answer_token":
                target_to_idx[target] = int(answer_idx)
            else:
                raise ValueError(f"Unsupported probe target: {target}")

        return target_to_idx

    return token_index_fn


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Text Token Probing")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="vqav2")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--probe_task", type=str, default="yes_no", choices=["yes_no", "color"])
    parser.add_argument(
        "--image_mode",
        type=str,
        default="original",
        choices=["original", "blind_black", "blind_mean", "blind_randomimage"],
        help="Image control mode for probing.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    probe_index_fn = _build_probe_index_fn(bundle)

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

    black_cache: dict[tuple[str, tuple[int, int]], Image.Image] = {}
    blind_mean_rgb: tuple[int, int, int] | None = None
    random_image_map: dict[str, Image.Image] = {}

    if args.image_mode == "blind_mean":
        rgb_means = []
        for sample in samples:
            image = sample["image"]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected PIL.Image in dataset, got {type(image)}")
            rgb_means.append(ImageStat.Stat(image.convert("RGB")).mean)
        if not rgb_means:
            raise RuntimeError("No samples available to compute blind_mean image")
        blind_mean_rgb = tuple(
            int(round(sum(ch[i] for ch in rgb_means) / len(rgb_means))) for i in range(3)
        )
        logger.info("blind_mean RGB color={}", blind_mean_rgb)

    if args.image_mode == "blind_randomimage":
        rng = random.Random(args.seed)
        indices = list(range(len(samples)))
        if len(indices) > 1:
            # Build a derangement so each sample gets a different image.
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
            sid = samples[idx].get("id", f"sample_{idx}")
            random_image_map[sid] = samples[pidx]["image"]
        logger.info("blind_randomimage map ready: {} samples, seed={}", len(random_image_map), args.seed)

    def make_image_control(sample: dict) -> Image.Image:
        image = sample["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL.Image in sample, got {type(image)}")

        if args.image_mode == "blind_black":
            key = (image.mode, image.size)
            if key not in black_cache:
                black_cache[key] = Image.new(image.mode, image.size, 0)
            return black_cache[key]

        if args.image_mode == "blind_mean":
            assert blind_mean_rgb is not None
            mean_rgb_img = Image.new("RGB", image.size, blind_mean_rgb)
            return mean_rgb_img.convert(image.mode)

        if args.image_mode == "blind_randomimage":
            sid = sample.get("id")
            if sid not in random_image_map:
                raise KeyError(f"Missing blind_randomimage mapping for sample id={sid}")
            return random_image_map[sid]

        return image

    def prepare_inputs_fn(sample: dict):
        sample_in = sample
        if args.image_mode != "original":
            controlled_image = make_image_control(sample)
            sample_in = {**sample, "image": controlled_image}
        model_inputs, _, metadata = prepare_model_input(
            sample_in,
            bundle,
            cfg.model.device,
            include_answer_in_prompt=True,
            return_metadata=True,
        )
        return model_inputs, metadata

    def visual_range_fn(sample: dict, model_inputs: dict) -> tuple[int, int]:
        return find_visual_token_positions(bundle, model_inputs["input_ids"])

    experiment = TextTokenProbeExperiment(bundle.model)
    features, labels, sample_ids = experiment.extract_features(
        samples=samples,
        prepare_inputs_fn=prepare_inputs_fn,
        visual_range_fn=visual_range_fn,
        label_fn=label_fn,
        probe_targets=probe_targets,
        token_index_fn=probe_index_fn,
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
        "image_mode": args.image_mode,
        "seed": args.seed,
        "blind_mean_rgb": blind_mean_rgb,
        "probe_targets": probe_targets,
        "probe_index_strategy": "assistant_boundary_v2_teacher_forced",
        "answer_token_source": "teacher_forced_first_label_token",
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
