"""Model-specific multimodal input preparation helpers."""

from __future__ import annotations

import torch


def prepare_model_input(sample: dict, model_bundle, device: str):
    """Prepare a single sample for model inference.

    Returns:
        (model_inputs, answer_token_ids)
    """
    model_name = model_bundle.model_name.lower()
    if "llava" in model_name:
        return _prepare_llava_input(sample, model_bundle.processor, device)
    if "qwen" in model_name:
        return _prepare_qwen_input(sample, model_bundle.processor, device)
    raise ValueError(f"Unsupported model family for input preparation: {model_bundle.model_name}")


def evaluate_contains_answer(model, model_inputs, sample, tokenizer, max_new_tokens: int = 128) -> bool:
    """Generate text and check whether the reference answer is contained in output."""
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = model_inputs["input_ids"].shape[1]
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip().lower()
    answer = sample["answer"].strip().lower()
    return answer in generated


def _prepare_llava_input(sample: dict, processor, device: str):
    image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]

    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_tokens = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)
    return inputs, answer_token_ids


def _prepare_qwen_input(sample: dict, processor, device: str):
    image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_tokens = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)
    return inputs, answer_token_ids
