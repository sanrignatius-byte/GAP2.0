"""Model-specific multimodal input preparation helpers."""

from __future__ import annotations

import torch


def _resolve_device(model_bundle, device: str) -> torch.device:
    """Resolve the actual device when device_map='auto' is used."""
    if device in ("auto", "auto:0"):
        # Infer from the model's first parameter
        return next(model_bundle.model.parameters()).device
    return torch.device(device)


def _find_subsequence_last(haystack: torch.Tensor, needle: torch.Tensor) -> int | None:
    """Find the last exact occurrence of ``needle`` inside ``haystack``.

    Both tensors are expected to be 1D token-id vectors on the same device.
    """
    if needle.numel() == 0 or haystack.numel() < needle.numel():
        return None

    h = haystack.tolist()
    n = needle.tolist()
    nlen = len(n)
    for start in range(len(h) - nlen, -1, -1):
        if h[start:start + nlen] == n:
            return int(start)
    return None


def prepare_model_input(
    sample: dict,
    model_bundle,
    device: str,
    include_answer_in_prompt: bool = False,
    return_metadata: bool = False,
):
    """Prepare a single sample for model inference.

    Returns:
        (model_inputs, answer_token_ids) by default.
        If ``return_metadata=True``, returns
        (model_inputs, answer_token_ids, metadata) where metadata can include:
          - answer_token_start_idx: first answer-token index in input_ids
          - answer_token_end_idx: exclusive end index in input_ids
    """
    resolved_device = _resolve_device(model_bundle, device)
    model_name = model_bundle.model_name.lower()
    if "llava" in model_name:
        return _prepare_llava_input(
            sample,
            model_bundle.processor,
            resolved_device,
            include_answer_in_prompt=include_answer_in_prompt,
            return_metadata=return_metadata,
        )
    if "qwen" in model_name:
        return _prepare_qwen_input(
            sample,
            model_bundle.processor,
            resolved_device,
            include_answer_in_prompt=include_answer_in_prompt,
            return_metadata=return_metadata,
        )
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


def _prepare_llava_input(
    sample: dict,
    processor,
    device: str,
    include_answer_in_prompt: bool = False,
    return_metadata: bool = False,
):
    image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]

    if include_answer_in_prompt:
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
    else:
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_tokens = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)
    metadata: dict[str, int] = {}
    if include_answer_in_prompt:
        start = _find_subsequence_last(inputs["input_ids"][0], answer_token_ids)
        if start is not None:
            metadata["answer_token_start_idx"] = int(start)
            metadata["answer_token_end_idx"] = int(start + answer_token_ids.shape[0])

    if return_metadata:
        return inputs, answer_token_ids, metadata
    return inputs, answer_token_ids


def _prepare_qwen_input(
    sample: dict,
    processor,
    device: str,
    include_answer_in_prompt: bool = False,
    return_metadata: bool = False,
):
    image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]

    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ],
    }]
    if include_answer_in_prompt:
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        })

    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=not include_answer_in_prompt,
    )
    inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    answer_tokens = processor.tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    answer_token_ids = answer_tokens["input_ids"][0].to(device)
    metadata: dict[str, int] = {}
    if include_answer_in_prompt:
        start = _find_subsequence_last(inputs["input_ids"][0], answer_token_ids)
        if start is not None:
            metadata["answer_token_start_idx"] = int(start)
            metadata["answer_token_end_idx"] = int(start + answer_token_ids.shape[0])

    if return_metadata:
        return inputs, answer_token_ids, metadata
    return inputs, answer_token_ids
