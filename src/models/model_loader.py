"""Model loading utilities for LLaVA and Qwen-VL model families."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import Optional
from loguru import logger


@dataclass
class ModelBundle:
    """Container for a loaded MLLM with its processor and metadata."""

    model: torch.nn.Module
    processor: object  # AutoProcessor or equivalent
    tokenizer: object
    model_name: str
    num_layers: int
    hidden_dim: int
    num_visual_tokens: Optional[int]  # None for dynamic-resolution models
    dtype: torch.dtype


def load_model(
    model_name: str,
    device: str = "cuda",
    dtype: str = "float16",
    attn_implementation: Optional[str] = None,
) -> ModelBundle:
    """Load an MLLM and return a ModelBundle.

    Args:
        model_name: HuggingFace model ID or local path.
        device: Target device.
        dtype: One of "float16", "bfloat16", "float32".
        attn_implementation: Optional attention implementation (e.g., "flash_attention_2").

    Returns:
        ModelBundle with model, processor, and metadata.
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    if "llava" in model_name.lower():
        return _load_llava(model_name, device, torch_dtype, attn_implementation)
    elif "qwen" in model_name.lower():
        return _load_qwen_vl(model_name, device, torch_dtype, attn_implementation)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: LLaVA, Qwen-VL.")


def _load_llava(
    model_name: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: Optional[str],
) -> ModelBundle:
    """Load LLaVA-1.5 model from HuggingFace."""
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    logger.info(f"Loading LLaVA model: {model_name}")

    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device if device != "cpu" else None,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = LlavaForConditionalGeneration.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()

    # LLaVA-1.5-7B: 32 layers, 4096 hidden dim, 576 visual tokens
    num_layers = model.config.text_config.num_hidden_layers
    hidden_dim = model.config.text_config.hidden_size

    # Visual tokens: CLIP ViT-L/14@336 produces 24x24 = 576 patches
    # Plus possibly a CLS token depending on config
    num_visual_tokens = _get_llava_visual_token_count(model, processor)

    logger.info(
        f"LLaVA loaded: {num_layers} layers, {hidden_dim}d, "
        f"{num_visual_tokens} visual tokens"
    )

    return ModelBundle(
        model=model,
        processor=processor,
        tokenizer=processor.tokenizer,
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_visual_tokens=num_visual_tokens,
        dtype=torch_dtype,
    )


def _load_qwen_vl(
    model_name: str,
    device: str,
    torch_dtype: torch.dtype,
    attn_implementation: Optional[str],
) -> ModelBundle:
    """Load Qwen2.5-VL model from HuggingFace."""
    from transformers import AutoProcessor, AutoModelForCausalLM

    logger.info(f"Loading Qwen-VL model: {model_name}")

    kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device if device != "cpu" else None,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)

    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    logger.info(
        f"Qwen-VL loaded: {num_layers} layers, {hidden_dim}d, "
        f"dynamic visual tokens"
    )

    return ModelBundle(
        model=model,
        processor=processor,
        tokenizer=processor.tokenizer,
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        num_visual_tokens=None,  # Dynamic resolution
        dtype=torch_dtype,
    )


def _get_llava_visual_token_count(model, processor) -> int:
    """Determine the number of visual tokens produced by LLaVA's vision encoder."""
    try:
        # Standard LLaVA-1.5: 576 tokens (24x24 patches from ViT-L/14@336)
        vision_config = model.config.vision_config
        image_size = vision_config.image_size
        patch_size = vision_config.patch_size
        num_patches = (image_size // patch_size) ** 2
        return num_patches
    except AttributeError:
        logger.warning("Could not determine visual token count; defaulting to 576")
        return 576


def find_visual_token_positions(
    model_bundle: ModelBundle,
    input_ids: torch.Tensor,
) -> tuple[int, int]:
    """Find the start and end positions of visual tokens in the input sequence.

    For LLaVA, visual tokens are inserted at the <image> token position.
    The <image> placeholder in input_ids is replaced by N visual tokens.

    Args:
        model_bundle: Loaded model bundle.
        input_ids: Token IDs of the input (before image token expansion).

    Returns:
        (start_idx, end_idx) of visual tokens in the expanded sequence.
    """
    if "llava" in model_bundle.model_name.lower():
        # Find the image token ID
        image_token_id = model_bundle.model.config.image_token_index

        # Find position of image token in input_ids
        positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(positions) == 0:
            raise ValueError("No image token found in input_ids")

        start = positions[0].item()
        end = start + model_bundle.num_visual_tokens
        return (start, end)

    elif "qwen" in model_bundle.model_name.lower():
        # For Qwen-VL, visual tokens are marked with special tokens
        # This needs model-specific handling based on the processor output
        raise NotImplementedError(
            "Qwen-VL visual token position detection requires processor output. "
            "Use processor.get_visual_token_mask() or equivalent."
        )

    else:
        raise ValueError(f"Unsupported model: {model_bundle.model_name}")
