"""Activation extraction hooks for transformer layers in MLLMs.

Registers forward hooks on each transformer layer to capture hidden states,
separating visual token representations from text token representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerActivations:
    """Stores hidden states for a single layer, split by modality."""

    layer_idx: int
    visual: Optional[torch.Tensor] = None   # (N_v, d)
    text: Optional[torch.Tensor] = None      # (N_t, d)
    full: Optional[torch.Tensor] = None      # (N_v + N_t, d)


class ActivationExtractor:
    """Hooks into transformer layers to extract per-layer hidden states.

    Supports LLaVA-1.5 and Qwen-VL model families. Automatically identifies
    visual vs text token positions based on model architecture.

    Usage:
        extractor = ActivationExtractor(model, visual_token_range=(1, 577))
        with extractor:
            output = model.generate(**inputs)
        activations = extractor.get_activations()
    """

    def __init__(
        self,
        model: nn.Module,
        visual_token_range: Optional[tuple[int, int]] = None,
        layers: Optional[list[int]] = None,
        detach: bool = True,
        store_on_cpu: bool = True,
    ):
        """
        Args:
            model: The MLLM (e.g., LLaVA model).
            visual_token_range: (start, end) indices of visual tokens in the
                input sequence. If None, must be set before forward pass via
                set_visual_token_range().
            layers: List of layer indices to hook. None = all layers.
            detach: Whether to detach tensors from computation graph.
            store_on_cpu: Whether to move captured activations to CPU.
        """
        self.model = model
        self.visual_token_range = visual_token_range
        self.target_layers = layers
        self.detach = detach
        self.store_on_cpu = store_on_cpu

        self._activations: dict[int, LayerActivations] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

        self._transformer_layers = self._find_transformer_layers()

    def _find_transformer_layers(self) -> list[nn.Module]:
        """Locate transformer decoder layers in the model.

        Supports multiple model architectures by searching for common
        layer container names.
        """
        # Qwen3-VL-MoE: model.model.language_model.layers
        if hasattr(self.model, "model"):
            inner = self.model.model
            if hasattr(inner, "language_model") and hasattr(inner.language_model, "layers"):
                return list(inner.language_model.layers)

        # LLaVA / LLaMA architecture: model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)

        # HuggingFace LlavaForConditionalGeneration
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return list(lm.model.layers)

        # Qwen-VL (old): model.transformer.h
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)

        raise ValueError(
            "Cannot find transformer layers. Supported: LLaVA, Qwen-VL, Qwen3-VL-MoE. "
            "Please subclass and override _find_transformer_layers()."
        )

    def set_visual_token_range(self, start: int, end: int):
        """Set the index range [start, end) of visual tokens in the sequence."""
        self.visual_token_range = (start, end)

    @property
    def num_layers(self) -> int:
        return len(self._transformer_layers)

    def _make_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""

        def hook_fn(module, input, output):
            # output is usually a tuple; hidden states are the first element
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # hidden: (batch, seq_len, hidden_dim) — we take batch=0
            h = hidden[0]  # (seq_len, hidden_dim)

            if self.detach:
                h = h.detach()
            if self.store_on_cpu:
                h = h.cpu()

            act = LayerActivations(layer_idx=layer_idx, full=h)

            if self.visual_token_range is not None:
                vstart, vend = self.visual_token_range
                act.visual = h[vstart:vend]
                act.text = torch.cat([h[:vstart], h[vend:]], dim=0)

            self._activations[layer_idx] = act

        return hook_fn

    def register_hooks(self):
        """Register forward hooks on target layers."""
        self.clear()
        for idx, layer in enumerate(self._transformer_layers):
            if self.target_layers is not None and idx not in self.target_layers:
                continue
            hook = layer.register_forward_hook(self._make_hook(idx))
            self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear(self):
        """Clear stored activations and remove hooks."""
        self._activations.clear()
        self.remove_hooks()

    def get_activations(self) -> dict[int, LayerActivations]:
        """Return all captured layer activations."""
        return dict(self._activations)

    def get_visual_activations(self) -> dict[int, torch.Tensor]:
        """Return visual token hidden states per layer. {layer_idx: (N_v, d)}"""
        return {
            idx: act.visual
            for idx, act in self._activations.items()
            if act.visual is not None
        }

    def get_text_activations(self) -> dict[int, torch.Tensor]:
        """Return text token hidden states per layer. {layer_idx: (N_t, d)}"""
        return {
            idx: act.text
            for idx, act in self._activations.items()
            if act.text is not None
        }

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, *args):
        self.remove_hooks()


class PatchingHook:
    """Hook that patches (corrupts) visual token hidden states at a target layer.

    Used for causal tracing: replaces visual token activations at layer l
    with corrupted versions to measure causal effect.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer_idx: int,
        corruption_fn: callable,
        visual_token_range: tuple[int, int],
        transformer_layers: Optional[list[nn.Module]] = None,
    ):
        """
        Args:
            model: The MLLM.
            target_layer_idx: Which layer to patch.
            corruption_fn: Function (hidden_states_visual) -> corrupted_states.
            visual_token_range: (start, end) of visual tokens.
            transformer_layers: Pre-found transformer layers (optional).
        """
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.corruption_fn = corruption_fn
        self.visual_token_range = visual_token_range

        if transformer_layers is not None:
            self._layers = transformer_layers
        else:
            # Reuse the same discovery logic
            temp = ActivationExtractor(model)
            self._layers = temp._transformer_layers

        self._hook = None

    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        vstart, vend = self.visual_token_range
        # Clone to avoid in-place modification issues
        hidden = hidden.clone()
        visual_states = hidden[:, vstart:vend, :]  # (batch, N_v, d)
        corrupted = self.corruption_fn(visual_states)
        hidden[:, vstart:vend, :] = corrupted

        if rest is not None:
            return (hidden,) + rest
        return hidden

    def register(self):
        layer = self._layers[self.target_layer_idx]
        self._hook = layer.register_forward_hook(self._hook_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


class TruncationHook:
    """Hook that removes visual tokens from the residual stream at layers > l.

    For the truncation experiment: at each layer beyond the truncation point,
    visual token hidden states are replaced with zeros (effectively removing
    their contribution to subsequent computation).
    """

    def __init__(
        self,
        model: nn.Module,
        truncation_layer: int,
        visual_token_range: tuple[int, int],
        transformer_layers: Optional[list[nn.Module]] = None,
    ):
        self.model = model
        self.truncation_layer = truncation_layer
        self.visual_token_range = visual_token_range

        if transformer_layers is not None:
            self._layers = transformer_layers
        else:
            temp = ActivationExtractor(model)
            self._layers = temp._transformer_layers

        self._hooks = []

    def _make_zero_hook(self):
        vstart, vend = self.visual_token_range

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = None

            hidden[:, vstart:vend, :] = 0.0

            if rest is not None:
                return (hidden,) + rest
            return hidden

        return hook_fn

    def register(self):
        self.remove()
        for idx in range(self.truncation_layer + 1, len(self._layers)):
            hook = self._layers[idx].register_forward_hook(self._make_zero_hook())
            self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


class SoftTruncationHook:
    """Replace visual tokens with controlled substitutes at layers > l.

    Unlike TruncationHook (hard zero), this preserves activation statistics
    to rule out distribution-shift artifacts.

    Supported replacement strategies:
        - "dataset_mean": Replace with pre-computed mean visual embedding
        - "norm_matched_noise": Gaussian noise matched to per-token L2 norm
        - "pca_matched_noise": Noise matching top-k singular directions
        - "shuffle": Permute visual tokens within the sequence
    """

    def __init__(
        self,
        model: nn.Module,
        truncation_layer: int,
        visual_token_range: tuple[int, int],
        strategy: str = "norm_matched_noise",
        reference_states: Optional[torch.Tensor] = None,
        pca_rank: int = 16,
        seed: int = 42,
        transformer_layers: Optional[list[nn.Module]] = None,
    ):
        self.model = model
        self.truncation_layer = truncation_layer
        self.visual_token_range = visual_token_range
        self.strategy = strategy
        self.reference_states = reference_states
        self.pca_rank = pca_rank
        self.seed = seed

        if transformer_layers is not None:
            self._layers = transformer_layers
        else:
            temp = ActivationExtractor(model)
            self._layers = temp._transformer_layers

        self._hooks = []

    def _make_replacement_hook(self):
        vstart, vend = self.visual_token_range
        strategy = self.strategy
        reference = self.reference_states
        pca_rank = self.pca_rank
        seed = self.seed
        call_counter = [0]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = None

            visual = hidden[:, vstart:vend, :]  # (batch, N_v, d)

            if strategy == "dataset_mean":
                if reference is not None:
                    replacement = reference.to(visual.device, visual.dtype)
                    if replacement.dim() == 2:
                        replacement = replacement.unsqueeze(0)
                    hidden[:, vstart:vend, :] = replacement
                else:
                    hidden[:, vstart:vend, :] = visual.mean(dim=1, keepdim=True)

            elif strategy == "norm_matched_noise":
                gen = torch.Generator(device=visual.device)
                gen.manual_seed(seed + call_counter[0])
                call_counter[0] += 1
                noise = torch.randn(visual.shape, generator=gen,
                                    device=visual.device, dtype=visual.dtype)
                token_norms = visual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                noise_norms = noise.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden[:, vstart:vend, :] = noise * (token_norms / noise_norms)

            elif strategy == "pca_matched_noise":
                gen = torch.Generator(device=visual.device)
                gen.manual_seed(seed + call_counter[0])
                call_counter[0] += 1
                v_flat = visual[0]  # (N_v, d)
                mean = v_flat.mean(dim=0, keepdim=True)
                centered = v_flat - mean
                U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
                k = min(pca_rank, S.shape[0])
                noise_coeffs = torch.randn(v_flat.shape[0], k,
                                           generator=gen, device=visual.device,
                                           dtype=visual.dtype)
                noise_coeffs = noise_coeffs * S[:k].unsqueeze(0)
                replacement = noise_coeffs @ Vt[:k, :] + mean
                hidden[:, vstart:vend, :] = replacement.unsqueeze(0)

            elif strategy == "shuffle":
                gen = torch.Generator(device=visual.device)
                gen.manual_seed(seed + call_counter[0])
                call_counter[0] += 1
                n_vis = vend - vstart
                perm = torch.randperm(n_vis, generator=gen, device=visual.device)
                hidden[:, vstart:vend, :] = visual[:, perm, :]

            else:
                raise ValueError(f"Unknown soft truncation strategy: {strategy}")

            if rest is not None:
                return (hidden,) + rest
            return hidden

        return hook_fn

    def register(self):
        self.remove()
        for idx in range(self.truncation_layer + 1, len(self._layers)):
            hook = self._layers[idx].register_forward_hook(
                self._make_replacement_hook()
            )
            self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()


class DeepLayerVisualMaskHook:
    """Mask visual tokens ONLY in a specified layer range.

    Unlike TruncationHook (which masks layers > l), this masks layers in
    [mask_start, mask_end). This tests whether deep layers functionally
    need visual tokens for re-grounding.

    Example: Keep visual tokens in layers 0-25, mask in layers 26-47.
    If accuracy drops, deep layers still need visual information.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_start_layer: int,
        mask_end_layer: int,
        visual_token_range: tuple[int, int],
        strategy: str = "zero",
        reference_states: Optional[torch.Tensor] = None,
        seed: int = 42,
        transformer_layers: Optional[list[nn.Module]] = None,
    ):
        self.model = model
        self.mask_start_layer = mask_start_layer
        self.mask_end_layer = mask_end_layer
        self.visual_token_range = visual_token_range
        self.strategy = strategy
        self.reference_states = reference_states
        self.seed = seed

        if transformer_layers is not None:
            self._layers = transformer_layers
        else:
            temp = ActivationExtractor(model)
            self._layers = temp._transformer_layers

        self._hooks = []

    def _make_mask_hook(self):
        vstart, vend = self.visual_token_range
        strategy = self.strategy
        reference = self.reference_states
        seed = self.seed
        call_counter = [0]

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0].clone()
                rest = output[1:]
            else:
                hidden = output.clone()
                rest = None

            visual = hidden[:, vstart:vend, :]

            if strategy == "zero":
                hidden[:, vstart:vend, :] = 0.0
            elif strategy == "norm_matched_noise":
                gen = torch.Generator(device=visual.device)
                gen.manual_seed(seed + call_counter[0])
                call_counter[0] += 1
                noise = torch.randn(visual.shape, generator=gen,
                                    device=visual.device, dtype=visual.dtype)
                token_norms = visual.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                noise_norms = noise.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden[:, vstart:vend, :] = noise * (token_norms / noise_norms)
            elif strategy == "dataset_mean":
                if reference is not None:
                    replacement = reference.to(visual.device, visual.dtype)
                    if replacement.dim() == 2:
                        replacement = replacement.unsqueeze(0)
                    hidden[:, vstart:vend, :] = replacement
                else:
                    hidden[:, vstart:vend, :] = visual.mean(dim=1, keepdim=True)
            else:
                raise ValueError(f"Unknown masking strategy: {strategy}")

            if rest is not None:
                return (hidden,) + rest
            return hidden

        return hook_fn

    def register(self):
        self.remove()
        for idx in range(self.mask_start_layer, min(self.mask_end_layer, len(self._layers))):
            hook = self._layers[idx].register_forward_hook(self._make_mask_hook())
            self._hooks.append(hook)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.register()
        return self

    def __exit__(self, *args):
        self.remove()
