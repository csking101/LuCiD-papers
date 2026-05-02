"""PyTorch hook management for extracting transformer hidden states.

This module provides:
- ``get_device()`` — auto-detect best available device (CUDA → MPS → CPU)
- ``load_model()`` — load a HuggingFace causal LM + tokenizer
- ``HiddenStateCapture`` — context manager that registers forward hooks
  on transformer layers and captures hidden states during a forward pass
- ``extract_last_token_features()`` — convenience function to extract the
  last-token hidden state from each layer for a batch of prompts

Architecture note (Qwen2.5-1.5B-Instruct):
    28 layers, hidden_size=1536.
    ``model.model.layers[i]`` is ``Qwen2DecoderLayer``.
    Forward hooks on decoder layers capture hidden states of shape
    ``[batch, seq_len, 1536]``.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Device detection ────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Auto-detect the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Model loading ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ModelInfo:
    """Metadata about the loaded model."""
    name: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    device: str
    dtype: str
    num_params: int


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: torch.device | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ModelInfo]:
    """Load a HuggingFace causal LM and tokenizer.

    Returns (model, tokenizer, info).
    """
    if device is None:
        device = get_device()

    # Choose dtype based on device capabilities
    if device.type == "cuda":
        dtype = torch.float16
    elif device.type == "mps":
        dtype = torch.float32  # MPS has limited fp16 support
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Extract architecture info
    config = model.config
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_params = sum(p.numel() for p in model.parameters())

    info = ModelInfo(
        name=model_name,
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        device=str(device),
        dtype=str(dtype),
        num_params=num_params,
    )

    return model, tokenizer, info


def get_transformer_layers(model: AutoModelForCausalLM) -> torch.nn.ModuleList:
    """Return the ModuleList of transformer decoder layers.

    Supports common HuggingFace architectures:
    - Qwen2: model.model.layers
    - GPT2:  model.transformer.h
    - Llama: model.model.layers
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Qwen2, Llama, Mistral, etc.
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT2, GPT-Neo, etc.
    raise AttributeError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h"
    )


# ─── Hidden state capture ────────────────────────────────────────────────────

@dataclass
class CapturedStates:
    """Container for hidden states captured from transformer layers.

    Attributes:
        states: dict mapping layer index to hidden state tensor
            [batch, seq_len, hidden_size].
        num_layers: number of layers that were hooked.
    """
    states: dict[int, torch.Tensor] = field(default_factory=dict)
    num_layers: int = 0

    def get_last_token(self, layer: int) -> torch.Tensor:
        """Return the hidden state at the last token position for a layer.

        Shape: [batch, hidden_size].
        """
        if layer not in self.states:
            raise KeyError(f"Layer {layer} not captured. Available: {sorted(self.states.keys())}")
        return self.states[layer][:, -1, :]

    def get_all_last_token(self) -> torch.Tensor:
        """Return last-token hidden states stacked across all layers.

        Shape: [num_layers, batch, hidden_size].
        """
        layers_sorted = sorted(self.states.keys())
        return torch.stack([self.states[l][:, -1, :] for l in layers_sorted])

    @property
    def layer_indices(self) -> list[int]:
        """Sorted list of captured layer indices."""
        return sorted(self.states.keys())


@contextmanager
def capture_hidden_states(
    model: AutoModelForCausalLM,
    layers: list[int] | None = None,
) -> Generator[CapturedStates, None, None]:
    """Context manager that hooks transformer layers and captures hidden states.

    Usage::

        with capture_hidden_states(model) as captured:
            outputs = model(input_ids)
        # captured.states[0] is the hidden state from layer 0

    Args:
        model: A HuggingFace causal LM.
        layers: List of layer indices to hook.  If None, hooks ALL layers.

    Yields:
        CapturedStates object that is populated during the forward pass.
    """
    transformer_layers = get_transformer_layers(model)
    total_layers = len(transformer_layers)

    if layers is None:
        layers = list(range(total_layers))
    else:
        for l in layers:
            if l < 0 or l >= total_layers:
                raise IndexError(
                    f"Layer index {l} out of range [0, {total_layers})"
                )

    captured = CapturedStates(num_layers=len(layers))
    handles: list[torch.utils.hooks.RemovableHook] = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Decoder layers return either a tensor or a tuple (hidden_state, ...)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured.states[layer_idx] = hidden.detach()
        return hook_fn

    try:
        for layer_idx in layers:
            handle = transformer_layers[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            handles.append(handle)
        yield captured
    finally:
        for h in handles:
            h.remove()


# ─── Feature extraction ──────────────────────────────────────────────────────

def tokenize_prompt(
    text: str,
    tokenizer: AutoTokenizer,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Tokenize a single prompt, using chat template if available.

    If the tokenizer has a chat_template, wraps the prompt as a user message.
    Otherwise falls back to raw tokenization (for models like GPT-2).

    Returns input_ids tensor on the specified device.
    """
    has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    if has_chat_template:
        messages = [{"role": "user", "content": text}]
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # Handle BatchEncoding vs raw tensor
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            input_ids = encoded["input_ids"]
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded
        else:
            input_ids = torch.tensor([encoded], dtype=torch.long)
    else:
        # Fallback: raw tokenization (e.g. GPT-2, test models)
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded["input_ids"]

    if device is not None:
        input_ids = input_ids.to(device)

    return input_ids


@torch.no_grad()
def extract_features(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layers: list[int] | None = None,
) -> dict[int, torch.Tensor]:
    """Extract last-token hidden states for a list of prompts.

    Processes prompts one at a time (variable lengths, no padding needed).

    Args:
        texts: List of prompt strings.
        model: Loaded HuggingFace model.
        tokenizer: Matching tokenizer.
        layers: Layer indices to capture. None = all layers.

    Returns:
        Dict mapping layer_index → tensor of shape [num_prompts, hidden_size].
    """
    device = next(model.parameters()).device
    all_features: dict[int, list[torch.Tensor]] = {}

    for text in texts:
        input_ids = tokenize_prompt(text, tokenizer, device=device)

        with capture_hidden_states(model, layers=layers) as captured:
            model(input_ids)

        for layer_idx in captured.layer_indices:
            last_tok = captured.get_last_token(layer_idx)  # [1, hidden_size]
            if layer_idx not in all_features:
                all_features[layer_idx] = []
            all_features[layer_idx].append(last_tok.squeeze(0))  # [hidden_size]

    # Stack into tensors
    result: dict[int, torch.Tensor] = {}
    for layer_idx, feat_list in all_features.items():
        result[layer_idx] = torch.stack(feat_list)  # [num_prompts, hidden_size]

    return result


def get_module_hierarchy(model: AutoModelForCausalLM, max_depth: int = 3) -> list[str]:
    """Return a list of named module paths up to max_depth.

    Useful for displaying the model architecture in the demo.
    """
    paths: list[str] = []
    for name, _ in model.named_modules():
        if name == "":
            continue
        depth = name.count(".") + 1
        if depth <= max_depth:
            paths.append(name)
    return paths
