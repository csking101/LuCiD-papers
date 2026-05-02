"""
Model loading and inference utilities for LLM KL divergence analysis.
===================================================================

Loads HuggingFace causal LMs (base + instruct) onto GPU in fp16.
Designed for reuse across multiple coding adventures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B"
DEFAULT_INSTRUCT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


# ── Data classes ────────────────────────────────────────────────────
@dataclass
class ModelInfo:
    """Metadata about a loaded model."""

    name: str
    num_params: int
    num_layers: int
    hidden_size: int
    vocab_size: int
    dtype: str
    device: str


@dataclass
class TokenLogits:
    """Logits and derived distributions for a single forward pass."""

    input_ids: torch.Tensor          # [1, seq_len]
    logits: torch.Tensor             # [1, seq_len, vocab_size]
    log_probs: torch.Tensor          # [1, seq_len, vocab_size]
    probs: torch.Tensor              # [1, seq_len, vocab_size]

    @property
    def seq_len(self) -> int:
        return self.input_ids.shape[1]

    @property
    def vocab_size(self) -> int:
        return self.logits.shape[-1]


@dataclass
class ModelPair:
    """A base + instruct model pair, ready for comparison."""

    base_model: AutoModelForCausalLM
    instruct_model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    base_info: ModelInfo
    instruct_info: ModelInfo


# ── Loading ─────────────────────────────────────────────────────────
def _resolve_device() -> str:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _model_info(model: AutoModelForCausalLM, name: str) -> ModelInfo:
    """Extract metadata from a loaded model."""
    config = model.config
    num_params = sum(p.numel() for p in model.parameters())
    dtype = str(next(model.parameters()).dtype)
    device = str(next(model.parameters()).device)
    return ModelInfo(
        name=name,
        num_params=num_params,
        num_layers=getattr(config, "num_hidden_layers", 0),
        hidden_size=getattr(config, "hidden_size", 0),
        vocab_size=getattr(config, "vocab_size", 0),
        dtype=dtype,
        device=device,
    )


def load_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a single causal LM and its tokenizer.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g. ``"Qwen/Qwen2.5-1.5B"``).
    device : str, optional
        Target device. Auto-detected if *None*.
    dtype : torch.dtype
        Model precision. Default ``torch.float16``.

    Returns
    -------
    tuple[model, tokenizer]
    """
    device = device or _resolve_device()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_model_pair(
    base_name: str = DEFAULT_BASE_MODEL,
    instruct_name: str = DEFAULT_INSTRUCT_MODEL,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> ModelPair:
    """Load a base + instruct model pair sharing the same tokenizer.

    Both models are placed on the same device in eval mode.
    The instruct model's tokenizer is used (superset of base vocab).
    """
    device = device or _resolve_device()

    base_model, _ = load_model(base_name, device=device, dtype=dtype)
    instruct_model, tokenizer = load_model(
        instruct_name, device=device, dtype=dtype,
    )

    return ModelPair(
        base_model=base_model,
        instruct_model=instruct_model,
        tokenizer=tokenizer,
        base_info=_model_info(base_model, base_name),
        instruct_info=_model_info(instruct_model, instruct_name),
    )


# ── Inference ───────────────────────────────────────────────────────
@torch.no_grad()
def get_logits(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
) -> TokenLogits:
    """Run a forward pass and return logits + distributions.

    Parameters
    ----------
    model : AutoModelForCausalLM
    input_ids : torch.Tensor, shape ``[1, seq_len]``

    Returns
    -------
    TokenLogits
    """
    outputs = model(input_ids)
    logits = outputs.logits.float()  # always compute in fp32
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return TokenLogits(
        input_ids=input_ids,
        logits=logits,
        log_probs=log_probs,
        probs=probs,
    )


@torch.no_grad()
def get_logits_pair(
    pair: ModelPair,
    text: str,
) -> tuple[TokenLogits, TokenLogits]:
    """Tokenize *text* and get logits from both models.

    Returns
    -------
    tuple[base_logits, instruct_logits]
    """
    device = str(next(pair.base_model.parameters()).device)
    input_ids = pair.tokenizer.encode(text, return_tensors="pt").to(device)
    base_tl = get_logits(pair.base_model, input_ids)
    inst_tl = get_logits(pair.instruct_model, input_ids)
    return base_tl, inst_tl


@torch.no_grad()
def generate_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> tuple[list[int], list[str]]:
    """Generate tokens from a single model.

    Returns
    -------
    tuple[token_ids, token_strings]
        Only the *newly generated* tokens (not the prompt).
    """
    device = str(next(model.parameters()).device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_ids = output_ids[0, prompt_len:].tolist()
    new_tokens = [tokenizer.decode([tid]) for tid in new_ids]
    return new_ids, new_tokens


@torch.no_grad()
def generate_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
) -> tuple[list[int], list[str]]:
    """Greedy (deterministic) generation from a single model.

    Returns
    -------
    tuple[token_ids, token_strings]
        Only the *newly generated* tokens.
    """
    device = str(next(model.parameters()).device)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_ids = output_ids[0, prompt_len:].tolist()
    new_tokens = [tokenizer.decode([tid]) for tid in new_ids]
    return new_ids, new_tokens
