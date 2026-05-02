"""
Model loading and inference utilities for system prompt steering analysis.
==========================================================================

Loads a single HuggingFace instruct model onto GPU in fp16.
Provides chat-template formatting and forward-pass utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


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
class FirstTokenDist:
    """Probability distribution over the first generated token."""

    log_probs: torch.Tensor          # [vocab_size]
    probs: torch.Tensor              # [vocab_size]
    top_k_tokens: list[str]          # decoded top-K token strings
    top_k_probs: list[float]         # corresponding probabilities
    top_k_ids: list[int]             # corresponding token IDs


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
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, ModelInfo]:
    """Load an instruct model, its tokenizer, and metadata.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID.
    device : str, optional
        Target device. Auto-detected if *None*.
    dtype : torch.dtype
        Model precision. Default ``torch.float16``.

    Returns
    -------
    tuple[model, tokenizer, info]
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

    info = _model_info(model, model_name)
    return model, tokenizer, info


# ── Chat template formatting ───────────────────────────────────────

def format_chat(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """Format a user prompt with an optional system prompt using the
    model's chat template.

    Parameters
    ----------
    tokenizer : AutoTokenizer
    user_prompt : str
    system_prompt : str, optional
        If *None*, the tokenizer's default system prompt is used.

    Returns
    -------
    str
        Formatted chat string with generation prompt appended.
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_chat(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt: Optional[str] = None,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Format and tokenize a chat, returning input_ids on the given device.

    Parameters
    ----------
    tokenizer : AutoTokenizer
    user_prompt : str
    system_prompt : str, optional
    device : str, optional

    Returns
    -------
    torch.Tensor, shape ``[1, seq_len]``
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template may return a BatchEncoding or a plain tensor
    if isinstance(result, torch.Tensor):
        ids = result
    else:
        ids = result["input_ids"]
    if device:
        ids = ids.to(device)
    return ids


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
def get_first_token_dist(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    top_k: int = 10,
) -> FirstTokenDist:
    """Get the probability distribution over the first generated token.

    The distribution is taken from the last position of ``input_ids``
    (which predicts the next token).

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    input_ids : torch.Tensor, shape ``[1, seq_len]``
    top_k : int

    Returns
    -------
    FirstTokenDist
    """
    outputs = model(input_ids)
    # Last position logits -> first generated token distribution
    last_logits = outputs.logits[0, -1, :].float()
    log_p = F.log_softmax(last_logits, dim=-1)
    p = log_p.exp()

    vals, idxs = p.topk(top_k)
    top_tokens = [tokenizer.decode([idx.item()]) for idx in idxs]
    top_probs = vals.tolist()
    top_ids = idxs.tolist()

    return FirstTokenDist(
        log_probs=log_p,
        probs=p,
        top_k_tokens=top_tokens,
        top_k_probs=top_probs,
        top_k_ids=top_ids,
    )


@torch.no_grad()
def generate_greedy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 64,
) -> tuple[list[int], list[str], str]:
    """Greedy (deterministic) generation.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    input_ids : torch.Tensor, shape ``[1, seq_len]``
    max_new_tokens : int

    Returns
    -------
    tuple[token_ids, token_strings, full_text]
        Only the *newly generated* tokens, plus the decoded text.
    """
    prompt_len = input_ids.shape[1]

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_ids = output_ids[0, prompt_len:].tolist()
    new_tokens = [tokenizer.decode([tid]) for tid in new_ids]
    full_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return new_ids, new_tokens, full_text
