"""
KL divergence computation and KL-constrained generation.
=========================================================

Core analysis utilities for comparing base vs instruct LLM distributions
at the token level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import ModelPair, TokenLogits, get_logits


# ── Data classes ────────────────────────────────────────────────────
@dataclass
class TokenKL:
    """Per-token KL divergence between two models for a single sequence."""

    tokens: list[str]                  # decoded token strings
    token_ids: list[int]               # token IDs
    kl_per_token: list[float]          # KL(instruct || base) per position
    base_top_k: list[list[tuple[str, float]]]     # top-K (token, prob) from base
    instruct_top_k: list[list[tuple[str, float]]]  # top-K (token, prob) from instruct
    total_kl: float                    # sum of per-token KL
    mean_kl: float                     # mean per-token KL

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def high_kl_positions(self, threshold: float = 0.5) -> list[int]:
        """Return indices of tokens with KL above *threshold*."""
        return [i for i, kl in enumerate(self.kl_per_token) if kl > threshold]


@dataclass
class SequenceKL:
    """KL analysis for a generated sequence (prompt + continuation)."""

    prompt: str
    base_text: str                     # greedy continuation from base
    instruct_text: str                 # greedy continuation from instruct
    prompt_token_kl: TokenKL           # KL over the prompt tokens
    total_kl: float
    mean_kl: float


@dataclass
class InterpolatedOutput:
    """Output from KL-constrained (interpolated) generation."""

    alpha: float                       # 0.0 = pure base, 1.0 = pure instruct
    text: str
    token_ids: list[int]
    tokens: list[str]
    kl_per_token: list[float]          # KL(mixed || base) per generated token
    total_kl: float


@dataclass
class CategoryKLSummary:
    """Summary KL statistics for a prompt category."""

    category: str
    num_prompts: int
    mean_kl: float
    max_kl: float
    min_kl: float


# ── Core KL computation ────────────────────────────────────────────
def compute_token_kl(
    base_logits: TokenLogits,
    instruct_logits: TokenLogits,
    tokenizer: AutoTokenizer,
    top_k: int = 5,
) -> TokenKL:
    """Compute per-token KL(instruct || base) for a shared input sequence.

    Parameters
    ----------
    base_logits, instruct_logits : TokenLogits
        From the same input_ids.
    tokenizer : AutoTokenizer
    top_k : int
        Number of top tokens to record per position.

    Returns
    -------
    TokenKL
    """
    # Both should have same input_ids
    assert base_logits.seq_len == instruct_logits.seq_len

    seq_len = base_logits.seq_len
    input_ids = base_logits.input_ids[0].tolist()

    # KL(P_inst || P_base) = sum_x P_inst(x) * [log P_inst(x) - log P_base(x)]
    # Compute per-position (sum over vocab)
    p_inst = instruct_logits.probs[0]       # [seq_len, vocab]
    log_p_inst = instruct_logits.log_probs[0]
    log_p_base = base_logits.log_probs[0]

    kl_per_pos = (p_inst * (log_p_inst - log_p_base)).sum(dim=-1)  # [seq_len]
    # Clamp to avoid numerical negatives
    kl_per_pos = kl_per_pos.clamp(min=0.0)

    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    kl_list = kl_per_pos.tolist()

    # Top-K per position
    base_probs = base_logits.probs[0]      # [seq_len, vocab]
    inst_probs = instruct_logits.probs[0]

    base_top_k_all: list[list[tuple[str, float]]] = []
    inst_top_k_all: list[list[tuple[str, float]]] = []

    for pos in range(seq_len):
        # Base top-K
        vals, idxs = base_probs[pos].topk(top_k)
        base_top_k_all.append([
            (tokenizer.decode([idx.item()]), val.item())
            for val, idx in zip(vals, idxs)
        ])
        # Instruct top-K
        vals, idxs = inst_probs[pos].topk(top_k)
        inst_top_k_all.append([
            (tokenizer.decode([idx.item()]), val.item())
            for val, idx in zip(vals, idxs)
        ])

    total = sum(kl_list)
    mean = total / max(len(kl_list), 1)

    return TokenKL(
        tokens=tokens,
        token_ids=input_ids,
        kl_per_token=kl_list,
        base_top_k=base_top_k_all,
        instruct_top_k=inst_top_k_all,
        total_kl=total,
        mean_kl=mean,
    )


def compute_sequence_kl(
    pair: ModelPair,
    prompt: str,
    max_new_tokens: int = 64,
    top_k: int = 5,
) -> SequenceKL:
    """Compute KL analysis for a prompt: generate from both models, then
    compute token-level KL over the prompt itself.

    Parameters
    ----------
    pair : ModelPair
    prompt : str
    max_new_tokens : int
    top_k : int

    Returns
    -------
    SequenceKL
    """
    from models import generate_greedy, get_logits_pair

    # Generate continuations
    _, base_tokens = generate_greedy(
        pair.base_model, pair.tokenizer, prompt,
        max_new_tokens=max_new_tokens,
    )
    _, inst_tokens = generate_greedy(
        pair.instruct_model, pair.tokenizer, prompt,
        max_new_tokens=max_new_tokens,
    )

    base_text = "".join(base_tokens)
    instruct_text = "".join(inst_tokens)

    # KL over prompt tokens
    base_tl, inst_tl = get_logits_pair(pair, prompt)
    token_kl = compute_token_kl(base_tl, inst_tl, pair.tokenizer, top_k=top_k)

    return SequenceKL(
        prompt=prompt,
        base_text=base_text,
        instruct_text=instruct_text,
        prompt_token_kl=token_kl,
        total_kl=token_kl.total_kl,
        mean_kl=token_kl.mean_kl,
    )


# ── Interpolated (KL-constrained) generation ───────────────────────
@torch.no_grad()
def generate_interpolated(
    pair: ModelPair,
    prompt: str,
    alpha: float = 0.5,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
) -> InterpolatedOutput:
    """Generate by mixing base and instruct log-probabilities.

    The mixed distribution is::

        log p_mixed = (1 - alpha) * log p_base + alpha * log p_instruct

    * ``alpha = 0.0`` → pure base model
    * ``alpha = 1.0`` → pure instruct model
    * ``0 < alpha < 1`` → interpolated (simulates varying KL budget)

    Parameters
    ----------
    pair : ModelPair
    prompt : str
    alpha : float
        Interpolation weight (0 = base, 1 = instruct).
    max_new_tokens : int
    temperature : float

    Returns
    -------
    InterpolatedOutput
    """
    device = str(next(pair.base_model.parameters()).device)
    tokenizer = pair.tokenizer

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids: list[int] = []
    kl_per_token: list[float] = []

    for _ in range(max_new_tokens):
        base_out = pair.base_model(input_ids)
        inst_out = pair.instruct_model(input_ids)

        # Last position logits in fp32
        base_logits = base_out.logits[:, -1, :].float()
        inst_logits = inst_out.logits[:, -1, :].float()

        base_log_p = F.log_softmax(base_logits / temperature, dim=-1)
        inst_log_p = F.log_softmax(inst_logits / temperature, dim=-1)

        # Interpolated log-probs
        mixed_log_p = (1.0 - alpha) * base_log_p + alpha * inst_log_p
        mixed_probs = F.softmax(mixed_log_p, dim=-1)

        # KL(mixed || base) for this token
        kl = (mixed_probs * (mixed_probs.log() - base_log_p)).sum(dim=-1)
        kl_per_token.append(kl.clamp(min=0.0).item())

        # Sample from the mixed distribution
        next_id = torch.multinomial(mixed_probs, num_samples=1)
        generated_ids.append(next_id.item())

        # Check for EOS
        if next_id.item() == tokenizer.eos_token_id:
            break

        # Append and continue
        input_ids = torch.cat([input_ids, next_id], dim=1)

    tokens = [tokenizer.decode([tid]) for tid in generated_ids]
    text = "".join(tokens)
    total_kl = sum(kl_per_token)

    return InterpolatedOutput(
        alpha=alpha,
        text=text,
        token_ids=generated_ids,
        tokens=tokens,
        kl_per_token=kl_per_token,
        total_kl=total_kl,
    )


# ── Batch analysis ─────────────────────────────────────────────────
def compute_global_kl(
    pair: ModelPair,
    prompts: list[str],
) -> tuple[float, list[float]]:
    """Compute average per-token KL across multiple prompts.

    Returns
    -------
    tuple[global_mean_kl, per_prompt_mean_kl]
    """
    from models import get_logits_pair

    per_prompt: list[float] = []
    for prompt in prompts:
        base_tl, inst_tl = get_logits_pair(pair, prompt)
        token_kl = compute_token_kl(base_tl, inst_tl, pair.tokenizer, top_k=1)
        per_prompt.append(token_kl.mean_kl)

    global_mean = sum(per_prompt) / max(len(per_prompt), 1)
    return global_mean, per_prompt


def compute_category_summaries(
    pair: ModelPair,
    prompts_by_category: dict[str, list[str]],
) -> list[CategoryKLSummary]:
    """Compute KL summaries grouped by prompt category.

    Parameters
    ----------
    pair : ModelPair
    prompts_by_category : dict[str, list[str]]
        Mapping from category name to list of prompts.

    Returns
    -------
    list[CategoryKLSummary]
    """
    summaries: list[CategoryKLSummary] = []
    for category, prompts in prompts_by_category.items():
        _, per_prompt_kl = compute_global_kl(pair, prompts)
        if not per_prompt_kl:
            continue
        summaries.append(CategoryKLSummary(
            category=category,
            num_prompts=len(per_prompt_kl),
            mean_kl=sum(per_prompt_kl) / len(per_prompt_kl),
            max_kl=max(per_prompt_kl),
            min_kl=min(per_prompt_kl),
        ))
    return summaries
