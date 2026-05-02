"""
Steering analysis: first-token KL, forced-continuation KL, steering matrix.
============================================================================

Core analysis utilities for measuring how system prompts steer
a single instruct model's token distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import (
    FirstTokenDist,
    TokenLogits,
    generate_greedy,
    get_first_token_dist,
    get_logits,
    tokenize_chat,
)


# ── Data classes ────────────────────────────────────────────────────

@dataclass
class FirstTokenComparison:
    """Comparison of first-token distributions between two system prompts."""

    user_prompt: str
    system_prompt_a: str               # name/label (baseline)
    system_prompt_b: str               # name/label (custom)
    kl_divergence: float               # KL(B || A)
    js_divergence: float               # Jensen-Shannon divergence (symmetric)
    top_k_a: list[tuple[str, float]]   # (token, prob) from condition A
    top_k_b: list[tuple[str, float]]   # (token, prob) from condition B
    top_shifts: list[tuple[str, float, float]]  # (token, prob_a, prob_b) for biggest |delta|
    dist_a: Optional[FirstTokenDist] = field(default=None, repr=False)
    dist_b: Optional[FirstTokenDist] = field(default=None, repr=False)


@dataclass
class ForcedContinuationKL:
    """KL profile when forcing one condition's continuation under another."""

    user_prompt: str
    source_system_prompt: str          # generated from this
    target_system_prompt: str          # forced under this
    continuation_tokens: list[str]     # decoded token strings
    continuation_ids: list[int]        # token IDs
    kl_per_token: list[float]          # KL(target || source) per position
    total_kl: float
    mean_kl: float
    source_text: str                   # greedy continuation from source
    target_text: str                   # greedy continuation from target


@dataclass
class SystemPromptProfile:
    """Aggregated steering power of a system prompt across user prompts."""

    system_prompt_name: str
    system_prompt_text: str
    category: str
    user_prompts: list[str]
    first_token_kls: list[float]       # KL for each user prompt
    mean_steering_power: float
    max_steering_power: float


@dataclass
class SteeringMatrix:
    """Full matrix of system_prompt x user_prompt first-token KL values."""

    system_prompt_names: list[str]
    user_prompts: list[str]
    kl_matrix: list[list[float]]       # [sys_prompt_idx][user_prompt_idx]
    row_means: list[float]             # mean KL per system prompt
    col_means: list[float]             # mean KL per user prompt
    global_mean: float


# ── Core KL computation ────────────────────────────────────────────

def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute KL(P || Q) between two probability distributions.

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability vectors (must sum to 1).

    Returns
    -------
    float
        KL divergence in nats. Clamped to >= 0.
    """
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    log_p = (p + eps).log()
    log_q = (q + eps).log()
    kl = (p * (log_p - log_q)).sum()
    return max(kl.item(), 0.0)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence between two distributions.

    JS(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = (P+Q)/2.
    Symmetric and bounded in [0, ln(2)].

    Parameters
    ----------
    p, q : torch.Tensor
        1-D probability vectors.

    Returns
    -------
    float
    """
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


# ── First-token comparison ─────────────────────────────────────────

@torch.no_grad()
def compare_first_token(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    system_prompt_a: str,
    system_prompt_b: str,
    name_a: str = "Default",
    name_b: str = "Custom",
    top_k: int = 10,
) -> FirstTokenComparison:
    """Compare first-token distributions between two system prompts.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    user_prompt : str
    system_prompt_a, system_prompt_b : str
        System prompt texts for the two conditions.
    name_a, name_b : str
        Display names for the conditions.
    top_k : int
        Number of top tokens to record.

    Returns
    -------
    FirstTokenComparison
    """
    device = str(next(model.parameters()).device)

    ids_a = tokenize_chat(tokenizer, user_prompt, system_prompt_a, device=device)
    ids_b = tokenize_chat(tokenizer, user_prompt, system_prompt_b, device=device)

    dist_a = get_first_token_dist(model, tokenizer, ids_a, top_k=top_k)
    dist_b = get_first_token_dist(model, tokenizer, ids_b, top_k=top_k)

    # KL(B || A) — how much B diverges from A
    kl = _kl_divergence(dist_b.probs, dist_a.probs)
    js = _js_divergence(dist_a.probs, dist_b.probs)

    # Top-K pairs
    top_k_a = list(zip(dist_a.top_k_tokens, dist_a.top_k_probs))
    top_k_b = list(zip(dist_b.top_k_tokens, dist_b.top_k_probs))

    # Compute top shifts: tokens with biggest |prob_b - prob_a|
    delta = dist_b.probs - dist_a.probs
    abs_delta = delta.abs()
    top_shift_vals, top_shift_idxs = abs_delta.topk(min(top_k, abs_delta.shape[0]))

    top_shifts = []
    for idx in top_shift_idxs:
        tok = tokenizer.decode([idx.item()])
        p_a = dist_a.probs[idx].item()
        p_b = dist_b.probs[idx].item()
        top_shifts.append((tok, p_a, p_b))

    return FirstTokenComparison(
        user_prompt=user_prompt,
        system_prompt_a=name_a,
        system_prompt_b=name_b,
        kl_divergence=kl,
        js_divergence=js,
        top_k_a=top_k_a,
        top_k_b=top_k_b,
        top_shifts=top_shifts,
        dist_a=dist_a,
        dist_b=dist_b,
    )


# ── Forced-continuation KL ─────────────────────────────────────────

@torch.no_grad()
def compute_forced_continuation_kl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_prompt: str,
    source_system_prompt: str,
    target_system_prompt: str,
    source_name: str = "Default",
    target_name: str = "Custom",
    max_new_tokens: int = 64,
) -> ForcedContinuationKL:
    """Generate from source condition, force those tokens under target,
    and compute per-position KL divergence.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    user_prompt : str
    source_system_prompt : str
        Generate continuation from this condition.
    target_system_prompt : str
        Force the continuation under this condition.
    source_name, target_name : str
        Display names.
    max_new_tokens : int

    Returns
    -------
    ForcedContinuationKL
    """
    device = str(next(model.parameters()).device)

    # Tokenize both conditions
    ids_source = tokenize_chat(
        tokenizer, user_prompt, source_system_prompt, device=device,
    )
    ids_target = tokenize_chat(
        tokenizer, user_prompt, target_system_prompt, device=device,
    )

    # Generate greedy continuation from source
    source_new_ids, source_tokens, source_text = generate_greedy(
        model, tokenizer, ids_source, max_new_tokens=max_new_tokens,
    )

    # Also generate from target for display
    _, _, target_text = generate_greedy(
        model, tokenizer, ids_target, max_new_tokens=max_new_tokens,
    )

    if not source_new_ids:
        return ForcedContinuationKL(
            user_prompt=user_prompt,
            source_system_prompt=source_name,
            target_system_prompt=target_name,
            continuation_tokens=[],
            continuation_ids=[],
            kl_per_token=[],
            total_kl=0.0,
            mean_kl=0.0,
            source_text="",
            target_text=target_text,
        )

    # Create forced sequences: prompt + continuation for both conditions
    continuation = torch.tensor(
        [source_new_ids], dtype=ids_source.dtype, device=device,
    )
    full_source = torch.cat([ids_source, continuation], dim=1)
    full_target = torch.cat([ids_target, continuation], dim=1)

    # Forward pass on both full sequences
    logits_source = model(full_source).logits.float()
    logits_target = model(full_target).logits.float()

    # Extract distributions at continuation positions
    # logits[i] predicts token i+1
    # Continuation starts at position P (prompt length)
    # So the distribution for continuation[0] is at logits[P-1]
    source_prompt_len = ids_source.shape[1]
    target_prompt_len = ids_target.shape[1]
    cont_len = len(source_new_ids)

    # Slice: from (prompt_len - 1) to (prompt_len - 1 + cont_len)
    source_cont_logits = logits_source[
        0, source_prompt_len - 1 : source_prompt_len - 1 + cont_len, :
    ]
    target_cont_logits = logits_target[
        0, target_prompt_len - 1 : target_prompt_len - 1 + cont_len, :
    ]

    # Compute per-position KL(target || source)
    source_probs = F.softmax(source_cont_logits, dim=-1)
    target_probs = F.softmax(target_cont_logits, dim=-1)
    source_log_probs = F.log_softmax(source_cont_logits, dim=-1)
    target_log_probs = F.log_softmax(target_cont_logits, dim=-1)

    kl_per_pos = (target_probs * (target_log_probs - source_log_probs)).sum(dim=-1)
    kl_per_pos = kl_per_pos.clamp(min=0.0)
    kl_list = kl_per_pos.tolist()

    total_kl = sum(kl_list)
    mean_kl = total_kl / max(len(kl_list), 1)

    return ForcedContinuationKL(
        user_prompt=user_prompt,
        source_system_prompt=source_name,
        target_system_prompt=target_name,
        continuation_tokens=source_tokens,
        continuation_ids=source_new_ids,
        kl_per_token=kl_list,
        total_kl=total_kl,
        mean_kl=mean_kl,
        source_text=source_text,
        target_text=target_text,
    )


# ── Steering matrix ────────────────────────────────────────────────

@torch.no_grad()
def compute_steering_matrix(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompts: list[tuple[str, str]],
    user_prompts: list[str],
    default_system_prompt: str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
) -> SteeringMatrix:
    """Compute first-token KL for each (system_prompt, user_prompt) pair.

    Each system prompt is compared against the default system prompt.

    Parameters
    ----------
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    system_prompts : list[tuple[name, text]]
    user_prompts : list[str]
    default_system_prompt : str
        Baseline system prompt text.

    Returns
    -------
    SteeringMatrix
    """
    device = str(next(model.parameters()).device)
    n_sys = len(system_prompts)
    n_user = len(user_prompts)

    # Pre-compute default distributions for each user prompt
    default_dists: list[FirstTokenDist] = []
    for user_p in user_prompts:
        ids = tokenize_chat(tokenizer, user_p, default_system_prompt, device=device)
        dist = get_first_token_dist(model, tokenizer, ids, top_k=1)
        default_dists.append(dist)

    # Compute KL matrix
    kl_matrix: list[list[float]] = []
    sys_names: list[str] = []

    for sys_name, sys_text in system_prompts:
        sys_names.append(sys_name)
        row: list[float] = []
        for j, user_p in enumerate(user_prompts):
            ids = tokenize_chat(tokenizer, user_p, sys_text, device=device)
            dist = get_first_token_dist(model, tokenizer, ids, top_k=1)
            kl = _kl_divergence(dist.probs, default_dists[j].probs)
            row.append(kl)
        kl_matrix.append(row)

    # Compute summary statistics
    row_means = [sum(row) / max(len(row), 1) for row in kl_matrix]
    col_means = []
    for j in range(n_user):
        col = [kl_matrix[i][j] for i in range(n_sys)]
        col_means.append(sum(col) / max(len(col), 1))

    all_kls = [kl for row in kl_matrix for kl in row]
    global_mean = sum(all_kls) / max(len(all_kls), 1)

    return SteeringMatrix(
        system_prompt_names=sys_names,
        user_prompts=user_prompts,
        kl_matrix=kl_matrix,
        row_means=row_means,
        col_means=col_means,
        global_mean=global_mean,
    )


# ── System prompt profiles ─────────────────────────────────────────

def compute_system_prompt_profiles(
    matrix: SteeringMatrix,
    system_prompts: list[tuple[str, str, str]],
) -> list[SystemPromptProfile]:
    """Derive per-system-prompt profiles from a steering matrix.

    Parameters
    ----------
    matrix : SteeringMatrix
    system_prompts : list[tuple[name, text, category]]
        Must be in the same order as ``matrix.system_prompt_names``.

    Returns
    -------
    list[SystemPromptProfile]
    """
    profiles: list[SystemPromptProfile] = []
    for i, (name, text, category) in enumerate(system_prompts):
        kls = matrix.kl_matrix[i]
        profiles.append(SystemPromptProfile(
            system_prompt_name=name,
            system_prompt_text=text,
            category=category,
            user_prompts=matrix.user_prompts,
            first_token_kls=kls,
            mean_steering_power=sum(kls) / max(len(kls), 1),
            max_steering_power=max(kls) if kls else 0.0,
        ))
    return profiles
