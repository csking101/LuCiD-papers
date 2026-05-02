"""
Tests for kl.py — KL divergence computation and constrained generation.

Uses a tiny GPT-2 model pair for fast CPU-only testing.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kl import (
    CategoryKLSummary,
    InterpolatedOutput,
    SequenceKL,
    TokenKL,
    compute_category_summaries,
    compute_global_kl,
    compute_sequence_kl,
    compute_token_kl,
    generate_interpolated,
)
from models import (
    ModelPair,
    TokenLogits,
    _model_info,
    get_logits,
    get_logits_pair,
)

# ── Fixtures ────────────────────────────────────────────────────────

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture(scope="module")
def tiny_pair(tiny_model, tiny_tokenizer):
    return ModelPair(
        base_model=tiny_model,
        instruct_model=tiny_model,
        tokenizer=tiny_tokenizer,
        base_info=_model_info(tiny_model, "tiny-base"),
        instruct_info=_model_info(tiny_model, "tiny-instruct"),
    )


# ── TokenKL ─────────────────────────────────────────────────────────

class TestTokenKL:
    def test_compute_token_kl_identical_models(self, tiny_pair):
        """KL between identical models should be ~0."""
        base_tl, inst_tl = get_logits_pair(tiny_pair, "hello")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer, top_k=3)
        assert tkl.num_tokens > 0
        assert tkl.total_kl == pytest.approx(0.0, abs=1e-4)
        assert tkl.mean_kl == pytest.approx(0.0, abs=1e-4)

    def test_token_kl_fields(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "hello world")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer, top_k=5)
        assert len(tkl.tokens) == tkl.num_tokens
        assert len(tkl.token_ids) == tkl.num_tokens
        assert len(tkl.kl_per_token) == tkl.num_tokens
        assert len(tkl.base_top_k) == tkl.num_tokens
        assert len(tkl.instruct_top_k) == tkl.num_tokens

    def test_top_k_correct_length(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "test")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer, top_k=3)
        for pos_top_k in tkl.base_top_k:
            assert len(pos_top_k) == 3
        for pos_top_k in tkl.instruct_top_k:
            assert len(pos_top_k) == 3

    def test_kl_non_negative(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "test prompt")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer)
        for kl in tkl.kl_per_token:
            assert kl >= 0.0

    def test_high_kl_positions_empty_for_identical(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "hello")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer)
        assert len(tkl.high_kl_positions(threshold=0.1)) == 0

    def test_top_k_probs_sum_leq_one(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "test")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer, top_k=5)
        for pos_top_k in tkl.base_top_k:
            total = sum(p for _, p in pos_top_k)
            assert total <= 1.0 + 1e-4


# ── SequenceKL ──────────────────────────────────────────────────────

class TestSequenceKL:
    def test_compute_sequence_kl_fields(self, tiny_pair):
        seq = compute_sequence_kl(tiny_pair, "hello", max_new_tokens=5)
        assert isinstance(seq, SequenceKL)
        assert seq.prompt == "hello"
        assert isinstance(seq.base_text, str)
        assert isinstance(seq.instruct_text, str)
        assert isinstance(seq.prompt_token_kl, TokenKL)
        assert seq.total_kl >= 0
        assert seq.mean_kl >= 0

    def test_identical_models_zero_kl(self, tiny_pair):
        seq = compute_sequence_kl(tiny_pair, "test", max_new_tokens=3)
        assert seq.total_kl == pytest.approx(0.0, abs=1e-4)

    def test_base_and_instruct_text_same_for_identical(self, tiny_pair):
        """When using the same model, greedy outputs should be identical."""
        seq = compute_sequence_kl(tiny_pair, "test", max_new_tokens=5)
        assert seq.base_text == seq.instruct_text


# ── InterpolatedOutput ──────────────────────────────────────────────

class TestInterpolatedGeneration:
    def test_generate_interpolated_alpha_zero(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "hello", alpha=0.0, max_new_tokens=5,
        )
        assert isinstance(out, InterpolatedOutput)
        assert out.alpha == 0.0
        assert len(out.token_ids) > 0
        assert len(out.tokens) == len(out.token_ids)
        assert out.total_kl >= 0

    def test_generate_interpolated_alpha_one(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "hello", alpha=1.0, max_new_tokens=5,
        )
        assert out.alpha == 1.0
        assert len(out.token_ids) > 0

    def test_generate_interpolated_mid(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "hello", alpha=0.5, max_new_tokens=5,
        )
        assert out.alpha == 0.5
        assert isinstance(out.text, str)

    def test_kl_per_token_length_matches(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "test", alpha=0.5, max_new_tokens=5,
        )
        assert len(out.kl_per_token) == len(out.token_ids)

    def test_kl_per_token_non_negative(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "test", alpha=0.5, max_new_tokens=5,
        )
        for kl in out.kl_per_token:
            assert kl >= 0.0

    def test_total_kl_is_sum(self, tiny_pair):
        out = generate_interpolated(
            tiny_pair, "test", alpha=0.5, max_new_tokens=5,
        )
        assert out.total_kl == pytest.approx(sum(out.kl_per_token), abs=1e-4)

    def test_identical_models_zero_kl(self, tiny_pair):
        """Interpolating between identical models should give ~0 KL."""
        out = generate_interpolated(
            tiny_pair, "hello", alpha=0.5, max_new_tokens=5,
        )
        assert out.total_kl == pytest.approx(0.0, abs=0.1)


# ── Batch analysis ──────────────────────────────────────────────────

class TestBatchAnalysis:
    def test_compute_global_kl(self, tiny_pair):
        prompts = ["hello", "test", "world"]
        global_mean, per_prompt = compute_global_kl(tiny_pair, prompts)
        assert len(per_prompt) == 3
        assert global_mean >= 0
        for kl in per_prompt:
            assert kl >= 0

    def test_global_kl_identical_models_near_zero(self, tiny_pair):
        prompts = ["hello", "test"]
        global_mean, _ = compute_global_kl(tiny_pair, prompts)
        assert global_mean == pytest.approx(0.0, abs=1e-4)

    def test_compute_category_summaries(self, tiny_pair):
        categories = {
            "cat_a": ["hello", "world"],
            "cat_b": ["test"],
        }
        summaries = compute_category_summaries(tiny_pair, categories)
        assert len(summaries) == 2
        for s in summaries:
            assert isinstance(s, CategoryKLSummary)
            assert s.num_prompts > 0
            assert s.mean_kl >= 0
            assert s.min_kl >= 0
            assert s.max_kl >= s.min_kl

    def test_empty_categories(self, tiny_pair):
        summaries = compute_category_summaries(tiny_pair, {})
        assert summaries == []


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_token_input(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "a")
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer)
        assert tkl.num_tokens >= 1

    def test_long_input(self, tiny_pair):
        text = "word " * 50
        base_tl, inst_tl = get_logits_pair(tiny_pair, text)
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer)
        assert tkl.num_tokens > 10

    def test_special_characters(self, tiny_pair):
        text = "Hello! @#$% ^&*()"
        base_tl, inst_tl = get_logits_pair(tiny_pair, text)
        tkl = compute_token_kl(base_tl, inst_tl, tiny_pair.tokenizer)
        assert tkl.num_tokens > 0
