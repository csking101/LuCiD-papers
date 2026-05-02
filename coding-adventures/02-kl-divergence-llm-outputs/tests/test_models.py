"""
Tests for models.py — model loading, inference, and data classes.

Uses a tiny GPT-2 model for fast CPU-only testing.
Real Qwen2.5 tests are marked with @pytest.mark.gpu and skip without CUDA.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import (
    ModelInfo,
    ModelPair,
    TokenLogits,
    _model_info,
    _resolve_device,
    generate_greedy,
    generate_tokens,
    get_logits,
    get_logits_pair,
    load_model,
)

# ── Fixtures ────────────────────────────────────────────────────────

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    """Load a tiny GPT-2 for fast testing."""
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
    """Fake model pair using the same tiny model twice."""
    return ModelPair(
        base_model=tiny_model,
        instruct_model=tiny_model,
        tokenizer=tiny_tokenizer,
        base_info=_model_info(tiny_model, "tiny-base"),
        instruct_info=_model_info(tiny_model, "tiny-instruct"),
    )


# ── ModelInfo ───────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_fields(self, tiny_model):
        info = _model_info(tiny_model, "test-model")
        assert info.name == "test-model"
        assert info.num_params > 0
        assert info.num_layers > 0
        assert info.hidden_size > 0
        assert info.vocab_size > 0
        assert "float" in info.dtype
        assert info.device in ("cpu", "cuda:0", "cuda")

    def test_model_info_param_count_positive(self, tiny_model):
        info = _model_info(tiny_model, "x")
        assert info.num_params == sum(p.numel() for p in tiny_model.parameters())


# ── TokenLogits ─────────────────────────────────────────────────────

class TestTokenLogits:
    def test_get_logits_shape(self, tiny_model, tiny_tokenizer):
        text = "Hello world"
        ids = tiny_tokenizer.encode(text, return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        assert tl.logits.shape[0] == 1
        assert tl.logits.shape[1] == ids.shape[1]
        assert tl.logits.shape[2] == tiny_model.config.vocab_size

    def test_log_probs_sum_to_one(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer.encode("test", return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        # exp(log_probs) should sum to ~1
        sums = tl.probs[0].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_seq_len_property(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer.encode("a b c d", return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        assert tl.seq_len == ids.shape[1]

    def test_vocab_size_property(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer.encode("test", return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        assert tl.vocab_size == tiny_model.config.vocab_size

    def test_probs_non_negative(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer.encode("test", return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        assert (tl.probs >= 0).all()

    def test_logits_dtype_float32(self, tiny_model, tiny_tokenizer):
        ids = tiny_tokenizer.encode("test", return_tensors="pt")
        tl = get_logits(tiny_model, ids)
        assert tl.logits.dtype == torch.float32


# ── get_logits_pair ─────────────────────────────────────────────────

class TestGetLogitsPair:
    def test_returns_two_token_logits(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "hello")
        assert isinstance(base_tl, TokenLogits)
        assert isinstance(inst_tl, TokenLogits)

    def test_same_seq_len(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "hello world")
        assert base_tl.seq_len == inst_tl.seq_len

    def test_same_vocab_size(self, tiny_pair):
        base_tl, inst_tl = get_logits_pair(tiny_pair, "test")
        assert base_tl.vocab_size == inst_tl.vocab_size


# ── Generation ──────────────────────────────────────────────────────

class TestGeneration:
    def test_generate_greedy_returns_tokens(self, tiny_model, tiny_tokenizer):
        ids, tokens = generate_greedy(
            tiny_model, tiny_tokenizer, "Hello", max_new_tokens=5,
        )
        assert len(ids) > 0
        assert len(ids) == len(tokens)
        assert len(ids) <= 5

    def test_generate_tokens_returns_tokens(self, tiny_model, tiny_tokenizer):
        ids, tokens = generate_tokens(
            tiny_model, tiny_tokenizer, "Hello", max_new_tokens=5,
        )
        assert len(ids) > 0
        assert len(ids) == len(tokens)
        assert len(ids) <= 5

    def test_generate_greedy_deterministic(self, tiny_model, tiny_tokenizer):
        ids1, _ = generate_greedy(tiny_model, tiny_tokenizer, "Test", max_new_tokens=5)
        ids2, _ = generate_greedy(tiny_model, tiny_tokenizer, "Test", max_new_tokens=5)
        assert ids1 == ids2

    def test_generate_short_prompt(self, tiny_model, tiny_tokenizer):
        ids, tokens = generate_greedy(
            tiny_model, tiny_tokenizer, "a", max_new_tokens=3,
        )
        assert isinstance(ids, list)
        assert isinstance(tokens, list)
        assert len(ids) > 0


# ── ModelPair ───────────────────────────────────────────────────────

class TestModelPair:
    def test_pair_has_both_models(self, tiny_pair):
        assert tiny_pair.base_model is not None
        assert tiny_pair.instruct_model is not None
        assert tiny_pair.tokenizer is not None

    def test_pair_info_populated(self, tiny_pair):
        assert tiny_pair.base_info.num_params > 0
        assert tiny_pair.instruct_info.num_params > 0


# ── Device ──────────────────────────────────────────────────────────

class TestDevice:
    def test_resolve_device_returns_string(self):
        device = _resolve_device()
        assert isinstance(device, str)
        assert device in ("cuda", "mps", "cpu")


# ── load_model ──────────────────────────────────────────────────────

class TestLoadModel:
    def test_load_tiny_model(self):
        model, tokenizer = load_model(TINY_MODEL, device="cpu", dtype=torch.float32)
        assert model is not None
        assert tokenizer is not None
        assert tokenizer.pad_token is not None
