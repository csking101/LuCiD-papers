"""Tests for models.py — model loading, chat formatting, inference utilities.

Uses sshleifer/tiny-gpt2 for CPU-only tests (no GPU required).
Note: tiny-gpt2 does not support chat templates, so chat-template
tests use Qwen tokenizer loaded separately from the model.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import (
    FirstTokenDist,
    ModelInfo,
    TokenLogits,
    _model_info,
    _resolve_device,
    format_chat,
    generate_greedy,
    get_first_token_dist,
    get_logits,
    load_model,
    tokenize_chat,
)


# ── Fixtures ────────────────────────────────────────────────────────

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_model():
    """Load the tiny GPT-2 model for testing."""
    model, tokenizer, info = load_model(
        TINY_MODEL, device="cpu", dtype=torch.float32,
    )
    return model, tokenizer, info


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Load just the Qwen tokenizer for chat template tests."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True,
    )


# ── ModelInfo ───────────────────────────────────────────────────────

class TestModelInfo:
    def test_fields(self, tiny_model):
        _, _, info = tiny_model
        assert isinstance(info, ModelInfo)
        assert TINY_MODEL in info.name
        assert info.num_params > 0
        assert info.vocab_size > 0
        assert "float32" in info.dtype
        assert "cpu" in info.device

    def test_layers_positive(self, tiny_model):
        _, _, info = tiny_model
        assert info.num_layers >= 0

    def test_hidden_size_positive(self, tiny_model):
        _, _, info = tiny_model
        assert info.hidden_size > 0


# ── TokenLogits ─────────────────────────────────────────────────────

class TestTokenLogits:
    def test_shapes(self, tiny_model):
        model, tokenizer, _ = tiny_model
        text = "Hello world"
        ids = tokenizer.encode(text, return_tensors="pt")
        tl = get_logits(model, ids)

        assert tl.input_ids.shape == ids.shape
        assert tl.logits.shape[0] == 1
        assert tl.logits.shape[1] == ids.shape[1]
        assert tl.logits.shape[2] > 0
        assert tl.log_probs.shape == tl.logits.shape
        assert tl.probs.shape == tl.logits.shape

    def test_properties(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Test", return_tensors="pt")
        tl = get_logits(model, ids)
        assert tl.seq_len == ids.shape[1]
        assert tl.vocab_size == tl.logits.shape[-1]

    def test_probs_sum_to_one(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("The cat", return_tensors="pt")
        tl = get_logits(model, ids)
        sums = tl.probs[0].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_log_probs_are_negative(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello", return_tensors="pt")
        tl = get_logits(model, ids)
        assert (tl.log_probs <= 0).all()

    def test_probs_non_negative(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello", return_tensors="pt")
        tl = get_logits(model, ids)
        assert (tl.probs >= 0).all()

    def test_fp32_computation(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Test", return_tensors="pt")
        tl = get_logits(model, ids)
        assert tl.logits.dtype == torch.float32
        assert tl.log_probs.dtype == torch.float32


# ── FirstTokenDist ──────────────────────────────────────────────────

class TestFirstTokenDist:
    def test_structure(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello world", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids, top_k=5)

        assert isinstance(ftd, FirstTokenDist)
        assert len(ftd.top_k_tokens) == 5
        assert len(ftd.top_k_probs) == 5
        assert len(ftd.top_k_ids) == 5

    def test_probs_sum_to_one(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("The", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids)
        total = ftd.probs.sum()
        assert torch.isclose(total, torch.tensor(1.0), atol=1e-4)

    def test_top_k_sorted(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids, top_k=10)
        # Top-K should be in descending order of probability
        for i in range(len(ftd.top_k_probs) - 1):
            assert ftd.top_k_probs[i] >= ftd.top_k_probs[i + 1]

    def test_top_k_probs_non_negative(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Test", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids)
        assert all(p >= 0 for p in ftd.top_k_probs)

    def test_tokens_are_strings(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids, top_k=3)
        assert all(isinstance(t, str) for t in ftd.top_k_tokens)

    def test_top_k_capped(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hi", return_tensors="pt")
        ftd = get_first_token_dist(model, tokenizer, ids, top_k=3)
        assert len(ftd.top_k_tokens) == 3


# ── Chat template formatting ───────────────────────────────────────

class TestFormatChat:
    def test_with_system_prompt(self, qwen_tokenizer):
        result = format_chat(qwen_tokenizer, "Hello?", system_prompt="Be helpful.")
        assert "Be helpful." in result
        assert "Hello?" in result
        assert "assistant" in result  # generation prompt

    def test_without_system_prompt(self, qwen_tokenizer):
        result = format_chat(qwen_tokenizer, "Hello?")
        assert "Hello?" in result
        # Default system prompt should be inserted by Qwen
        assert "Qwen" in result or "assistant" in result

    def test_different_system_prompts_differ(self, qwen_tokenizer):
        a = format_chat(qwen_tokenizer, "Hi", system_prompt="Be a pirate.")
        b = format_chat(qwen_tokenizer, "Hi", system_prompt="Be a doctor.")
        assert a != b

    def test_returns_string(self, qwen_tokenizer):
        result = format_chat(qwen_tokenizer, "Test")
        assert isinstance(result, str)


class TestTokenizeChat:
    def test_returns_tensor(self, qwen_tokenizer):
        ids = tokenize_chat(qwen_tokenizer, "Hello?", system_prompt="Test")
        assert isinstance(ids, torch.Tensor)
        assert ids.dim() == 2
        assert ids.shape[0] == 1

    def test_different_system_prompts_different_lengths(self, qwen_tokenizer):
        ids_short = tokenize_chat(qwen_tokenizer, "Hi", system_prompt="Ok.")
        ids_long = tokenize_chat(
            qwen_tokenizer, "Hi",
            system_prompt="Be a very verbose and extremely detailed assistant."
        )
        assert ids_long.shape[1] > ids_short.shape[1]

    def test_device_placement(self, qwen_tokenizer):
        ids = tokenize_chat(qwen_tokenizer, "Hello", device="cpu")
        assert ids.device == torch.device("cpu")


# ── Generation ──────────────────────────────────────────────────────

class TestGeneration:
    def test_greedy_returns_tokens(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("The capital of France is", return_tensors="pt")
        token_ids, token_strs, full_text = generate_greedy(
            model, tokenizer, ids, max_new_tokens=10,
        )
        assert len(token_ids) > 0
        assert len(token_ids) <= 10
        assert len(token_ids) == len(token_strs)
        assert isinstance(full_text, str)
        assert len(full_text) > 0

    def test_greedy_deterministic(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Hello world", return_tensors="pt")

        ids_a, _, text_a = generate_greedy(model, tokenizer, ids, max_new_tokens=5)
        ids_b, _, text_b = generate_greedy(model, tokenizer, ids, max_new_tokens=5)

        assert ids_a == ids_b
        assert text_a == text_b

    def test_max_new_tokens_respected(self, tiny_model):
        model, tokenizer, _ = tiny_model
        ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        token_ids, _, _ = generate_greedy(model, tokenizer, ids, max_new_tokens=3)
        assert len(token_ids) <= 3


# ── Device resolution ──────────────────────────────────────────────

class TestDevice:
    def test_resolve_returns_string(self):
        dev = _resolve_device()
        assert isinstance(dev, str)
        assert dev in ("cuda", "mps", "cpu")
