"""Tests for pipeline.py — guardrail pipeline."""

import sys
from pathlib import Path

_adventure_root = Path(__file__).resolve().parent.parent
if str(_adventure_root) not in sys.path:
    sys.path.insert(0, str(_adventure_root))

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from classifier import ConfusionMatrix, IntentProbe
from pipeline import (
    ComparisonResult,
    Decision,
    Guardrail,
    GuardrailConfig,
    GuardrailResult,
    StressTestResult,
)

TINY_MODEL = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny_setup():
    """Load tiny GPT-2 and a dummy probe for pipeline tests."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    model.eval()

    hidden_size = model.config.hidden_size
    probe = IntentProbe(hidden_size)
    probe.eval()

    return model, tokenizer, probe


# ─── GuardrailConfig ─────────────────────────────────────────────────────────

class TestGuardrailConfig:
    def test_creation(self):
        config = GuardrailConfig(layer=10)
        assert config.layer == 10
        assert config.threshold == 0.5

    def test_custom_threshold(self):
        config = GuardrailConfig(layer=5, threshold=0.8)
        assert config.threshold == 0.8

    def test_frozen(self):
        config = GuardrailConfig(layer=0)
        with pytest.raises(AttributeError):
            config.layer = 5  # type: ignore[misc]


# ─── GuardrailResult ─────────────────────────────────────────────────────────

class TestGuardrailResult:
    def test_denied(self):
        r = GuardrailResult(
            prompt="test", decision="deny", prob_harmful=0.9,
            response="refused", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=0.0, latency_total_ms=1.1, layer=10,
        )
        assert r.was_denied is True
        assert r.was_allowed is False

    def test_allowed(self):
        r = GuardrailResult(
            prompt="test", decision="allow", prob_harmful=0.1,
            response="Hello!", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=5.0, latency_total_ms=6.1, layer=10,
        )
        assert r.was_allowed is True
        assert r.was_denied is False

    def test_frozen(self):
        r = GuardrailResult(
            prompt="test", decision="allow", prob_harmful=0.1,
            response="Hello!", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=5.0, latency_total_ms=6.1, layer=10,
        )
        with pytest.raises(AttributeError):
            r.decision = "deny"  # type: ignore[misc]


# ─── StressTestResult ────────────────────────────────────────────────────────

class TestStressTestResult:
    def _make_result(self, decision: Decision, prob: float) -> GuardrailResult:
        return GuardrailResult(
            prompt="test", decision=decision, prob_harmful=prob,
            response="r", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=1.0, latency_total_ms=2.1, layer=0,
        )

    def test_empty(self):
        st = StressTestResult()
        assert st.num_total == 0
        assert st.denial_rate == 0.0

    def test_counts(self):
        st = StressTestResult(
            results=[
                self._make_result("deny", 0.9),
                self._make_result("allow", 0.1),
                self._make_result("deny", 0.8),
            ],
            true_labels=[1, 0, 1],
        )
        assert st.num_total == 3
        assert st.num_denied == 2
        assert st.num_allowed == 1

    def test_denial_rate(self):
        st = StressTestResult(
            results=[
                self._make_result("deny", 0.9),
                self._make_result("allow", 0.1),
            ],
            true_labels=[1, 0],
        )
        assert st.denial_rate == pytest.approx(0.5)

    def test_confusion_matrix(self):
        st = StressTestResult(
            results=[
                self._make_result("deny", 0.9),   # correctly denied harmful
                self._make_result("allow", 0.1),  # correctly allowed benign
                self._make_result("deny", 0.7),   # false positive (benign denied)
                self._make_result("allow", 0.3),  # false negative (harmful allowed)
            ],
            true_labels=[1, 0, 0, 1],
        )
        cm = st.confusion_matrix()
        assert cm.tp == 1
        assert cm.tn == 1
        assert cm.fp == 1
        assert cm.fn == 1

    def test_confusion_matrix_length_mismatch(self):
        st = StressTestResult(
            results=[self._make_result("deny", 0.9)],
            true_labels=[1, 0],  # wrong length
        )
        with pytest.raises(ValueError, match="same length"):
            st.confusion_matrix()

    def test_mean_latency(self):
        st = StressTestResult(
            results=[
                self._make_result("deny", 0.9),
                self._make_result("allow", 0.1),
            ]
        )
        assert st.mean_latency_ms == pytest.approx(2.1)


# ─── Guardrail pipeline (with tiny model) ────────────────────────────────────

class TestGuardrail:
    def test_process_returns_result(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Hello world")
        assert isinstance(result, GuardrailResult)
        assert result.prompt == "Hello world"
        assert result.decision in ("allow", "deny")
        assert 0.0 <= result.prob_harmful <= 1.0

    def test_process_latency_fields(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Test")
        assert result.latency_hook_ms > 0
        assert result.latency_classify_ms >= 0
        assert result.latency_total_ms > 0
        assert result.layer == 0

    def test_deny_with_low_threshold(self, tiny_setup):
        """With threshold=0.0, everything should be denied."""
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0, threshold=0.0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Benign question")
        assert result.was_denied

    def test_allow_with_high_threshold(self, tiny_setup):
        """With threshold=1.0, everything should be allowed."""
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0, threshold=1.0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Any question at all")
        assert result.was_allowed

    def test_denied_returns_refusal_message(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0, threshold=0.0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Test")
        assert "harmful intent" in result.response.lower()

    def test_allowed_returns_generated_text(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0, threshold=1.0, max_new_tokens=10)
        guardrail = Guardrail(model, tokenizer, probe, config)
        result = guardrail.process("Hello")
        assert len(result.response) > 0
        assert result.response != config.refusal_message

    def test_stress_test(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0, threshold=0.5)
        guardrail = Guardrail(model, tokenizer, probe, config)
        stress = guardrail.stress_test(
            prompts=["Hello", "Goodbye"],
            true_labels=[0, 0],
        )
        assert isinstance(stress, StressTestResult)
        assert stress.num_total == 2

    def test_stress_test_length_mismatch_raises(self, tiny_setup):
        model, tokenizer, probe = tiny_setup
        config = GuardrailConfig(layer=0)
        guardrail = Guardrail(model, tokenizer, probe, config)
        with pytest.raises(ValueError, match="same length"):
            guardrail.stress_test(["a", "b"], [0])


# ─── ComparisonResult ────────────────────────────────────────────────────────

class TestComparisonResult:
    def _make_guardrail_result(self, decision: Decision) -> GuardrailResult:
        return GuardrailResult(
            prompt="test", decision=decision, prob_harmful=0.5,
            response="r", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=1.0, latency_total_ms=2.1, layer=0,
        )

    def test_both_refuse(self):
        cr = ComparisonResult(
            prompt="bad prompt",
            model_response="I'm sorry, I cannot help with that.",
            guardrail_result=self._make_guardrail_result("deny"),
        )
        assert cr.model_refused is True
        assert cr.hook_denied is True
        assert cr.agreement == "both_refuse"

    def test_both_allow(self):
        cr = ComparisonResult(
            prompt="hello",
            model_response="Hello! How can I help you?",
            guardrail_result=self._make_guardrail_result("allow"),
        )
        assert cr.model_refused is False
        assert cr.hook_denied is False
        assert cr.agreement == "both_allow"

    def test_hook_catches(self):
        cr = ComparisonResult(
            prompt="jailbreak",
            model_response="Sure, here is how to make a bomb...",
            guardrail_result=self._make_guardrail_result("deny"),
        )
        assert cr.agreement == "hook_catches"

    def test_model_refuses_only(self):
        cr = ComparisonResult(
            prompt="ambiguous",
            model_response="I cannot help with that request.",
            guardrail_result=self._make_guardrail_result("allow"),
        )
        assert cr.agreement == "model_refuses"

    def test_model_refused_heuristic(self):
        # Test various refusal patterns
        for response in [
            "I can't do that",
            "Sorry, I won't help",
            "That's inappropriate",
            "I apologize, but I'm not able to",
        ]:
            cr = ComparisonResult(
                prompt="test",
                model_response=response,
                guardrail_result=self._make_guardrail_result("allow"),
            )
            assert cr.model_refused is True, f"Should detect refusal in: {response}"

    def test_model_not_refused(self):
        cr = ComparisonResult(
            prompt="test",
            model_response="The speed of light is approximately 299,792,458 meters per second.",
            guardrail_result=self._make_guardrail_result("allow"),
        )
        assert cr.model_refused is False
