"""Tests for viz.py — Rich terminal rendering."""

import sys
from pathlib import Path

_adventure_root = Path(__file__).resolve().parent.parent
if str(_adventure_root) not in sys.path:
    sys.path.insert(0, str(_adventure_root))

import pytest
from rich.panel import Panel
from rich.table import Table

from classifier import ConfusionMatrix, LayerSweepResult, ProbeResult
from hooks import ModelInfo
from pipeline import ComparisonResult, GuardrailResult, StressTestResult
from viz import (
    render_adventure_connections,
    render_comparison,
    render_conclusion,
    render_confusion_matrix,
    render_dataset_summary,
    render_guardrail_result,
    render_hook_demo,
    render_layer_sweep,
    render_llm_parallel_table,
    render_model_info,
    render_module_hierarchy,
    render_phase_header,
    render_probe_detail,
    render_prompt_samples,
    render_stress_test,
    render_welcome,
)


# ─── Welcome and headers ────────────────────────────────────────────────────

class TestWelcome:
    def test_returns_panel(self):
        result = render_welcome()
        assert isinstance(result, Panel)

    def test_contains_title(self):
        result = render_welcome()
        assert "Adventure 05" in str(result.title)


class TestPhaseHeader:
    def test_returns_panel(self):
        result = render_phase_header(1, "Test Phase")
        assert isinstance(result, Panel)

    def test_with_description(self):
        result = render_phase_header(1, "Test", "A description")
        assert isinstance(result, Panel)


# ─── Phase 1: Hook anatomy ──────────────────────────────────────────────────

class TestModelInfo:
    def test_returns_panel(self):
        info = ModelInfo("test", 12, 768, 50000, "cpu", "float32", 1000000)
        result = render_model_info(info)
        assert isinstance(result, Panel)


class TestModuleHierarchy:
    def test_returns_panel(self):
        paths = ["model", "model.layers", "model.layers.0"]
        result = render_module_hierarchy(paths)
        assert isinstance(result, Panel)

    def test_empty_paths(self):
        result = render_module_hierarchy([])
        assert isinstance(result, Panel)


class TestHookDemo:
    def test_returns_panel(self):
        result = render_hook_demo(5, (1, 10, 768), "Hello world")
        assert isinstance(result, Panel)


# ─── Phase 2: Dataset ────────────────────────────────────────────────────────

class TestDatasetSummary:
    def test_returns_panel(self):
        summary = {"benign": 40, "harmful": 40, "ambiguous": 10, "jailbreak": 10}
        result = render_dataset_summary(summary)
        assert isinstance(result, Panel)

    def test_empty_summary(self):
        result = render_dataset_summary({})
        assert isinstance(result, Panel)


class TestPromptSamples:
    def test_returns_panel(self):
        samples = [
            ("Hello", "benign", "factual"),
            ("Bad thing", "harmful", "violence"),
        ]
        result = render_prompt_samples(samples)
        assert isinstance(result, Panel)

    def test_long_prompt_truncated(self):
        samples = [("x" * 200, "benign", "test")]
        result = render_prompt_samples(samples)
        assert isinstance(result, Panel)

    def test_empty(self):
        result = render_prompt_samples([])
        assert isinstance(result, Panel)


# ─── Phase 3: Layer sweep ────────────────────────────────────────────────────

def _make_probe_result(layer: int, accuracy: float) -> ProbeResult:
    return ProbeResult(
        layer=layer, accuracy=accuracy, precision=accuracy,
        recall=accuracy, f1=accuracy, train_loss=0.1,
        num_train=60, num_test=20,
    )


class TestLayerSweep:
    def test_returns_panel(self):
        sweep = LayerSweepResult(results=[
            _make_probe_result(0, 0.6),
            _make_probe_result(1, 0.9),
        ])
        result = render_layer_sweep(sweep)
        assert isinstance(result, Panel)

    def test_single_layer(self):
        sweep = LayerSweepResult(results=[_make_probe_result(0, 0.8)])
        result = render_layer_sweep(sweep)
        assert isinstance(result, Panel)


class TestProbeDetail:
    def test_returns_panel(self):
        pr = _make_probe_result(5, 0.85)
        result = render_probe_detail(pr)
        assert isinstance(result, Panel)


# ─── Phase 4: Guardrail ──────────────────────────────────────────────────────

class TestGuardrailResult:
    def _make_result(self, decision, prob):
        return GuardrailResult(
            prompt="test", decision=decision, prob_harmful=prob,
            response="response text", latency_hook_ms=5.0,
            latency_classify_ms=0.1, latency_generate_ms=10.0,
            latency_total_ms=15.1, layer=10,
        )

    def test_denied_returns_panel(self):
        result = render_guardrail_result(self._make_result("deny", 0.9))
        assert isinstance(result, Panel)

    def test_allowed_returns_panel(self):
        result = render_guardrail_result(self._make_result("allow", 0.1))
        assert isinstance(result, Panel)


# ─── Phase 5: Stress test ────────────────────────────────────────────────────

class TestConfusionMatrixViz:
    def test_returns_panel(self):
        cm = ConfusionMatrix(tp=10, fp=2, tn=8, fn=1)
        result = render_confusion_matrix(cm)
        assert isinstance(result, Panel)

    def test_all_zeros(self):
        cm = ConfusionMatrix(tp=0, fp=0, tn=0, fn=0)
        result = render_confusion_matrix(cm)
        assert isinstance(result, Panel)


class TestStressTest:
    def _make_gr(self, decision, prob):
        return GuardrailResult(
            prompt="test prompt", decision=decision, prob_harmful=prob,
            response="r", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=1.0, latency_total_ms=2.1, layer=0,
        )

    def test_returns_panel(self):
        stress = StressTestResult(
            results=[self._make_gr("deny", 0.9), self._make_gr("allow", 0.1)],
            true_labels=[1, 0],
        )
        result = render_stress_test(stress)
        assert isinstance(result, Panel)


class TestComparison:
    def test_returns_panel(self):
        gr = GuardrailResult(
            prompt="test", decision="deny", prob_harmful=0.9,
            response="refused", latency_hook_ms=1.0, latency_classify_ms=0.1,
            latency_generate_ms=0.0, latency_total_ms=1.1, layer=10,
        )
        comp = ComparisonResult(
            prompt="test",
            model_response="I'm sorry, I cannot help.",
            guardrail_result=gr,
        )
        result = render_comparison(comp)
        assert isinstance(result, Panel)


# ─── Phase 6: Conclusion ────────────────────────────────────────────────────

class TestConclusion:
    def test_returns_panel(self):
        cm = ConfusionMatrix(tp=10, fp=2, tn=8, fn=1)
        result = render_conclusion(
            best_layer=15, best_accuracy=0.95,
            num_layers=28, cm=cm, jailbreak_catch_rate=0.8,
        )
        assert isinstance(result, Panel)


class TestLLMParallel:
    def test_returns_panel(self):
        result = render_llm_parallel_table()
        assert isinstance(result, Panel)


class TestAdventureConnections:
    def test_returns_panel(self):
        result = render_adventure_connections()
        assert isinstance(result, Panel)
