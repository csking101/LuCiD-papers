"""
Tests for viz.py — Rich rendering components.

These tests verify that renderables are created without errors.
They do NOT require GPU or real models.
"""

from __future__ import annotations

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rich.panel import Panel
from rich.text import Text

from kl import CategoryKLSummary, InterpolatedOutput, SequenceKL, TokenKL
from viz import (
    _kl_bar,
    _kl_color,
    render_adventure_connection,
    render_category_summaries,
    render_conclusion,
    render_global_kl,
    render_interpolated_outputs,
    render_kl_heatmap,
    render_llm_parallel_table,
    render_metrics_table,
    render_phase_header,
    render_sequence_comparison,
    render_token_kl_table,
    render_welcome,
    sparkline,
)


# ── Sparkline ───────────────────────────────────────────────────────

class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value(self):
        result = sparkline([1.0])
        assert len(result) == 1

    def test_ascending(self):
        result = sparkline([0, 1, 2, 3, 4, 5])
        assert len(result) == 6

    def test_all_same(self):
        result = sparkline([1.0, 1.0, 1.0])
        # All same → all map to same block
        assert len(set(result)) == 1

    def test_width_truncation(self):
        result = sparkline(list(range(100)), width=10)
        assert len(result) == 10

    def test_negative_values(self):
        result = sparkline([-2, -1, 0, 1, 2])
        assert len(result) == 5


# ── KL helpers ──────────────────────────────────────────────────────

class TestKLHelpers:
    def test_kl_color_dim(self):
        assert _kl_color(0.05) == "dim"

    def test_kl_color_green(self):
        assert _kl_color(0.3) == "green"

    def test_kl_color_yellow(self):
        assert _kl_color(0.7) == "yellow"

    def test_kl_color_red(self):
        assert _kl_color(1.5) == "bright_red"

    def test_kl_color_bold_red(self):
        assert _kl_color(3.0) == "bold bright_red"

    def test_kl_bar_empty(self):
        bar = _kl_bar(0.0, max_kl=3.0, width=20)
        assert len(bar) == 20
        assert "\u2588" not in bar

    def test_kl_bar_full(self):
        bar = _kl_bar(3.0, max_kl=3.0, width=20)
        assert len(bar) == 20
        assert "\u2591" not in bar

    def test_kl_bar_half(self):
        bar = _kl_bar(1.5, max_kl=3.0, width=20)
        assert len(bar) == 20


# ── Welcome ─────────────────────────────────────────────────────────

class TestWelcome:
    def test_renders(self):
        panel = render_welcome()
        assert isinstance(panel, Panel)


# ── Phase headers ───────────────────────────────────────────────────

class TestPhaseHeaders:
    @pytest.mark.parametrize("phase", [1, 2, 3, 4, 5, 6])
    def test_all_phases(self, phase):
        panel = render_phase_header(phase)
        assert isinstance(panel, Panel)

    def test_unknown_phase(self):
        panel = render_phase_header(99)
        assert isinstance(panel, Panel)


# ── Token KL fixtures ──────────────────────────────────────────────

@pytest.fixture
def sample_token_kl():
    return TokenKL(
        tokens=["Hello", " world", "!"],
        token_ids=[100, 200, 300],
        kl_per_token=[0.1, 1.5, 0.3],
        base_top_k=[
            [("Hello", 0.5), ("Hi", 0.3), ("Hey", 0.1)],
            [(" world", 0.4), (" there", 0.3), (" everyone", 0.1)],
            [("!", 0.6), (".", 0.2), ("?", 0.1)],
        ],
        instruct_top_k=[
            [("Hello", 0.6), ("Hi", 0.2), ("Hey", 0.1)],
            [(" world", 0.2), (" there", 0.4), (" everyone", 0.2)],
            [("!", 0.7), (".", 0.1), ("?", 0.1)],
        ],
        total_kl=1.9,
        mean_kl=0.633,
    )


@pytest.fixture
def sample_seq_kl(sample_token_kl):
    return SequenceKL(
        prompt="Hello",
        base_text=" world of possibilities and wonder",
        instruct_text=" there! How can I help you today?",
        prompt_token_kl=sample_token_kl,
        total_kl=1.9,
        mean_kl=0.633,
    )


@pytest.fixture
def sample_summaries():
    return [
        CategoryKLSummary("safety", 3, 2.5, 3.0, 2.0),
        CategoryKLSummary("helpfulness", 3, 0.8, 1.2, 0.5),
        CategoryKLSummary("style", 3, 1.2, 1.5, 0.9),
    ]


@pytest.fixture
def sample_interpolated():
    return [
        InterpolatedOutput(0.0, "base text here", [1, 2, 3], ["base", " text", " here"], [0.0, 0.0, 0.0], 0.0),
        InterpolatedOutput(0.5, "mixed text here", [1, 2, 3], ["mixed", " text", " here"], [0.2, 0.3, 0.1], 0.6),
        InterpolatedOutput(1.0, "instruct text", [1, 2, 3], ["instruct", " text", ""], [0.5, 0.4, 0.3], 1.2),
    ]


# ── Render tests ────────────────────────────────────────────────────

class TestRenderTokenKLTable:
    def test_renders(self, sample_token_kl):
        panel = render_token_kl_table(sample_token_kl)
        assert isinstance(panel, Panel)

    def test_renders_with_max_rows(self, sample_token_kl):
        panel = render_token_kl_table(sample_token_kl, max_rows=2)
        assert isinstance(panel, Panel)

    def test_custom_title(self, sample_token_kl):
        panel = render_token_kl_table(sample_token_kl, title="Custom")
        assert isinstance(panel, Panel)


class TestRenderGlobalKL:
    def test_renders(self):
        panel = render_global_kl(0.5, [0.3, 0.5, 0.7], ["a", "b", "c"])
        assert isinstance(panel, Panel)

    def test_empty(self):
        panel = render_global_kl(0.0, [], [])
        assert isinstance(panel, Panel)


class TestRenderSequenceComparison:
    def test_renders(self, sample_seq_kl):
        panel = render_sequence_comparison(sample_seq_kl)
        assert isinstance(panel, Panel)


class TestRenderCategorySummaries:
    def test_renders(self, sample_summaries):
        panel = render_category_summaries(sample_summaries)
        assert isinstance(panel, Panel)

    def test_empty(self):
        panel = render_category_summaries([])
        assert isinstance(panel, Panel)

    def test_single_category(self):
        panel = render_category_summaries([
            CategoryKLSummary("test", 1, 0.5, 0.5, 0.5),
        ])
        assert isinstance(panel, Panel)


class TestRenderInterpolated:
    def test_renders(self, sample_interpolated):
        panel = render_interpolated_outputs(sample_interpolated, "test prompt")
        assert isinstance(panel, Panel)

    def test_two_outputs(self):
        outs = [
            InterpolatedOutput(0.0, "a", [1], ["a"], [0.0], 0.0),
            InterpolatedOutput(1.0, "b", [2], ["b"], [0.5], 0.5),
        ]
        panel = render_interpolated_outputs(outs, "p")
        assert isinstance(panel, Panel)


class TestRenderKLHeatmap:
    def test_renders(self, sample_token_kl):
        panel = render_kl_heatmap(sample_token_kl)
        assert isinstance(panel, Panel)


class TestRenderMetricsTable:
    def test_renders(self):
        panel = render_metrics_table({"Key": ("Value", "spark")})
        assert isinstance(panel, Panel)

    def test_empty(self):
        panel = render_metrics_table({})
        assert isinstance(panel, Panel)


class TestRenderLLMParallel:
    def test_renders(self):
        panel = render_llm_parallel_table()
        assert isinstance(panel, Panel)


class TestRenderAdventureConnection:
    def test_renders(self):
        panel = render_adventure_connection()
        assert isinstance(panel, Panel)


class TestRenderConclusion:
    def test_renders(self, sample_summaries):
        panel = render_conclusion(0.5, sample_summaries)
        assert isinstance(panel, Panel)

    def test_renders_empty_summaries(self):
        panel = render_conclusion(0.0, [])
        assert isinstance(panel, Panel)
