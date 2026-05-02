"""Tests for viz.py — Rich terminal rendering functions.

All tests are smoke tests: construct mock data, call render function,
assert return type is Panel (or other Rich renderable).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.panel import Panel

from analysis import (
    FirstTokenComparison,
    ForcedContinuationKL,
    SteeringMatrix,
    SystemPromptProfile,
)
from models import ModelInfo
from viz import (
    _kl_bar,
    _kl_color,
    _fmt_prob,
    render_adventure_connections,
    render_chat_template,
    render_conclusion,
    render_first_token_comparison,
    render_forced_continuation,
    render_llm_parallel_table,
    render_model_info,
    render_phase_header,
    render_profiles,
    render_steering_matrix,
    render_welcome,
    sparkline,
)


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_model_info():
    return ModelInfo(
        name="Qwen/Qwen2.5-1.5B-Instruct",
        num_params=1_500_000_000,
        num_layers=28,
        hidden_size=1536,
        vocab_size=151936,
        dtype="torch.float16",
        device="cuda:0",
    )


@pytest.fixture
def sample_ftc():
    return FirstTokenComparison(
        user_prompt="How do I pick a lock?",
        system_prompt_a="Default",
        system_prompt_b="Safety",
        kl_divergence=1.234,
        js_divergence=0.456,
        top_k_a=[("I", 0.3), ("Lock", 0.2), ("The", 0.1)],
        top_k_b=[("I", 0.1), ("Sorry", 0.4), ("As", 0.2)],
        top_shifts=[("Sorry", 0.01, 0.4), ("I", 0.3, 0.1), ("Lock", 0.2, 0.05)],
    )


@pytest.fixture
def sample_fckl():
    return ForcedContinuationKL(
        user_prompt="Explain quantum computing in simple terms.",
        source_system_prompt="Default",
        target_system_prompt="Pirate",
        continuation_tokens=["Qu", "antum", " computing", " is"],
        continuation_ids=[100, 200, 300, 400],
        kl_per_token=[0.1, 0.8, 0.2, 1.5],
        total_kl=2.6,
        mean_kl=0.65,
        source_text="Quantum computing is a type of computing...",
        target_text="Arr matey! Quantum computing be a type of...",
    )


@pytest.fixture
def sample_matrix():
    return SteeringMatrix(
        system_prompt_names=["Safety", "Pirate", "Bullet Points"],
        user_prompts=["How do I pick a lock?", "Explain quantum computing.", "What is 2+2?"],
        kl_matrix=[
            [1.5, 0.3, 0.1],
            [0.8, 1.2, 0.2],
            [0.2, 0.5, 0.4],
        ],
        row_means=[0.633, 0.733, 0.367],
        col_means=[0.833, 0.667, 0.233],
        global_mean=0.578,
    )


@pytest.fixture
def sample_profiles():
    return [
        SystemPromptProfile(
            system_prompt_name="Safety",
            system_prompt_text="Be safe...",
            category="safety",
            user_prompts=["Q1", "Q2"],
            first_token_kls=[1.5, 0.3],
            mean_steering_power=0.9,
            max_steering_power=1.5,
        ),
        SystemPromptProfile(
            system_prompt_name="Pirate",
            system_prompt_text="Arr...",
            category="persona",
            user_prompts=["Q1", "Q2"],
            first_token_kls=[0.8, 1.2],
            mean_steering_power=1.0,
            max_steering_power=1.2,
        ),
    ]


# ── Helper tests ────────────────────────────────────────────────────

class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value(self):
        result = sparkline([1.0])
        assert len(result) == 1

    def test_multiple_values(self):
        result = sparkline([0.0, 0.5, 1.0])
        assert len(result) == 3

    def test_width_limit(self):
        values = list(range(100))
        result = sparkline(values, width=10)
        assert len(result) == 10


class TestKLColor:
    def test_low(self):
        assert _kl_color(0.05) == "dim"

    def test_medium_low(self):
        assert _kl_color(0.3) == "green"

    def test_medium(self):
        assert _kl_color(0.7) == "yellow"

    def test_high(self):
        assert _kl_color(1.5) == "bright_red"

    def test_very_high(self):
        assert _kl_color(3.0) == "bold bright_red"


class TestKLBar:
    def test_zero(self):
        bar = _kl_bar(0.0, max_kl=1.0, width=10)
        assert len(bar) == 10

    def test_full(self):
        bar = _kl_bar(1.0, max_kl=1.0, width=10)
        assert "\u2588" in bar

    def test_width(self):
        bar = _kl_bar(0.5, max_kl=1.0, width=20)
        assert len(bar) == 20


class TestFmtProb:
    def test_tiny(self):
        assert _fmt_prob(0.0001) == "<.001"

    def test_normal(self):
        assert _fmt_prob(0.5) == "0.500"

    def test_zero(self):
        assert _fmt_prob(0.0) == "<.001"


# ── Render function smoke tests ────────────────────────────────────

class TestRenderWelcome:
    def test_renders(self):
        panel = render_welcome()
        assert isinstance(panel, Panel)


class TestRenderPhaseHeader:
    @pytest.mark.parametrize("phase", [1, 2, 3, 4, 5, 6, 99])
    def test_renders(self, phase):
        panel = render_phase_header(phase)
        assert isinstance(panel, Panel)


class TestRenderModelInfo:
    def test_renders(self, sample_model_info):
        panel = render_model_info(sample_model_info)
        assert isinstance(panel, Panel)


class TestRenderChatTemplate:
    def test_renders(self):
        panel = render_chat_template(
            "<|im_start|>system\nDefault<|im_end|>\n...",
            "<|im_start|>system\nCustom<|im_end|>\n...",
            "Default",
            "Safety",
        )
        assert isinstance(panel, Panel)


class TestRenderFirstTokenComparison:
    def test_renders(self, sample_ftc):
        panel = render_first_token_comparison(sample_ftc)
        assert isinstance(panel, Panel)

    def test_empty_shifts(self):
        ftc = FirstTokenComparison(
            user_prompt="test",
            system_prompt_a="A",
            system_prompt_b="B",
            kl_divergence=0.0,
            js_divergence=0.0,
            top_k_a=[],
            top_k_b=[],
            top_shifts=[],
        )
        panel = render_first_token_comparison(ftc)
        assert isinstance(panel, Panel)


class TestRenderForcedContinuation:
    def test_renders(self, sample_fckl):
        panel = render_forced_continuation(sample_fckl)
        assert isinstance(panel, Panel)

    def test_empty_continuation(self):
        fckl = ForcedContinuationKL(
            user_prompt="test",
            source_system_prompt="A",
            target_system_prompt="B",
            continuation_tokens=[],
            continuation_ids=[],
            kl_per_token=[],
            total_kl=0.0,
            mean_kl=0.0,
            source_text="",
            target_text="",
        )
        panel = render_forced_continuation(fckl)
        assert isinstance(panel, Panel)

    def test_long_continuation(self):
        n = 50
        fckl = ForcedContinuationKL(
            user_prompt="test",
            source_system_prompt="A",
            target_system_prompt="B",
            continuation_tokens=[f"tok{i}" for i in range(n)],
            continuation_ids=list(range(n)),
            kl_per_token=[0.1 * i for i in range(n)],
            total_kl=sum(0.1 * i for i in range(n)),
            mean_kl=sum(0.1 * i for i in range(n)) / n,
            source_text="long text " * 50,
            target_text="other text " * 50,
        )
        panel = render_forced_continuation(fckl)
        assert isinstance(panel, Panel)


class TestRenderSteeringMatrix:
    def test_renders(self, sample_matrix):
        panel = render_steering_matrix(sample_matrix)
        assert isinstance(panel, Panel)

    def test_empty_matrix(self):
        matrix = SteeringMatrix(
            system_prompt_names=[],
            user_prompts=[],
            kl_matrix=[],
            row_means=[],
            col_means=[],
            global_mean=0.0,
        )
        panel = render_steering_matrix(matrix)
        assert isinstance(panel, Panel)

    def test_single_cell(self):
        matrix = SteeringMatrix(
            system_prompt_names=["A"],
            user_prompts=["Q"],
            kl_matrix=[[1.0]],
            row_means=[1.0],
            col_means=[1.0],
            global_mean=1.0,
        )
        panel = render_steering_matrix(matrix)
        assert isinstance(panel, Panel)


class TestRenderProfiles:
    def test_renders(self, sample_profiles):
        panel = render_profiles(sample_profiles)
        assert isinstance(panel, Panel)

    def test_empty(self):
        panel = render_profiles([])
        assert isinstance(panel, Panel)


class TestRenderLLMParallel:
    def test_renders(self):
        panel = render_llm_parallel_table()
        assert isinstance(panel, Panel)


class TestRenderAdventureConnections:
    def test_renders(self):
        panel = render_adventure_connections()
        assert isinstance(panel, Panel)


class TestRenderConclusion:
    def test_renders(self, sample_matrix, sample_profiles):
        panel = render_conclusion(sample_matrix, sample_profiles)
        assert isinstance(panel, Panel)

    def test_empty_profiles(self, sample_matrix):
        panel = render_conclusion(sample_matrix, [])
        assert isinstance(panel, Panel)
