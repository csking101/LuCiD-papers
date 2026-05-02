"""Tests for viz.py — Rich rendering functions for the Pocket Cube adventure.

Covers:
    - Every public render function returns the correct Rich type
    - Content sanity checks (expected strings, row counts)
    - Helper functions (sparkline, bar)
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cube import PocketCube, ACTION_NAMES
from train import DepthStats, TrainingStats, SolveResult, TestResults
from viz import (
    render_cube, render_phase_header, render_env_info,
    render_move_demo, render_training_progress, render_curriculum_summary,
    render_solve_attempt, render_solve_sequence,
    render_difficulty_test, render_comparison,
    render_llm_parallel, render_summary,
    _spark, _bar, _pct,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestHelpers:
    def test_spark_empty(self):
        assert _spark([]) == ""

    def test_spark_single(self):
        result = _spark([0.5])
        assert len(result) == 1

    def test_spark_monotonic(self):
        result = _spark([0.0, 0.25, 0.5, 0.75, 1.0])
        assert len(result) == 5

    def test_spark_resamples(self):
        result = _spark(list(range(100)), width=10)
        assert len(result) == 10

    def test_bar_full(self):
        result = _bar(1.0, width=10)
        assert "█" * 10 == result

    def test_bar_empty(self):
        result = _bar(0.0, width=10)
        assert "░" * 10 == result

    def test_bar_half(self):
        result = _bar(0.5, width=10)
        assert len(result) == 10

    def test_pct(self):
        assert _pct(0.85) == "85.0%"
        assert _pct(0.0) == "0.0%"
        assert _pct(1.0) == "100.0%"


# ═══════════════════════════════════════════════════════════════════════════════
# Cube Rendering
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderCube:
    def test_returns_panel(self):
        cube = PocketCube()
        result = render_cube(cube)
        assert isinstance(result, Panel)

    def test_with_label(self):
        cube = PocketCube()
        result = render_cube(cube, label="Solved")
        assert isinstance(result, Panel)

    def test_scrambled_cube(self):
        cube = PocketCube()
        cube.scramble(3, rng=random.Random(42))
        result = render_cube(cube, "Scrambled")
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase Header
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseHeader:
    def test_returns_panel(self):
        result = render_phase_header(1, "Test Phase", "A description")
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Info
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvInfo:
    def test_returns_panel(self):
        result = render_env_info()
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Move Demo
# ═══════════════════════════════════════════════════════════════════════════════

class TestMoveDemo:
    def test_returns_panel(self):
        before = PocketCube()
        after = before.clone()
        after.apply_move(0)
        result = render_move_demo(before, after, "U")
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Training Progress
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingProgress:
    def _make_data(self):
        ds = DepthStats(depth=1, episodes_trained=128, solve_rate=0.65,
                        avg_return=0.5, avg_episode_length=3.2,
                        policy_loss=0.1, value_loss=0.2, entropy=1.5)
        stats = TrainingStats(
            current_depth=1, total_episodes=128, total_updates=1,
            solve_rate_history=[(128, 0.65)],
        )
        return ds, stats

    def test_returns_panel(self):
        ds, stats = self._make_data()
        result = render_training_progress(1, ds, stats)
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurriculumSummary:
    def test_returns_table(self):
        stats = TrainingStats(depth_stats=[
            DepthStats(depth=1, episodes_trained=256, solve_rate=0.9, advanced=True, time_seconds=5.0),
            DepthStats(depth=2, episodes_trained=512, solve_rate=0.85, advanced=True, time_seconds=10.0),
        ])
        result = render_curriculum_summary(stats)
        assert isinstance(result, Table)

    def test_row_count(self):
        stats = TrainingStats(depth_stats=[
            DepthStats(depth=i, time_seconds=1.0) for i in range(1, 4)
        ])
        result = render_curriculum_summary(stats)
        assert result.row_count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Solve Attempt
# ═══════════════════════════════════════════════════════════════════════════════

class TestSolveAttempt:
    def test_returns_panel_solved(self):
        sr = SolveResult(depth=2, solved=True, steps=2, moves=[1, 3], states=[])
        result = render_solve_attempt(sr)
        assert isinstance(result, Panel)

    def test_returns_panel_failed(self):
        sr = SolveResult(depth=3, solved=False, steps=11, moves=[0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3], states=[])
        result = render_solve_attempt(sr)
        assert isinstance(result, Panel)


class TestSolveSequence:
    def test_returns_panel(self):
        cube = PocketCube()
        initial = list(cube.state)
        cube.scramble(1, rng=random.Random(42))
        sr = SolveResult(depth=1, solved=True, steps=1, moves=[0],
                         states=[list(cube.state), initial])
        result = render_solve_sequence(sr)
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Difficulty Test
# ═══════════════════════════════════════════════════════════════════════════════

class TestDifficultyTest:
    def test_returns_panel(self):
        results = [
            TestResults(depth=d, n_episodes=50, solve_rate=max(0, 1.0 - d * 0.15),
                        avg_steps=d * 2.0, results=[])
            for d in range(1, 6)
        ]
        result = render_difficulty_test(results)
        assert isinstance(result, Panel)


# ═══════════════════════════════════════════════════════════════════════════════
# Comparison
# ═══════════════════════════════════════════════════════════════════════════════

class TestComparison:
    def test_returns_table(self):
        rand_results = [
            TestResults(depth=d, n_episodes=50, solve_rate=0.01, avg_steps=10.0, results=[])
            for d in range(1, 4)
        ]
        trained_results = [
            TestResults(depth=d, n_episodes=50, solve_rate=0.9, avg_steps=d * 1.5, results=[])
            for d in range(1, 4)
        ]
        result = render_comparison(rand_results, trained_results)
        assert isinstance(result, Table)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Parallel & Summary
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMParallel:
    def test_returns_panel(self):
        rows = [("Cube state", "Token sequence"), ("Move", "Next token")]
        result = render_llm_parallel("Analogy", rows)
        assert isinstance(result, Panel)


class TestSummary:
    def test_returns_panel(self):
        stats = TrainingStats(
            depth_stats=[
                DepthStats(depth=1, solve_rate=0.9, advanced=True, time_seconds=3.0),
            ],
            total_episodes=256, total_updates=2, total_time=10.0,
        )
        results = [
            TestResults(depth=1, n_episodes=50, solve_rate=0.9, avg_steps=1.5, results=[]),
        ]
        result = render_summary(stats, results)
        assert isinstance(result, Panel)
