"""
Tests for Rich Visualization functions.
========================================

Covers:
- Sparkline generation (empty, single, normal, all-same)
- Grid rendering (path, walls, start/goal)
- Policy arrow rendering
- Heatmap rendering
- Metrics table
- NN forward pass view
- Phase headers
- Preference pair display
- Policy comparison
- Welcome and summary panels
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import DOWN, RIGHT, UP, COIN, GEM, GridWorld, Trajectory
from viz import (
    render_grid,
    render_heatmap,
    render_llm_parallel_table,
    render_metrics_table,
    render_nn_forward,
    render_phase_header,
    render_policy_arrows,
    render_policy_comparison,
    render_preference_pair,
    render_results_summary,
    render_rm_architecture,
    render_rm_spot_check,
    render_welcome,
    sparkline,
)


@pytest.fixture
def env():
    return GridWorld()


@pytest.fixture
def tiny_env():
    return GridWorld(size=4, walls=[], pickups=[])


@pytest.fixture
def console():
    return Console(width=120, force_terminal=True)


# Helper to check a renderable can be printed without error
def _can_render(console, renderable):
    """Ensure the renderable produces output without exceptions."""
    with console.capture() as cap:
        console.print(renderable)
    output = cap.get()
    assert len(output) > 0
    return output


# -----------------------------------------------------------------------
# Sparkline
# -----------------------------------------------------------------------
class TestSparkline:
    def test_empty(self):
        assert sparkline([]) == ""

    def test_single_value(self):
        s = sparkline([5.0])
        assert len(s) == 1

    def test_ascending(self):
        s = sparkline([1, 2, 3, 4, 5])
        assert len(s) == 5
        # First char should be lowest, last highest
        assert s[0] <= s[-1]  # lexicographic, but sparkline chars are ordered

    def test_all_same(self):
        s = sparkline([3.0, 3.0, 3.0])
        # All same value → all same char
        assert len(set(s)) == 1

    def test_width_truncation(self):
        s = sparkline(list(range(100)), width=10)
        assert len(s) == 10

    def test_negative_values(self):
        s = sparkline([-5, -3, -1, 0, 2])
        assert len(s) == 5


# -----------------------------------------------------------------------
# Grid rendering
# -----------------------------------------------------------------------
class TestRenderGrid:
    def test_basic_render(self, env, console):
        panel = render_grid(env)
        output = _can_render(console, panel)
        assert "S" in output
        assert "G" in output

    def test_with_path(self, env, console):
        path = [(0, 0), (0, 1), (0, 2)]
        panel = render_grid(env, path=path)
        output = _can_render(console, panel)
        assert "\u25cf" in output  # bullet character

    def test_with_stats(self, tiny_env, console):
        traj = Trajectory(
            actions=[RIGHT, RIGHT, DOWN],
            positions=[(0, 0), (0, 1), (0, 2), (1, 2)],
            states=[np.zeros(2)] * 3,
            rewards=[0.0] * 3,
        )
        panel = render_grid(tiny_env, path=traj.positions, show_stats=True, traj=traj)
        output = _can_render(console, panel)
        assert "Steps: 3" in output

    def test_custom_title(self, env, console):
        panel = render_grid(env, title="My Grid")
        output = _can_render(console, panel)
        assert "My Grid" in output

    def test_returns_panel(self, env):
        result = render_grid(env)
        assert isinstance(result, Panel)


# -----------------------------------------------------------------------
# Policy arrows
# -----------------------------------------------------------------------
class TestRenderPolicyArrows:
    def test_renders(self, env, console):
        policy_map = np.zeros((env.size, env.size), dtype=np.int32)
        panel = render_policy_arrows(policy_map, env)
        output = _can_render(console, panel)
        assert "\u2191" in output or "\u2193" in output  # arrows

    def test_walls_shown(self, env, console):
        policy_map = np.zeros((env.size, env.size), dtype=np.int32)
        panel = render_policy_arrows(policy_map, env)
        output = _can_render(console, panel)
        assert "\u2588" in output  # block char for walls


# -----------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------
class TestRenderHeatmap:
    def test_renders(self, env, console):
        values = np.random.randn(env.size, env.size)
        panel = render_heatmap(values, env)
        output = _can_render(console, panel)
        assert "min=" in output
        assert "max=" in output

    def test_uniform_values(self, env, console):
        values = np.ones((env.size, env.size))
        panel = render_heatmap(values, env, title="Uniform")
        output = _can_render(console, panel)
        assert "Uniform" in output

    def test_custom_title(self, env, console):
        values = np.zeros((env.size, env.size))
        panel = render_heatmap(values, env, title="RM Rewards")
        output = _can_render(console, panel)
        assert "RM Rewards" in output


# -----------------------------------------------------------------------
# Metrics table
# -----------------------------------------------------------------------
class TestRenderMetricsTable:
    def test_renders(self, console):
        metrics = {
            "Episode": ("150", ""),
            "Avg Reward": ("4.32", sparkline([1, 2, 3, 4])),
            "Goal Rate": ("72%", sparkline([0.5, 0.6, 0.7])),
        }
        panel = render_metrics_table(metrics)
        output = _can_render(console, panel)
        assert "Episode" in output
        assert "150" in output

    def test_empty_metrics(self, console):
        panel = render_metrics_table({})
        output = _can_render(console, panel)
        assert len(output) > 0


# -----------------------------------------------------------------------
# NN forward pass
# -----------------------------------------------------------------------
class TestRenderNNForward:
    def test_basic_render(self, console):
        info = {
            "input": [0.42, 0.67],
            "hidden_1": np.random.rand(32),
            "hidden_2": np.random.rand(32),
            "logits": np.array([-1.2, 2.8, -0.5, 1.1]),
            "probs": np.array([0.02, 0.72, 0.03, 0.23]),
            "value": 5.67,
        }
        panel = render_nn_forward(info)
        output = _can_render(console, panel)
        assert "Input" in output
        assert "V(s)" in output

    def test_with_reference_probs(self, console):
        info = {
            "input": [0.5, 0.5],
            "hidden_1": np.random.rand(32),
            "hidden_2": np.random.rand(32),
            "logits": np.array([0.0, 1.0, -1.0, 0.5]),
            "probs": np.array([0.2, 0.4, 0.1, 0.3]),
            "value": 3.0,
        }
        ref_probs = np.array([0.25, 0.25, 0.25, 0.25])
        panel = render_nn_forward(info, ref_probs=ref_probs)
        output = _can_render(console, panel)
        assert "Pre-trained" in output
        assert "KL" in output


# -----------------------------------------------------------------------
# RM architecture and spot-check
# -----------------------------------------------------------------------
class TestRMPanels:
    def test_rm_architecture(self, console):
        panel = render_rm_architecture()
        output = _can_render(console, panel)
        assert "Bradley-Terry" in output

    def test_rm_spot_check_correct(self, console):
        panel = render_rm_spot_check(
            pair_idx=7, r_a=3.42, r_b=1.87, prob_a=0.82, human_label=0.0
        )
        output = _can_render(console, panel)
        assert "Correct" in output

    def test_rm_spot_check_incorrect(self, console):
        panel = render_rm_spot_check(
            pair_idx=3, r_a=1.0, r_b=3.0, prob_a=0.2, human_label=0.0
        )
        output = _can_render(console, panel)
        assert "Incorrect" in output


# -----------------------------------------------------------------------
# Policy comparison
# -----------------------------------------------------------------------
class TestPolicyComparison:
    def test_identical_policies(self, env, console):
        policy_map = np.zeros((env.size, env.size), dtype=np.int32)
        panel = render_policy_comparison(policy_map, policy_map, env)
        output = _can_render(console, panel)
        assert "0.0%" in output  # no changes

    def test_different_policies(self, env, console):
        map_a = np.zeros((env.size, env.size), dtype=np.int32)
        map_b = np.ones((env.size, env.size), dtype=np.int32)
        panel = render_policy_comparison(map_a, map_b, env)
        output = _can_render(console, panel)
        assert "\u2260" in output  # difference markers


# -----------------------------------------------------------------------
# Phase headers and LLM parallels
# -----------------------------------------------------------------------
class TestPhaseHeaders:
    @pytest.mark.parametrize("phase", [1, 2, 3, 4])
    def test_all_phases(self, phase, console):
        panel = render_phase_header(phase)
        output = _can_render(console, panel)
        assert "Phase" in output

    def test_llm_parallel_table(self, console):
        panel = render_llm_parallel_table()
        output = _can_render(console, panel)
        assert "LLM" in output
        assert "PPO" in output


# -----------------------------------------------------------------------
# Results summary
# -----------------------------------------------------------------------
class TestResultsSummary:
    def test_renders(self, console):
        pt = {"Avg Reward": 4.0, "Goal Rate": 0.7}
        rl = {"Avg Reward": 6.0, "Goal Rate": 0.95}
        panel = render_results_summary(pt, rl)
        output = _can_render(console, panel)
        assert "4.00" in output
        assert "6.00" in output


# -----------------------------------------------------------------------
# Welcome and preference pair
# -----------------------------------------------------------------------
class TestMiscPanels:
    def test_welcome(self, console):
        panel = render_welcome()
        output = _can_render(console, panel)
        assert "RLHF" in output

    def test_preference_pair(self, tiny_env, console):
        traj_a = Trajectory(
            actions=[RIGHT, RIGHT],
            positions=[(0, 0), (0, 1), (0, 2)],
            states=[np.zeros(2)] * 2,
            rewards=[0.0] * 2,
        )
        traj_b = Trajectory(
            actions=[DOWN, DOWN],
            positions=[(0, 0), (1, 0), (2, 0)],
            states=[np.zeros(2)] * 2,
            rewards=[0.0] * 2,
        )
        group = render_preference_pair(tiny_env, traj_a, traj_b, 5, 20)
        output = _can_render(console, group)
        assert "Path A" in output
        assert "Path B" in output
        assert "5 / 20" in output


# -----------------------------------------------------------------------
# Pickup rendering
# -----------------------------------------------------------------------
class TestPickupRendering:
    def test_coins_shown(self, console):
        """Grid with coins should render the ¢ character."""
        pickups = [(1, 1, COIN)]
        env = GridWorld(size=4, walls=[], pickups=pickups)
        panel = render_grid(env)
        output = _can_render(console, panel)
        assert "\u00a2" in output  # ¢

    def test_gems_shown(self, console):
        """Grid with gems should render the ◆ character."""
        pickups = [(1, 1, GEM)]
        env = GridWorld(size=4, walls=[], pickups=pickups)
        panel = render_grid(env)
        output = _can_render(console, panel)
        assert "\u25c6" in output  # ◆

    def test_default_env_shows_pickups(self, env, console):
        """Default 8x8 env should show both coin and gem characters."""
        panel = render_grid(env)
        output = _can_render(console, panel)
        assert "\u00a2" in output  # ¢
        assert "\u25c6" in output  # ◆

    def test_pickup_stats_in_subtitle(self, console):
        """When show_stats=True with a trajectory, subtitle should show Pickups."""
        pickups = [(0, 1, COIN), (0, 2, GEM)]
        env = GridWorld(size=4, walls=[], pickups=pickups)
        traj = Trajectory(
            actions=[RIGHT, RIGHT],
            positions=[(0, 0), (0, 1), (0, 2)],
            states=[np.zeros(2)] * 2,
            rewards=[0.49, 1.99],
            pickups_collected=[(0, 1, COIN), (0, 2, GEM)],
        )
        panel = render_grid(
            env, path=traj.positions, show_stats=True, traj=traj,
        )
        output = _can_render(console, panel)
        assert "Pickups: 2/2" in output
