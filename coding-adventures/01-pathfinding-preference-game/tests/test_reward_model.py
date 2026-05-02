"""
Tests for the Bradley-Terry Reward Model.
==========================================

Covers:
- Network architecture and forward shapes
- Per-state vs trajectory reward
- Bradley-Terry loss computation and gradient flow
- Bradley-Terry probability function
- Reward heatmap generation
- Forward-with-activations introspection
- Training loop convergence on synthetic preferences
- Training with callbacks
- Edge cases (single-state trajectories, empty data)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reward_model import (
    RMTrainMetrics,
    RewardModel,
    bradley_terry_loss,
    bradley_terry_prob,
    train_reward_model,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture
def rm():
    torch.manual_seed(42)
    return RewardModel(state_dim=2, hidden_dim=32)


@pytest.fixture
def simple_prefs():
    """
    Synthetic preference data where trajectory A (near goal) should
    be preferred over trajectory B (near start).
    """
    # Trajectory A: states near (1.0, 1.0) — close to goal
    states_a = np.array([[0.8, 0.8], [0.9, 0.9], [1.0, 1.0]], dtype=np.float32)
    # Trajectory B: states near (0.0, 0.0) — close to start
    states_b = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], dtype=np.float32)
    # Label: 0.0 means A is preferred
    return [(states_a, states_b, 0.0)] * 20


# -----------------------------------------------------------------------
# Architecture
# -----------------------------------------------------------------------
class TestArchitecture:
    def test_output_shape_single_state(self, rm):
        x = torch.randn(2)
        out = rm(x)
        assert out.shape == ()  # scalar

    def test_output_shape_batch(self, rm):
        x = torch.randn(10, 2)
        out = rm(x)
        assert out.shape == (10,)

    def test_trajectory_reward_is_sum(self, rm):
        states = torch.randn(5, 2)
        per_state = rm(states)
        traj_reward = rm.trajectory_reward(states)
        assert torch.allclose(traj_reward, per_state.sum())

    def test_parameter_count(self, rm):
        """2→32→32→1 should have known param count."""
        total = sum(p.numel() for p in rm.parameters())
        # Layer 1: 2*32 + 32 = 96
        # Layer 2: 32*32 + 32 = 1056
        # Layer 3: 32*1 + 1 = 33
        expected = 96 + 1056 + 33
        assert total == expected


# -----------------------------------------------------------------------
# Bradley-Terry loss
# -----------------------------------------------------------------------
class TestBradleyTerryLoss:
    def test_loss_is_scalar(self, rm):
        sa = torch.randn(3, 2)
        sb = torch.randn(4, 2)
        loss = bradley_terry_loss(rm, sa, sb, label=0.0)
        assert loss.shape == ()

    def test_loss_is_positive(self, rm):
        sa = torch.randn(3, 2)
        sb = torch.randn(3, 2)
        loss = bradley_terry_loss(rm, sa, sb, label=1.0)
        assert loss.item() > 0

    def test_gradient_flows(self, rm):
        sa = torch.randn(3, 2)
        sb = torch.randn(3, 2)
        loss = bradley_terry_loss(rm, sa, sb, label=0.0)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in rm.parameters())
        assert has_grad

    def test_perfect_preference_low_loss(self, rm):
        """If RM already assigns higher reward to A and label says A preferred,
        loss should be relatively low (compared to opposite case)."""
        # Force RM to give high reward to specific input
        sa = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
        sb = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

        with torch.no_grad():
            ra = rm.trajectory_reward(sa)
            rb = rm.trajectory_reward(sb)

        if ra > rb:
            # A gets more reward, label=0 (A preferred) should have lower loss
            loss_correct = bradley_terry_loss(rm, sa, sb, 0.0).item()
            loss_wrong = bradley_terry_loss(rm, sa, sb, 1.0).item()
        else:
            loss_correct = bradley_terry_loss(rm, sa, sb, 1.0).item()
            loss_wrong = bradley_terry_loss(rm, sa, sb, 0.0).item()

        assert loss_correct < loss_wrong

    def test_tie_label(self, rm):
        """label=0.5 should produce a valid loss."""
        sa = torch.randn(3, 2)
        sb = torch.randn(3, 2)
        loss = bradley_terry_loss(rm, sa, sb, label=0.5)
        assert not torch.isnan(loss)
        assert loss.item() > 0


# -----------------------------------------------------------------------
# Bradley-Terry probability
# -----------------------------------------------------------------------
class TestBradleyTerryProb:
    def test_prob_range(self, rm):
        sa = torch.randn(3, 2)
        sb = torch.randn(3, 2)
        p = bradley_terry_prob(rm, sa, sb)
        assert 0.0 <= p <= 1.0

    def test_prob_symmetry(self, rm):
        """P(A≻B) + P(B≻A) should ≈ 1."""
        sa = torch.randn(3, 2)
        sb = torch.randn(3, 2)
        p_ab = bradley_terry_prob(rm, sa, sb)
        p_ba = bradley_terry_prob(rm, sb, sa)
        assert abs(p_ab + p_ba - 1.0) < 1e-5

    def test_identical_trajectories(self, rm):
        """Same trajectory should give P ≈ 0.5."""
        sa = torch.randn(5, 2)
        p = bradley_terry_prob(rm, sa, sa)
        assert abs(p - 0.5) < 1e-5


# -----------------------------------------------------------------------
# Reward heatmap
# -----------------------------------------------------------------------
class TestRewardHeatmap:
    def test_heatmap_shape(self, rm):
        hm = rm.reward_heatmap(grid_size=12)
        assert hm.shape == (12, 12)

    def test_heatmap_small_grid(self, rm):
        hm = rm.reward_heatmap(grid_size=3)
        assert hm.shape == (3, 3)

    def test_heatmap_values_finite(self, rm):
        hm = rm.reward_heatmap(grid_size=12)
        assert np.all(np.isfinite(hm))


# -----------------------------------------------------------------------
# Forward with activations
# -----------------------------------------------------------------------
class TestForwardWithActivations:
    def test_returns_activations(self, rm):
        x = torch.randn(2)
        out, acts = rm.forward_with_activations(x)
        assert len(acts) == 2  # two ReLU layers
        assert out.shape == ()

    def test_activations_non_negative(self, rm):
        """Post-ReLU activations should be >= 0."""
        x = torch.randn(5, 2)
        _, acts = rm.forward_with_activations(x)
        for a in acts:
            assert (a >= 0).all()

    def test_batch_activations(self, rm):
        x = torch.randn(8, 2)
        out, acts = rm.forward_with_activations(x)
        assert out.shape == (8,)
        assert acts[0].shape == (8, 32)  # hidden_dim = 32


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
class TestTraining:
    def test_training_reduces_loss(self, rm, simple_prefs):
        """Loss should decrease over training."""
        metrics = train_reward_model(rm, simple_prefs, epochs=50, lr=1e-3)
        assert metrics.loss_history[-1] < metrics.loss_history[0]

    def test_training_improves_accuracy(self, rm, simple_prefs):
        metrics = train_reward_model(rm, simple_prefs, epochs=50, lr=1e-3)
        # Accuracy should be well above chance (50%) for this easy data
        assert metrics.accuracy >= 0.8

    def test_training_metrics_structure(self, rm, simple_prefs):
        metrics = train_reward_model(rm, simple_prefs, epochs=10, lr=1e-3)
        assert metrics.epoch == 10
        assert len(metrics.loss_history) == 10
        assert len(metrics.accuracy_history) == 10

    def test_callback_invoked(self, rm, simple_prefs):
        call_log = []

        def cb(m):
            call_log.append(m.epoch)

        train_reward_model(rm, simple_prefs, epochs=5, lr=1e-3, callback=cb)
        assert call_log == [1, 2, 3, 4, 5]

    def test_empty_data(self, rm):
        metrics = train_reward_model(rm, [], epochs=10)
        assert metrics.epoch == 0
        assert metrics.loss == 0.0

    def test_single_preference(self, rm):
        sa = np.array([[0.5, 0.5]], dtype=np.float32)
        sb = np.array([[0.0, 0.0]], dtype=np.float32)
        data = [(sa, sb, 0.0)]
        metrics = train_reward_model(rm, data, epochs=20, lr=1e-3)
        assert metrics.epoch == 20

    def test_learned_preference_direction(self, rm, simple_prefs):
        """After training, RM should assign higher reward to preferred trajectories."""
        train_reward_model(rm, simple_prefs, epochs=80, lr=1e-3)

        sa_t = torch.tensor(simple_prefs[0][0], dtype=torch.float32)
        sb_t = torch.tensor(simple_prefs[0][1], dtype=torch.float32)

        with torch.no_grad():
            ra = rm.trajectory_reward(sa_t).item()
            rb = rm.trajectory_reward(sb_t).item()

        # A was preferred (label=0), so R(A) should be > R(B)
        assert ra > rb, f"Expected R(A)={ra:.3f} > R(B)={rb:.3f}"

    def test_training_with_mixed_labels(self, rm):
        """Train with both A-preferred and B-preferred examples."""
        data = []
        for _ in range(10):
            sa = np.random.randn(3, 2).astype(np.float32) + np.array([0.8, 0.8])
            sb = np.random.randn(3, 2).astype(np.float32) + np.array([0.2, 0.2])
            data.append((sa, sb, 0.0))  # A preferred
        for _ in range(10):
            sa = np.random.randn(3, 2).astype(np.float32) + np.array([0.2, 0.2])
            sb = np.random.randn(3, 2).astype(np.float32) + np.array([0.8, 0.8])
            data.append((sa, sb, 1.0))  # B preferred

        metrics = train_reward_model(rm, data, epochs=50, lr=1e-3)
        # Should learn that near-(0.8, 0.8) is preferred
        assert metrics.accuracy >= 0.7


# -----------------------------------------------------------------------
# RMTrainMetrics dataclass
# -----------------------------------------------------------------------
class TestRMTrainMetrics:
    def test_defaults(self):
        m = RMTrainMetrics()
        assert m.epoch == 0
        assert m.loss == 0.0
        assert m.accuracy == 0.0
        assert m.loss_history == []
        assert m.accuracy_history == []
