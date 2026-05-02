"""Tests for policy.py — PPO actor-critic network for the Pocket Cube.

Covers:
    - Forward pass shapes (single and batch)
    - Output finiteness and validity
    - Action selection (valid range, log_prob, value)
    - Batch evaluation (shapes, log_probs, entropy)
    - Gradient flow through both heads
    - Orthogonal initialization
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from cube import PocketCube, STATE_DIM, NUM_ACTIONS
from policy import CubePolicy, select_action, evaluate_actions


DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Forward Pass
# ═══════════════════════════════════════════════════════════════════════════════

class TestForward:
    def test_single_input_shapes(self):
        policy = CubePolicy()
        x = torch.randn(STATE_DIM)
        logits, value = policy(x)
        assert logits.shape == (NUM_ACTIONS,)
        assert value.shape == (1,)

    def test_batch_input_shapes(self):
        policy = CubePolicy()
        x = torch.randn(8, STATE_DIM)
        logits, value = policy(x)
        assert logits.shape == (8, NUM_ACTIONS)
        assert value.shape == (8, 1)

    def test_logits_are_finite(self):
        policy = CubePolicy()
        x = PocketCube().to_tensor()
        logits, _ = policy(x)
        assert torch.isfinite(logits).all()

    def test_value_is_finite(self):
        policy = CubePolicy()
        x = PocketCube().to_tensor()
        _, value = policy(x)
        assert torch.isfinite(value).all()

    def test_different_inputs_different_outputs(self):
        policy = CubePolicy()
        c1 = PocketCube()
        c2 = PocketCube()
        c2.scramble(3)
        l1, v1 = policy(c1.to_tensor())
        l2, v2 = policy(c2.to_tensor())
        assert not torch.equal(l1, l2)

    def test_custom_dims(self):
        policy = CubePolicy(state_dim=10, hidden_dim=16, n_actions=3)
        x = torch.randn(10)
        logits, value = policy(x)
        assert logits.shape == (3,)
        assert value.shape == (1,)


# ═══════════════════════════════════════════════════════════════════════════════
# Action Selection
# ═══════════════════════════════════════════════════════════════════════════════

class TestSelectAction:
    def test_returns_valid_action(self):
        policy = CubePolicy()
        state = PocketCube().to_tensor()
        action, _, _ = select_action(policy, state, DEVICE)
        assert 0 <= action < NUM_ACTIONS

    def test_returns_log_prob(self):
        policy = CubePolicy()
        state = PocketCube().to_tensor()
        _, log_prob, _ = select_action(policy, state, DEVICE)
        assert isinstance(log_prob, float)
        assert log_prob <= 0.0  # log of probability

    def test_returns_value(self):
        policy = CubePolicy()
        state = PocketCube().to_tensor()
        _, _, value = select_action(policy, state, DEVICE)
        assert isinstance(value, float)

    def test_explores_multiple_actions(self):
        """Over many samples, we should see multiple different actions."""
        policy = CubePolicy()
        state = PocketCube().to_tensor()
        actions = set()
        for _ in range(200):
            a, _, _ = select_action(policy, state, DEVICE)
            actions.add(a)
        assert len(actions) >= 3  # at least 3 distinct actions


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluate Actions
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateActions:
    def _make_batch(self, n=16):
        states = torch.stack([PocketCube().to_tensor() for _ in range(n)])
        actions = torch.randint(0, NUM_ACTIONS, (n,))
        return states, actions

    def test_shapes(self):
        policy = CubePolicy()
        states, actions = self._make_batch(16)
        log_probs, entropy, values = evaluate_actions(policy, states, actions)
        assert log_probs.shape == (16,)
        assert entropy.shape == (16,)
        assert values.shape == (16,)

    def test_log_probs_negative(self):
        policy = CubePolicy()
        states, actions = self._make_batch()
        log_probs, _, _ = evaluate_actions(policy, states, actions)
        assert (log_probs <= 0.0).all()

    def test_entropy_non_negative(self):
        policy = CubePolicy()
        states, actions = self._make_batch()
        _, entropy, _ = evaluate_actions(policy, states, actions)
        assert (entropy >= 0.0).all()

    def test_values_finite(self):
        policy = CubePolicy()
        states, actions = self._make_batch()
        _, _, values = evaluate_actions(policy, states, actions)
        assert torch.isfinite(values).all()

    def test_log_probs_consistent_with_forward(self):
        """log_probs from evaluate should match manual computation."""
        policy = CubePolicy()
        states, actions = self._make_batch(4)
        log_probs, _, _ = evaluate_actions(policy, states, actions)
        # Manual computation
        logits, _ = policy(states)
        from torch.distributions import Categorical
        dist = Categorical(logits=logits)
        expected = dist.log_prob(actions)
        assert torch.allclose(log_probs, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# Gradient Flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradients:
    def test_policy_head_gradients(self):
        policy = CubePolicy()
        x = PocketCube().to_tensor().unsqueeze(0)
        logits, _ = policy(x)
        loss = logits.sum()
        loss.backward()
        assert policy.policy_head.weight.grad is not None
        assert policy.policy_head.weight.grad.abs().sum() > 0

    def test_value_head_gradients(self):
        policy = CubePolicy()
        x = PocketCube().to_tensor().unsqueeze(0)
        _, value = policy(x)
        loss = value.sum()
        loss.backward()
        assert policy.value_head.weight.grad is not None
        assert policy.value_head.weight.grad.abs().sum() > 0

    def test_shared_backbone_gradients(self):
        policy = CubePolicy()
        x = PocketCube().to_tensor().unsqueeze(0)
        logits, value = policy(x)
        loss = logits.sum() + value.sum()
        loss.backward()
        for layer in policy.shared:
            if hasattr(layer, "weight"):
                assert layer.weight.grad is not None
                assert layer.weight.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_policy_head_small_init(self):
        """Policy head should have small weights (gain=0.01)."""
        policy = CubePolicy()
        assert policy.policy_head.weight.abs().max().item() < 0.5

    def test_biases_zero(self):
        policy = CubePolicy()
        assert (policy.policy_head.bias == 0).all()
        assert (policy.value_head.bias == 0).all()
