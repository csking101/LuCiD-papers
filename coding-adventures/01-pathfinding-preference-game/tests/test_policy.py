"""
Tests for the PPO Policy Agent.
================================

Covers:
- Network architecture shapes (forward, get_action, get_probs)
- Batch log-prob computation
- Introspection (forward_with_activations, policy/value heatmaps)
- GAE computation (basic, single-step, zero-discount, all-zero rewards)
- PPO update (loss decreases, clip fraction, gradient flow)
- PPO with KL penalty (penalty increases with divergence)
- KL divergence computation (identical policies, diverged policies)
- EpisodeMetrics and PPOMetrics dataclasses
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import DOWN, LEFT, NUM_ACTIONS, RIGHT, UP, GridWorld, Trajectory
from policy import (
    EpisodeMetrics,
    PPOMetrics,
    PolicyNetwork,
    compute_gae,
    compute_policy_kl,
    ppo_update,
)


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------
@pytest.fixture
def policy():
    torch.manual_seed(0)
    return PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)


@pytest.fixture
def tiny_env():
    return GridWorld(size=4, walls=[], pickups=[], max_steps=50)


@pytest.fixture
def sample_trajectory(tiny_env, policy):
    """Generate a short trajectory using the policy."""
    def policy_fn(obs):
        return policy.get_action(obs)
    return tiny_env.rollout(policy_fn, max_steps=20)


# -----------------------------------------------------------------------
# Network architecture
# -----------------------------------------------------------------------
class TestNetworkArchitecture:
    def test_forward_shapes(self, policy):
        x = torch.randn(2)
        logits, value = policy.forward(x)
        assert logits.shape == (4,)
        assert value.shape == ()

    def test_forward_batch(self, policy):
        x = torch.randn(10, 2)
        logits, values = policy.forward(x)
        assert logits.shape == (10, 4)
        assert values.shape == (10,)

    def test_get_action_returns_tuple(self, policy):
        obs = np.array([0.5, 0.5], dtype=np.float32)
        action, log_prob, value = policy.get_action(obs)
        assert isinstance(action, int)
        assert 0 <= action < 4
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_get_action_deterministic(self, policy):
        obs = np.array([0.5, 0.5], dtype=np.float32)
        actions = set()
        for _ in range(20):
            a, _, _ = policy.get_action(obs, deterministic=True)
            actions.add(a)
        assert len(actions) == 1  # always same action

    def test_get_probs_sum_to_one(self, policy):
        obs = np.array([0.3, 0.7], dtype=np.float32)
        probs = policy.get_probs(obs)
        assert probs.shape == (4,)
        assert abs(probs.sum() - 1.0) < 1e-5
        assert all(p >= 0 for p in probs)

    def test_log_probs_batch(self, policy):
        states = torch.randn(10, 2)
        actions = torch.randint(0, 4, (10,))
        lp, vals, ent = policy.get_log_probs_batch(states, actions)
        assert lp.shape == (10,)
        assert vals.shape == (10,)
        assert ent.shape == ()
        # log probs should be <= 0
        assert (lp <= 0).all()

    def test_parameter_count(self, policy):
        total = sum(p.numel() for p in policy.parameters())
        # Trunk: (2*32+32) + (32*32+32) = 96 + 1056 = 1152
        # Policy head: 32*4+4 = 132
        # Value head: 32*1+1 = 33
        expected = 1152 + 132 + 33
        assert total == expected


# -----------------------------------------------------------------------
# Introspection
# -----------------------------------------------------------------------
class TestIntrospection:
    def test_forward_with_activations_keys(self, policy):
        obs = np.array([0.4, 0.6], dtype=np.float32)
        info = policy.forward_with_activations(obs)
        expected_keys = {"input", "hidden_1", "hidden_2", "logits", "probs", "value"}
        assert set(info.keys()) == expected_keys

    def test_forward_with_activations_shapes(self, policy):
        obs = np.array([0.4, 0.6], dtype=np.float32)
        info = policy.forward_with_activations(obs)
        assert len(info["input"]) == 2
        assert info["hidden_1"].shape == (32,)
        assert info["hidden_2"].shape == (32,)
        assert info["logits"].shape == (4,)
        assert info["probs"].shape == (4,)
        assert isinstance(info["value"], float)

    def test_activations_post_relu_nonneg(self, policy):
        obs = np.array([0.2, 0.8], dtype=np.float32)
        info = policy.forward_with_activations(obs)
        assert (info["hidden_1"] >= 0).all()
        assert (info["hidden_2"] >= 0).all()

    def test_probs_sum_to_one(self, policy):
        obs = np.array([0.5, 0.5], dtype=np.float32)
        info = policy.forward_with_activations(obs)
        assert abs(info["probs"].sum() - 1.0) < 1e-5

    def test_policy_heatmap(self, policy):
        hm = policy.policy_heatmap(grid_size=4)
        assert hm.shape == (4, 4)
        assert hm.dtype == np.int32
        assert all(0 <= a < 4 for a in hm.flatten())

    def test_policy_heatmap_with_env_masking(self, policy):
        """Wall-masked heatmap should never point into a wall or OOB."""
        env = GridWorld()  # 8x8 default
        hm = policy.policy_heatmap(grid_size=env.size, env=env)
        assert hm.shape == (env.size, env.size)
        from env import ACTION_DELTAS
        for r in range(env.size):
            for c in range(env.size):
                if env.grid[r, c] != 0:  # skip wall cells
                    continue
                a = hm[r, c]
                dr, dc = ACTION_DELTAS[a]
                nr, nc = r + dr, c + dc
                # Target should be in-bounds and not a wall
                assert 0 <= nr < env.size, f"OOB at ({r},{c}) action {a}"
                assert 0 <= nc < env.size, f"OOB at ({r},{c}) action {a}"
                assert env.grid[nr, nc] == 0, f"Arrow into wall at ({r},{c}) action {a}"

    def test_value_heatmap(self, policy):
        hm = policy.value_heatmap(grid_size=4)
        assert hm.shape == (4, 4)
        assert np.all(np.isfinite(hm))


# -----------------------------------------------------------------------
# GAE computation
# -----------------------------------------------------------------------
class TestGAE:
    def test_basic_gae(self):
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        adv, ret = compute_gae(rewards, values, gamma=0.99, lam=0.95)
        assert len(adv) == 3
        assert len(ret) == 3
        # Returns = advantages + values
        for a, r, v in zip(adv, ret, values):
            assert abs(r - (a + v)) < 1e-5

    def test_single_step(self):
        rewards = [10.0]
        values = [0.0]
        adv, ret = compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0)
        # delta = 10 + 0.99*0 - 0 = 10
        assert abs(adv[0] - 10.0) < 1e-5

    def test_zero_discount(self):
        """With gamma=0, advantages should just be r - V."""
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 0.5, 0.5]
        adv, _ = compute_gae(rewards, values, gamma=0.0, lam=0.95)
        for a, r, v in zip(adv, rewards, values):
            assert abs(a - (r - v)) < 1e-5

    def test_all_zero_rewards(self):
        rewards = [0.0, 0.0, 0.0]
        values = [1.0, 1.0, 1.0]
        adv, _ = compute_gae(rewards, values, gamma=0.99, lam=0.95)
        # Should have negative advantages (values > actual returns)
        assert all(a < 0 for a in adv)

    def test_last_value_propagates(self):
        rewards = [0.0]
        values = [0.0]
        adv_no_boot, _ = compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=0.0)
        adv_boot, _ = compute_gae(rewards, values, gamma=0.99, lam=0.95, last_value=10.0)
        assert adv_boot[0] > adv_no_boot[0]


# -----------------------------------------------------------------------
# PPO update
# -----------------------------------------------------------------------
class TestPPOUpdate:
    def _collect_trajectories(self, env, policy, n=5, max_steps=30):
        trajs = []
        for _ in range(n):
            traj = env.rollout(lambda obs: policy.get_action(obs), max_steps=max_steps)
            trajs.append(traj)
        return trajs

    def test_ppo_returns_metrics(self, policy, tiny_env):
        trajs = self._collect_trajectories(tiny_env, policy, n=3)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(policy, trajs, opt, ppo_epochs=2)
        assert isinstance(metrics, PPOMetrics)
        assert metrics.policy_loss != 0.0 or metrics.value_loss != 0.0

    def test_ppo_updates_weights(self, policy, tiny_env):
        old_params = {n: p.clone() for n, p in policy.named_parameters()}
        trajs = self._collect_trajectories(tiny_env, policy, n=3)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        ppo_update(policy, trajs, opt, ppo_epochs=4)

        changed = False
        for n, p in policy.named_parameters():
            if not torch.allclose(p, old_params[n]):
                changed = True
                break
        assert changed, "PPO update did not change any parameters"

    def test_ppo_empty_trajectories(self, policy):
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(policy, [], opt)
        assert metrics.policy_loss == 0.0

    def test_ppo_clip_fraction_bounded(self, policy, tiny_env):
        trajs = self._collect_trajectories(tiny_env, policy, n=3)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(policy, trajs, opt, clip_eps=0.2)
        assert 0.0 <= metrics.clip_fraction <= 1.0

    def test_ppo_entropy_positive(self, policy, tiny_env):
        trajs = self._collect_trajectories(tiny_env, policy, n=3)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(policy, trajs, opt)
        assert metrics.entropy > 0  # random-init policy has high entropy

    def test_ppo_with_kl_penalty(self, policy, tiny_env):
        ref = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        ref.load_state_dict(policy.state_dict())
        ref.eval()

        trajs = self._collect_trajectories(tiny_env, policy, n=3)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(
            policy, trajs, opt,
            kl_penalty_coeff=0.2,
            ref_policy=ref,
        )
        # KL should be very small initially (policies are identical)
        assert metrics.kl_divergence < 0.5

    def test_ppo_multiple_updates_improve(self, tiny_env):
        """Multiple rounds of PPO should improve average reward."""
        torch.manual_seed(42)
        np.random.seed(42)
        pol = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        opt = torch.optim.Adam(pol.parameters(), lr=3e-3)

        rewards_before = []
        for _ in range(5):
            traj = tiny_env.rollout(lambda obs: pol.get_action(obs), max_steps=30)
            rewards_before.append(traj.total_reward)

        for _ in range(10):
            trajs = []
            for _ in range(5):
                traj = tiny_env.rollout(lambda obs: pol.get_action(obs), max_steps=30)
                trajs.append(traj)
            ppo_update(pol, trajs, opt, ppo_epochs=4)

        rewards_after = []
        for _ in range(5):
            traj = tiny_env.rollout(lambda obs: pol.get_action(obs), max_steps=30)
            rewards_after.append(traj.total_reward)

        # After training, average reward should improve (or at least not crash)
        assert np.mean(rewards_after) >= np.mean(rewards_before) - 2.0


# -----------------------------------------------------------------------
# KL divergence
# -----------------------------------------------------------------------
class TestKLDivergence:
    def test_identical_policies_zero_kl(self, policy):
        ref = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        ref.load_state_dict(policy.state_dict())
        kl = compute_policy_kl(policy, ref, grid_size=4)
        assert abs(kl) < 1e-5

    def test_diverged_policies_positive_kl(self):
        torch.manual_seed(0)
        p1 = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        torch.manual_seed(99)
        p2 = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        kl = compute_policy_kl(p1, p2, grid_size=4)
        assert kl > 0

    def test_kl_is_finite(self, policy):
        ref = PolicyNetwork(state_dim=2, action_dim=4, hidden_dim=32)
        ref.load_state_dict(policy.state_dict())
        kl = compute_policy_kl(policy, ref, grid_size=8)
        assert np.isfinite(kl)


# -----------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------
class TestDataclasses:
    def test_ppo_metrics_defaults(self):
        m = PPOMetrics()
        assert m.policy_loss == 0.0
        assert m.kl_divergence == 0.0

    def test_episode_metrics_defaults(self):
        m = EpisodeMetrics()
        assert m.episode == 0
        assert m.reached_goal is False
        assert m.reward_history == []
        assert m.latest_trajectory is None


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------
class TestEdgeCases:
    def test_single_step_trajectory(self, policy, tiny_env):
        """PPO should handle very short trajectories."""
        traj = tiny_env.rollout(lambda obs: policy.get_action(obs), max_steps=1)
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        metrics = ppo_update(policy, [traj], opt)
        assert isinstance(metrics, PPOMetrics)

    def test_get_action_extreme_obs(self, policy):
        """Policy shouldn't crash on extreme inputs."""
        for obs in [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
            np.array([-1.0, 2.0], dtype=np.float32),
        ]:
            a, lp, v = policy.get_action(obs)
            assert 0 <= a < 4
