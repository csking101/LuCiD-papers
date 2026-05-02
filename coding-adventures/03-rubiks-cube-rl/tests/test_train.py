"""Tests for train.py — Curriculum PPO training loop.

Covers:
    - TrainConfig defaults and max_steps computation
    - GAE computation (known cases)
    - Rollout collection (shapes, validity)
    - PPO update (runs without error, gradient flow)
    - Evaluation (solve rate range, result structure)
    - Curriculum trainer (depth progression, stats)
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from cube import PocketCube, NUM_ACTIONS, INVERSE_ACTION
from policy import CubePolicy
from train import (
    TrainConfig, DepthStats, TrainingStats, SolveResult, TestResults,
    Rollout, compute_gae, collect_rollouts, ppo_update,
    evaluate_policy, CurriculumTrainer,
)


DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainConfig:
    def test_default_config(self):
        cfg = TrainConfig()
        assert cfg.max_depth == 7
        assert cfg.advance_threshold == 0.80
        assert cfg.clip_epsilon == 0.2

    def test_max_steps(self):
        cfg = TrainConfig()
        assert cfg.max_steps(1) == 1 * 3 + 2  # 5
        assert cfg.max_steps(3) == 3 * 3 + 2  # 11

    def test_custom_config(self):
        cfg = TrainConfig(max_depth=3, lr=1e-4)
        assert cfg.max_depth == 3
        assert cfg.lr == 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# GAE
# ═══════════════════════════════════════════════════════════════════════════════

class TestGAE:
    def test_single_step_terminal(self):
        """Single step with terminal reward, gamma=0.99, lambda=0.95."""
        rewards = [1.0]
        values = [0.5]
        dones = [True]
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        # delta = 1.0 + 0.99*0 - 0.5 = 0.5  (done → next_value=0)
        # gae = 0.5  (single step)
        # return = 0.5 + 0.5 = 1.0
        assert adv[0] == pytest.approx(0.5)
        assert ret[0] == pytest.approx(1.0)

    def test_two_steps_not_terminal(self):
        """Two non-terminal steps then terminal."""
        rewards = [0.0, 1.0]
        values = [0.3, 0.7]
        dones = [False, True]
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        # Step 1 (t=1, terminal): delta = 1.0 + 0 - 0.7 = 0.3, gae = 0.3
        # Step 0 (t=0, not terminal): delta = 0.0 + 0.99*0.7 - 0.3 = 0.393
        #   gae = 0.393 + 0.99*0.95*0.3 = 0.393 + 0.28215 = 0.67515
        assert len(adv) == 2
        assert adv[1] == pytest.approx(0.3)
        assert adv[0] == pytest.approx(0.67515, abs=1e-4)

    def test_gae_zero_rewards(self):
        rewards = [0.0, 0.0, 0.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        for a in adv:
            assert a == pytest.approx(0.0)

    def test_gae_length(self):
        n = 10
        rewards = [0.1] * n
        values = [0.5] * n
        dones = [False] * (n - 1) + [True]
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        assert len(adv) == n
        assert len(ret) == n

    def test_gae_multi_episode(self):
        """Two back-to-back episodes: dones reset the gae."""
        rewards = [0.0, 1.0, 0.0, 0.5]
        values = [0.1, 0.2, 0.3, 0.4]
        dones = [False, True, False, True]
        adv, ret = compute_gae(rewards, values, dones, 0.99, 0.95)
        # At t=3 (done): delta = 0.5 + 0 - 0.4 = 0.1, gae = 0.1
        assert adv[3] == pytest.approx(0.1)
        # At t=1 (done): delta = 1.0 + 0 - 0.2 = 0.8, gae = 0.8
        assert adv[1] == pytest.approx(0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout Collection
# ═══════════════════════════════════════════════════════════════════════════════

class TestRolloutCollection:
    def _small_config(self):
        return TrainConfig(episodes_per_rollout=8, max_depth=2)

    def test_collect_rollouts_runs(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert isinstance(rollout, Rollout)

    def test_rollout_episode_counts(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert len(rollout.episode_returns) == 8
        assert len(rollout.episode_lengths) == 8
        assert len(rollout.episode_solved) == 8

    def test_rollout_states_shape(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert rollout.states.dim() == 2
        assert rollout.states.shape[1] == 144

    def test_rollout_actions_valid(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert (rollout.actions >= 0).all()
        assert (rollout.actions < NUM_ACTIONS).all()

    def test_rollout_advantages_shape(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert rollout.advantages.shape == rollout.actions.shape

    def test_rollout_returns_shape(self):
        cfg = self._small_config()
        policy = CubePolicy()
        rollout = collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))
        assert rollout.returns.shape == rollout.actions.shape


# ═══════════════════════════════════════════════════════════════════════════════
# PPO Update
# ═══════════════════════════════════════════════════════════════════════════════

class TestPPOUpdate:
    def _small_rollout(self):
        cfg = TrainConfig(episodes_per_rollout=8)
        policy = CubePolicy()
        return policy, cfg, collect_rollouts(policy, 1, cfg, DEVICE, random.Random(42))

    def test_ppo_update_runs(self):
        policy, cfg, rollout = self._small_rollout()
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        pl, vl, ent = ppo_update(policy, optimizer, rollout, cfg, DEVICE)
        assert isinstance(pl, float)
        assert isinstance(vl, float)
        assert isinstance(ent, float)

    def test_ppo_update_entropy_positive(self):
        policy, cfg, rollout = self._small_rollout()
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        _, _, ent = ppo_update(policy, optimizer, rollout, cfg, DEVICE)
        assert ent > 0.0

    def test_ppo_update_modifies_weights(self):
        policy, cfg, rollout = self._small_rollout()
        old_w = policy.policy_head.weight.clone()
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        ppo_update(policy, optimizer, rollout, cfg, DEVICE)
        assert not torch.equal(policy.policy_head.weight, old_w)


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluation:
    def test_evaluate_returns_test_results(self):
        cfg = TrainConfig()
        policy = CubePolicy()
        results = evaluate_policy(policy, 1, 10, cfg, DEVICE, random.Random(42))
        assert isinstance(results, TestResults)

    def test_solve_rate_range(self):
        cfg = TrainConfig()
        policy = CubePolicy()
        results = evaluate_policy(policy, 1, 20, cfg, DEVICE, random.Random(42))
        assert 0.0 <= results.solve_rate <= 1.0

    def test_result_count(self):
        cfg = TrainConfig()
        policy = CubePolicy()
        results = evaluate_policy(policy, 1, 15, cfg, DEVICE, random.Random(42))
        assert results.n_episodes == 15
        assert len(results.results) == 15

    def test_record_moves(self):
        cfg = TrainConfig()
        policy = CubePolicy()
        results = evaluate_policy(
            policy, 1, 5, cfg, DEVICE, random.Random(42), record_moves=True,
        )
        for r in results.results:
            assert len(r.moves) > 0
            assert len(r.states) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Curriculum Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurriculumTrainer:
    def test_trainer_creates_policy(self):
        trainer = CurriculumTrainer(TrainConfig(max_depth=1))
        assert isinstance(trainer.policy, CubePolicy)

    def test_trainer_short_run(self):
        """A very short training run should complete without errors."""
        cfg = TrainConfig(
            max_depth=1,
            episodes_per_rollout=16,
            eval_episodes=10,
            max_episodes_per_depth=32,
            ppo_epochs=1,
        )
        trainer = CurriculumTrainer(cfg, device=DEVICE, seed=42)
        stats = trainer.train()
        assert isinstance(stats, TrainingStats)
        assert len(stats.depth_stats) >= 1
        assert stats.total_episodes > 0

    def test_trainer_callback(self):
        """Callback should be invoked at least once."""
        cfg = TrainConfig(
            max_depth=1,
            episodes_per_rollout=16,
            eval_episodes=10,
            max_episodes_per_depth=32,
            ppo_epochs=1,
        )
        calls = []
        def cb(depth, ds, ts):
            calls.append((depth, ds.solve_rate))
        trainer = CurriculumTrainer(cfg, device=DEVICE, seed=42)
        trainer.train(callback=cb)
        assert len(calls) >= 1

    def test_depth_stats_populated(self):
        cfg = TrainConfig(
            max_depth=1,
            episodes_per_rollout=16,
            eval_episodes=10,
            max_episodes_per_depth=32,
            ppo_epochs=1,
        )
        trainer = CurriculumTrainer(cfg, device=DEVICE, seed=42)
        stats = trainer.train()
        ds = stats.depth_stats[0]
        assert ds.depth == 1
        assert ds.episodes_trained > 0
        assert 0.0 <= ds.solve_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataClasses:
    def test_depth_stats_defaults(self):
        ds = DepthStats(depth=3)
        assert ds.depth == 3
        assert ds.episodes_trained == 0
        assert ds.advanced is False

    def test_training_stats_defaults(self):
        ts = TrainingStats()
        assert ts.current_depth == 1
        assert ts.total_episodes == 0

    def test_solve_result(self):
        sr = SolveResult(depth=2, solved=True, steps=3, moves=[0, 1, 2], states=[])
        assert sr.solved
        assert sr.steps == 3

    def test_test_results(self):
        tr = TestResults(depth=1, n_episodes=10, solve_rate=0.5, avg_steps=3.0, results=[])
        assert tr.solve_rate == 0.5
