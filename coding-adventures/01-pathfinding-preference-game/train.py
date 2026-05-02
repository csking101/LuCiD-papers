"""
Training Orchestrator
======================

Coordinates the four RLHF phases:

  Phase 1 — Pre-train the agent with shaped rewards
  Phase 2 — Collect human preferences (handled by app.py)
  Phase 3 — Train the reward model on preferences
  Phase 4 — PPO fine-tune with learned RM + KL penalty

Each function accepts a callback for live UI updates.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim

from env import GridWorld, Trajectory
from policy import (
    EpisodeMetrics,
    PPOMetrics,
    PolicyNetwork,
    compute_gae,
    compute_policy_kl,
    ppo_update,
)
from preferences import PreferenceDatabase
from reward_model import (
    RMTrainMetrics,
    RewardModel,
    bradley_terry_prob,
    train_reward_model,
)


# ---------------------------------------------------------------------------
# Phase 1: Pre-training
# ---------------------------------------------------------------------------
def pretrain(
    env: GridWorld,
    policy: PolicyNetwork,
    episodes: int = 500,
    lr: float = 3e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    batch_episodes: int = 10,
    entropy_coeff: float = 0.05,
    callback: Optional[Callable[[EpisodeMetrics], None]] = None,
) -> EpisodeMetrics:
    """
    Pre-train the agent using the environment's shaped reward.

    This is analogous to language model pre-training: the agent learns
    basic competence (reaching the goal) before being fine-tuned with
    human preferences.

    Parameters
    ----------
    env : GridWorld
    policy : PolicyNetwork
    episodes : int
        Total training episodes.
    lr : float
    gamma, gae_lambda : float
        GAE parameters.
    batch_episodes : int
        Number of episodes per PPO update.
    entropy_coeff : float
        Weight for the entropy bonus.  Higher values encourage exploration
        of all corridors before the policy commits to one route.
    callback : callable(EpisodeMetrics) or None
        Called after each batch for live UI updates.

    Returns
    -------
    metrics : EpisodeMetrics  (final state with history)
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    metrics = EpisodeMetrics()
    goal_count = 0

    for ep in range(1, episodes + 1):
        # Collect a batch of trajectories
        batch_trajs: List[Trajectory] = []
        batch_reward = 0.0
        batch_steps = 0
        batch_goals = 0

        for _ in range(batch_episodes):
            traj = env.rollout(lambda obs: policy.get_action(obs))
            batch_trajs.append(traj)
            batch_reward += traj.total_reward
            batch_steps += traj.length
            if traj.reached_goal:
                batch_goals += 1
                goal_count += 1

        # PPO update
        ppo_metrics = ppo_update(
            policy, batch_trajs, optimizer,
            gamma=gamma, gae_lambda=gae_lambda,
            entropy_coeff=entropy_coeff,
        )

        # Fresh deterministic rollout with the *updated* policy for live display
        latest_traj = env.rollout(
            lambda obs: policy.get_action(obs, deterministic=True)
        )

        # Update episode metrics
        metrics.episode = ep
        metrics.total_reward = batch_reward / batch_episodes
        metrics.shaped_reward = metrics.total_reward
        metrics.steps = int(batch_steps / batch_episodes)
        metrics.reached_goal = batch_goals > 0
        metrics.ppo = ppo_metrics
        metrics.latest_trajectory = latest_traj

        # Running history
        metrics.reward_history.append(metrics.total_reward)
        metrics.steps_history.append(float(metrics.steps))
        metrics.policy_loss_history.append(ppo_metrics.policy_loss)
        metrics.value_loss_history.append(ppo_metrics.value_loss)
        metrics.entropy_history.append(ppo_metrics.entropy)

        # Goal rate: rolling window of last 20 batches
        window = metrics.goal_rate_history[-19:] + [batch_goals / batch_episodes]
        metrics.goal_rate_history.append(batch_goals / batch_episodes)

        if callback is not None:
            callback(metrics)

    return metrics


# ---------------------------------------------------------------------------
# Phase 3: Train reward model
# ---------------------------------------------------------------------------
def train_rm(
    rm: RewardModel,
    pref_db: PreferenceDatabase,
    epochs: int = 100,
    lr: float = 1e-3,
    callback: Optional[Callable[[RMTrainMetrics], None]] = None,
) -> RMTrainMetrics:
    """
    Train the reward model on collected preferences.

    Wraps ``reward_model.train_reward_model`` with preference DB conversion.
    """
    training_data = pref_db.to_training_data()
    return train_reward_model(
        rm, training_data, epochs=epochs, lr=lr, callback=callback,
    )


# ---------------------------------------------------------------------------
# Phase 4: RLHF with PPO
# ---------------------------------------------------------------------------
def rlhf_train(
    env: GridWorld,
    policy: PolicyNetwork,
    rm: RewardModel,
    pretrained_policy: PolicyNetwork,
    episodes: int = 200,
    lr: float = 1e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    kl_coeff: float = 0.2,
    batch_episodes: int = 10,
    callback: Optional[Callable[[EpisodeMetrics], None]] = None,
) -> EpisodeMetrics:
    """
    RLHF phase: PPO with learned reward model and KL penalty.

    The reward for each trajectory is:
        R_total = R_RM(trajectory) - beta * KL(policy || pretrained)

    Parameters
    ----------
    env : GridWorld
    policy : PolicyNetwork
        The policy to fine-tune (starts from pre-trained weights).
    rm : RewardModel
        Trained reward model.
    pretrained_policy : PolicyNetwork
        Frozen pre-trained policy for KL penalty.
    episodes : int
    lr : float
    gamma, gae_lambda : float
    kl_coeff : float
        Beta for KL penalty.
    batch_episodes : int
    callback : callable or None

    Returns
    -------
    metrics : EpisodeMetrics
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    pretrained_policy.eval()
    metrics = EpisodeMetrics()

    for ep in range(1, episodes + 1):
        batch_trajs: List[Trajectory] = []
        batch_rm_score = 0.0
        batch_shaped = 0.0
        batch_kl = 0.0
        batch_steps = 0
        batch_goals = 0

        for _ in range(batch_episodes):
            traj = env.rollout(lambda obs: policy.get_action(obs))

            # Replace rewards with RM scores + shaped bonus
            rm_rewards = _compute_rm_rewards(rm, traj)

            # Original shaped reward (for monitoring)
            shaped_total = traj.total_reward

            # Replace trajectory rewards with RM rewards
            traj.rewards = rm_rewards

            batch_trajs.append(traj)
            batch_rm_score += sum(rm_rewards)
            batch_shaped += shaped_total
            batch_steps += traj.length
            if traj.reached_goal:
                batch_goals += 1

        # PPO update with KL penalty
        ppo_metrics = ppo_update(
            policy, batch_trajs, optimizer,
            gamma=gamma, gae_lambda=gae_lambda,
            kl_penalty_coeff=kl_coeff,
            ref_policy=pretrained_policy,
        )

        # Compute global KL for monitoring
        global_kl = compute_policy_kl(policy, pretrained_policy, grid_size=env.size)

        # Fresh deterministic rollout with the *updated* policy for live display
        latest_traj = env.rollout(
            lambda obs: policy.get_action(obs, deterministic=True)
        )

        # Update metrics
        metrics.episode = ep
        metrics.rm_score = batch_rm_score / batch_episodes
        metrics.shaped_reward = batch_shaped / batch_episodes
        metrics.kl_penalty = kl_coeff * global_kl
        metrics.net_reward = metrics.rm_score - metrics.kl_penalty
        metrics.steps = int(batch_steps / batch_episodes)
        metrics.reached_goal = batch_goals > 0
        metrics.ppo = ppo_metrics
        metrics.latest_trajectory = latest_traj

        metrics.rm_score_history.append(metrics.rm_score)
        metrics.reward_history.append(metrics.net_reward)
        metrics.kl_history.append(global_kl)
        metrics.steps_history.append(float(metrics.steps))
        metrics.goal_rate_history.append(batch_goals / batch_episodes)
        metrics.policy_loss_history.append(ppo_metrics.policy_loss)
        metrics.value_loss_history.append(ppo_metrics.value_loss)
        metrics.entropy_history.append(ppo_metrics.entropy)

        if callback is not None:
            callback(metrics)

    return metrics


def _compute_rm_rewards(rm: RewardModel, traj: Trajectory) -> List[float]:
    """Compute per-step rewards from the reward model for a trajectory."""
    if len(traj.states) == 0:
        return []
    states_t = torch.tensor(np.array(traj.states), dtype=torch.float32)
    with torch.no_grad():
        rewards = rm(states_t).numpy().tolist()
    return rewards


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def evaluate_policy(
    env: GridWorld,
    policy: PolicyNetwork,
    n_episodes: int = 50,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a policy over multiple episodes.

    Returns dict with: avg_reward, avg_steps, goal_rate, avg_turns
    """
    total_reward = 0.0
    total_steps = 0
    total_goals = 0
    total_turns = 0

    for _ in range(n_episodes):
        traj = env.rollout(
            lambda obs: policy.get_action(obs, deterministic=deterministic)
        )
        total_reward += traj.total_reward
        total_steps += traj.length
        total_turns += traj.num_turns
        if traj.reached_goal:
            total_goals += 1

    return {
        "Avg Reward": total_reward / n_episodes,
        "Avg Steps": total_steps / n_episodes,
        "Goal Rate": total_goals / n_episodes,
        "Avg Turns": total_turns / n_episodes,
    }


def generate_diverse_trajectories(
    env: GridWorld,
    policy: PolicyNetwork,
    n: int = 10,
    temperature: float = 1.5,
) -> List[Trajectory]:
    """
    Generate diverse trajectories by sampling with higher temperature.
    Used for preference pair generation.
    """
    trajectories = []

    for _ in range(n):
        def _temp_policy(obs):
            import torch as _torch
            with _torch.no_grad():
                t = _torch.tensor(obs, dtype=_torch.float32)
                logits, value = policy.forward(t)
                # Scale logits by temperature for diversity
                scaled = logits / temperature
                dist = _torch.distributions.Categorical(logits=scaled)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

        traj = env.rollout(_temp_policy)
        trajectories.append(traj)

    return trajectories


def select_preference_pairs(
    trajectories: List[Trajectory],
    n_pairs: int = 10,
) -> List[Tuple[Trajectory, Trajectory]]:
    """
    Select diverse pairs from a pool of trajectories for preference
    comparison.  Tries to pick pairs that are meaningfully different.
    """
    pairs = []
    n = len(trajectories)
    if n < 2:
        return pairs

    # Sort by total reward for variety
    sorted_trajs = sorted(trajectories, key=lambda t: t.total_reward)

    for i in range(min(n_pairs, n // 2)):
        # Pick from different parts of the distribution
        idx_a = i % n
        idx_b = (n - 1 - i) % n
        if idx_a == idx_b:
            idx_b = (idx_a + 1) % n
        pairs.append((sorted_trajs[idx_a], sorted_trajs[idx_b]))

    # Fill remaining with random pairs
    while len(pairs) < n_pairs and n >= 2:
        a, b = np.random.choice(n, 2, replace=False)
        pairs.append((trajectories[a], trajectories[b]))

    return pairs[:n_pairs]
