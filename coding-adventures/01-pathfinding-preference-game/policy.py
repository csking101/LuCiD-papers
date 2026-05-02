"""
PPO Policy Agent
=================

MLP policy (state → action probabilities) and value head (state → scalar)
with a full PPO update step including:

- Clipped surrogate objective
- GAE (Generalised Advantage Estimation)
- Entropy bonus
- **KL penalty** against a frozen pre-trained policy (the RLHF ingredient)

LLM parallel
------------
The policy network is the language model.  The value head is the same as
the value head added during RLHF.  The KL penalty keeps the fine-tuned
model close to the SFT checkpoint, preventing reward hacking.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from env import NUM_ACTIONS, Trajectory


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    """
    Shared-trunk MLP with separate policy and value heads.

    Architecture
    ------------
    state (2) → [hidden, ReLU] → [hidden, ReLU] → policy logits (4)
                                                  → value (1)
    """

    def __init__(self, state_dim: int = 2, action_dim: int = NUM_ACTIONS, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits : Tensor (*batch, action_dim)
        values : Tensor (*batch,)
        """
        h = self.trunk(states)
        logits = self.policy_head(h)
        values = self.value_head(h).squeeze(-1)
        return logits, values

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Sample an action for a single observation.

        Returns (action, log_prob, value).
        """
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32)
            logits, value = self.forward(t)
            dist = Categorical(logits=logits)
            if deterministic:
                action = logits.argmax().item()
                log_prob = dist.log_prob(torch.tensor(action)).item()
            else:
                action_t = dist.sample()
                action = action_t.item()
                log_prob = dist.log_prob(action_t).item()
        return action, log_prob, value.item()

    def get_probs(self, obs: np.ndarray) -> np.ndarray:
        """Return action probabilities for a single observation."""
        with torch.no_grad():
            t = torch.tensor(obs, dtype=torch.float32)
            logits, _ = self.forward(t)
            probs = torch.softmax(logits, dim=-1).numpy()
        return probs

    def get_log_probs_batch(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch computation of log_probs, values, and entropy.

        Parameters
        ----------
        states : (N, state_dim)
        actions : (N,)  — integer actions

        Returns
        -------
        log_probs : (N,)
        values    : (N,)
        entropy   : scalar
        """
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        return log_probs, values, entropy

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def forward_with_activations(
        self, obs: np.ndarray
    ) -> Dict:
        """
        Detailed forward pass for visualization.

        Returns dict with keys:
          input, hidden_1, hidden_2, logits, probs, value, chosen_action
        """
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32)
            # Manual forward through trunk
            h1 = self.trunk[0](x)    # Linear
            h1_act = self.trunk[1](h1)  # ReLU
            h2 = self.trunk[2](h1_act)  # Linear
            h2_act = self.trunk[3](h2)  # ReLU

            logits = self.policy_head(h2_act)
            value = self.value_head(h2_act).squeeze(-1)
            probs = torch.softmax(logits, dim=-1)

        return {
            "input": obs.tolist() if isinstance(obs, np.ndarray) else obs,
            "hidden_1": h1_act.numpy(),
            "hidden_2": h2_act.numpy(),
            "logits": logits.numpy(),
            "probs": probs.numpy(),
            "value": value.item(),
        }

    def policy_heatmap(
        self,
        grid_size: int = 8,
        env: "object | None" = None,
    ) -> np.ndarray:
        """
        Greedy action at every cell, with optional wall-masking.

        If *env* is provided, invalid actions (walls / out-of-bounds) are
        masked with ``-inf`` before ``argmax`` so arrows never point into
        walls.

        Returns ndarray of shape (grid_size, grid_size) with action indices.
        """
        from env import ACTION_DELTAS  # local import to avoid circular at module level

        actions = np.zeros((grid_size, grid_size), dtype=np.int32)
        for r in range(grid_size):
            for c in range(grid_size):
                obs = np.array(
                    [r / max(grid_size - 1, 1), c / max(grid_size - 1, 1)],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    t = torch.tensor(obs)
                    logits, _ = self.forward(t)

                    # Mask invalid moves when env is available
                    if env is not None:
                        for a, (dr, dc) in enumerate(ACTION_DELTAS):
                            nr, nc = r + dr, c + dc
                            if (
                                nr < 0
                                or nr >= grid_size
                                or nc < 0
                                or nc >= grid_size
                                or env.grid[nr, nc] != 0  # WALL
                            ):
                                logits[a] = float("-inf")

                    actions[r, c] = logits.argmax().item()
        return actions

    def value_heatmap(self, grid_size: int = 8) -> np.ndarray:
        """
        Value estimate at every cell.

        Returns ndarray of shape (grid_size, grid_size).
        """
        values = np.zeros((grid_size, grid_size), dtype=np.float32)
        for r in range(grid_size):
            for c in range(grid_size):
                obs = np.array(
                    [r / max(grid_size - 1, 1), c / max(grid_size - 1, 1)],
                    dtype=np.float32,
                )
                with torch.no_grad():
                    t = torch.tensor(obs)
                    _, v = self.forward(t)
                    values[r, c] = v.item()
        return values


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------
def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[List[float], List[float]]:
    """
    Generalised Advantage Estimation.

    Returns
    -------
    advantages : list of float (same length as rewards)
    returns    : list of float (advantage + value = return)
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = [adv + val for adv, val in zip(advantages, values[:T])]
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO training metrics
# ---------------------------------------------------------------------------
@dataclass
class PPOMetrics:
    """Metrics from a single PPO update."""
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    kl_divergence: float = 0.0
    clip_fraction: float = 0.0
    mean_advantage: float = 0.0


@dataclass
class EpisodeMetrics:
    """Metrics from a completed episode during training."""
    episode: int = 0
    total_reward: float = 0.0
    shaped_reward: float = 0.0
    rm_score: float = 0.0
    kl_penalty: float = 0.0
    net_reward: float = 0.0
    steps: int = 0
    reached_goal: bool = False
    ppo: Optional[PPOMetrics] = None
    latest_trajectory: Optional["Trajectory"] = None  # best trajectory from latest batch

    # Running averages for sparklines
    reward_history: List[float] = field(default_factory=list)
    rm_score_history: List[float] = field(default_factory=list)
    kl_history: List[float] = field(default_factory=list)
    goal_rate_history: List[float] = field(default_factory=list)
    steps_history: List[float] = field(default_factory=list)
    policy_loss_history: List[float] = field(default_factory=list)
    value_loss_history: List[float] = field(default_factory=list)
    entropy_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------
def ppo_update(
    policy: PolicyNetwork,
    trajectories: List[Trajectory],
    optimizer: optim.Optimizer,
    clip_eps: float = 0.2,
    entropy_coeff: float = 0.01,
    value_coeff: float = 0.5,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    ppo_epochs: int = 4,
    mini_batch_size: int = 64,
    kl_penalty_coeff: float = 0.0,
    ref_policy: Optional[PolicyNetwork] = None,
) -> PPOMetrics:
    """
    Perform one PPO update on collected trajectories.

    Parameters
    ----------
    policy : PolicyNetwork
        The policy to update.
    trajectories : list of Trajectory
        Collected rollout data.
    optimizer : Optimizer
    clip_eps : float
        PPO clipping epsilon.
    entropy_coeff : float
        Weight for the entropy bonus.
    value_coeff : float
        Weight for the value loss.
    gamma, gae_lambda : float
        GAE parameters.
    ppo_epochs : int
        Number of passes over the data.
    mini_batch_size : int
    kl_penalty_coeff : float
        Beta for KL penalty against ref_policy (0 = no penalty).
    ref_policy : PolicyNetwork or None
        Frozen pre-trained policy for KL computation.

    Returns
    -------
    metrics : PPOMetrics
    """
    # Flatten all trajectories into arrays
    all_states = []
    all_actions = []
    all_old_log_probs = []
    all_advantages = []
    all_returns = []

    for traj in trajectories:
        advantages, returns = compute_gae(
            traj.rewards, traj.values, gamma, gae_lambda
        )
        all_states.extend(traj.states)
        all_actions.extend(traj.actions)
        all_old_log_probs.extend(traj.log_probs)
        all_advantages.extend(advantages)
        all_returns.extend(returns)

    if len(all_states) == 0:
        return PPOMetrics()

    states_t = torch.tensor(np.array(all_states), dtype=torch.float32)
    actions_t = torch.tensor(all_actions, dtype=torch.long)
    old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32)
    advantages_t = torch.tensor(all_advantages, dtype=torch.float32)
    returns_t = torch.tensor(all_returns, dtype=torch.float32)

    # Normalise advantages
    if len(advantages_t) > 1 and advantages_t.std() > 1e-8:
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = len(all_states)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip_frac = 0.0
    num_updates = 0

    for _ in range(ppo_epochs):
        indices = np.random.permutation(N)
        for start in range(0, N, mini_batch_size):
            mb_idx = indices[start : start + mini_batch_size]
            mb_states = states_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_lp = old_log_probs_t[mb_idx]
            mb_adv = advantages_t[mb_idx]
            mb_ret = returns_t[mb_idx]

            new_lp, new_values, entropy = policy.get_log_probs_batch(
                mb_states, mb_actions
            )

            # PPO clipped objective
            ratio = torch.exp(new_lp - mb_old_lp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(new_values, mb_ret)

            # KL penalty (if reference policy provided)
            kl_loss = torch.tensor(0.0)
            if kl_penalty_coeff > 0 and ref_policy is not None:
                with torch.no_grad():
                    ref_logits, _ = ref_policy.forward(mb_states)
                    ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
                cur_logits, _ = policy.forward(mb_states)
                cur_log_probs = torch.log_softmax(cur_logits, dim=-1)
                # KL(current || reference) per state, averaged
                kl = (torch.exp(cur_log_probs) * (cur_log_probs - ref_log_probs)).sum(-1).mean()
                kl_loss = kl_penalty_coeff * kl

            total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy + kl_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            # Track metrics
            clip_frac = ((ratio - 1).abs() > clip_eps).float().mean().item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += kl_loss.item() / max(kl_penalty_coeff, 1e-8) if kl_penalty_coeff > 0 else 0.0
            total_clip_frac += clip_frac
            num_updates += 1

    n = max(num_updates, 1)
    return PPOMetrics(
        policy_loss=total_policy_loss / n,
        value_loss=total_value_loss / n,
        entropy=total_entropy / n,
        kl_divergence=total_kl / n,
        clip_fraction=total_clip_frac / n,
        mean_advantage=advantages_t.mean().item(),
    )


# ---------------------------------------------------------------------------
# KL divergence computation (standalone, for monitoring)
# ---------------------------------------------------------------------------
def compute_policy_kl(
    policy: PolicyNetwork,
    ref_policy: PolicyNetwork,
    grid_size: int = 8,
) -> float:
    """
    Compute average KL(policy || ref_policy) across all grid cells.
    Used for monitoring the "alignment tax".
    """
    total_kl = 0.0
    n = 0
    for r in range(grid_size):
        for c in range(grid_size):
            obs = np.array(
                [r / max(grid_size - 1, 1), c / max(grid_size - 1, 1)],
                dtype=np.float32,
            )
            with torch.no_grad():
                t = torch.tensor(obs)
                cur_logits, _ = policy.forward(t)
                ref_logits, _ = ref_policy.forward(t)
                cur_lp = torch.log_softmax(cur_logits, dim=-1)
                ref_lp = torch.log_softmax(ref_logits, dim=-1)
                kl = (torch.exp(cur_lp) * (cur_lp - ref_lp)).sum().item()
            total_kl += kl
            n += 1
    return total_kl / max(n, 1)
