"""Curriculum PPO training for the 2x2 Pocket Cube.

Training strategy:
    1. Start at scramble depth 1 (one move from solved).
    2. Collect rollouts, run PPO update.
    3. Evaluate solve rate; if >= threshold, advance to next depth.
    4. Repeat until max_depth reached or time budget exhausted.

Key PPO components (from paper 1707.06347):
    - Clipped surrogate objective with epsilon
    - Generalised Advantage Estimation (GAE)
    - Separate value loss and entropy bonus
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

from cube import PocketCube, NUM_ACTIONS, INVERSE_ACTION
from policy import CubePolicy, select_action, evaluate_actions


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """All hyperparameters for curriculum PPO training."""

    # Curriculum
    max_depth: int = 7
    advance_threshold: float = 0.80
    eval_episodes: int = 100
    max_episodes_per_depth: int = 3000
    episodes_per_rollout: int = 128

    # Episode
    max_steps_factor: int = 3
    max_steps_offset: int = 2

    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 64
    max_grad_norm: float = 0.5

    # Reward
    solve_reward: float = 1.0
    step_penalty: float = -0.02

    def max_steps(self, depth: int) -> int:
        return depth * self.max_steps_factor + self.max_steps_offset


# ─── Statistics ───────────────────────────────────────────────────────────────

@dataclass
class DepthStats:
    """Training statistics for a single curriculum depth level."""

    depth: int
    episodes_trained: int = 0
    solve_rate: float = 0.0
    avg_return: float = 0.0
    avg_episode_length: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    advanced: bool = False
    time_seconds: float = 0.0


@dataclass
class TrainingStats:
    """Aggregate training statistics across all depth levels."""

    depth_stats: list[DepthStats] = field(default_factory=list)
    current_depth: int = 1
    total_episodes: int = 0
    total_updates: int = 0
    solve_rate_history: list[tuple[int, float]] = field(default_factory=list)
    total_time: float = 0.0


# ─── Solve results ────────────────────────────────────────────────────────────

@dataclass
class SolveResult:
    """Result of one solve attempt by the agent."""

    depth: int
    solved: bool
    steps: int
    moves: list[int]
    states: list[list[int]]  # state after each move


@dataclass
class TestResults:
    """Aggregate test results at a specific scramble depth."""

    depth: int
    n_episodes: int
    solve_rate: float
    avg_steps: float
    results: list[SolveResult]


# ─── GAE computation ─────────────────────────────────────────────────────────

def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Compute Generalised Advantage Estimation.

    Args:
        rewards: per-step rewards.
        values: per-step value estimates from the critic.
        dones: per-step episode termination flags.
        gamma: discount factor.
        gae_lambda: GAE lambda parameter.

    Returns:
        (advantages, returns) — both lists of the same length.
    """
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        if dones[t]:
            next_value = 0.0
            gae = 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
        next_value = values[t]

    return advantages, returns


# ─── Rollout collection ──────────────────────────────────────────────────────

@dataclass
class Rollout:
    """A batch of transitions collected from the environment."""

    states: torch.Tensor       # (N, 144)
    actions: torch.Tensor      # (N,) long
    log_probs: torch.Tensor    # (N,)
    returns: torch.Tensor      # (N,)
    advantages: torch.Tensor   # (N,)
    episode_returns: list[float]
    episode_lengths: list[int]
    episode_solved: list[bool]


def collect_rollouts(
    policy: CubePolicy,
    depth: int,
    config: TrainConfig,
    device: torch.device,
    rng: random.Random,
) -> Rollout:
    """Run *config.episodes_per_rollout* episodes and return batched transitions."""
    all_states: list[torch.Tensor] = []
    all_actions: list[int] = []
    all_log_probs: list[float] = []
    all_rewards: list[float] = []
    all_dones: list[bool] = []
    all_values: list[float] = []

    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    episode_solved: list[bool] = []

    policy.eval()

    for _ in range(config.episodes_per_rollout):
        cube = PocketCube()
        cube.reset()
        cube.scramble(depth, rng=rng)
        cube._max_steps = config.max_steps(depth)

        ep_return = 0.0
        ep_len = 0
        solved = False

        done = False
        while not done:
            state_t = cube.to_tensor()
            action, log_prob, value = select_action(policy, state_t, device)

            result = cube.step(action)

            all_states.append(state_t)
            all_actions.append(action)
            all_log_probs.append(log_prob)
            all_rewards.append(result.reward)
            all_dones.append(result.done)
            all_values.append(value)

            ep_return += result.reward
            ep_len += 1
            done = result.done
            if result.info["solved"]:
                solved = True

        episode_returns.append(ep_return)
        episode_lengths.append(ep_len)
        episode_solved.append(solved)

    # Compute GAE
    advantages, returns = compute_gae(
        all_rewards, all_values, all_dones,
        config.gamma, config.gae_lambda,
    )

    return Rollout(
        states=torch.stack(all_states),
        actions=torch.tensor(all_actions, dtype=torch.long),
        log_probs=torch.tensor(all_log_probs),
        returns=torch.tensor(returns),
        advantages=torch.tensor(advantages),
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        episode_solved=episode_solved,
    )


# ─── PPO update ──────────────────────────────────────────────────────────────

def ppo_update(
    policy: CubePolicy,
    optimizer: Adam,
    rollout: Rollout,
    config: TrainConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """Run PPO clipped update on collected rollouts.

    Returns:
        (policy_loss, value_loss, entropy) — averaged over all mini-batches.
    """
    policy.train()

    states = rollout.states.to(device)
    actions = rollout.actions.to(device)
    old_log_probs = rollout.log_probs.to(device)
    returns = rollout.returns.to(device)
    advantages = rollout.advantages.to(device)

    # Normalise advantages
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    n = states.size(0)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(config.ppo_epochs):
        indices = torch.randperm(n, device=device)
        for start in range(0, n, config.mini_batch_size):
            end = min(start + config.mini_batch_size, n)
            idx = indices[start:end]

            new_log_probs, entropy, new_values = evaluate_actions(
                policy, states[idx], actions[idx],
            )

            ratio = torch.exp(new_log_probs - old_log_probs[idx])
            adv = advantages[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(
                ratio,
                1.0 - config.clip_epsilon,
                1.0 + config.clip_epsilon,
            ) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns[idx])
            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy_mean
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_mean.item()
            n_updates += 1

    if n_updates == 0:
        return 0.0, 0.0, 0.0

    return (
        total_policy_loss / n_updates,
        total_value_loss / n_updates,
        total_entropy / n_updates,
    )


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_policy(
    policy: CubePolicy,
    depth: int,
    n_episodes: int,
    config: TrainConfig,
    device: torch.device,
    rng: random.Random,
    record_moves: bool = False,
) -> TestResults:
    """Evaluate the policy at a given scramble depth.

    Args:
        record_moves: if True, record full move+state sequences (for viz).
    """
    policy.eval()
    results: list[SolveResult] = []

    for _ in range(n_episodes):
        cube = PocketCube()
        cube.reset()
        cube.scramble(depth, rng=rng)
        cube._max_steps = config.max_steps(depth)

        moves: list[int] = []
        states: list[list[int]] = []
        if record_moves:
            states.append(list(cube.state))

        done = False
        while not done:
            state_t = cube.to_tensor()
            action, _, _ = select_action(policy, state_t, device)
            result = cube.step(action)
            moves.append(action)
            if record_moves:
                states.append(list(cube.state))
            done = result.done

        results.append(SolveResult(
            depth=depth,
            solved=cube.is_solved(),
            steps=len(moves),
            moves=moves,
            states=states if record_moves else [],
        ))

    solved_count = sum(1 for r in results if r.solved)
    avg_steps = sum(r.steps for r in results) / max(len(results), 1)

    return TestResults(
        depth=depth,
        n_episodes=n_episodes,
        solve_rate=solved_count / max(n_episodes, 1),
        avg_steps=avg_steps,
        results=results,
    )


# ─── Curriculum trainer ──────────────────────────────────────────────────────

class CurriculumTrainer:
    """Trains a PPO agent on the Pocket Cube with curriculum learning.

    Starts at depth 1 and advances when solve rate exceeds the threshold.
    """

    def __init__(
        self,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        self.policy = CubePolicy().to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=self.config.lr)
        self.stats = TrainingStats()

    def train(
        self,
        callback: Optional[Callable[[int, DepthStats, TrainingStats], None]] = None,
    ) -> TrainingStats:
        """Run the full curriculum training loop.

        Args:
            callback: called after each rollout cycle with
                (depth, current_depth_stats, overall_stats).

        Returns:
            TrainingStats with per-depth results.
        """
        start_time = time.time()

        for depth in range(1, self.config.max_depth + 1):
            self.stats.current_depth = depth
            depth_start = time.time()
            ds = DepthStats(depth=depth)
            episodes_done = 0

            while episodes_done < self.config.max_episodes_per_depth:
                # Collect rollouts
                rollout = collect_rollouts(
                    self.policy, depth, self.config, self.device, self.rng,
                )

                # PPO update
                pl, vl, ent = ppo_update(
                    self.policy, self.optimizer, rollout, self.config, self.device,
                )

                episodes_done += self.config.episodes_per_rollout
                self.stats.total_episodes += self.config.episodes_per_rollout
                self.stats.total_updates += 1

                # Track stats
                ds.episodes_trained = episodes_done
                ds.policy_loss = pl
                ds.value_loss = vl
                ds.entropy = ent
                ds.avg_return = sum(rollout.episode_returns) / len(rollout.episode_returns)
                ds.avg_episode_length = sum(rollout.episode_lengths) / len(rollout.episode_lengths)

                # Evaluate
                eval_results = evaluate_policy(
                    self.policy, depth, self.config.eval_episodes,
                    self.config, self.device, self.rng,
                )
                ds.solve_rate = eval_results.solve_rate
                ds.time_seconds = time.time() - depth_start

                self.stats.solve_rate_history.append(
                    (self.stats.total_episodes, ds.solve_rate)
                )

                if callback:
                    callback(depth, ds, self.stats)

                if ds.solve_rate >= self.config.advance_threshold:
                    ds.advanced = True
                    break

            self.stats.depth_stats.append(ds)

        self.stats.total_time = time.time() - start_time
        return self.stats
