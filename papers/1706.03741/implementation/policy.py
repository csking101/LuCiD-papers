# Policy
#
# A policy gradient agent that optimizes against the learned reward
# model r_hat. Since the reward function is non-stationary (it changes
# as more preferences are collected), policy gradient methods are
# preferred over value-based methods.
#
# Uses REINFORCE — good starting point for this grid world.
# The paper uses A2C (Atari) and TRPO (MuJoCo), but for an 8x8 grid
# REINFORCE is sufficient.
#
# Key additions over vanilla REINFORCE:
#   - Entropy bonus: prevents premature convergence to deterministic policies
#   - Exploration epsilon: floor on random action rate during collection
#   - Reward normalization: zero mean, constant std (Section 2.1)

import random as py_random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Simple feedforward network: (x, y) -> action probabilities."""

    def __init__(self, policy_hidden_size: int):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, policy_hidden_size),
            nn.ReLU(),
            nn.Linear(policy_hidden_size, 4),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits  # raw logits — Categorical handles softmax internally


class Policy:
    """
    REINFORCE policy gradient agent with entropy regularization.
    Produces actions by sampling from the policy distribution.
    Updates via policy gradient using rewards from the learned reward model.
    """

    def __init__(self, policy_hidden_size: int, policy_lr: float,
                 policy_epochs: int, gamma: float,
                 entropy_beta: float = 0.01,
                 grad_clip_norm: float = 1.0,
                 device: str = "cpu"):
        self.lr = policy_lr
        self.epochs = policy_epochs
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.grad_clip_norm = grad_clip_norm
        self.device = torch.device(device)

        self.network = PolicyNetwork(policy_hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

    def get_action(self, obs: list, exploration_epsilon: float = 0.0) -> tuple:
        """
        Sample an action from the policy.

        With probability exploration_epsilon, sample uniformly at random
        (ensures diverse trajectories even if policy has collapsed).

        Returns (action_int, log_prob) where log_prob is a scalar tensor.
        """
        # Epsilon-greedy exploration
        if exploration_epsilon > 0 and py_random.random() < exploration_epsilon:
            action_int = py_random.randint(0, 3)
            # Still compute log_prob from the policy (for correct gradients)
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.network(obs_tensor)
                dist = Categorical(logits=logits)
                log_prob = dist.log_prob(torch.tensor(action_int, device=self.device))
            return action_int, log_prob

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        logits = self.network(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

    def get_action_probabilities(self, obs: list) -> list:
        """Return action probabilities as a plain list (for export/debug)."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().tolist()

    def get_entropy(self, obs: list) -> float:
        """Get the entropy of the policy at a given observation (for logging)."""
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.network(obs_tensor)
            dist = Categorical(logits=logits)
            return dist.entropy().item()

    def get_avg_entropy(self, env) -> float:
        """Average policy entropy across all non-wall cells."""
        total = 0.0
        count = 0
        for y in range(env.height):
            for x in range(env.width):
                if (x, y) not in env.wall_placement:
                    total += self.get_entropy([x, y])
                    count += 1
        return total / max(count, 1)

    def _recompute_log_probs_and_entropy(self, obs_list: list,
                                          actions: list) -> tuple:
        """
        Recompute log probabilities and entropy from stored obs/actions
        using the current network weights.

        Returns (log_probs, entropy) both as tensors.
        """
        obs_tensor = torch.tensor(obs_list, dtype=torch.float32, device=self.device)
        logits = self.network(obs_tensor)  # (T, 4)
        dist = Categorical(logits=logits)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        log_probs = dist.log_prob(actions_tensor)  # (T,)
        entropy = dist.entropy()  # (T,)
        return log_probs, entropy

    def update(self, trajectories: list, reward_ensemble) -> tuple:
        """
        REINFORCE update with entropy bonus using rewards from the reward model.

        For each trajectory:
          1. Compute reward r_t = reward_model(o_t, a_t) for each step
          2. Normalize rewards to zero mean, unit std
          3. Compute discounted returns G_t
          4. Recompute log_probs + entropy from current network weights
          5. Loss = -mean(log_prob_t * G_t) - beta * mean(entropy)

        Returns (avg_policy_loss, avg_entropy).
        """
        total_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # Pre-compute rewards for each trajectory (these don't depend on policy)
        traj_data = []
        for traj in trajectories:
            if len(traj) == 0:
                continue
            obs_list = [step["obs"] for step in traj]
            actions = [step["action"] for step in traj]

            # Compute rewards from the reward model
            rewards = []
            for obs, action in zip(obs_list, actions):
                r = reward_ensemble.predict_reward(obs, action)
                rewards.append(r)

            # Normalize rewards (Section 2.1 — zero mean, constant std)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            if len(rewards_tensor) > 1 and rewards_tensor.std() > 1e-8:
                rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

            # Compute discounted returns G_t = sum_{k=0}^{T-t} gamma^k * r_{t+k}
            returns = []
            G = 0.0
            for r in reversed(rewards_tensor.tolist()):
                G = r + self.gamma * G
                returns.insert(0, G)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Normalize returns for stability
            if len(returns_tensor) > 1 and returns_tensor.std() > 1e-8:
                returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            traj_data.append((obs_list, actions, returns_tensor))

        if not traj_data:
            return 0.0, 0.0

        for _ in range(self.epochs):
            epoch_loss = 0.0
            epoch_entropy = 0.0
            for obs_list, actions, returns_tensor in traj_data:
                # Recompute from current weights (fresh graph each time)
                log_probs, entropy = self._recompute_log_probs_and_entropy(
                    obs_list, actions)

                # Policy gradient loss with entropy bonus
                pg_loss = -(log_probs * returns_tensor.detach()).mean()
                entropy_bonus = -self.entropy_beta * entropy.mean()
                loss = pg_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                epoch_loss += pg_loss.item()
                epoch_entropy += entropy.mean().item()
                num_updates += 1

            total_loss += epoch_loss
            total_entropy += epoch_entropy

        avg_loss = total_loss / max(num_updates, 1)
        avg_entropy = total_entropy / max(num_updates, 1)
        return avg_loss, avg_entropy
