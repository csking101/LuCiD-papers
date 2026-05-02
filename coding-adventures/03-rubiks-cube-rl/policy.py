"""PPO policy network for the 2x2 Pocket Cube.

Architecture:
    Shared MLP backbone (144 → 256 → 128, ReLU activations)
    ├── Policy head  → 6 action logits (Categorical distribution)
    └── Value head   → 1 scalar state-value estimate

Provides helper functions for action selection and batch evaluation
used by the PPO training loop.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from cube import STATE_DIM, NUM_ACTIONS


class CubePolicy(nn.Module):
    """Actor-critic network for the Pocket Cube."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 256,
        n_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, n_actions)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Orthogonal init (helps PPO convergence)
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=2**0.5)
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: state tensor, shape ``(batch, 144)`` or ``(144,)``.

        Returns:
            (logits, value) where logits has shape ``(..., 6)``
            and value has shape ``(..., 1)``.
        """
        features = self.shared(x)
        logits = self.policy_head(features)
        value = self.value_head(features)
        return logits, value


def select_action(
    policy: CubePolicy,
    state_tensor: torch.Tensor,
    device: torch.device,
) -> tuple[int, float, float]:
    """Sample one action from the policy.

    Args:
        policy: the actor-critic network (in eval or train mode).
        state_tensor: 1-D tensor of shape ``(144,)``.
        device: torch device for the forward pass.

    Returns:
        (action, log_prob, value) — all Python scalars.
    """
    with torch.no_grad():
        x = state_tensor.unsqueeze(0).to(device)
        logits, value = policy(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
    return action.item(), log_prob.item(), value.squeeze(-1).item()


def evaluate_actions(
    policy: CubePolicy,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate a batch of (state, action) pairs.

    Used in the PPO update to compute new log-probs, entropy, and values
    for a batch of transitions collected under the old policy.

    Args:
        policy: the actor-critic network.
        states: ``(batch, 144)``.
        actions: ``(batch,)`` long tensor.

    Returns:
        (log_probs, entropy, values) — each shape ``(batch,)``.
    """
    logits, values = policy(states)
    dist = Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    entropy = dist.entropy()
    return log_probs, entropy, values.squeeze(-1)
