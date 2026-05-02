"""
Bradley-Terry Neural Reward Model
==================================

A small MLP that assigns a scalar reward to each grid-world state.
The total trajectory reward is the **sum** of per-state rewards.

Preferences are modelled with the Bradley-Terry formula:

    P(trajectory A ≻ B) = σ( R(A) − R(B) )

where R(τ) = Σ_t  r_θ(s_t)  and σ is the logistic sigmoid.

The loss is binary cross-entropy on the preference label.

LLM parallel
------------
This is *exactly* how reward models are trained in RLHF for LLMs.
The only difference is the input representation — positions (r, c)
instead of token embeddings — and the summation over states instead
of over tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Reward model network
# ---------------------------------------------------------------------------
class RewardModel(nn.Module):
    """
    MLP that maps a single state (2-D normalised position) to a scalar
    reward.  Trajectory-level reward is computed by summing over states.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the state vector (default 2 for row, col).
    hidden_dim : int
        Width of each hidden layer.
    """

    def __init__(self, state_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Per-state reward.

        Parameters
        ----------
        states : Tensor of shape (*, state_dim)

        Returns
        -------
        rewards : Tensor of shape (*,)
        """
        return self.net(states).squeeze(-1)

    def trajectory_reward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Sum of per-state rewards for a trajectory.

        Parameters
        ----------
        states : Tensor of shape (T, state_dim)

        Returns
        -------
        total : scalar Tensor
        """
        return self.forward(states).sum()

    # ------------------------------------------------------------------
    # Introspection helpers (for the viz layer)
    # ------------------------------------------------------------------
    def forward_with_activations(
        self, states: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run forward pass and capture intermediate activations.

        Returns
        -------
        output : Tensor of shape (*,)
        activations : list of Tensors, one per hidden layer (post-ReLU)
        """
        activations: List[torch.Tensor] = []
        x = states
        for i, layer in enumerate(self.net):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x.detach().clone())
        return x.squeeze(-1), activations

    def reward_heatmap(self, grid_size: int = 12) -> np.ndarray:
        """
        Compute the learned reward for every cell in the grid.

        Returns
        -------
        heatmap : ndarray of shape (grid_size, grid_size)
        """
        coords = []
        for r in range(grid_size):
            for c in range(grid_size):
                coords.append(
                    [r / max(grid_size - 1, 1), c / max(grid_size - 1, 1)]
                )
        with torch.no_grad():
            t = torch.tensor(coords, dtype=torch.float32)
            rewards = self.forward(t).numpy()
        return rewards.reshape(grid_size, grid_size)


# ---------------------------------------------------------------------------
# Bradley-Terry preference loss
# ---------------------------------------------------------------------------
def bradley_terry_loss(
    rm: RewardModel,
    states_a: torch.Tensor,
    states_b: torch.Tensor,
    label: float,
) -> torch.Tensor:
    """
    Compute the Bradley-Terry cross-entropy loss for a single preference
    pair.

    Parameters
    ----------
    rm : RewardModel
    states_a : Tensor (T_a, 2)  — states of trajectory A
    states_b : Tensor (T_b, 2)  — states of trajectory B
    label : float
        0.0  if A is preferred,  1.0  if B is preferred,
        0.5  for a tie / equally-preferred.

    Returns
    -------
    loss : scalar Tensor
    """
    r_a = rm.trajectory_reward(states_a)
    r_b = rm.trajectory_reward(states_b)
    # logit = R(B) - R(A);  label=1 means B preferred
    logit = r_b - r_a
    target = torch.tensor(label, dtype=torch.float32)
    return nn.functional.binary_cross_entropy_with_logits(logit, target)


def bradley_terry_prob(
    rm: RewardModel,
    states_a: torch.Tensor,
    states_b: torch.Tensor,
) -> float:
    """
    Return P(A ≻ B) under the current reward model.
    """
    with torch.no_grad():
        r_a = rm.trajectory_reward(states_a)
        r_b = rm.trajectory_reward(states_b)
        prob = torch.sigmoid(r_a - r_b).item()
    return prob


# ---------------------------------------------------------------------------
# Training metrics
# ---------------------------------------------------------------------------
@dataclass
class RMTrainMetrics:
    """Metrics collected during reward model training."""
    epoch: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    accuracy_history: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_reward_model(
    rm: RewardModel,
    preference_data: List[Tuple[np.ndarray, np.ndarray, float]],
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 16,
    callback: Optional[callable] = None,
) -> RMTrainMetrics:
    """
    Train the reward model on collected preference data.

    Parameters
    ----------
    rm : RewardModel
    preference_data : list of (states_a, states_b, label)
        Each entry has numpy arrays for the two trajectories' states and a
        float label (0 = A preferred, 1 = B preferred).
    epochs : int
    lr : float
    batch_size : int
    callback : callable(metrics: RMTrainMetrics) or None
        Called after each epoch for live UI updates.

    Returns
    -------
    metrics : RMTrainMetrics  (final state)
    """
    optimizer = optim.Adam(rm.parameters(), lr=lr)
    metrics = RMTrainMetrics()
    n = len(preference_data)

    if n == 0:
        return metrics

    for epoch in range(1, epochs + 1):
        # Shuffle data
        indices = np.random.permutation(n)
        epoch_loss = 0.0
        correct = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_loss = torch.tensor(0.0)

            for idx in batch_idx:
                sa, sb, label = preference_data[idx]
                ta = torch.tensor(sa, dtype=torch.float32)
                tb = torch.tensor(sb, dtype=torch.float32)
                loss = bradley_terry_loss(rm, ta, tb, label)
                batch_loss = batch_loss + loss

                # Accuracy: does the RM agree with the label?
                with torch.no_grad():
                    r_a = rm.trajectory_reward(ta).item()
                    r_b = rm.trajectory_reward(tb).item()
                    pred = 0.0 if r_a > r_b else 1.0
                    if abs(label - 0.5) < 0.01:
                        correct += 1  # ties always "correct"
                    elif pred == label:
                        correct += 1

            batch_loss = batch_loss / len(batch_idx)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * len(batch_idx)

        metrics.epoch = epoch
        metrics.loss = epoch_loss / n
        metrics.accuracy = correct / n
        metrics.loss_history.append(metrics.loss)
        metrics.accuracy_history.append(metrics.accuracy)

        if callback is not None:
            callback(metrics)

    return metrics
