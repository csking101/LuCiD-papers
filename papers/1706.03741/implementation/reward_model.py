# Reward Model
#
# An ensemble of neural networks that predict scalar rewards from
# (observation, action) pairs.
#
# Trained using the Bradley-Terry model + cross-entropy loss:
#
#   P[sigma_1 > sigma_2] = exp(sum r(o,a) for sigma_1)
#                          / (exp(sum for sigma_1) + exp(sum for sigma_2))
#
#   loss = -sum[ mu(1)*log(P[s1>s2]) + mu(2)*log(P[s2>s1]) ]
#
# The ensemble provides uncertainty estimates for active query selection.
# Each predictor is trained on a bootstrap sample of the preference DB.
# Final reward = average of independently normalized predictor outputs.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


class RewardNetwork(nn.Module):
    """
    Single reward predictor: (obs, action_onehot) -> scalar reward.
    Input: obs [x, y] (2D) concatenated with action one-hot (4D) = 6D.
    Output: scalar reward.
    """

    def __init__(self, hidden_size: int = 64):
        super(RewardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # Running stats for normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0

    def forward(self, obs_action: torch.Tensor) -> torch.Tensor:
        """obs_action shape: (..., 6) -> (..., 1)"""
        return self.net(obs_action)


def encode_step(obs: list, action: int, device: torch.device) -> torch.Tensor:
    """Encode a single (obs, action) pair as a 6D tensor."""
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
    action_onehot = torch.zeros(4, dtype=torch.float32, device=device)
    action_onehot[action] = 1.0
    return torch.cat([obs_tensor, action_onehot])


def encode_segment(segment: list, device: torch.device) -> torch.Tensor:
    """Encode a segment as a (segment_length, 6) tensor."""
    encoded = [encode_step(step["obs"], step["action"], device) for step in segment]
    return torch.stack(encoded)


class RewardEnsemble:
    """
    Ensemble of reward predictors (Section 2.2.3 of the paper).

    Each predictor is trained on a bootstrap sample of the preference DB.
    The final reward is the average of independently normalized outputs.
    """

    def __init__(self, n_predictors: int, hidden_size: int, lr: float,
                 human_error_rate: float = 0.1, device: str = "cpu"):
        self.device = torch.device(device)
        self.human_error_rate = human_error_rate

        self.predictors = [
            RewardNetwork(hidden_size).to(self.device)
            for _ in range(n_predictors)
        ]
        self.optimizers = [
            optim.Adam(pred.parameters(), lr=lr, weight_decay=1e-4)
            for pred in self.predictors
        ]

    def predict_reward(self, obs: list, action: int) -> float:
        """
        Predict the reward for a single (obs, action) pair.
        Returns the average of independently normalized predictor outputs.
        """
        encoded = encode_step(obs, action, self.device).unsqueeze(0)

        rewards = []
        with torch.no_grad():
            for pred in self.predictors:
                raw = pred(encoded).item()
                # Normalize using predictor's running stats
                normalized = (raw - pred.reward_mean) / (pred.reward_std + 1e-8)
                rewards.append(normalized)

        return sum(rewards) / len(rewards)

    def _segment_reward_sum(self, predictor: RewardNetwork,
                            segment_encoded: torch.Tensor) -> torch.Tensor:
        """
        Sum of predicted rewards over a segment.
        segment_encoded shape: (segment_length, 6)
        Returns: scalar tensor.
        """
        rewards = predictor(segment_encoded)  # (segment_length, 1)
        return rewards.sum()

    def _preference_probability(self, predictor: RewardNetwork,
                                seg1_encoded: torch.Tensor,
                                seg2_encoded: torch.Tensor) -> torch.Tensor:
        """
        Bradley-Terry preference probability: P[s1 > s2].
        Returns: scalar tensor in (0, 1).

        Includes the human error rate (epsilon) from Section 2.2.3:
        P_adjusted = (1 - epsilon) * P_bradley_terry + epsilon * 0.5
        """
        r1 = self._segment_reward_sum(predictor, seg1_encoded)
        r2 = self._segment_reward_sum(predictor, seg2_encoded)

        # Bradley-Terry: softmax over [r1, r2]
        logits = torch.stack([r1, r2])
        p_bt = torch.softmax(logits, dim=0)[0]  # P[s1 > s2]

        # Mix with uniform noise (human error assumption)
        eps = self.human_error_rate
        p_adjusted = (1.0 - eps) * p_bt + eps * 0.5

        return p_adjusted

    def train_on_preferences(self, preference_db, epochs: int,
                             batch_size: int) -> float:
        """
        Train the ensemble on the preference database.

        Each predictor trains on a bootstrap sample (sampling with replacement).
        Loss = cross-entropy between predicted and actual preferences.
        Batched for efficiency — all preferences in a batch are processed in
        a single forward/backward pass.

        Returns the average loss across all predictors.
        """
        if len(preference_db) == 0:
            return 0.0

        total_loss = 0.0

        for pred_idx, (predictor, optimizer) in enumerate(
                zip(self.predictors, self.optimizers)):
            predictor.train()

            for epoch in range(epochs):
                # Bootstrap sample from the preference DB
                batch = preference_db.sample(batch_size)
                if not batch:
                    continue

                # Pre-encode all segments and collect mus
                seg1_sums = []
                seg2_sums = []
                mus = []
                for sigma_1, sigma_2, mu in batch:
                    seg1_enc = encode_segment(sigma_1, self.device)
                    seg2_enc = encode_segment(sigma_2, self.device)
                    # Sum rewards over each segment
                    r1 = predictor(seg1_enc).sum()
                    r2 = predictor(seg2_enc).sum()
                    seg1_sums.append(r1)
                    seg2_sums.append(r2)
                    mus.append(mu)

                # Stack into tensors for batched computation
                r1_batch = torch.stack(seg1_sums)       # (B,)
                r2_batch = torch.stack(seg2_sums)       # (B,)
                mu_batch = torch.tensor(mus, dtype=torch.float32,
                                        device=self.device)  # (B, 2)

                # Bradley-Terry with human error mixing
                logits = torch.stack([r1_batch, r2_batch], dim=1)  # (B, 2)
                p_bt = torch.softmax(logits, dim=1)[:, 0]  # P[s1>s2], (B,)
                eps = self.human_error_rate
                p1 = (1.0 - eps) * p_bt + eps * 0.5
                p2 = 1.0 - p1

                # Cross-entropy loss (batched)
                loss = -(mu_batch[:, 0] * torch.log(p1 + 1e-8) +
                         mu_batch[:, 1] * torch.log(p2 + 1e-8)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            predictor.eval()

        # Update running normalization stats for each predictor
        self._update_normalization_stats(preference_db)

        avg_loss = total_loss / (len(self.predictors) * epochs)
        return avg_loss

    def _update_normalization_stats(self, preference_db):
        """Update each predictor's running mean/std from recent data."""
        # Collect a sample of (obs, action) pairs from the DB
        all_data = preference_db.get_all()
        if not all_data:
            return

        # Gather all steps from all segments
        steps = []
        for sigma_1, sigma_2, _ in all_data[-200:]:  # last 200 for efficiency
            for step in sigma_1 + sigma_2:
                steps.append(encode_step(step["obs"], step["action"], self.device))

        if not steps:
            return

        encoded_batch = torch.stack(steps)

        with torch.no_grad():
            for predictor in self.predictors:
                rewards = predictor(encoded_batch).squeeze()
                predictor.reward_mean = rewards.mean().item()
                predictor.reward_std = rewards.std().item() if len(rewards) > 1 else 1.0

    def get_pair_uncertainty(self, sigma_1: list, sigma_2: list) -> float:
        """
        Estimate uncertainty for a pair of segments.
        Returns the variance of preference predictions across ensemble members.
        Used for active query selection (Section 2.2.4).
        """
        seg1_enc = encode_segment(sigma_1, self.device)
        seg2_enc = encode_segment(sigma_2, self.device)

        probs = []
        with torch.no_grad():
            for predictor in self.predictors:
                p = self._preference_probability(predictor, seg1_enc, seg2_enc)
                probs.append(p.item())

        # Variance across ensemble members
        mean_p = sum(probs) / len(probs)
        variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
        return variance
