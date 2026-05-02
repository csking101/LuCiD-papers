"""
Preference Database
====================

Stores (trajectory_A, trajectory_B, label) tuples collected from the
human annotator and provides sampling utilities for reward-model training.

Also includes lightweight analytics on what the user tends to prefer,
used by the Rich UI to show preference patterns.

LLM parallel
------------
This is the human-feedback dataset.  In real RLHF, companies like Anthropic
and OpenAI collect hundreds of thousands of comparison pairs.  Here, the
user provides ~20-50 comparisons — just enough to train a small RM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import Trajectory


# ---------------------------------------------------------------------------
# A single preference pair
# ---------------------------------------------------------------------------
@dataclass
class PreferencePair:
    """One human preference comparison."""

    traj_a: Trajectory
    traj_b: Trajectory
    label: float  # 0.0 = A preferred, 1.0 = B preferred, 0.5 = tie

    @property
    def preferred(self) -> str:
        if self.label < 0.25:
            return "A"
        elif self.label > 0.75:
            return "B"
        return "tie"


# ---------------------------------------------------------------------------
# Preference Database
# ---------------------------------------------------------------------------
class PreferenceDatabase:
    """
    Collection of human preference pairs with sampling and analytics.
    """

    def __init__(self):
        self.pairs: List[PreferencePair] = []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(self, traj_a: Trajectory, traj_b: Trajectory, label: float) -> None:
        """
        Record a preference.

        Parameters
        ----------
        traj_a, traj_b : Trajectory
        label : float
            0.0 = A preferred,  1.0 = B preferred,  0.5 = tie.
        """
        assert 0.0 <= label <= 1.0, f"Label must be in [0, 1], got {label}"
        self.pairs.append(PreferencePair(traj_a, traj_b, label))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_batch(self, batch_size: int) -> List[PreferencePair]:
        """Sample a random batch (without replacement if possible)."""
        n = len(self.pairs)
        if n == 0:
            return []
        k = min(batch_size, n)
        indices = np.random.choice(n, k, replace=False)
        return [self.pairs[i] for i in indices]

    def to_training_data(
        self,
    ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Convert to the format expected by ``train_reward_model``.

        Returns list of (states_a, states_b, label) tuples.
        """
        data = []
        for pair in self.pairs:
            sa = pair.traj_a.state_tensor()
            sb = pair.traj_b.state_tensor()
            data.append((sa, sb, pair.label))
        return data

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def count_by_preference(self) -> Dict[str, int]:
        """Return counts: {'A': n, 'B': m, 'tie': k}."""
        counts = {"A": 0, "B": 0, "tie": 0}
        for p in self.pairs:
            counts[p.preferred] += 1
        return counts

    def preference_patterns(self) -> Dict[str, str]:
        """
        Analyse what the user tends to prefer.

        Returns dict with human-readable pattern descriptions.
        """
        if len(self.pairs) == 0:
            return {"status": "No preferences yet"}

        shorter_preferred = 0
        fewer_turns_preferred = 0
        total = 0

        for pair in self.pairs:
            if pair.label == 0.5:
                continue
            total += 1
            if pair.label < 0.5:
                preferred = pair.traj_a
                other = pair.traj_b
            else:
                preferred = pair.traj_b
                other = pair.traj_a

            if preferred.length < other.length:
                shorter_preferred += 1
            if preferred.num_turns < other.num_turns:
                fewer_turns_preferred += 1

        if total == 0:
            return {"status": "All ties"}

        return {
            "shorter_paths": f"{shorter_preferred}/{total}",
            "fewer_turns": f"{fewer_turns_preferred}/{total}",
            "total_comparisons": str(len(self.pairs)),
            "total_non_tie": str(total),
        }

    def summary_stats(self) -> Dict[str, float]:
        """
        Numerical summary stats for display.

        Returns dict with:
          avg_length_preferred, avg_length_other,
          avg_turns_preferred, avg_turns_other
        """
        if len(self.pairs) == 0:
            return {}

        pref_lengths = []
        other_lengths = []
        pref_turns = []
        other_turns = []

        for pair in self.pairs:
            if pair.label == 0.5:
                continue
            if pair.label < 0.5:
                pref, other = pair.traj_a, pair.traj_b
            else:
                pref, other = pair.traj_b, pair.traj_a

            pref_lengths.append(pref.length)
            other_lengths.append(other.length)
            pref_turns.append(pref.num_turns)
            other_turns.append(other.num_turns)

        if not pref_lengths:
            return {}

        return {
            "avg_length_preferred": float(np.mean(pref_lengths)),
            "avg_length_other": float(np.mean(other_lengths)),
            "avg_turns_preferred": float(np.mean(pref_turns)),
            "avg_turns_other": float(np.mean(other_turns)),
        }
