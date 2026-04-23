# Preference Collection
#
# Manages the preference database D of triples (sigma_1, sigma_2, mu).
# Provides a synthetic oracle for automated testing.
#
# The synthetic oracle compares segments using a hidden "true" reward
# (e.g., terminal object values + step penalty), so you can verify the
# algorithm works before hooking up real human feedback.
#
# mu is the preference distribution over {1, 2}:
#   - Human prefers sigma_1: mu = [1, 0]
#   - Human prefers sigma_2: mu = [0, 1]
#   - Equal preference:      mu = [0.5, 0.5]
#   - Incomparable:          not added to database

import random


class PreferenceDB:
    """Database of preference triples (sigma_1, sigma_2, mu)."""

    def __init__(self, max_size: int = 5000):
        self.data = []
        self.max_size = max_size

    def add(self, sigma_1: list, sigma_2: list, mu: list):
        """Add a preference triple. If full, drop the oldest entry."""
        self.data.append((sigma_1, sigma_2, mu))
        if len(self.data) > self.max_size:
            self.data.pop(0)

    def sample(self, batch_size: int) -> list:
        """Sample a batch of preference triples (with replacement)."""
        if len(self.data) == 0:
            return []
        return random.choices(self.data, k=min(batch_size, len(self.data)))

    def get_all(self) -> list:
        """Return all stored preference triples."""
        return list(self.data)

    def __len__(self):
        return len(self.data)


class SyntheticOracle:
    """
    Synthetic oracle that compares segments using a hidden true reward.

    Two reward modes:
      1. Sparse (use_distance_shaping=False):
         - terminal_value if the step lands on a terminal cell
         - step_penalty otherwise
         Only informative when segments hit different terminals.

      2. Distance-shaped (use_distance_shaping=True, default):
         - terminal_value if the step lands on a terminal cell
         - step_penalty + proximity bonus for getting closer to the goal
         Every step is informative — mimics a human who prefers seeing
         the cat move toward the goal, not just reaching it.

    Includes a HUMAN_ERROR_RATE chance of responding randomly,
    matching the paper's 10% noise assumption (Section 2.2.3).
    """

    def __init__(self, terminal_objects_placement: dict,
                 human_error_rate: float = 0.1,
                 step_penalty: float = -1.0,
                 use_distance_shaping: bool = True):
        self.terminal_objects_placement = terminal_objects_placement or {}
        self.human_error_rate = human_error_rate
        self.step_penalty = step_penalty
        self.use_distance_shaping = use_distance_shaping

        # Find the goal position (highest-value terminal) for distance shaping
        if self.terminal_objects_placement:
            self.goal_pos = max(self.terminal_objects_placement,
                                key=self.terminal_objects_placement.get)
            # Max possible Manhattan distance (used for normalization)
            self.max_dist = abs(self.goal_pos[0]) + abs(self.goal_pos[1]) + 2
        else:
            self.goal_pos = None
            self.max_dist = 1

    def true_step_reward(self, step: dict) -> float:
        """
        Compute the hidden true reward for a single step.

        With distance shaping: reward = step_penalty + proximity_bonus
        where proximity_bonus = (max_dist - dist_to_goal) / max_dist
        This ranges from ~0 (far from goal) to ~1 (at goal).

        Without shaping: flat step_penalty for non-terminal steps.
        """
        next_pos = tuple(step["next_obs"])

        if next_pos in self.terminal_objects_placement:
            return float(self.terminal_objects_placement[next_pos])

        if self.use_distance_shaping and self.goal_pos is not None:
            dist = abs(next_pos[0] - self.goal_pos[0]) + abs(next_pos[1] - self.goal_pos[1])
            proximity_bonus = (self.max_dist - dist) / self.max_dist
            return self.step_penalty + proximity_bonus

        return self.step_penalty

    def true_segment_reward(self, segment: list) -> float:
        """Sum of true rewards over a segment."""
        return sum(self.true_step_reward(step) for step in segment)

    def compare(self, sigma_1: list, sigma_2: list):
        """
        Compare two segments. Returns mu = [p1, p2] or None if incomparable.

        With probability human_error_rate, returns a random preference.
        Otherwise, returns based on the true reward comparison.
        """
        # Human error: random response
        if random.random() < self.human_error_rate:
            coin = random.random()
            if coin < 0.5:
                return [1.0, 0.0]
            else:
                return [0.0, 1.0]

        r1 = self.true_segment_reward(sigma_1)
        r2 = self.true_segment_reward(sigma_2)

        if r1 > r2:
            return [1.0, 0.0]
        elif r2 > r1:
            return [0.0, 1.0]
        else:
            return [0.5, 0.5]
