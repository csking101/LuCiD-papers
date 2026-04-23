# Trajectory Collection
#
# Roll out the current policy to collect trajectories.
# Slice trajectories into segments (flexible length).
# Pair segments for human comparison.
#
# Key data structures:
#   trajectory = list of step dicts: {"obs", "action", "log_prob", "next_obs", "done"}
#   segment    = list of step dicts: {"obs", "action", "next_obs", "done"}
#   pair       = (segment_1, segment_2)

import random


def collect_trajectory(env, policy, exploration_epsilon: float = 0.0) -> list:
    """
    Roll out one episode using the current policy.

    If exploration_epsilon > 0, the policy uses epsilon-greedy exploration
    to ensure diverse trajectories even when the policy has collapsed.

    Returns a list of step dicts with keys:
        obs, action, log_prob, next_obs, done
    """
    trajectory = []
    obs = env.reset()

    while True:
        action, log_prob = policy.get_action(obs, exploration_epsilon)
        next_obs, done, info = env.step(action)

        trajectory.append({
            "obs": list(obs),
            "action": action,
            "log_prob": log_prob,
            "next_obs": list(next_obs),
            "done": done,
            "info": info,
        })

        obs = next_obs
        if done:
            break

    return trajectory


def collect_trajectories(env, policy, n_episodes: int,
                         exploration_epsilon: float = 0.0) -> list:
    """Collect multiple trajectories with optional exploration."""
    return [collect_trajectory(env, policy, exploration_epsilon)
            for _ in range(n_episodes)]


def slice_segments(trajectory: list, segment_length: int,
                   min_length: int = 3) -> list:
    """
    Slice a trajectory into segments for preference comparison.

    Uses overlapping windows (stride = segment_length // 2) to generate
    more segments from the same trajectory.

    If the trajectory is shorter than segment_length but >= min_length,
    the whole trajectory is returned as a single shorter segment.

    Each segment is a list of step dicts (without log_prob — not needed
    for preferences).
    """
    if len(trajectory) < min_length:
        return []

    def _strip_step(step):
        return {
            "obs": step["obs"],
            "action": step["action"],
            "next_obs": step["next_obs"],
            "done": step["done"],
        }

    segments = []

    if len(trajectory) < segment_length:
        # Trajectory is short but viable — use it as one segment
        segments.append([_strip_step(s) for s in trajectory])
        return segments

    # Overlapping windows: stride = half the segment length
    stride = max(segment_length // 2, 1)
    for i in range(0, len(trajectory) - min_length + 1, stride):
        end = min(i + segment_length, len(trajectory))
        segment = [_strip_step(s) for s in trajectory[i:end]]
        if len(segment) >= min_length:
            segments.append(segment)

    return segments


def pair_segments(segments: list, n_pairs: int,
                  reward_ensemble=None) -> list:
    """
    Pair segments for preference comparison.

    If reward_ensemble is provided, use ensemble disagreement for active
    query selection (Section 2.2.4 of the paper): pick pairs where the
    ensemble's predictions have the highest variance.

    Otherwise, pair randomly.
    """
    if len(segments) < 2:
        return []

    n_pairs = min(n_pairs, len(segments) * (len(segments) - 1) // 2)

    if reward_ensemble is None:
        # Random pairing
        pairs = []
        indices = list(range(len(segments)))
        for _ in range(n_pairs):
            i, j = random.sample(indices, 2)
            pairs.append((segments[i], segments[j]))
        return pairs

    # Active query selection: sample candidate pairs, pick highest variance
    n_candidates = min(n_pairs * 5, len(segments) * (len(segments) - 1) // 2)
    candidates = []
    indices = list(range(len(segments)))
    seen = set()

    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 10:
        i, j = random.sample(indices, 2)
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            candidates.append((i, j))
        attempts += 1

    # Compute variance across ensemble for each candidate pair
    scored = []
    for i, j in candidates:
        variance = reward_ensemble.get_pair_uncertainty(segments[i], segments[j])
        scored.append((variance, i, j))

    # Pick the top-n_pairs by variance
    scored.sort(reverse=True)
    pairs = [(segments[i], segments[j]) for _, i, j in scored[:n_pairs]]
    return pairs
