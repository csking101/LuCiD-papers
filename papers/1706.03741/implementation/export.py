# Export
#
# Exports training results to JSON for the JavaScript demo:
#   - Grid configuration (size, goals, obstacles)
#   - Trajectory history (before and after training)
#   - Reward model heatmap (learned reward at each grid cell)
#   - Preference database sample (for replaying the rating UI)
#   - Training metrics over time

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import Environment, Action
from reward_model import RewardEnsemble
from optimal import compute_baseline


def export_grid_config(env: Environment) -> dict:
    """Export the grid layout: dimensions, terminals, walls."""
    return {
        "width": env.width,
        "height": env.height,
        "start": {"x": 0, "y": 0},
        "terminals": {
            f"{x},{y}": val
            for (x, y), val in env.terminal_objects_placement.items()
        },
        "walls": [
            {"x": x, "y": y}
            for (x, y) in env.wall_placement
        ],
        "max_steps": env.max_steps,
    }


def export_reward_heatmap(env: Environment,
                          reward_ensemble: RewardEnsemble) -> dict:
    """
    Compute the learned reward at each (x, y) cell for each action,
    and also the max-reward action per cell.

    Returns:
      {
        "per_action": { "0,0": {"UP": r, "DOWN": r, ...}, ... },
        "best_action": { "0,0": {"action": "RIGHT", "reward": r}, ... },
        "max_reward": float,
        "min_reward": float,
      }
    """
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    per_action = {}
    best_action = {}
    all_rewards = []

    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                continue

            key = f"{x},{y}"
            per_action[key] = {}
            best_r = float("-inf")
            best_a = None

            for a in range(4):
                r = reward_ensemble.predict_reward([x, y], a)
                per_action[key][action_names[a]] = round(r, 4)
                all_rewards.append(r)
                if r > best_r:
                    best_r = r
                    best_a = action_names[a]

            best_action[key] = {
                "action": best_a,
                "reward": round(best_r, 4),
            }

    return {
        "per_action": per_action,
        "best_action": best_action,
        "max_reward": round(max(all_rewards), 4) if all_rewards else 0.0,
        "min_reward": round(min(all_rewards), 4) if all_rewards else 0.0,
    }


def export_policy_map(env: Environment, policy) -> dict:
    """
    Export action probabilities for every cell.

    Returns:
      {
        "0,0": {"UP": p, "DOWN": p, "LEFT": p, "RIGHT": p, "best": "RIGHT"},
        ...
      }
    """
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
    policy_map = {}

    for y in range(env.height):
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                continue

            probs = policy.get_action_probabilities([x, y])
            key = f"{x},{y}"
            entry = {}
            for i, name in enumerate(action_names):
                entry[name] = round(probs[i], 4)
            entry["best"] = action_names[probs.index(max(probs))]
            policy_map[key] = entry

    return policy_map


def export_trajectories(env: Environment, policy,
                        n_episodes: int = 5,
                        label: str = "trained") -> list:
    """
    Collect and export sample trajectories.

    Each trajectory is a list of:
      {"x": int, "y": int, "action": str, "step": int}
    """
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    exported = []

    for ep in range(n_episodes):
        obs = env.reset()
        traj_steps = [{"x": obs[0], "y": obs[1], "action": None, "step": 0}]
        step = 0

        while not env.done:
            action, _ = policy.get_action(obs)
            obs, done, info = env.step(action)
            step += 1
            traj_steps.append({
                "x": obs[0],
                "y": obs[1],
                "action": action_names[action],
                "step": step,
            })

        exported.append({
            "label": label,
            "episode": ep,
            "length": step,
            "outcome": info.get("event", "unknown"),
            "steps": traj_steps,
        })

    return exported


def export_preferences_sample(preference_db, n: int = 20) -> list:
    """
    Export a sample of preference triples for the UI replay.

    Each entry:
      {"segment_1": [...], "segment_2": [...], "mu": [p1, p2]}
    where each segment step is {"obs": [x,y], "action": int}.
    """
    all_prefs = preference_db.get_all()
    sample = all_prefs[-n:] if len(all_prefs) >= n else all_prefs

    exported = []
    for sigma_1, sigma_2, mu in sample:
        exported.append({
            "segment_1": [
                {"obs": s["obs"], "action": s["action"]}
                for s in sigma_1
            ],
            "segment_2": [
                {"obs": s["obs"], "action": s["action"]}
                for s in sigma_2
            ],
            "mu": mu,
        })

    return exported


def export_optimal_policy(env: Environment, step_penalty: float = -1.0,
                          gamma: float = 0.99) -> dict:
    """
    Compute and export the DP optimal policy for the grid.

    Returns:
      {
        "policy": { "0,0": {"action": "DOWN", "action_int": 1}, ... },
        "optimal_return": float,
      }
    """
    action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    V_star, Q_star, pi_star, opt_return = compute_baseline(
        env, step_penalty, gamma)

    policy_map = {}
    for (x, y), action_int in pi_star.items():
        key = f"{x},{y}"
        policy_map[key] = {
            "action": action_names[action_int],
            "action_int": action_int,
        }

    return {
        "policy": policy_map,
        "optimal_return": round(opt_return, 2),
    }


def export_metrics(metrics_history: list) -> list:
    """Export the training metrics history as-is (already JSON-serializable)."""
    return metrics_history


def export_all(env, policy, reward_ensemble, preference_db,
               metrics_history, output_dir: str = None,
               step_penalty: float = -1.0, gamma: float = 0.99):
    """
    Export everything to a single JSON file.

    Output file: {output_dir}/rlhf_results.json
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    print("Exporting training results...")

    data = {
        "grid": export_grid_config(env),
        "reward_heatmap": export_reward_heatmap(env, reward_ensemble),
        "policy_map": export_policy_map(env, policy),
        "optimal": export_optimal_policy(env, step_penalty, gamma),
        "trajectories": export_trajectories(env, policy, n_episodes=5,
                                            label="trained"),
        "preferences_sample": export_preferences_sample(preference_db, n=20),
        "metrics": export_metrics(metrics_history),
    }

    output_path = os.path.join(output_dir, "rlhf_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Exported to {output_path} ({size_kb:.1f} KB)")
    print(f"  Grid: {data['grid']['width']}x{data['grid']['height']}")
    print(f"  Reward heatmap: {len(data['reward_heatmap']['per_action'])} cells")
    print(f"  Policy map: {len(data['policy_map'])} cells")
    print(f"  Optimal policy: {len(data['optimal']['policy'])} cells")
    print(f"  Trajectories: {len(data['trajectories'])} episodes")
    print(f"  Preferences: {len(data['preferences_sample'])} samples")
    print(f"  Metrics: {len(data['metrics'])} iterations")

    return output_path


if __name__ == "__main__":
    # Disable human feedback for non-interactive export run
    import config
    config.HUMAN_FEEDBACK_INTERVAL = 0

    # Run training first, then export
    from train import main as train_main

    metrics_history, policy, reward_ensemble, preference_db = train_main()

    from train import make_environment
    env = make_environment()

    export_all(env, policy, reward_ensemble, preference_db, metrics_history,
               step_penalty=config.STEP_PENALTY, gamma=config.GAMMA)
