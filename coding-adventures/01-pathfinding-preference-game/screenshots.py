#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 01 from real captured data.

Loads ``screenshot_data.json`` (produced by ``app.py --capture``) and
reconstructs the data objects needed by ``viz.py``, so the screenshots
are pixel-perfect renderings of actual training runs — not synthetic data.

Usage::

    python screenshots.py                     # uses ./screenshot_data.json
    python screenshots.py path/to/data.json   # custom path
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add adventure dir to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from env import GridWorld, Trajectory
from viz import (
    render_grid,
    render_heatmap,
    render_metrics_table,
    render_llm_parallel_table,
    render_phase_header,
    render_policy_arrows,
    render_policy_comparison,
    render_preference_pair,
    render_results_summary,
    render_rm_architecture,
    render_rm_spot_check,
    sparkline,
)

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "01"


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _dict_to_traj(d: dict) -> Trajectory:
    """Reconstruct a Trajectory from a serialized JSON dict.

    The JSON format comes from ``_traj_to_dict()`` in ``app.py``:
    positions, actions, rewards, done, reached_goal, pickups_collected,
    length, num_turns, unique_cells, pickup_reward, total_reward.
    """
    positions = [tuple(p) for p in d["positions"]]
    actions = d["actions"]
    rewards = d["rewards"]
    pickups_collected = [tuple(pc) for pc in d["pickups_collected"]]

    # Reconstruct normalised state vectors from positions
    size = 8  # default grid size
    states = [np.array([r / size, c / size], dtype=np.float32)
              for r, c in positions]

    # log_probs and values aren't captured — fill with plausible dummies
    n_actions = len(actions)
    log_probs = [-1.0] * n_actions
    values = [float(i) / max(n_actions, 1) * 8.0 for i in range(n_actions)]

    return Trajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        positions=positions,
        log_probs=log_probs,
        values=values,
        done=d["done"],
        reached_goal=d["reached_goal"],
        pickups_collected=pickups_collected,
    )


def _load_data(path: Path) -> dict:
    """Load and return the captured JSON data."""
    with open(path) as f:
        return json.load(f)


def _save(console: Console, filename: str, title: str) -> None:
    svg = console.export_svg(title=title)
    (OUTPUT_DIR / filename).write_text(svg)
    print(f"  ✓ {filename}")


# ---------------------------------------------------------------------------
# Screenshot 1: Grid World with a real demo trajectory
# ---------------------------------------------------------------------------

def generate_grid_world(data: dict):
    console = Console(record=True, width=80)
    env = GridWorld()

    traj = _dict_to_traj(data["phase1"]["demo_traj"])

    grid = render_grid(
        env,
        path=traj.positions,
        path_style="bold cyan",
        title=f"Agent Path (Episode — Pre-training Demo)",
        show_stats=True,
        traj=traj,
    )
    console.print(grid)

    _save(console, "01_grid_world.svg", "Grid World Environment")


# ---------------------------------------------------------------------------
# Screenshot 2: Preference pair comparison
# ---------------------------------------------------------------------------

def generate_preferences(data: dict):
    console = Console(record=True, width=90)
    env = GridWorld()

    traj_a = _dict_to_traj(data["phase2"]["traj_a"])
    traj_b = _dict_to_traj(data["phase2"]["traj_b"])
    pair_num = data["phase2"]["pair_num"]
    total_pairs = data["phase2"]["total_pairs"]

    pair_display = render_preference_pair(
        env, traj_a, traj_b,
        pair_num=pair_num, total_pairs=total_pairs,
    )
    console.print(pair_display)

    _save(console, "02_preferences.svg", "Human Preference Collection")


# ---------------------------------------------------------------------------
# Screenshot 3: Reward Model training
# ---------------------------------------------------------------------------

def generate_reward_model(data: dict):
    console = Console(record=True, width=80)
    p3 = data["phase3"]

    # RM architecture (static display)
    arch = render_rm_architecture()
    console.print(arch)

    # Metrics from real training
    loss_vals = p3["loss_history"]
    acc_vals = p3["accuracy_history"]

    metrics = {
        "Train Loss": (f"{p3['final_loss']:.4f}", sparkline(loss_vals)),
        "Accuracy": (f"{p3['final_accuracy']:.1%}", sparkline(acc_vals)),
        "Epoch": (f"{len(loss_vals)} / {len(loss_vals)}", ""),
    }
    metrics_panel = render_metrics_table(metrics, title="Reward Model Training")
    console.print(metrics_panel)

    # RM heatmap — real learned reward values
    rm_heatmap = np.array(p3["rm_heatmap"], dtype=np.float32)
    env = GridWorld()
    heatmap = render_heatmap(
        rm_heatmap, env,
        title="Learned Reward r(s)",
        border_style="cyan",
    )
    console.print(heatmap)

    # Spot-check of a real RM prediction
    sc = p3["spot_check"]
    spot = render_rm_spot_check(
        pair_idx=sc["pair_idx"],
        r_a=sc["r_a"],
        r_b=sc["r_b"],
        prob_a=sc["prob_a"],
        human_label=sc["human_label"],
    )
    console.print(spot)

    _save(console, "03_reward_model.svg", "Reward Model Training")


# ---------------------------------------------------------------------------
# Screenshot 4: PPO fine-tuning dashboard
# ---------------------------------------------------------------------------

def generate_ppo_training(data: dict):
    console = Console(record=True, width=90)
    env = GridWorld()
    p4 = data["phase4"]

    # Phase header
    header = render_phase_header(4)
    console.print(header)

    # Training metrics from real PPO run
    reward_vals = p4["reward_history"]
    kl_vals = p4["kl_history"]
    goal_vals = p4["goal_rate_history"]
    entropy_vals = p4["entropy_history"]

    metrics = {
        "RM Score": (f"{reward_vals[-1]:.2f}", sparkline(reward_vals)),
        "Goal Rate": (f"{goal_vals[-1]:.0%}", sparkline(goal_vals)),
        "KL Divergence": (f"{kl_vals[-1]:.3f}", sparkline(kl_vals)),
        "Entropy": (f"{entropy_vals[-1]:.3f}", sparkline(entropy_vals)),
        "Episode": (f"{len(reward_vals)} / {len(reward_vals)}", ""),
    }
    metrics_panel = render_metrics_table(metrics, title="PPO Fine-Tuning")
    console.print(metrics_panel)

    # Value heatmap from real training
    val_map = np.array(p4["val_map"], dtype=np.float32)
    heatmap = render_heatmap(val_map, env, title="Value Heatmap V(s)", border_style="magenta")
    console.print(heatmap)

    # Policy arrows from real RLHF policy
    rlhf_policy = np.array(p4["rlhf_policy_map"], dtype=int)
    arrows = render_policy_arrows(rlhf_policy, env, title="RLHF Policy (Greedy)")
    console.print(arrows)

    _save(console, "04_ppo_training.svg", "PPO Fine-Tuning with KL Penalty")


# ---------------------------------------------------------------------------
# Screenshot 5: Pre-trained vs RLHF comparison
# ---------------------------------------------------------------------------

def generate_comparison(data: dict):
    console = Console(record=True, width=90)
    env = GridWorld()
    conc = data["conclusion"]

    # Real policy maps
    pt_map = np.array(conc["pt_map"], dtype=int)
    rl_map = np.array(conc["rl_map"], dtype=int)

    # Policy comparison (3-column: pre-trained, RLHF, diff)
    comparison = render_policy_comparison(pt_map, rl_map, env)
    console.print(comparison)

    # Results summary from real evaluation
    summary = render_results_summary(conc["pt_stats"], conc["rl_stats"])
    console.print(summary)

    # LLM parallel mapping table
    parallel = render_llm_parallel_table()
    console.print(parallel)

    _save(console, "05_comparison.svg", "Pre-training vs RLHF Policy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "screenshot_data.json"

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Run 'python app.py --capture screenshot_data.json' first.")
        sys.exit(1)

    data = _load_data(data_path)
    print(f"Loaded capture data from {data_path}")
    print("Generating Adventure 01 screenshots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_grid_world(data)
    generate_preferences(data)
    generate_reward_model(data)
    generate_ppo_training(data)
    generate_comparison(data)

    print("Done!")
