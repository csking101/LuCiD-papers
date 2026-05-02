#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 01 using the real viz.py functions.

Constructs mock data objects (GridWorld, Trajectory, policy maps, etc.) and
passes them to the actual rendering functions, so the screenshots are
pixel-perfect matches of what users see in the terminal.

No GPU or torch training required -- just numpy + rich.
"""

import sys
from pathlib import Path

import numpy as np

# Add adventure dir to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console, Group

from env import (
    ACTION_DELTAS,
    COIN,
    GEM,
    NUM_ACTIONS,
    SIZE,
    WALL,
    GridWorld,
    Trajectory,
)
from viz import (
    render_grid,
    render_heatmap,
    render_llm_parallel_table,
    render_metrics_table,
    render_nn_forward,
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
# Helpers
# ---------------------------------------------------------------------------

def _build_trajectory(
    env: GridWorld,
    positions: list[tuple[int, int]],
    reached_goal: bool = True,
) -> Trajectory:
    """Build a Trajectory from a list of (row, col) positions."""
    states = [np.array([r / env.size, c / env.size], dtype=np.float32)
              for r, c in positions]

    # Derive actions from consecutive positions
    actions = []
    for (r1, c1), (r2, c2) in zip(positions[:-1], positions[1:]):
        dr, dc = r2 - r1, c2 - c1
        for a, (ar, ac) in enumerate(ACTION_DELTAS):
            if (ar, ac) == (dr, dc):
                actions.append(a)
                break
        else:
            actions.append(3)  # default RIGHT

    # Determine collected pickups along this path
    collected = []
    seen = set()
    for r, c in positions:
        if (r, c) in env.initial_pickups and (r, c) not in seen:
            collected.append((r, c, env.initial_pickups[(r, c)]))
            seen.add((r, c))

    # Simple synthetic rewards
    rewards = [-0.01] * len(actions)
    if reached_goal:
        rewards[-1] = 10.0

    return Trajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        positions=positions,
        log_probs=[-1.2] * len(actions),
        values=[float(i) / len(actions) * 8.0 for i in range(len(actions))],
        done=True,
        reached_goal=reached_goal,
        pickups_collected=collected,
    )


def _save(console: Console, filename: str, title: str) -> None:
    svg = console.export_svg(title=title)
    (OUTPUT_DIR / filename).write_text(svg)
    print(f"  ✓ {filename}")


# ---------------------------------------------------------------------------
# Screenshot 1: Grid World with a sample path
# ---------------------------------------------------------------------------

def generate_grid_world():
    console = Console(record=True, width=80)
    env = GridWorld()

    # A realistic path: S→ right along top, down through gap, across middle,
    # down through bottom gap, to goal. Collects some pickups along the way.
    path_positions = [
        (0, 0), (0, 1), (1, 1),  # start, grab coin at (1,1)
        (1, 0), (2, 0), (2, 1),  # down and through wall gap
        (3, 1),                    # grab coin at (3,1)
        (3, 2), (4, 2),           # middle corridor
        (4, 4), (3, 4), (3, 5),   # around barrier, grab coin at (3,5)
        (3, 6),                    # grab gem at (3,6)
        (4, 6),                    # grab coin at (4,6)
        (5, 6), (5, 7),           # through bottom gap
        (6, 7), (7, 7),           # to goal
    ]

    traj = _build_trajectory(env, path_positions, reached_goal=True)

    grid = render_grid(
        env,
        path=traj.positions,
        path_style="bold cyan",
        title="Agent Path (Episode 142)",
        show_stats=True,
        traj=traj,
    )
    console.print(grid)

    svg = console.export_svg(title="Grid World Environment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "01_grid_world.svg").write_text(svg)
    print("  ✓ 01_grid_world.svg")


# ---------------------------------------------------------------------------
# Screenshot 2: Preference pair comparison
# ---------------------------------------------------------------------------

def generate_preferences():
    console = Console(record=True, width=90)
    env = GridWorld()

    # Path A: efficient, shorter, fewer pickups
    path_a = [
        (0, 0), (0, 1), (1, 1), (1, 0),
        (2, 0), (2, 1), (3, 1), (3, 2),
        (4, 2), (4, 4), (5, 4 + 2), (5, 7),  # skip to gap
        (6, 7), (7, 7),
    ]
    # Fix: walk through valid cells only
    path_a = [
        (0, 0), (0, 1), (1, 1), (1, 0),
        (2, 0), (2, 1), (3, 1), (3, 2),
        (4, 2), (4, 4), (3, 4), (3, 5),
        (4, 5 + 1), (5, 6), (5, 7),
        (6, 7), (7, 7),
    ]

    # Path B: scenic detour, more pickups including gem at (0,7)
    path_b = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),  # coin at (0,4)
        (0, 5), (0, 6), (0, 7),  # gem at (0,7)
        (1, 7), (1, 6), (1, 5), (1, 3), (1, 2),
        (2, 1), (2, 0), (3, 0), (3, 1),  # coin at (3,1)
        (3, 2), (4, 2), (4, 4),
        (3, 4), (3, 5), (3, 6),  # gem at (3,6)
        (4, 6),  # coin at (4,6)
        (5, 6), (5, 7), (6, 7), (7, 7),
    ]

    traj_a = _build_trajectory(env, path_a, reached_goal=True)
    traj_b = _build_trajectory(env, path_b, reached_goal=True)

    pair_display = render_preference_pair(env, traj_a, traj_b, pair_num=12, total_pairs=30)
    console.print(pair_display)

    _save(console, "02_preferences.svg", "Human Preference Collection")


# ---------------------------------------------------------------------------
# Screenshot 3: Reward Model training
# ---------------------------------------------------------------------------

def generate_reward_model():
    console = Console(record=True, width=80)

    # RM architecture (static display)
    arch = render_rm_architecture()
    console.print(arch)

    # Metrics with realistic training curves
    loss_vals = [0.693, 0.641, 0.584, 0.512, 0.447, 0.381, 0.319, 0.264,
                 0.218, 0.189, 0.156, 0.134, 0.119, 0.111]
    acc_vals = [0.50, 0.55, 0.62, 0.68, 0.73, 0.78, 0.82, 0.84,
                0.87, 0.89, 0.91, 0.92, 0.93, 0.933]

    metrics = {
        "Train Loss": (f"{loss_vals[-1]:.4f}", sparkline(loss_vals)),
        "Val Loss": (f"{loss_vals[-1] * 1.85:.4f}", sparkline([v * 1.85 for v in loss_vals])),
        "Accuracy": (f"{acc_vals[-1]:.1%}", sparkline(acc_vals)),
        "Epoch": ("100 / 100", ""),
    }
    metrics_panel = render_metrics_table(metrics, title="Reward Model Training")
    console.print(metrics_panel)

    # Spot-check of an RM prediction
    spot = render_rm_spot_check(
        pair_idx=7,
        r_a=4.82,
        r_b=2.31,
        prob_a=0.92,
        human_label=0.0,  # human preferred A
    )
    console.print(spot)

    _save(console, "03_reward_model.svg", "Reward Model Training")


# ---------------------------------------------------------------------------
# Screenshot 4: PPO fine-tuning dashboard
# ---------------------------------------------------------------------------

def generate_ppo_training():
    console = Console(record=True, width=90)
    env = GridWorld()

    # Phase header
    header = render_phase_header(4)
    console.print(header)

    # Training metrics with realistic RLHF curves
    reward_vals = [1.2, 2.1, 3.4, 4.8, 5.9, 6.7, 7.5, 8.1, 8.7, 9.1, 9.5, 9.8, 10.0]
    kl_vals = [0.0, 0.008, 0.019, 0.035, 0.058, 0.082, 0.109, 0.134, 0.157, 0.178, 0.195, 0.209, 0.218]
    goal_vals = [0.42, 0.46, 0.52, 0.58, 0.64, 0.70, 0.75, 0.80, 0.84, 0.87, 0.90, 0.91, 0.92]
    entropy_vals = [1.386, 1.35, 1.30, 1.25, 1.19, 1.14, 1.10, 1.06, 1.03, 1.00, 0.98, 0.96, 0.958]

    metrics = {
        "RM Score": (f"{reward_vals[-1]:.2f}", sparkline(reward_vals)),
        "Goal Rate": (f"{goal_vals[-1]:.0%}", sparkline(goal_vals)),
        "KL Divergence": (f"{kl_vals[-1]:.3f}", sparkline(kl_vals)),
        "Entropy": (f"{entropy_vals[-1]:.3f}", sparkline(entropy_vals)),
        "β (KL coeff)": ("0.200", ""),
        "Episode": ("100 / 100", ""),
    }
    metrics_panel = render_metrics_table(metrics, title="PPO Fine-Tuning")
    console.print(metrics_panel)

    # Value heatmap — higher values near goal, lower near start
    values = np.zeros((SIZE, SIZE), dtype=np.float32)
    for r in range(SIZE):
        for c in range(SIZE):
            if env.grid[r, c] != WALL:
                # Distance-based value: closer to goal = higher
                dist = abs(r - 7) + abs(c - 7)
                values[r, c] = max(0, 10.0 - dist * 0.7) + np.random.uniform(0, 0.5)
    heatmap = render_heatmap(values, env, title="Value Heatmap V(s)", border_style="magenta")
    console.print(heatmap)

    # Policy arrows — mostly pointing toward goal with some exploration
    policy_map = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(SIZE):
        for c in range(SIZE):
            if env.grid[r, c] == WALL:
                continue
            # Generally: go down if above goal row, go right if left of goal col
            if r < 7 and c < 7:
                policy_map[r, c] = 1 if r < c else 3  # DOWN or RIGHT
            elif r < 7:
                policy_map[r, c] = 1  # DOWN
            elif c < 7:
                policy_map[r, c] = 3  # RIGHT
            else:
                policy_map[r, c] = 3  # at goal, default RIGHT
    arrows = render_policy_arrows(policy_map, env, title="RLHF Policy (Greedy)")
    console.print(arrows)

    _save(console, "04_ppo_training.svg", "PPO Fine-Tuning with KL Penalty")


# ---------------------------------------------------------------------------
# Screenshot 5: Pre-trained vs RLHF comparison
# ---------------------------------------------------------------------------

def generate_comparison():
    console = Console(record=True, width=90)
    env = GridWorld()

    # Pre-trained policy: somewhat random, biased toward goal but noisy
    rng = np.random.RandomState(42)
    pretrained_map = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(SIZE):
        for c in range(SIZE):
            if env.grid[r, c] == WALL:
                continue
            # Noisy goal-seeking: 60% optimal, 40% random
            if rng.random() < 0.6:
                if r < 7 and c < 7:
                    pretrained_map[r, c] = 1 if r <= c else 3
                elif r < 7:
                    pretrained_map[r, c] = 1
                elif c < 7:
                    pretrained_map[r, c] = 3
            else:
                pretrained_map[r, c] = rng.randint(0, NUM_ACTIONS)

    # RLHF policy: cleaner, more consistent goal-seeking + detours for pickups
    rlhf_map = np.zeros((SIZE, SIZE), dtype=int)
    for r in range(SIZE):
        for c in range(SIZE):
            if env.grid[r, c] == WALL:
                continue
            # Strong goal-seeking with pickup awareness
            if r < 7 and c < 7:
                rlhf_map[r, c] = 1 if r < c else 3  # DOWN or RIGHT
            elif r < 7:
                rlhf_map[r, c] = 1  # DOWN
            elif c < 7:
                rlhf_map[r, c] = 3  # RIGHT
            else:
                rlhf_map[r, c] = 3

    # Policy comparison (3-column: pre-trained, RLHF, diff)
    comparison = render_policy_comparison(pretrained_map, rlhf_map, env)
    console.print(comparison)

    # Results summary table
    pretrained_stats = {
        "Goal Rate (%)": 44.0,
        "Avg Steps": 23.6,
        "Avg Reward": 3.21,
        "Avg Pickups": 1.8,
    }
    rlhf_stats = {
        "Goal Rate (%)": 92.0,
        "Avg Steps": 15.4,
        "Avg Reward": 9.87,
        "Avg Pickups": 4.2,
    }
    summary = render_results_summary(pretrained_stats, rlhf_stats)
    console.print(summary)

    # LLM parallel mapping table
    parallel = render_llm_parallel_table()
    console.print(parallel)

    _save(console, "05_comparison.svg", "Pre-training vs RLHF Policy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating Adventure 01 screenshots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)

    generate_grid_world()
    generate_preferences()
    generate_reward_model()
    generate_ppo_training()
    generate_comparison()

    print("Done!")
