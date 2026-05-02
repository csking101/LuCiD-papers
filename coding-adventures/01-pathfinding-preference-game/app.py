#!/usr/bin/env python3
"""
Path-Finding Preference Game — Interactive Terminal App
========================================================

Run with:  python app.py
Capture:   python app.py --capture screenshot_data.json

An interactive RLHF demo in the terminal using Rich.  Walks through
four phases that mirror LLM alignment:

  Phase 1 — Pre-training (agent learns basic navigation)
  Phase 2 — Human preference collection (you rate path pairs)
  Phase 3 — Reward model training (RM learns from your ratings)
  Phase 4 — RLHF / PPO fine-tuning (agent optimises for your preferences)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from env import COIN, GEM, PICKUP_VALUES, GridWorld, Trajectory
from policy import EpisodeMetrics, PolicyNetwork, compute_policy_kl
from preferences import PreferenceDatabase
from reward_model import RewardModel, RMTrainMetrics, bradley_terry_prob
from train import (
    evaluate_policy,
    generate_diverse_trajectories,
    pretrain,
    rlhf_train,
    select_preference_pairs,
    train_rm,
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
    render_welcome,
    sparkline,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PRETRAIN_EPISODES = 150
PRETRAIN_BATCH = 10
PRETRAIN_ENTROPY_COEFF = 0.1
NUM_PREFERENCE_PAIRS = 30
RM_EPOCHS = 100
RLHF_EPISODES = 100
RLHF_BATCH = 10
KL_COEFF = 0.2
TRAJ_POOL_SIZE = 40
TRAJ_TEMPERATURE = 1.5

console = Console()

# Capture mode: set by --capture, controls auto-pause/auto-preference
CAPTURE_MODE = False
CAPTURE_DATA: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pause(msg: str = "[dim]Press Enter to continue...[/dim]") -> None:
    """Block until the user presses Enter (skipped in capture mode)."""
    if CAPTURE_MODE:
        return
    console.print()
    input(msg)


def auto_preference(traj_a: Trajectory, traj_b: Trajectory) -> str:
    """Mixed/natural auto-preference: prefer shorter paths with more pickups.

    Score = -length + 3 * num_pickups_collected + 1 * reached_goal * 5
    Ties → skip.
    """
    def score(t: Trajectory) -> float:
        return (-t.length
                + 3 * len(t.pickups_collected)
                + 5 * int(t.reached_goal)
                - 0.5 * t.num_turns)

    sa, sb = score(traj_a), score(traj_b)
    if sa > sb + 0.5:
        return "1"
    elif sb > sa + 0.5:
        return "2"
    return "s"


def _traj_to_dict(traj: Trajectory) -> dict:
    """Serialize a Trajectory to a JSON-safe dict."""
    return {
        "positions": traj.positions,
        "actions": traj.actions,
        "rewards": traj.rewards,
        "done": traj.done,
        "reached_goal": traj.reached_goal,
        "pickups_collected": traj.pickups_collected,
        "length": traj.length,
        "num_turns": traj.num_turns,
        "unique_cells": traj.unique_cells,
        "pickup_reward": traj.pickup_reward,
        "total_reward": traj.total_reward,
    }


# ---------------------------------------------------------------------------
# Phase 1: Pre-training
# ---------------------------------------------------------------------------
def run_phase1(env: GridWorld, policy: PolicyNetwork) -> None:
    """Pre-train the agent with shaped rewards, showing live progress."""
    console.print(render_phase_header(1))
    console.print()

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        def callback(metrics: EpisodeMetrics):
            time.sleep(0.3)

            # Use the actual stochastic training trajectory
            traj = metrics.latest_trajectory

            # --- Compact live view: grid + metrics only ---
            grid = render_grid(
                env,
                path=traj.positions if traj else None,
                title=f"Latest Path (ep {metrics.episode})",
                show_stats=True,
                traj=traj,
            )

            goal_rate = (
                np.mean(metrics.goal_rate_history[-20:])
                if metrics.goal_rate_history else 0
            )
            metrics_dict = {
                "Episode": (
                    f"{metrics.episode} / {PRETRAIN_EPISODES}",
                    "",
                ),
                "Avg Reward": (
                    f"{metrics.total_reward:.2f}",
                    sparkline(metrics.reward_history),
                ),
                "Goal Rate": (
                    f"{goal_rate:.0%}",
                    sparkline(metrics.goal_rate_history),
                ),
                "Avg Steps": (
                    f"{metrics.steps}",
                    sparkline(metrics.steps_history),
                ),
                "Policy Loss": (
                    f"{metrics.ppo.policy_loss:.4f}" if metrics.ppo else "—",
                    sparkline(metrics.policy_loss_history),
                ),
                "Value Loss": (
                    f"{metrics.ppo.value_loss:.4f}" if metrics.ppo else "—",
                    sparkline(metrics.value_loss_history),
                ),
                "Entropy": (
                    f"{metrics.ppo.entropy:.3f}" if metrics.ppo else "—",
                    sparkline(metrics.entropy_history),
                ),
            }
            metrics_panel = render_metrics_table(metrics_dict)

            top_row = Columns([grid, metrics_panel], equal=True, expand=True)
            live.update(Group(top_row))

        pretrain(
            env, policy,
            episodes=PRETRAIN_EPISODES,
            batch_episodes=PRETRAIN_BATCH,
            entropy_coeff=PRETRAIN_ENTROPY_COEFF,
            callback=callback,
        )

    console.print(
        Panel("[bold green]Pre-training complete![/]", border_style="green")
    )

    # --- Post-phase summary: detailed views ---
    pause("[dim]Press Enter to see detailed policy views...[/dim]")
    console.clear()
    console.print(Panel("[bold]Phase 1 Summary — Learned Policy[/]", border_style="green"))
    console.print()

    # Policy arrows (wall-masked)
    policy_map = policy.policy_heatmap(env.size, env=env)
    console.print(render_policy_arrows(policy_map, env, title="Current Policy"))

    # Value heatmap
    val_map = policy.value_heatmap(env.size)
    console.print(render_heatmap(val_map, env, title="Value V(s)"))

    # NN forward pass at centre
    obs = np.array([0.5, 0.5], dtype=np.float32)
    nn_info = policy.forward_with_activations(obs)
    console.print(render_nn_forward(nn_info))
    console.print()

    # Capture Phase 1 data
    if CAPTURE_MODE:
        # Find a good representative trajectory (deterministic rollout)
        demo_traj = env.rollout(
            lambda o: policy.get_action(o, deterministic=True)
        )
        CAPTURE_DATA["phase1"] = {
            "demo_traj": _traj_to_dict(demo_traj),
            "policy_map": policy_map.tolist(),
            "val_map": val_map.tolist(),
        }


# ---------------------------------------------------------------------------
# Phase 2: Preference collection
# ---------------------------------------------------------------------------
def run_phase2(
    env: GridWorld,
    policy: PolicyNetwork,
    pref_db: PreferenceDatabase,
) -> None:
    """Collect human preferences by showing path pairs."""
    console.print(render_phase_header(2))
    console.print()

    # Generate a diverse pool of trajectories
    console.print("[dim]Generating trajectory pool...[/]")
    pool = generate_diverse_trajectories(
        env, policy, n=TRAJ_POOL_SIZE, temperature=TRAJ_TEMPERATURE
    )
    pairs = select_preference_pairs(pool, n_pairs=NUM_PREFERENCE_PAIRS)

    for i, (traj_a, traj_b) in enumerate(pairs):
        console.clear()
        console.print(render_phase_header(2))
        console.print()
        console.print(
            render_preference_pair(env, traj_a, traj_b, i, NUM_PREFERENCE_PAIRS)
        )

        # Show preference patterns so far
        if len(pref_db) > 0:
            patterns = pref_db.preference_patterns()
            pattern_text = Text()
            pattern_text.append("\nYou tend to prefer: ", style="bold")
            pattern_text.append(
                f"shorter paths ({patterns.get('shorter_paths', '?')}), "
                f"fewer turns ({patterns.get('fewer_turns', '?')})",
                style="cyan",
            )
            console.print(pattern_text)

        console.print()

        if CAPTURE_MODE:
            choice = auto_preference(traj_a, traj_b)
            console.print(f"[dim]Auto-preference: {choice}[/]")
        else:
            choice = Prompt.ask(
                "Which path do you prefer?",
                choices=["1", "2", "s"],
                default="1",
            )

        if choice == "1":
            pref_db.add(traj_a, traj_b, 0.0)  # A preferred
        elif choice == "2":
            pref_db.add(traj_a, traj_b, 1.0)  # B preferred
        else:
            pref_db.add(traj_a, traj_b, 0.5)  # skip / tie

        # Capture a representative pair (around the middle, pair ~12)
        if CAPTURE_MODE and i == min(12, len(pairs) - 1):
            CAPTURE_DATA["phase2"] = {
                "traj_a": _traj_to_dict(traj_a),
                "traj_b": _traj_to_dict(traj_b),
                "pair_num": i,
                "total_pairs": NUM_PREFERENCE_PAIRS,
            }

    console.print()
    counts = pref_db.count_by_preference()
    console.print(
        Panel(
            f"[bold]Collected {len(pref_db)} preferences![/]\n"
            f"A preferred: {counts['A']}  |  B preferred: {counts['B']}  |  "
            f"Ties: {counts['tie']}",
            border_style="yellow",
            title="Preference Collection Complete",
        )
    )
    console.print()


# ---------------------------------------------------------------------------
# Phase 3: Reward model training
# ---------------------------------------------------------------------------
def run_phase3(
    rm: RewardModel,
    pref_db: PreferenceDatabase,
    env: GridWorld,
) -> None:
    """Train the reward model and show live metrics."""
    console.print(render_phase_header(3))
    console.print(render_rm_architecture())
    console.print()

    # Track last metrics + last spot check for capture
    _last_rm_metrics = {}
    _last_spot_check = {}

    with Live(console=console, refresh_per_second=8, transient=False) as live:
        def callback(metrics: RMTrainMetrics):
            # RM reward heatmap
            heatmap_data = rm.reward_heatmap(env.size)
            heatmap = render_heatmap(
                heatmap_data, env,
                title="RM Learned Rewards",
                border_style="cyan",
            )

            # Training metrics
            metrics_dict = {
                "Epoch": (f"{metrics.epoch} / {RM_EPOCHS}", ""),
                "Loss": (
                    f"{metrics.loss:.4f}",
                    sparkline(metrics.loss_history),
                ),
                "Accuracy": (
                    f"{metrics.accuracy:.1%}",
                    sparkline(metrics.accuracy_history),
                ),
            }
            metrics_panel = render_metrics_table(metrics_dict, title="RM Training")

            # Spot check on a random preference
            if len(pref_db) > 0 and metrics.epoch % 10 == 0:
                idx = np.random.randint(len(pref_db))
                pair = pref_db[idx]
                sa = torch.tensor(pair.traj_a.state_tensor(), dtype=torch.float32)
                sb = torch.tensor(pair.traj_b.state_tensor(), dtype=torch.float32)
                with torch.no_grad():
                    r_a = rm.trajectory_reward(sa).item()
                    r_b = rm.trajectory_reward(sb).item()
                prob_a = bradley_terry_prob(rm, sa, sb)
                spot = render_rm_spot_check(idx, r_a, r_b, prob_a, pair.label)
                _last_spot_check.update({
                    "pair_idx": int(idx), "r_a": r_a, "r_b": r_b,
                    "prob_a": prob_a, "human_label": pair.label,
                })
            else:
                spot = Text("")

            row = Columns([heatmap, metrics_panel], equal=True, expand=True)
            live.update(Group(row, spot))

            # Store for capture
            _last_rm_metrics.update({
                "loss_history": [float(x) for x in metrics.loss_history],
                "accuracy_history": [float(x) for x in metrics.accuracy_history],
                "final_loss": float(metrics.loss),
                "final_accuracy": float(metrics.accuracy),
                "rm_heatmap": heatmap_data.tolist(),
            })

        train_rm(rm, pref_db, epochs=RM_EPOCHS, callback=callback)

    console.print(
        Panel("[bold cyan]Reward model training complete![/]", border_style="cyan")
    )
    console.print()

    if CAPTURE_MODE:
        CAPTURE_DATA["phase3"] = {
            **_last_rm_metrics,
            "spot_check": _last_spot_check,
        }


# ---------------------------------------------------------------------------
# Phase 4: RLHF / PPO
# ---------------------------------------------------------------------------
def run_phase4(
    env: GridWorld,
    policy: PolicyNetwork,
    rm: RewardModel,
    pretrained_policy: PolicyNetwork,
) -> None:
    """RLHF fine-tuning with live visualization."""
    console.print(render_phase_header(4))
    console.print()

    # Get pre-trained reference trajectory
    pretrained_traj = env.rollout(
        lambda obs: pretrained_policy.get_action(obs, deterministic=True)
    )

    _last_rlhf_metrics = {}

    with Live(console=console, refresh_per_second=6, transient=False) as live:
        def callback(metrics: EpisodeMetrics):
            time.sleep(0.3)

            # Use the actual stochastic training trajectory
            rlhf_traj = metrics.latest_trajectory

            # --- Compact live view: side-by-side paths + metrics ---
            grid_pt = render_grid(
                env, path=pretrained_traj.positions,
                path_style="dim cyan",
                title="Pre-trained Path",
                show_stats=True, traj=pretrained_traj,
            )
            grid_rl = render_grid(
                env,
                path=rlhf_traj.positions if rlhf_traj else None,
                path_style="bold bright_green",
                title="RLHF Path (evolving)",
                show_stats=True, traj=rlhf_traj,
            )
            grids_row = Columns([grid_pt, grid_rl], equal=True, expand=True)

            goal_rate = (
                np.mean(metrics.goal_rate_history[-20:])
                if metrics.goal_rate_history else 0
            )
            metrics_dict = {
                "Episode": (f"{metrics.episode} / {RLHF_EPISODES}", ""),
                "RM Score": (
                    f"{metrics.rm_score:.2f}",
                    sparkline(metrics.rm_score_history),
                ),
                "Shaped Reward": (
                    f"{metrics.shaped_reward:.2f}",
                    sparkline(metrics.reward_history),
                ),
                "KL Penalty": (f"{metrics.kl_penalty:.3f}", ""),
                "Net Reward": (f"{metrics.net_reward:.2f}", ""),
                "Policy Loss": (
                    f"{metrics.ppo.policy_loss:.4f}" if metrics.ppo else "—",
                    sparkline(metrics.policy_loss_history),
                ),
                "Goal Rate": (
                    f"{goal_rate:.0%}",
                    sparkline(metrics.goal_rate_history),
                ),
            }
            metrics_panel = render_metrics_table(metrics_dict, title="RLHF Metrics")

            # KL dynamics
            kl_metrics = {
                "\u03b2 (KL coeff)": (f"{KL_COEFF:.2f}", ""),
                "KL(\u03c0_RL \u2016 \u03c0_PT)": (
                    f"{metrics.kl_history[-1]:.4f}" if metrics.kl_history else "—",
                    sparkline(metrics.kl_history),
                ),
                "RM score trend": (
                    "",
                    sparkline(metrics.rm_score_history),
                ),
            }
            kl_panel = render_metrics_table(kl_metrics, title="KL & Reward Dynamics")

            mid_row = Columns([metrics_panel, kl_panel], equal=True, expand=True)

            live.update(Group(grids_row, mid_row))

            # Store for capture
            _last_rlhf_metrics.update({
                "reward_history": [float(x) for x in metrics.rm_score_history],
                "kl_history": [float(x) for x in metrics.kl_history],
                "goal_rate_history": [float(x) for x in metrics.goal_rate_history],
                "entropy_history": [float(x) for x in (metrics.entropy_history or [])],
                "policy_loss_history": [float(x) for x in metrics.policy_loss_history],
            })

        rlhf_train(
            env, policy, rm, pretrained_policy,
            episodes=RLHF_EPISODES,
            kl_coeff=KL_COEFF,
            batch_episodes=RLHF_BATCH,
            callback=callback,
        )

    console.print(
        Panel("[bold magenta]RLHF training complete![/]", border_style="magenta")
    )

    # --- Post-phase summary: detailed views ---
    pause("[dim]Press Enter to see detailed RLHF analysis...[/dim]")
    console.clear()
    console.print(Panel("[bold]Phase 4 Summary — RLHF Analysis[/]", border_style="magenta"))
    console.print()

    # NN forward pass with reference comparison
    obs = np.array([0.5, 0.5], dtype=np.float32)
    nn_info = policy.forward_with_activations(obs)
    ref_probs = pretrained_policy.get_probs(obs)
    console.print(render_nn_forward(nn_info, ref_probs=ref_probs))

    # Policy comparison
    pt_map = pretrained_policy.policy_heatmap(env.size, env=env)
    rl_map = policy.policy_heatmap(env.size, env=env)
    console.print(render_policy_comparison(pt_map, rl_map, env))
    console.print()

    # Capture Phase 4 data
    if CAPTURE_MODE:
        val_map = policy.value_heatmap(env.size)
        CAPTURE_DATA["phase4"] = {
            **_last_rlhf_metrics,
            "val_map": val_map.tolist(),
            "rlhf_policy_map": rl_map.tolist(),
            "pretrained_traj": _traj_to_dict(pretrained_traj),
        }


# ---------------------------------------------------------------------------
# Phase 5: Conclusion
# ---------------------------------------------------------------------------
def run_conclusion(
    env: GridWorld,
    policy: PolicyNetwork,
    pretrained_policy: PolicyNetwork,
) -> None:
    """Show final results and LLM parallel mapping."""
    console.print()
    console.print(Panel("[bold]Results", border_style="bright_green"))

    # Evaluate both policies
    pt_stats = evaluate_policy(env, pretrained_policy, n_episodes=50)
    rl_stats = evaluate_policy(env, policy, n_episodes=50)

    console.print(render_results_summary(pt_stats, rl_stats))

    # Final policy comparison
    pt_map = pretrained_policy.policy_heatmap(env.size, env=env)
    rl_map = policy.policy_heatmap(env.size, env=env)
    console.print(render_policy_comparison(pt_map, rl_map, env))

    # Final paths side-by-side
    pt_traj = env.rollout(
        lambda obs: pretrained_policy.get_action(obs, deterministic=True)
    )
    rl_traj = env.rollout(
        lambda obs: policy.get_action(obs, deterministic=True)
    )
    grid_pt = render_grid(
        env, path=pt_traj.positions, path_style="dim cyan",
        title="Pre-trained Final Path", show_stats=True, traj=pt_traj,
    )
    grid_rl = render_grid(
        env, path=rl_traj.positions, path_style="bold bright_green",
        title="RLHF Final Path", show_stats=True, traj=rl_traj,
    )
    console.print(Columns([grid_pt, grid_rl], equal=True, expand=True))

    # KL divergence
    kl = compute_policy_kl(policy, pretrained_policy, grid_size=env.size)
    console.print(
        Panel(
            f"Final KL(\u03c0_RLHF \u2016 \u03c0_pretrained) = {kl:.4f}\n"
            f"This is the 'alignment tax' \u2014 how far the model moved from its\n"
            f"pre-trained behaviour to accommodate your preferences.",
            title="Alignment Tax",
            border_style="yellow",
        )
    )

    # LLM parallel table
    console.print(render_llm_parallel_table())

    # Capture conclusion data
    if CAPTURE_MODE:
        CAPTURE_DATA["conclusion"] = {
            "pt_stats": pt_stats,
            "rl_stats": rl_stats,
            "pt_traj": _traj_to_dict(pt_traj),
            "rl_traj": _traj_to_dict(rl_traj),
            "kl": float(kl),
            "pt_map": pt_map.tolist(),
            "rl_map": rl_map.tolist(),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global CAPTURE_MODE

    parser = argparse.ArgumentParser(description="Path-Finding Preference Game")
    parser.add_argument(
        "--capture", type=str, default=None,
        help="Path to write captured visualization data (JSON) for screenshots",
    )
    args = parser.parse_args()

    if args.capture:
        CAPTURE_MODE = True
        console.print("[bold yellow]Capture mode enabled[/] — auto-preferences, no pauses\n")

    # Seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialise components
    env = GridWorld()
    policy = PolicyNetwork()
    rm = RewardModel()
    pref_db = PreferenceDatabase()

    # Welcome
    console.print(render_welcome())
    if not CAPTURE_MODE:
        input()  # wait for Enter

    # Phase 1: Pre-training
    console.clear()
    run_phase1(env, policy)

    # Snapshot pre-trained policy (frozen reference for KL)
    pretrained_policy = PolicyNetwork()
    pretrained_policy.load_state_dict(copy.deepcopy(policy.state_dict()))
    pretrained_policy.eval()

    pause("[dim]Press Enter for Phase 2 (Preference Collection)...[/dim]")

    # Phase 2: Preference collection
    console.clear()
    run_phase2(env, policy, pref_db)

    pause("[dim]Press Enter for Phase 3 (Reward Model Training)...[/dim]")

    # Phase 3: RM training
    console.clear()
    run_phase3(rm, pref_db, env)

    pause("[dim]Press Enter for Phase 4 (RLHF / PPO)...[/dim]")

    # Phase 4: RLHF
    console.clear()
    run_phase4(env, policy, rm, pretrained_policy)

    pause("[dim]Press Enter for Results...[/dim]")

    # Conclusion
    console.clear()
    run_conclusion(env, policy, pretrained_policy)

    console.print()
    console.print(
        "[bold bright_cyan]Thank you for playing the Path-Finding "
        "Preference Game![/]"
    )

    # Dump captured data
    if CAPTURE_MODE and args.capture:
        out_path = Path(args.capture)
        out_path.write_text(json.dumps(CAPTURE_DATA, indent=2))
        console.print(f"\n[bold green]Captured data written to {out_path}[/]")


if __name__ == "__main__":
    main()
