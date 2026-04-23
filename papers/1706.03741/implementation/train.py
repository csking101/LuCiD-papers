# Training Loop
#
# The main RLHF training loop that orchestrates the 3 processes:
#
#   Process 1: Policy interacts with environment, collects trajectories
#              Policy parameters updated to maximize predicted reward r_hat
#
#   Process 2: Segments are selected from trajectories, paired, and
#              sent for comparison (synthetic oracle or human)
#
#   Process 3: Reward model r_hat is updated via supervised learning
#              on the preference database
#
# In the paper these run asynchronously. Here we run them sequentially
# per iteration for simplicity.
#
# This version includes:
#   - Rich live dashboard with real-time metrics
#   - Warm-up phase (random exploration to seed preference DB)
#   - Entropy regularization (prevents policy collapse)
#   - DP baseline comparison (policy accuracy, reward MSE, true return)
#   - Interactive human feedback rounds at configurable intervals

import sys
import os
import time

# Allow running as a script from within the implementation directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.live import Live
from rich.text import Text
from rich.layout import Layout
from rich import box

from env import Environment
from policy import Policy
from trajectory import collect_trajectories, slice_segments, pair_segments
from preferences import PreferenceDB, SyntheticOracle
from reward_model import RewardEnsemble
from optimal import (
    compute_baseline, compare_policy_accuracy, compare_reward_mse,
    evaluate_true_return, render_policy_grid, get_learned_policy_dict,
)
from human_feedback import human_feedback_round
from config import (
    GRID_HEIGHT, GRID_WIDTH, TERMINAL_OBJECTS_PLACEMENT, WALL_PLACEMENT,
    MAX_STEPS_PER_EPISODE, STEP_PENALTY, SEGMENT_LENGTH, MIN_SEGMENT_LENGTH,
    TRAJECTORIES_PER_ITER, PAIRS_PER_ITER, PREFERENCE_DB_MAX,
    REWARD_ENSEMBLE_SIZE, REWARD_HIDDEN_SIZE, REWARD_LR, REWARD_EPOCHS,
    REWARD_BATCH_SIZE, HUMAN_ERROR_RATE, POLICY_HIDDEN_SIZE, POLICY_LR,
    POLICY_EPOCHS, GAMMA, ENTROPY_BETA_START, ENTROPY_BETA_END,
    EXPLORATION_EPSILON, GRAD_CLIP_NORM, POLICY_UPDATE_INTERVAL,
    WARMUP_TRAJECTORIES, EVAL_INTERVAL, HUMAN_FEEDBACK_INTERVAL,
    HUMAN_PAIRS_PER_ROUND, NUM_ITERATIONS, DEVICE,
)

console = Console()


# ─── Factory functions ────────────────────────────────────────────────

def make_environment():
    return Environment(
        height=GRID_HEIGHT, width=GRID_WIDTH,
        terminal_objects_placement=TERMINAL_OBJECTS_PLACEMENT,
        wall_placement=WALL_PLACEMENT, max_steps=MAX_STEPS_PER_EPISODE,
    )


def make_policy():
    return Policy(
        policy_hidden_size=POLICY_HIDDEN_SIZE, policy_lr=POLICY_LR,
        policy_epochs=POLICY_EPOCHS, gamma=GAMMA,
        entropy_beta=ENTROPY_BETA_START, grad_clip_norm=GRAD_CLIP_NORM,
        device=DEVICE,
    )


def make_reward_ensemble():
    return RewardEnsemble(
        n_predictors=REWARD_ENSEMBLE_SIZE, hidden_size=REWARD_HIDDEN_SIZE,
        lr=REWARD_LR, human_error_rate=HUMAN_ERROR_RATE, device=DEVICE,
    )


def make_preference_db():
    return PreferenceDB(max_size=PREFERENCE_DB_MAX)


def make_oracle():
    return SyntheticOracle(
        terminal_objects_placement=TERMINAL_OBJECTS_PLACEMENT,
        human_error_rate=HUMAN_ERROR_RATE, step_penalty=STEP_PENALTY,
    )


# ─── Evaluation ──────────────────────────────────────────────────────

def evaluate_policy(env, policy, n_episodes: int = 10) -> dict:
    total_steps = 0
    goals = 0
    traps = 0
    timeouts = 0

    for _ in range(n_episodes):
        obs = env.reset()
        info = {}
        while not env.done:
            action, _ = policy.get_action(obs)
            obs, done, info = env.step(action)

        total_steps += env.step_count
        if info.get("event") == "terminal":
            if info.get("terminal_value", 0) > 0:
                goals += 1
            else:
                traps += 1
        elif info.get("event") == "max_steps":
            timeouts += 1

    return {
        "avg_steps": total_steps / max(n_episodes, 1),
        "goal_rate": goals / max(n_episodes, 1),
        "trap_rate": traps / max(n_episodes, 1),
        "timeout_rate": timeouts / max(n_episodes, 1),
    }


# ─── Rich dashboard builder ─────────────────────────────────────────

def _color_delta(current, previous, higher_is_better=True):
    """Return a colored arrow string showing direction of change."""
    if previous is None:
        return ""
    diff = current - previous
    if abs(diff) < 1e-4:
        return "[dim]-[/]"
    if (diff > 0) == higher_is_better:
        return f"[green]+{abs(diff):.2f}[/]"
    else:
        return f"[red]-{abs(diff):.2f}[/]"


def _pct(val):
    return f"{val:.0%}"


def _fmt(val, decimals=4):
    return f"{val:.{decimals}f}"


def build_dashboard(metrics_history: list, dp_metrics: dict,
                    env, policy, pi_star: dict,
                    phase: str = "training") -> Panel:
    """Build the full Rich dashboard panel."""

    # ── Iteration metrics table (last 12 rows) ──
    iter_table = Table(
        title=None, box=box.SIMPLE_HEAVY, show_header=True,
        header_style="bold", pad_edge=False, collapse_padding=True,
    )
    iter_table.add_column("Iter", style="dim", width=5, justify="right")
    iter_table.add_column("R_Loss", width=8, justify="right")
    iter_table.add_column("P_Loss", width=8, justify="right")
    iter_table.add_column("Entropy", width=8, justify="right")
    iter_table.add_column("Prefs", width=6, justify="right")
    iter_table.add_column("Segs", width=5, justify="right")
    iter_table.add_column("Steps", width=6, justify="right")
    iter_table.add_column("Goal", width=6, justify="right")
    iter_table.add_column("Trap", width=6, justify="right")
    iter_table.add_column("T/O", width=5, justify="right")

    display_rows = metrics_history[-12:]
    for m in display_rows:
        goal_style = "green" if m["goal_rate"] > 0.5 else ("yellow" if m["goal_rate"] > 0.1 else "red")
        trap_style = "green" if m["trap_rate"] < 0.1 else ("yellow" if m["trap_rate"] < 0.3 else "red")

        iter_table.add_row(
            str(m["iteration"]),
            _fmt(m["reward_loss"]),
            _fmt(m["policy_loss"]),
            _fmt(m.get("entropy", 0), 3),
            str(m["preference_db_size"]),
            str(m.get("segments_collected", 0)),
            _fmt(m["avg_steps"], 1),
            f"[{goal_style}]{_pct(m['goal_rate'])}[/]",
            f"[{trap_style}]{_pct(m['trap_rate'])}[/]",
            _pct(m["timeout_rate"]),
        )

    # ── DP comparison panel ──
    if dp_metrics:
        acc = dp_metrics["policy_accuracy"]
        mse = dp_metrics["reward_mse"]
        true_ret = dp_metrics["true_return"]
        opt_ret = dp_metrics["optimal_return"]
        gap = opt_ret - true_ret

        acc_bar_len = int(acc * 20)
        acc_bar = "[green]" + "█" * acc_bar_len + "[/][dim]" + "░" * (20 - acc_bar_len) + "[/]"

        prev_acc = dp_metrics.get("prev_accuracy")
        acc_delta = _color_delta(acc, prev_acc, higher_is_better=True)

        dp_text = (
            f"  Policy Accuracy: [bold cyan]{_pct(acc)}[/]  {acc_bar}  {acc_delta}\n"
            f"  Reward MSE:      [bold]{_fmt(mse, 3)}[/]\n"
            f"  True Return:     [bold]{_fmt(true_ret, 1)}[/]  "
            f"(Optimal: [green]{_fmt(opt_ret, 1)}[/]  Gap: [yellow]{_fmt(gap, 1)}[/])"
        )
        dp_panel = Panel(dp_text, title="[bold]vs Optimal (DP)[/]",
                         border_style="cyan", padding=(0, 1))
    else:
        dp_panel = Panel("[dim]Waiting for first evaluation...[/]",
                         title="[bold]vs Optimal (DP)[/]",
                         border_style="dim", padding=(0, 1))

    # ── Side-by-side policy grids ──
    learned_dict = get_learned_policy_dict(policy, env)
    learned_grid = render_policy_grid(env, learned_dict)
    optimal_grid = render_policy_grid(env, pi_star)

    grid_left = Panel(learned_grid, title="[bold blue]Learned[/]",
                      border_style="blue", width=22)
    grid_right = Panel(optimal_grid, title="[bold green]Optimal[/]",
                       border_style="green", width=22)
    grids = Columns([grid_left, grid_right], padding=1)

    # ── Assemble ──
    phase_color = {"warmup": "yellow", "training": "cyan",
                   "human_feedback": "magenta"}.get(phase, "white")
    title = f"[bold {phase_color}]Cat Grid World — RLHF Training  [{phase.upper()}][/]"

    # Build as a group of renderables
    from rich.console import Group
    dashboard = Group(iter_table, dp_panel, grids)

    return Panel(dashboard, title=title, border_style=phase_color,
                 padding=(0, 1))


# ─── Warm-up phase ───────────────────────────────────────────────────

def warmup(env, policy, reward_ensemble, preference_db, oracle):
    """
    Seed the preference DB with random exploration before training starts.
    Collect trajectories with full exploration (epsilon=1.0), slice, pair,
    get oracle preferences, and train the reward model.
    """
    console.print("\n[bold yellow]Warm-up phase:[/] collecting random "
                  f"trajectories ({WARMUP_TRAJECTORIES} episodes)...")

    # Collect with full random exploration
    trajectories = collect_trajectories(
        env, policy, WARMUP_TRAJECTORIES, exploration_epsilon=1.0)

    # Slice into segments
    all_segments = []
    for traj in trajectories:
        all_segments.extend(slice_segments(traj, SEGMENT_LENGTH, MIN_SEGMENT_LENGTH))

    console.print(f"  Segments extracted: {len(all_segments)}")

    # Pair and get preferences
    n_warmup_pairs = min(PAIRS_PER_ITER * 5, len(all_segments) * (len(all_segments) - 1) // 2)
    pairs = pair_segments(all_segments, n_warmup_pairs)

    new_prefs = 0
    for seg1, seg2 in pairs:
        mu = oracle.compare(seg1, seg2)
        if mu is not None:
            preference_db.add(seg1, seg2, mu)
            new_prefs += 1

    console.print(f"  Preferences collected: {new_prefs}")

    # Train reward model on initial preferences
    if len(preference_db) >= REWARD_BATCH_SIZE:
        loss = reward_ensemble.train_on_preferences(
            preference_db, epochs=REWARD_EPOCHS * 2, batch_size=REWARD_BATCH_SIZE)
        console.print(f"  Reward model trained: loss={loss:.4f}")

    console.print(f"  Preference DB size: {len(preference_db)}")
    console.print("[bold green]Warm-up complete.[/]\n")

    return all_segments


# ─── Main training loop ─────────────────────────────────────────────

def main():
    env = make_environment()
    policy = make_policy()
    reward_ensemble = make_reward_ensemble()
    preference_db = make_preference_db()
    oracle = make_oracle()

    # ── Compute DP baseline ──
    console.print("[bold]Computing optimal baseline (value iteration)...[/]")
    V_star, Q_star, pi_star, optimal_return = compute_baseline(
        env, STEP_PENALTY, GAMMA)
    console.print(f"  Optimal return from (0,0): [green]{optimal_return:.1f}[/]")
    console.print(f"  Optimal policy grid:")
    console.print(Panel(render_policy_grid(env, pi_star),
                        title="[bold green]Optimal Policy[/]",
                        border_style="green", width=22))

    # ── Warm-up ──
    warmup_segments = warmup(env, policy, reward_ensemble, preference_db, oracle)

    # ── Initial evaluation ──
    eval_metrics = evaluate_policy(env, policy)
    init_accuracy = compare_policy_accuracy(policy, pi_star, env)
    console.print(f"[Init] Random policy — "
                  f"goal={eval_metrics['goal_rate']:.0%}, "
                  f"trap={eval_metrics['trap_rate']:.0%}, "
                  f"accuracy={init_accuracy:.0%}")

    metrics_history = []
    dp_metrics = {}
    prev_accuracy = None
    last_all_segments = warmup_segments  # keep recent segments for human feedback

    # ── Main loop with Rich Live display ──
    with Live(console=console, refresh_per_second=2, transient=False) as live:
        for iteration in range(1, NUM_ITERATIONS + 1):

            # ── Human feedback round? ──
            if (HUMAN_FEEDBACK_INTERVAL > 0 and
                    iteration > 1 and
                    (iteration - 1) % HUMAN_FEEDBACK_INTERVAL == 0 and
                    len(last_all_segments) >= 2):

                live.stop()
                human_prefs = human_feedback_round(
                    last_all_segments, env,
                    n_pairs=HUMAN_PAIRS_PER_ROUND,
                    iteration=iteration,
                    policy=policy,
                    policy_accuracy=dp_metrics.get("policy_accuracy"),
                    goal_rate=eval_metrics.get("goal_rate"),
                )

                # Add human preferences to DB
                for seg1, seg2, mu in human_prefs:
                    preference_db.add(seg1, seg2, mu)

                # Retrain reward model immediately on new preferences
                if human_prefs and len(preference_db) >= REWARD_BATCH_SIZE:
                    reward_ensemble.train_on_preferences(
                        preference_db, epochs=REWARD_EPOCHS,
                        batch_size=REWARD_BATCH_SIZE)
                    console.print("[bold green]Reward model updated with "
                                  f"human feedback ({len(human_prefs)} prefs)[/]")

                live.start()

            # ── Collect trajectories ──
            # EXPLORATION trajectories (with epsilon) → for segments & preferences
            # These have diverse behavior for informative preference comparisons
            explore_trajs = collect_trajectories(
                env, policy, TRAJECTORIES_PER_ITER // 2,
                exploration_epsilon=EXPLORATION_EPSILON)

            # ON-POLICY trajectories (no epsilon) → for policy gradient update
            # REINFORCE requires on-policy data for valid gradients
            onpolicy_trajs = collect_trajectories(
                env, policy, TRAJECTORIES_PER_ITER // 2,
                exploration_epsilon=0.0)

            all_trajs = explore_trajs + onpolicy_trajs

            # ── Process 2: Get preferences (from exploration trajectories) ──
            all_segments = []
            for traj in all_trajs:
                all_segments.extend(
                    slice_segments(traj, SEGMENT_LENGTH, MIN_SEGMENT_LENGTH))
            last_all_segments = all_segments

            use_active = iteration > 3 and len(preference_db) > REWARD_BATCH_SIZE
            pairs = pair_segments(
                all_segments, PAIRS_PER_ITER,
                reward_ensemble=reward_ensemble if use_active else None)

            new_prefs = 0
            for seg1, seg2 in pairs:
                mu = oracle.compare(seg1, seg2)
                if mu is not None:
                    preference_db.add(seg1, seg2, mu)
                    new_prefs += 1

            # ── Process 3: Update reward model ──
            reward_loss = 0.0
            if len(preference_db) >= REWARD_BATCH_SIZE:
                reward_loss = reward_ensemble.train_on_preferences(
                    preference_db, epochs=REWARD_EPOCHS,
                    batch_size=REWARD_BATCH_SIZE)

            # ── Update policy (on-policy trajectories, every K iterations) ──
            policy_loss = 0.0
            entropy = 0.0

            # Decay entropy bonus: high early (explore), low late (exploit)
            progress = iteration / NUM_ITERATIONS
            current_entropy_beta = (ENTROPY_BETA_START * (1 - progress) +
                                    ENTROPY_BETA_END * progress)
            policy.entropy_beta = current_entropy_beta

            if (len(preference_db) > 0 and
                    iteration % POLICY_UPDATE_INTERVAL == 0):
                policy_loss, entropy = policy.update(
                    onpolicy_trajs, reward_ensemble)
            elif len(preference_db) > 0:
                # Still measure entropy even when not updating
                entropy = policy.get_avg_entropy(env)

            # ── Evaluate ──
            eval_metrics = evaluate_policy(env, policy)

            # ── DP comparison (every EVAL_INTERVAL) ──
            if iteration % EVAL_INTERVAL == 0 or iteration == 1:
                accuracy = compare_policy_accuracy(policy, pi_star, env)
                mse = compare_reward_mse(reward_ensemble, Q_star, env)
                true_return = evaluate_true_return(
                    env, policy, TERMINAL_OBJECTS_PLACEMENT,
                    STEP_PENALTY, GAMMA, n_episodes=20)

                dp_metrics = {
                    "policy_accuracy": accuracy,
                    "reward_mse": mse,
                    "true_return": true_return,
                    "optimal_return": optimal_return,
                    "prev_accuracy": prev_accuracy,
                }
                prev_accuracy = accuracy

            # ── Log ──
            iteration_metrics = {
                "iteration": iteration,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
                "entropy": entropy,
                "preference_db_size": len(preference_db),
                "new_preferences": new_prefs,
                "segments_collected": len(all_segments),
                "active_queries": use_active,
                **eval_metrics,
            }
            if dp_metrics:
                iteration_metrics["policy_accuracy"] = dp_metrics["policy_accuracy"]
                iteration_metrics["reward_mse"] = dp_metrics["reward_mse"]
                iteration_metrics["true_return"] = dp_metrics["true_return"]
            metrics_history.append(iteration_metrics)

            # ── Update dashboard ──
            dashboard = build_dashboard(
                metrics_history, dp_metrics, env, policy, pi_star,
                phase="training")
            live.update(dashboard)

    # ── Final evaluation ──
    console.print("\n")
    console.rule("[bold green]Training Complete[/]")

    final = evaluate_policy(env, policy, n_episodes=50)
    final_accuracy = compare_policy_accuracy(policy, pi_star, env)
    final_return = evaluate_true_return(
        env, policy, TERMINAL_OBJECTS_PLACEMENT, STEP_PENALTY, GAMMA, n_episodes=50)

    console.print(f"  Iterations:       {NUM_ITERATIONS}")
    console.print(f"  Preference DB:    {len(preference_db)} entries")
    console.print(f"  Avg steps:        {final['avg_steps']:.1f}")
    console.print(f"  Goal rate:        [green]{final['goal_rate']:.0%}[/]")
    console.print(f"  Trap rate:        [red]{final['trap_rate']:.0%}[/]")
    console.print(f"  Timeout rate:     {final['timeout_rate']:.0%}")
    console.print(f"  Policy accuracy:  [cyan]{final_accuracy:.0%}[/]")
    console.print(f"  True return:      {final_return:.1f}  "
                  f"(Optimal: [green]{optimal_return:.1f}[/])")

    # Side-by-side final grids
    learned_dict = get_learned_policy_dict(policy, env)
    learned_grid = render_policy_grid(env, learned_dict)
    optimal_grid = render_policy_grid(env, pi_star)
    console.print(Columns([
        Panel(learned_grid, title="[bold blue]Learned[/]",
              border_style="blue", width=22),
        Panel(optimal_grid, title="[bold green]Optimal[/]",
              border_style="green", width=22),
    ], padding=1))

    return metrics_history, policy, reward_ensemble, preference_db


if __name__ == "__main__":
    metrics_history, policy, reward_ensemble, preference_db = main()
