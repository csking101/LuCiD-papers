"""
Rich Terminal Visualizations
=============================

All rendering functions for the terminal UI — grids, heatmaps, sparklines,
neural network forward-pass views, metrics tables, and comparison panels.

Every function returns a Rich renderable (Text, Table, Panel, Columns, etc.)
that can be composed into the live display.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from env import (
    ACTION_ARROWS,
    COIN,
    EMPTY,
    GEM,
    NUM_ACTIONS,
    PICKUP_VALUES,
    SIZE,
    WALL,
    GridWorld,
    Trajectory,
)

# ---------------------------------------------------------------------------
# Unicode characters
# ---------------------------------------------------------------------------
SPARKLINE_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"  # ▁▂▃▄▅▆▇█
HEATMAP_CHARS = "\u2591\u2592\u2593\u2588"  # ░▒▓█
BLOCK_FULL = "\u2588\u2588"  # ██


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------
def sparkline(values: List[float], width: int = 20) -> str:
    """
    Render a list of floats as a Unicode sparkline.

    Parameters
    ----------
    values : list of float
    width : int
        Max characters to display (takes the last *width* values).

    Returns
    -------
    str : sparkline string, e.g. "▁▂▃▅▆▇█▇▆"
    """
    if not values:
        return ""
    vals = values[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo if hi - lo > 1e-8 else 1.0
    n = len(SPARKLINE_CHARS) - 1
    return "".join(SPARKLINE_CHARS[min(int((v - lo) / span * n), n)] for v in vals)


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------
def render_grid(
    env: GridWorld,
    path: Optional[List[Tuple[int, int]]] = None,
    path_style: str = "bold cyan",
    title: str = "Grid World",
    show_stats: bool = False,
    traj: Optional[Trajectory] = None,
) -> Panel:
    """
    Render the grid as a Rich Panel with colored cells.

    - Start ``S`` in green
    - Goal ``G`` in red
    - Walls ``██`` in bright_white
    - Coins ``¢ `` in bold yellow
    - Gems ``◆ `` in bold bright_magenta
    - Path ``● `` in *path_style*
    - Empty ``· `` in dim

    Pickups that were collected by *traj* are shown dimmed; uncollected
    pickups keep their bright colour.
    """
    path_set: Set[Tuple[int, int]] = set(path) if path else set()

    # Build set of positions where pickups were collected by this trajectory
    collected_set: Set[Tuple[int, int]] = set()
    if traj is not None:
        collected_set = {(r, c) for r, c, _ in traj.pickups_collected}

    text = Text()

    for r in range(env.size):
        for c in range(env.size):
            if (r, c) == env.start:
                text.append("S ", style="bold green")
            elif (r, c) == env.goal:
                text.append("G ", style="bold red")
            elif env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            elif (r, c) in path_set:
                text.append("\u25cf ", style=path_style)
            elif (r, c) in env.initial_pickups:
                ptype = env.initial_pickups[(r, c)]
                if (r, c) in collected_set:
                    # Collected — show dimmed
                    if ptype == COIN:
                        text.append("\u00a2 ", style="dim yellow")
                    else:
                        text.append("\u25c6 ", style="dim magenta")
                else:
                    # Still available
                    if ptype == COIN:
                        text.append("\u00a2 ", style="bold yellow")
                    else:
                        text.append("\u25c6 ", style="bold bright_magenta")
            else:
                text.append("\u00b7 ", style="dim")
        if r < env.size - 1:
            text.append("\n")

    subtitle = None
    if show_stats and traj is not None:
        n_collected = len(traj.pickups_collected)
        n_total = len(env.initial_pickups)
        bonus = traj.pickup_reward
        subtitle = (
            f"Steps: {traj.length}  Turns: {traj.num_turns}  "
            f"Unique: {traj.unique_cells}  "
            f"Pickups: {n_collected}/{n_total} (+{bonus:.1f})"
        )

    return Panel(text, title=title, subtitle=subtitle, border_style="blue")


# ---------------------------------------------------------------------------
# Policy arrow grid
# ---------------------------------------------------------------------------
def render_policy_arrows(
    policy_map: np.ndarray,
    env: GridWorld,
    title: str = "Current Policy",
) -> Panel:
    """
    Render greedy policy actions as arrows on the grid.

    Parameters
    ----------
    policy_map : ndarray (size, size) of action indices
    env : GridWorld (for wall positions)
    """
    text = Text()
    for r in range(env.size):
        for c in range(env.size):
            if (r, c) == env.start:
                text.append("S ", style="bold green")
            elif (r, c) == env.goal:
                text.append("G ", style="bold red")
            elif env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            else:
                arrow = ACTION_ARROWS[policy_map[r, c]]
                text.append(f"{arrow} ", style="bold yellow")
        if r < env.size - 1:
            text.append("\n")

    return Panel(text, title=title, border_style="yellow")


# ---------------------------------------------------------------------------
# Value / reward heatmap
# ---------------------------------------------------------------------------
def render_heatmap(
    values: np.ndarray,
    env: GridWorld,
    title: str = "Value Heatmap",
    border_style: str = "magenta",
) -> Panel:
    """
    Render a 2D array as a brightness heatmap on the grid.

    Uses ``░▒▓█`` characters scaled to the value range.
    """
    text = Text()
    lo, hi = values.min(), values.max()
    span = hi - lo if hi - lo > 1e-8 else 1.0
    n = len(HEATMAP_CHARS) - 1

    for r in range(env.size):
        for c in range(env.size):
            if env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            else:
                level = min(int((values[r, c] - lo) / span * n), n)
                char = HEATMAP_CHARS[level]
                # Color gradient: dim → bright green
                styles = ["dim green", "green", "bright_green", "bold bright_green"]
                text.append(f"{char}{char}", style=styles[level])
        if r < env.size - 1:
            text.append("\n")

    text.append(f"\n(min={lo:.2f}  max={hi:.2f})", style="dim")
    return Panel(text, title=title, border_style=border_style)


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------
def render_metrics_table(
    metrics: Dict[str, Tuple[str, str]],
    title: str = "Training Metrics",
) -> Panel:
    """
    Render a metrics dashboard.

    Parameters
    ----------
    metrics : dict mapping label → (value_str, sparkline_str)
    """
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Metric", style="bold", min_width=16)
    table.add_column("Value", min_width=10, justify="right")
    table.add_column("Trend", style="cyan", min_width=20)

    for label, (value, spark) in metrics.items():
        table.add_row(label, value, spark)

    return Panel(table, title=title, border_style="green")


# ---------------------------------------------------------------------------
# Neural network forward pass view
# ---------------------------------------------------------------------------
def render_nn_forward(
    info: Dict,
    title: str = "Neural Network Forward Pass",
    ref_probs: Optional[np.ndarray] = None,
    ref_label: str = "Pre-trained",
) -> Panel:
    """
    Render a detailed view of the NN's forward pass.

    Parameters
    ----------
    info : dict from PolicyNetwork.forward_with_activations()
        Keys: input, hidden_1, hidden_2, logits, probs, value
    ref_probs : optional ndarray of reference policy probs (for comparison)
    ref_label : label for the reference policy
    """
    text = Text()

    # Input
    inp = info["input"]
    text.append("Input: ", style="bold")
    text.append(f"[row={inp[0]:.2f}, col={inp[1]:.2f}]\n", style="cyan")

    # Hidden layers
    for i, key in enumerate(["hidden_1", "hidden_2"], 1):
        h = info[key]
        text.append(f"\u2500\u2192 Hidden\u2081" if i == 1 else f"\u2500\u2192 Hidden\u2082",
                     style="bold")
        text.append(f" ({len(h)} neurons, ReLU): ", style="dim")
        text.append(
            f"min={h.min():.2f}  mean={h.mean():.2f}  max={h.max():.2f}  ",
            style="white",
        )
        # Mini activation sparkline (sample 12 neurons)
        indices = np.linspace(0, len(h) - 1, min(12, len(h)), dtype=int)
        sampled = h[indices].tolist()
        text.append(sparkline(sampled), style="bright_cyan")
        text.append("\n")

    # Logits
    logits = info["logits"]
    text.append("\u2500\u2192 Logits : ", style="bold")
    for a in range(NUM_ACTIONS):
        style = "bright_yellow" if logits[a] == logits.max() else "white"
        text.append(f"[{ACTION_ARROWS[a]} {logits[a]:+.2f}] ", style=style)
    text.append("\n")

    # Probabilities
    probs = info["probs"]
    text.append("\u2500\u2192 Probs  : ", style="bold")
    chosen = probs.argmax()
    for a in range(NUM_ACTIONS):
        style = "bold bright_green" if a == chosen else "white"
        text.append(f"[{ACTION_ARROWS[a]} {probs[a]:.2f}] ", style=style)
    text.append(f"  \u2190 chose: {ACTION_ARROWS[chosen]}", style="bold green")
    text.append("\n")

    # Value
    text.append(f"\u2500\u2192 V(s) = {info['value']:.3f}\n", style="bold magenta")

    # Reference comparison (if provided)
    if ref_probs is not None:
        text.append(f"\n{ref_label} probs: ", style="dim bold")
        for a in range(NUM_ACTIONS):
            text.append(f"[{ACTION_ARROWS[a]} {ref_probs[a]:.2f}] ", style="dim")
        # Per-state KL
        cur_lp = np.log(np.clip(probs, 1e-8, 1.0))
        ref_lp = np.log(np.clip(ref_probs, 1e-8, 1.0))
        kl = float(np.sum(probs * (cur_lp - ref_lp)))
        text.append(f"\nKL at this state: {kl:.4f}", style="dim yellow")

    return Panel(text, title=title, border_style="bright_blue")


# ---------------------------------------------------------------------------
# RM architecture display
# ---------------------------------------------------------------------------
def render_rm_architecture() -> Panel:
    """Static display of the reward model architecture and loss formula."""
    text = Text()
    text.append("State (r, c)  \u2500\u2192  ", style="bold")
    text.append("[64 ReLU]", style="cyan")
    text.append("  \u2500\u2192  ", style="bold")
    text.append("[64 ReLU]", style="cyan")
    text.append("  \u2500\u2192  ", style="bold")
    text.append("Scalar Reward r(s)\n", style="bright_green")

    text.append("Trajectory reward: R(\u03c4) = \u03a3 r(s\u209c)\n", style="white")
    text.append(
        "Loss: -[ \u03bc\u00b7log \u03c3(R\u2081-R\u2082) + (1-\u03bc)\u00b7log \u03c3(R\u2082-R\u2081) ]",
        style="yellow",
    )
    text.append("   (Bradley-Terry cross-entropy)", style="dim")

    return Panel(text, title="RM Architecture", border_style="cyan")


# ---------------------------------------------------------------------------
# Preference spot-check
# ---------------------------------------------------------------------------
def render_rm_spot_check(
    pair_idx: int,
    r_a: float,
    r_b: float,
    prob_a: float,
    human_label: float,
) -> Panel:
    """Show how the RM scores one preference pair vs the human label."""
    text = Text()
    text.append(f"Pair #{pair_idx}:  ", style="bold")
    text.append(f"R(Path A) = {r_a:.2f}   R(Path B) = {r_b:.2f}   ", style="white")

    rm_preferred = "A" if r_a > r_b else "B"
    text.append(f"RM says: {rm_preferred} \u227b other (\u03c3={prob_a:.2f})\n", style="cyan")

    human_preferred = "A" if human_label < 0.5 else "B"
    correct = rm_preferred == human_preferred
    symbol = "\u2713" if correct else "\u2717"
    style = "bold green" if correct else "bold red"
    text.append(f"You said:  {human_preferred} \u227b other   {symbol} ", style=style)
    text.append("Correct" if correct else "Incorrect", style=style)

    return Panel(text, title="RM Preference Check", border_style="cyan")


# ---------------------------------------------------------------------------
# Policy comparison (pre-trained vs RLHF)
# ---------------------------------------------------------------------------
def render_policy_comparison(
    pretrained_map: np.ndarray,
    rlhf_map: np.ndarray,
    env: GridWorld,
) -> Panel:
    """
    Side-by-side policy comparison with a difference column.
    Shows where the RLHF policy diverges from pre-trained.
    """
    text = Text()
    changed = 0
    total = 0

    # Header
    text.append("Pre-trained         RLHF              Diff\n", style="bold dim")

    for r in range(env.size):
        # Pre-trained
        for c in range(env.size):
            if env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            elif (r, c) == env.start:
                text.append("S ", style="green")
            elif (r, c) == env.goal:
                text.append("G ", style="red")
            else:
                text.append(f"{ACTION_ARROWS[pretrained_map[r, c]]} ", style="dim yellow")
        text.append("  ")
        # RLHF
        for c in range(env.size):
            if env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            elif (r, c) == env.start:
                text.append("S ", style="green")
            elif (r, c) == env.goal:
                text.append("G ", style="red")
            else:
                text.append(f"{ACTION_ARROWS[rlhf_map[r, c]]} ", style="bright_yellow")
        text.append("  ")
        # Diff
        for c in range(env.size):
            if env.grid[r, c] == WALL:
                text.append(BLOCK_FULL, style="bright_white")
            else:
                total += 1
                if pretrained_map[r, c] != rlhf_map[r, c]:
                    changed += 1
                    text.append("\u2260 ", style="bold red")
                else:
                    text.append("\u00b7 ", style="dim")
        if r < env.size - 1:
            text.append("\n")

    pct = changed / total * 100 if total > 0 else 0
    subtitle = f"Cells changed: {changed} / {total} ({pct:.1f}%)"
    return Panel(text, title="Policy Comparison", subtitle=subtitle, border_style="yellow")


# ---------------------------------------------------------------------------
# LLM parallel panels
# ---------------------------------------------------------------------------
_PHASE_PARALLELS = {
    1: (
        "Phase 1: Pre-training",
        "Like pre-training an LLM on next-token prediction, the agent learns\n"
        "basic navigation from a simple shaped reward (+10 goal, -0.01/step).\n"
        "This gives it basic competence before fine-tuning with preferences.",
    ),
    2: (
        "Phase 2: Human Preference Collection",
        "You are the human annotator! Just as annotators compare two LLM\n"
        "responses and pick the better one, you compare two paths and pick\n"
        "your preferred trajectory.",
    ),
    3: (
        "Phase 3: Reward Model Training",
        "Training a reward model on your preferences using the Bradley-Terry\n"
        "model: P(A\u227bB) = \u03c3(R(A) - R(B)). This is exactly how reward models\n"
        "are trained in RLHF for LLMs.",
    ),
    4: (
        "Phase 4: RLHF \u2014 PPO Fine-tuning",
        "PPO optimizes the policy against the learned RM, with a KL penalty\n"
        "\u03b2\u00b7KL(\u03c0_RL \u2016 \u03c0_pretrained) to prevent reward hacking. Same as RLHF.",
    ),
}


def render_phase_header(phase: int) -> Panel:
    """Render the LLM-parallel explanation banner for a given phase."""
    title, body = _PHASE_PARALLELS.get(phase, ("", ""))
    text = Text(body, style="italic")
    return Panel(text, title=f"LLM Parallel: {title}", border_style="bright_magenta")


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
def render_results_summary(
    pretrained_stats: Dict[str, float],
    rlhf_stats: Dict[str, float],
) -> Panel:
    """Final comparison table: pre-trained vs RLHF metrics."""
    table = Table(title="Results Summary", border_style="green", show_lines=True)
    table.add_column("Metric", style="bold", min_width=20)
    table.add_column("Pre-trained", justify="right", min_width=12)
    table.add_column("RLHF", justify="right", min_width=12)
    table.add_column("Change", justify="right", min_width=12)

    for key in pretrained_stats:
        pt = pretrained_stats[key]
        rl = rlhf_stats.get(key, 0)
        if pt != 0:
            change = f"{(rl - pt) / abs(pt) * 100:+.1f}%"
        else:
            change = f"{rl:+.2f}"
        table.add_row(key, f"{pt:.2f}", f"{rl:.2f}", change)

    return Panel(table, border_style="bright_green")


def render_llm_parallel_table() -> Panel:
    """Full mapping table for the conclusion."""
    table = Table(show_lines=True, border_style="bright_magenta")
    table.add_column("What you just did", style="bold cyan", min_width=30)
    table.add_column("What happens in LLM RLHF", style="bold yellow", min_width=38)

    rows = [
        ("Pre-trained agent on grid", "Pre-train LLM on internet text"),
        ("Rated path pairs", "Annotators rate response pairs"),
        ("Trained RM on comparisons", "Train reward model on comparisons"),
        ("PPO with KL penalty", "PPO fine-tune LLM with KL penalty"),
        ("Agent prefers YOUR path style", "LLM prefers human-approved responses"),
        ("Reward hacking when \u03b2 too low", "LLM gaming metrics when \u03b2 too low"),
    ]
    for a, b in rows:
        table.add_row(a, b)

    return Panel(table, title="LLM Parallel: The Full Picture", border_style="bright_magenta")


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------
def render_welcome() -> Panel:
    """Opening panel with project overview."""
    text = Text()
    text.append("Path-Finding Preference Game\n", style="bold bright_cyan")
    text.append("=" * 40 + "\n\n", style="dim")
    text.append(
        "An interactive demo of Reinforcement Learning from Human Feedback\n"
        "(RLHF) applied to grid-world navigation.\n\n",
        style="white",
    )
    text.append("You will go through 4 phases:\n\n", style="bold")
    text.append("  1. ", style="dim")
    text.append("Pre-training", style="bold green")
    text.append("    \u2014 Agent learns basic navigation\n")
    text.append("  2. ", style="dim")
    text.append("Preferences", style="bold yellow")
    text.append("     \u2014 You rate pairs of paths\n")
    text.append("  3. ", style="dim")
    text.append("Reward Model", style="bold cyan")
    text.append("    \u2014 RM learns from your ratings\n")
    text.append("  4. ", style="dim")
    text.append("RLHF (PPO)", style="bold magenta")
    text.append("     \u2014 Agent optimises for your preferences\n\n")
    text.append(
        "Each phase mirrors a step in LLM alignment.\n"
        "Press Enter to begin...",
        style="italic dim",
    )

    return Panel(text, border_style="bright_blue", title="Welcome")


# ---------------------------------------------------------------------------
# Preference collection display
# ---------------------------------------------------------------------------
def render_preference_pair(
    env: GridWorld,
    traj_a: Trajectory,
    traj_b: Trajectory,
    pair_num: int,
    total_pairs: int,
) -> Group:
    """
    Render two trajectories side-by-side for preference comparison.
    Returns a Group containing Columns + progress info.
    """
    grid_a = render_grid(
        env,
        path=traj_a.positions,
        path_style="bold cyan",
        title=f"[1] Path A",
        show_stats=True,
        traj=traj_a,
    )
    grid_b = render_grid(
        env,
        path=traj_b.positions,
        path_style="bold bright_yellow",
        title=f"[2] Path B",
        show_stats=True,
        traj=traj_b,
    )
    grids = Columns([grid_a, grid_b], equal=True, expand=True)

    # Progress bar
    filled = int(pair_num / total_pairs * 30)
    bar = "\u2588" * filled + "\u2591" * (30 - filled)
    progress = Text()
    progress.append(f"\nPreferences collected: {pair_num} / {total_pairs}   ", style="bold")
    progress.append(bar, style="cyan")
    progress.append(f"  {pair_num / total_pairs * 100:.0f}%", style="dim")

    return Group(grids, progress)
