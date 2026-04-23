# Human Feedback
#
# Interactive terminal UI for collecting human preferences on segment pairs.
# Uses Rich for nice rendering of grid paths side-by-side.
#
# During training, the loop pauses at configurable intervals and presents
# segment pairs to the user. The user picks which behavior they prefer,
# and these preferences are added to the DB. Training resumes and the
# Rich dashboard shows the immediate impact.

import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.prompt import Prompt

console = Console()

ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_ARROWS = {0: "^", 1: "v", 2: "<", 3: ">"}


def _render_segment_grid(segment: list, env, label: str) -> str:
    """
    Render a segment as a text grid showing the cat's path.
    Steps are numbered 1, 2, 3... on the cells visited.
    Start position marked 'S', final position marked 'E'.
    """
    # Track which cells were visited and in what order
    visit_order = {}  # (x, y) -> step number

    if segment:
        # Mark starting position
        start_pos = tuple(segment[0]["obs"])
        visit_order[start_pos] = "S"

        # Mark each step
        for i, step in enumerate(segment):
            next_pos = tuple(step["next_obs"])
            step_num = i + 1
            visit_order[next_pos] = str(step_num) if step_num <= 9 else chr(ord('a') + step_num - 10)

        # Mark end position (overwrite the number)
        end_pos = tuple(segment[-1]["next_obs"])
        if end_pos != start_pos:
            # Keep the number but we'll color it differently
            pass

    # Build grid
    lines = []
    for y in range(env.height):
        row = []
        for x in range(env.width):
            if (x, y) in env.wall_placement:
                row.append("#")
            elif (x, y) in env.terminal_objects_placement:
                val = env.terminal_objects_placement[(x, y)]
                if (x, y) in visit_order:
                    row.append(visit_order[(x, y)])
                else:
                    row.append("G" if val > 0 else "T")
            elif (x, y) in visit_order:
                row.append(visit_order[(x, y)])
            else:
                row.append(".")
        lines.append(" ".join(row))

    # Summary info
    n_steps = len(segment)
    start = segment[0]["obs"] if segment else [0, 0]
    end = segment[-1]["next_obs"] if segment else [0, 0]

    terminal_hit = None
    for step in segment:
        next_pos = tuple(step["next_obs"])
        if next_pos in env.terminal_objects_placement:
            val = env.terminal_objects_placement[next_pos]
            terminal_hit = f"{'Goal' if val > 0 else 'Trap'} ({val:+d})"

    summary = f"{n_steps} steps: ({start[0]},{start[1]})->({end[0]},{end[1]})"
    if terminal_hit:
        summary += f"  [{terminal_hit}]"

    grid_text = "\n".join(lines) + "\n\n" + summary
    return grid_text


def render_segment_pair(seg1: list, seg2: list, env, pair_idx: int,
                        total_pairs: int):
    """Display two segments side-by-side using Rich panels."""
    grid_a = _render_segment_grid(seg1, env, "A")
    grid_b = _render_segment_grid(seg2, env, "B")

    panel_a = Panel(grid_a, title=f"[bold cyan]Segment A[/]",
                    border_style="cyan", width=28)
    panel_b = Panel(grid_b, title=f"[bold magenta]Segment B[/]",
                    border_style="magenta", width=28)

    console.print()
    console.print(f"[bold]Pair {pair_idx + 1} of {total_pairs}[/]")
    console.print(Columns([panel_a, panel_b], padding=2))


def get_human_preference() -> list | None:
    """
    Prompt the user for their preference.
    Returns mu = [p1, p2] or None to skip.
    """
    while True:
        choice = Prompt.ask(
            "[bold]Which is better?[/]  "
            "[cyan][1][/] A   [magenta][2][/] B   "
            "[yellow][=][/] Equal   [dim][s][/] Skip",
            default="s"
        )
        choice = choice.strip().lower()

        if choice == "1":
            return [1.0, 0.0]
        elif choice == "2":
            return [0.0, 1.0]
        elif choice in ("=", "e"):
            return [0.5, 0.5]
        elif choice == "s":
            return None
        else:
            console.print("[red]Invalid input. Enter 1, 2, =, or s[/]")


def human_feedback_round(segments: list, env, n_pairs: int,
                         iteration: int, policy=None,
                         policy_accuracy: float = None,
                         goal_rate: float = None) -> list:
    """
    Run an interactive human feedback round.

    Selects n_pairs segment pairs, shows them to the user, collects preferences.
    Returns a list of (seg1, seg2, mu) tuples.

    Args:
        segments: all available segments from recent trajectories
        env: the grid environment (for rendering)
        n_pairs: how many pairs to show
        iteration: current training iteration
        policy: current policy (for displaying the policy grid)
        policy_accuracy: current accuracy vs optimal (for context)
        goal_rate: current goal rate (for context)
    """
    console.print()
    console.rule(f"[bold yellow]Human Feedback Round (iter {iteration})[/]")

    # Show context
    context_parts = []
    if goal_rate is not None:
        context_parts.append(f"Goal rate: [green]{goal_rate:.0%}[/]")
    if policy_accuracy is not None:
        context_parts.append(f"Policy accuracy: [cyan]{policy_accuracy:.0%}[/]")
    if context_parts:
        console.print("  " + "  |  ".join(context_parts))

    # Show current policy grid if available
    if policy is not None:
        from optimal import get_learned_policy_dict, render_policy_grid
        policy_dict = get_learned_policy_dict(policy, env)
        grid_str = render_policy_grid(env, policy_dict)
        console.print(Panel(grid_str, title="[bold]Current Learned Policy[/]",
                            border_style="blue", width=28))

    # Select pairs
    if len(segments) < 2:
        console.print("[red]Not enough segments for comparison. Skipping.[/]")
        console.rule()
        return []

    n_pairs = min(n_pairs, len(segments) * (len(segments) - 1) // 2)
    indices = list(range(len(segments)))

    results = []
    for pair_idx in range(n_pairs):
        i, j = random.sample(indices, 2)
        seg1, seg2 = segments[i], segments[j]

        render_segment_pair(seg1, seg2, env, pair_idx, n_pairs)
        mu = get_human_preference()

        if mu is not None:
            results.append((seg1, seg2, mu))
            pref_str = ("A" if mu[0] > mu[1] else
                        "B" if mu[1] > mu[0] else "Equal")
            console.print(f"  -> Recorded: [bold]{pref_str}[/]")
        else:
            console.print("  -> [dim]Skipped[/]")

    console.print(f"\n[bold green]Collected {len(results)} preferences[/]")
    console.rule()
    return results
