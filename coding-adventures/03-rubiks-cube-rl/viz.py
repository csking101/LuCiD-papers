"""Rich terminal visualization for the Pocket Cube RL adventure.

Every public function returns a Rich renderable (Panel, Table, Text,
Group, Columns).  Nothing prints directly — the caller (app.py) decides
when and how to display each renderable.
"""

from __future__ import annotations

from typing import Sequence

from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cube import (
    PocketCube, FACE_NAMES, FACE_COLORS, ACTION_NAMES,
    NUM_STICKERS, NUM_ACTIONS, STATE_DIM, GODS_NUMBER,
    STICKERS_PER_FACE,
)
from train import DepthStats, TrainingStats, SolveResult, TestResults

# ─── Colour mapping ──────────────────────────────────────────────────────────
# Rich style for each cube face colour index

STICKER_STYLES = {
    0: "bold white",        # U = White
    1: "bold yellow",       # D = Yellow
    2: "bold green",        # F = Green
    3: "bold blue",         # B = Blue
    4: "dark_orange",       # L = Orange
    5: "bold red",          # R = Red
}

STICKER_BLOCK = "██"        # Two full-block chars per sticker
STICKER_GAP = " "           # Between stickers on the same face
FACE_GAP = "  "             # Between adjacent faces in the unfolded view

# Sparkline characters for mini bar charts
SPARK_CHARS = "▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_PARTIAL = "░"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _spark(values: Sequence[float], width: int = 20) -> str:
    """Return a sparkline string for *values*, scaled to *width*."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    # Resample to width if needed
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = list(values)
    chars = []
    for v in sampled:
        idx = int((v - mn) / rng * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        chars.append(SPARK_CHARS[idx])
    return "".join(chars)


def _bar(fraction: float, width: int = 20) -> str:
    """Return a horizontal bar for *fraction* (0.0-1.0)."""
    filled = int(fraction * width)
    return BAR_FULL * filled + BAR_PARTIAL * (width - filled)


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


# ─── Cube rendering ──────────────────────────────────────────────────────────

def _render_face_row(face: list[list[int]], row: int) -> Text:
    """Render one row (2 stickers) of a face as styled Text."""
    t = Text()
    for col in range(2):
        color_idx = face[row][col]
        t.append(STICKER_BLOCK, style=STICKER_STYLES[color_idx])
        if col == 0:
            t.append(STICKER_GAP)
    return t


def render_cube(cube: PocketCube, label: str = "") -> Panel:
    """Render the cube in unfolded cross layout.

    Layout::
                 U
            L    F    R    B
                 D
    """
    faces = cube.render_faces()

    lines: list[Text] = []

    indent = " " * (len(STICKER_BLOCK) * 2 + len(STICKER_GAP) + len(FACE_GAP))

    # U face (top)
    for row in range(2):
        t = Text(indent)
        t.append_text(_render_face_row(faces["U"], row))
        lines.append(t)

    # Middle band: L F R B
    for row in range(2):
        t = Text()
        for fi, fname in enumerate(["L", "F", "R", "B"]):
            t.append_text(_render_face_row(faces[fname], row))
            if fi < 3:
                t.append(FACE_GAP)
        lines.append(t)

    # D face (bottom)
    for row in range(2):
        t = Text(indent)
        t.append_text(_render_face_row(faces["D"], row))
        lines.append(t)

    combined = Text("\n").join(lines)
    title = f"Cube{' — ' + label if label else ''}"
    return Panel(combined, title=title, border_style="bright_cyan", padding=(1, 2))


# ─── Phase header ─────────────────────────────────────────────────────────────

def render_phase_header(num: int, title: str, desc: str) -> Panel:
    """Render a styled phase header panel."""
    t = Text()
    t.append(f"Phase {num}", style="bold bright_cyan")
    t.append(" — ", style="dim")
    t.append(title, style="bold white")
    t.append("\n")
    t.append(desc, style="dim")
    return Panel(t, border_style="bright_cyan", padding=(1, 2))


# ─── Environment info ─────────────────────────────────────────────────────────

def render_env_info() -> Panel:
    """Render a panel with cube environment statistics."""
    t = Text()
    t.append("2x2 Pocket Cube Environment\n\n", style="bold bright_cyan")

    rows = [
        ("State space", "3,674,160 reachable positions"),
        ("State encoding", f"One-hot: {NUM_STICKERS} stickers × {6} colours = {STATE_DIM} dims"),
        ("Action space", f"{NUM_ACTIONS} moves: {', '.join(ACTION_NAMES)}"),
        ("God's number", f"{GODS_NUMBER} quarter-turns (worst case)"),
        ("Fixed corner", "DLB — removes rotational equivalence"),
        ("Reward", "+1.0 on solve, −0.02 per step"),
    ]
    for label, value in rows:
        t.append(f"  {label:<18}", style="bold")
        t.append(f"{value}\n", style="dim")

    return Panel(t, title="Environment", border_style="bright_cyan", padding=(1, 2))


# ─── Move demonstration ──────────────────────────────────────────────────────

def render_move_demo(before: PocketCube, after: PocketCube, move_name: str) -> Panel:
    """Show before → after state for a single move."""
    cols = Columns([
        render_cube(before, "Before"),
        Text("  →  ", style="bold bright_cyan"),
        render_cube(after, f"After {move_name}"),
    ], padding=(0, 1))
    return Panel(cols, title=f"Move: {move_name}", border_style="dim")


# ─── Training progress ───────────────────────────────────────────────────────

def render_training_progress(
    depth: int,
    ds: DepthStats,
    stats: TrainingStats,
) -> Panel:
    """Live training status panel, updated during training."""
    t = Text()
    t.append(f"Curriculum Depth: {depth}\n", style="bold bright_cyan")
    t.append(f"Episodes trained:  {ds.episodes_trained}\n")
    t.append(f"Solve rate:        ", style="bold")
    t.append(f"{_pct(ds.solve_rate)}", style="bold green" if ds.solve_rate >= 0.8 else "bold yellow")
    t.append(f"  {_bar(ds.solve_rate)}\n")
    t.append(f"Avg return:        {ds.avg_return:.3f}\n")
    t.append(f"Avg ep length:     {ds.avg_episode_length:.1f}\n")
    t.append(f"Policy loss:       {ds.policy_loss:.4f}\n")
    t.append(f"Value loss:        {ds.value_loss:.4f}\n")
    t.append(f"Entropy:           {ds.entropy:.4f}\n")

    if stats.solve_rate_history:
        rates = [r for _, r in stats.solve_rate_history]
        t.append(f"\nSolve rate trend:  {_spark(rates)}\n", style="dim")

    t.append(f"\nTotal episodes:    {stats.total_episodes}\n", style="dim")
    t.append(f"Total updates:     {stats.total_updates}", style="dim")

    return Panel(
        t, title="PPO Training", border_style="bright_cyan", padding=(1, 2),
    )


# ─── Curriculum summary ──────────────────────────────────────────────────────

def render_curriculum_summary(stats: TrainingStats) -> Table:
    """Table summarising training results per depth level."""
    table = Table(
        title="Curriculum Summary",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
    )
    table.add_column("Depth", justify="center", style="bold")
    table.add_column("Solve Rate", justify="center")
    table.add_column("Bar", justify="left")
    table.add_column("Episodes", justify="right")
    table.add_column("Avg Steps", justify="right")
    table.add_column("Policy Loss", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Status", justify="center")

    for ds in stats.depth_stats:
        rate_style = "bold green" if ds.solve_rate >= 0.8 else (
            "bold yellow" if ds.solve_rate >= 0.4 else "bold red"
        )
        status = "✓ Advanced" if ds.advanced else "— Max eps"
        status_style = "green" if ds.advanced else "dim"

        table.add_row(
            str(ds.depth),
            Text(_pct(ds.solve_rate), style=rate_style),
            Text(_bar(ds.solve_rate, 12)),
            str(ds.episodes_trained),
            f"{ds.avg_episode_length:.1f}",
            f"{ds.policy_loss:.4f}",
            f"{ds.time_seconds:.1f}s",
            Text(status, style=status_style),
        )

    return table


# ─── Solve attempt ────────────────────────────────────────────────────────────

def render_solve_attempt(result: SolveResult) -> Panel:
    """Step-by-step visualisation of the agent solving a cube."""
    t = Text()
    t.append(f"Scramble depth: {result.depth}\n", style="bold")
    t.append(f"Result: ", style="bold")
    if result.solved:
        t.append(f"SOLVED in {result.steps} moves ✓\n", style="bold green")
    else:
        t.append(f"FAILED after {result.steps} moves ✗\n", style="bold red")

    t.append("Moves: ", style="bold")
    for i, m in enumerate(result.moves):
        t.append(ACTION_NAMES[m], style="bright_cyan")
        if i < len(result.moves) - 1:
            t.append(" → ", style="dim")
    t.append("\n")

    return Panel(t, title=f"Solve Attempt (depth {result.depth})", border_style="bright_cyan", padding=(1, 2))


def render_solve_sequence(result: SolveResult) -> Panel:
    """Render a solve attempt with inline cube states at key steps."""
    parts: list[Text | Panel] = []

    header = Text()
    header.append(f"Scramble depth {result.depth}: ", style="bold")
    if result.solved:
        header.append(f"Solved in {result.steps} moves ✓", style="bold green")
    else:
        header.append(f"Failed after {result.steps} moves ✗", style="bold red")
    parts.append(header)

    # Show initial state
    if result.states:
        cube = PocketCube()
        cube.state = list(result.states[0])
        parts.append(render_cube(cube, "Scrambled"))

    # Show moves as text
    moves_text = Text("Moves: ", style="bold")
    for i, m in enumerate(result.moves):
        moves_text.append(ACTION_NAMES[m], style="bright_cyan")
        if i < len(result.moves) - 1:
            moves_text.append(" → ", style="dim")
    parts.append(moves_text)

    # Show final state
    if len(result.states) > 1:
        cube = PocketCube()
        cube.state = list(result.states[-1])
        label = "Solved ✓" if result.solved else "Final state"
        parts.append(render_cube(cube, label))

    combined = Text("\n").join(
        [p if isinstance(p, Text) else Text(str(p)) for p in parts[:1]]
    )

    # Build as group using Text for the header and moves
    inner = Text()
    inner.append_text(header)
    inner.append("\n\n")
    inner.append_text(moves_text)

    return Panel(inner, title=f"Solve — Depth {result.depth}", border_style="bright_cyan", padding=(1, 2))


# ─── Difficulty test ──────────────────────────────────────────────────────────

def render_difficulty_test(results: list[TestResults]) -> Panel:
    """Bar chart of solve rates across scramble depths."""
    t = Text()
    t.append("Solve Rate by Scramble Depth\n\n", style="bold bright_cyan")

    for tr in results:
        rate_style = "bold green" if tr.solve_rate >= 0.8 else (
            "bold yellow" if tr.solve_rate >= 0.4 else "bold red"
        )
        t.append(f"  Depth {tr.depth:>2}  ", style="bold")
        t.append(f"{_bar(tr.solve_rate, 25)} ", style=rate_style)
        t.append(f"{_pct(tr.solve_rate):>6}", style=rate_style)
        t.append(f"  (avg {tr.avg_steps:.1f} steps)\n")

    # Capability frontier
    frontier = 0
    for tr in results:
        if tr.solve_rate >= 0.5:
            frontier = tr.depth
    t.append(f"\n  Capability frontier (≥50%): depth {frontier}", style="bold bright_cyan")

    return Panel(t, title="Stress Test", border_style="bright_cyan", padding=(1, 2))


# ─── Comparison ───────────────────────────────────────────────────────────────

def render_comparison(
    random_results: list[TestResults],
    trained_results: list[TestResults],
) -> Table:
    """Side-by-side table: random agent vs trained agent."""
    table = Table(
        title="Random vs Trained Agent",
        border_style="bright_cyan",
        header_style="bold bright_cyan",
        show_lines=True,
    )
    table.add_column("Depth", justify="center", style="bold")
    table.add_column("Random Solve %", justify="center")
    table.add_column("Trained Solve %", justify="center")
    table.add_column("Random Steps", justify="center")
    table.add_column("Trained Steps", justify="center")

    for rand_tr, trained_tr in zip(random_results, trained_results):
        r_style = "bold green" if rand_tr.solve_rate >= 0.5 else "dim"
        t_style = "bold green" if trained_tr.solve_rate >= 0.5 else (
            "bold yellow" if trained_tr.solve_rate >= 0.2 else "bold red"
        )
        table.add_row(
            str(rand_tr.depth),
            Text(_pct(rand_tr.solve_rate), style=r_style),
            Text(_pct(trained_tr.solve_rate), style=t_style),
            f"{rand_tr.avg_steps:.1f}",
            f"{trained_tr.avg_steps:.1f}",
        )

    return table


# ─── LLM parallels ───────────────────────────────────────────────────────────

def render_llm_parallel(title: str, rows: list[tuple[str, str]]) -> Panel:
    """Side-by-side comparison table: cube concept ↔ LLM concept."""
    table = Table(
        show_header=True,
        header_style="bold bright_cyan",
        border_style="dim",
        show_lines=True,
    )
    table.add_column("Cube RL", style="bold")
    table.add_column("LLM Alignment", style="bold")
    for left, right in rows:
        table.add_row(left, right)
    return Panel(table, title=title, border_style="bright_cyan", padding=(1, 2))


# ─── Summary ──────────────────────────────────────────────────────────────────

def render_summary(stats: TrainingStats, trained_results: list[TestResults]) -> Panel:
    """Final summary panel."""
    t = Text()
    t.append("Adventure Complete!\n\n", style="bold bright_cyan")

    depths_passed = sum(1 for ds in stats.depth_stats if ds.advanced)
    max_depth_solved = max(
        (ds.depth for ds in stats.depth_stats if ds.solve_rate >= 0.5),
        default=0,
    )

    t.append(f"  Depths passed:       {depths_passed}/{len(stats.depth_stats)}\n", style="bold")
    t.append(f"  Max depth ≥50%:      {max_depth_solved}\n", style="bold")
    t.append(f"  Total episodes:      {stats.total_episodes}\n")
    t.append(f"  Total PPO updates:   {stats.total_updates}\n")
    t.append(f"  Training time:       {stats.total_time:.1f}s\n")

    if trained_results:
        t.append(f"\n  Stress test results:\n", style="bold")
        for tr in trained_results:
            marker = "✓" if tr.solve_rate >= 0.5 else "✗"
            style = "green" if tr.solve_rate >= 0.5 else "red"
            t.append(f"    Depth {tr.depth}: ", style="bold")
            t.append(f"{_pct(tr.solve_rate)} {marker}\n", style=style)

    return Panel(t, title="Summary", border_style="bright_cyan", padding=(1, 2))
