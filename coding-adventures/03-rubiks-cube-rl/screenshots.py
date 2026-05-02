#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 03 from real captured data.

Loads ``screenshot_data.json`` (produced by ``app.py --capture``) and
reconstructs the dataclass instances needed by ``viz.py``, so the
screenshots are pixel-perfect renderings of actual training runs — not
synthetic data.

Usage::

    python screenshots.py                     # uses ./screenshot_data.json
    python screenshots.py path/to/data.json   # custom path
"""

import json
import sys
from pathlib import Path

# Add adventure dir to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from cube import PocketCube, ACTION_NAMES
from train import DepthStats, TrainingStats, SolveResult, TestResults
from viz import (
    render_cube,
    render_env_info,
    render_llm_parallel,
    render_curriculum_summary,
    render_difficulty_test,
    render_comparison,
    render_solve_attempt,
    render_summary,
)

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "03"


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _dict_to_depth_stats(d: dict) -> DepthStats:
    """Reconstruct a DepthStats from a JSON dict."""
    return DepthStats(
        depth=d["depth"],
        episodes_trained=d["episodes_trained"],
        solve_rate=d["solve_rate"],
        avg_return=d["avg_return"],
        avg_episode_length=d["avg_episode_length"],
        policy_loss=d["policy_loss"],
        value_loss=d["value_loss"],
        entropy=d["entropy"],
        advanced=d["advanced"],
        time_seconds=d["time_seconds"],
    )


def _dict_to_solve_result(d: dict) -> SolveResult:
    """Reconstruct a SolveResult from a JSON dict."""
    return SolveResult(
        depth=d["depth"],
        solved=d["solved"],
        steps=d["steps"],
        moves=d["moves"],
        states=d["states"],
    )


def _dict_to_test_results(d: dict) -> TestResults:
    """Reconstruct a TestResults from a JSON dict."""
    return TestResults(
        depth=d["depth"],
        n_episodes=d["n_episodes"],
        solve_rate=d["solve_rate"],
        avg_steps=d["avg_steps"],
        results=[_dict_to_solve_result(r) for r in d["results"]],
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
# Screenshot 1: Cube World — environment overview
# ---------------------------------------------------------------------------

def generate_cube_world(data: dict):
    console = Console(record=True, width=80)

    # Solved state
    cube = PocketCube()
    env_info = render_env_info()
    console.print(render_cube(cube, "Solved State"))
    console.print(env_info)

    # Scrambled state with solution
    scrambled = PocketCube()
    scrambled.state = list(data["phase1"]["scrambled_state"])
    scramble_move_names = [ACTION_NAMES[m] for m in data["phase1"]["scramble_moves"]]
    solution_names = [ACTION_NAMES[m] for m in data["phase1"]["solution"]]
    console.print(render_cube(
        scrambled,
        f"Scrambled (depth {len(scramble_move_names)}): {' '.join(scramble_move_names)}",
    ))

    _save(console, "01_cube_world.svg", "2x2 Pocket Cube Environment")


# ---------------------------------------------------------------------------
# Screenshot 2: Curriculum training summary
# ---------------------------------------------------------------------------

def generate_training(data: dict):
    console = Console(record=True, width=80)

    p2 = data["phase2"]
    depth_stats = [_dict_to_depth_stats(d) for d in p2["depth_stats"]]
    stats = TrainingStats(
        depth_stats=depth_stats,
        current_depth=depth_stats[-1].depth,
        total_episodes=p2["total_episodes"],
        total_updates=p2["total_updates"],
        solve_rate_history=[(ep, sr) for ep, sr in p2["solve_rate_history"]],
        total_time=p2["total_time"],
    )

    table = render_curriculum_summary(stats)
    console.print(table)

    _save(console, "02_training.svg", "Curriculum PPO Training")


# ---------------------------------------------------------------------------
# Screenshot 3: Live solve demonstrations
# ---------------------------------------------------------------------------

def generate_solve_demos(data: dict):
    console = Console(record=True, width=80)

    p3 = data["phase3"]
    for sr_dict in p3["solve_results"]:
        result = _dict_to_solve_result(sr_dict)
        console.print(render_solve_attempt(result))

        # Show scrambled and solved cube states if available
        if result.states:
            scrambled = PocketCube()
            scrambled.state = list(result.states[0])
            console.print(render_cube(scrambled, f"Scrambled (depth {result.depth})"))

            if result.solved and len(result.states) > 1:
                solved = PocketCube()
                solved.state = list(result.states[-1])
                console.print(render_cube(solved, "Solved ✓"))

    _save(console, "03_solve_demos.svg", "Agent Solving Cubes")


# ---------------------------------------------------------------------------
# Screenshot 4: Stress test bar chart
# ---------------------------------------------------------------------------

def generate_stress_test(data: dict):
    console = Console(record=True, width=80)

    p4 = data["phase4"]
    stress_results = [_dict_to_test_results(d) for d in p4["stress_results"]]

    panel = render_difficulty_test(stress_results)
    console.print(panel)

    _save(console, "04_stress_test.svg", "Stress Test — Solve Rate by Depth")


# ---------------------------------------------------------------------------
# Screenshot 5: Random vs trained comparison + summary
# ---------------------------------------------------------------------------

def generate_comparison(data: dict):
    console = Console(record=True, width=90)

    p5 = data["phase5"]
    random_results = [_dict_to_test_results(d) for d in p5["random_results"]]
    trained_results = [_dict_to_test_results(d) for d in p5["trained_results"]]

    table = render_comparison(random_results, trained_results)
    console.print(table)

    # Summary
    p2 = data["phase2"]
    depth_stats = [_dict_to_depth_stats(d) for d in p2["depth_stats"]]
    stats = TrainingStats(
        depth_stats=depth_stats,
        current_depth=depth_stats[-1].depth,
        total_episodes=p2["total_episodes"],
        total_updates=p2["total_updates"],
        solve_rate_history=[(ep, sr) for ep, sr in p2["solve_rate_history"]],
        total_time=p2["total_time"],
    )

    # Use stress test results for summary
    p4 = data["phase4"]
    stress_results = [_dict_to_test_results(d) for d in p4["stress_results"]]
    summary = render_summary(stats, stress_results)
    console.print(summary)

    # LLM parallel mapping
    parallel_rows = [
        ("24 stickers → 144-dim one-hot", "Tokens → embedding vectors"),
        ("6 moves (U, U', R, R', F, F')", "Vocabulary of next-token choices"),
        ("Curriculum depth 1→7", "Train on easy tasks first, scale up"),
        ("PPO clipped objective", "Same optimizer used in ChatGPT/Claude RLHF"),
        ("Dense reward shaping", "Reward model signal guiding token choices"),
        ("Solve rate plateau at deep scrambles", "Alignment tax — harder alignment = lower capability"),
    ]
    parallel = render_llm_parallel("Takeaway: Cube RL ↔ LLM Alignment", parallel_rows)
    console.print(parallel)

    _save(console, "05_comparison.svg", "Random vs Trained Agent + Summary")


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
    print("Generating Adventure 03 screenshots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_cube_world(data)
    generate_training(data)
    generate_solve_demos(data)
    generate_stress_test(data)
    generate_comparison(data)

    print("Done!")
