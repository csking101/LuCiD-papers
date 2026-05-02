#!/usr/bin/env python3
"""Adventure 03 — Solve a Rubik's Cube with RL

Train a PPO agent to solve a 2x2 Pocket Cube using curriculum learning.

Run interactively:
    python app.py

Capture screenshot data for docs:
    python app.py --capture screenshot_data.json

Phases:
    1. Cube World           — Explore the 2x2 Pocket Cube environment
    2. PPO Training         — Curriculum training from depth 1 upward
    3. Live Solving         — Watch the agent solve cubes step-by-step
    4. Stress Test          — Test across scramble depths 1-10
    5. Comparison           — Random agent vs trained agent
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch
from rich.console import Console
from rich.live import Live
from rich.prompt import Prompt

from cube import PocketCube, ACTION_NAMES, INVERSE_ACTION, NUM_ACTIONS
from policy import CubePolicy, select_action
from train import (
    TrainConfig, CurriculumTrainer, evaluate_policy, TestResults,
)
from viz import (
    render_phase_header, render_cube, render_env_info, render_move_demo,
    render_scramble_info,
    render_training_progress, render_curriculum_summary,
    render_solve_attempt, render_solve_sequence,
    render_difficulty_test, render_comparison,
    render_llm_parallel, render_summary,
)

console = Console()

# ─── Capture helper ───────────────────────────────────────────────────────────

def _serialize_test_results(results: list[TestResults]) -> list[dict]:
    """Convert TestResults to JSON-safe dicts."""
    out = []
    for tr in results:
        out.append({
            "depth": tr.depth,
            "n_episodes": tr.n_episodes,
            "solve_rate": tr.solve_rate,
            "avg_steps": tr.avg_steps,
            "results": [
                {
                    "depth": sr.depth,
                    "solved": sr.solved,
                    "steps": sr.steps,
                    "moves": sr.moves,
                    "states": sr.states,
                }
                for sr in tr.results
            ],
        })
    return out


# ─── Random agent baseline ───────────────────────────────────────────────────

def random_agent_test(
    depths: list[int], n_episodes: int, config: TrainConfig, rng: random.Random,
) -> list[TestResults]:
    """Evaluate a uniformly random agent at each depth."""
    results = []
    for depth in depths:
        solved_count = 0
        total_steps = 0
        for _ in range(n_episodes):
            cube = PocketCube()
            cube.reset()
            cube.scramble(depth, rng=rng)
            max_steps = config.max_steps(depth)
            steps = 0
            done = False
            while not done:
                action = rng.randint(0, NUM_ACTIONS - 1)
                result = cube.step(action)
                steps += 1
                done = result.done or steps >= max_steps
            total_steps += steps
            if cube.is_solved():
                solved_count += 1
        results.append(TestResults(
            depth=depth,
            n_episodes=n_episodes,
            solve_rate=solved_count / n_episodes,
            avg_steps=total_steps / n_episodes,
            results=[],
        ))
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--capture", metavar="FILE",
        help="Capture screenshot data to JSON file instead of interactive mode",
    )
    args = parser.parse_args()

    capture_mode = args.capture is not None
    capture_data: dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    rng = random.Random(seed)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Cube World
    # ══════════════════════════════════════════════════════════════════════

    console.print(render_phase_header(
        1, "Cube World",
        "Explore the 2x2 Pocket Cube — state space, moves, and reward structure.",
    ))

    # Solved cube
    cube = PocketCube()
    console.print(render_cube(cube, "Solved State"))
    console.print(render_env_info())

    # Demonstrate a move
    before = PocketCube()
    after = before.clone()
    after.apply_move(0)
    console.print(render_move_demo(before, after, "U"))

    # Show a scramble
    scrambled = PocketCube()
    scramble_moves = scrambled.scramble(9, rng=random.Random(seed))
    console.print(render_cube(scrambled, "Scrambled (depth 9)"))

    # Solution for the scramble
    solution = PocketCube.solution_for_scramble(scramble_moves)
    console.print(render_scramble_info(scramble_moves, solution))

    # LLM parallel
    console.print(render_llm_parallel("Cube ↔ LLM Alignment", [
        ("24 stickers → 144-dim one-hot", "Tokens → embedding vectors"),
        ("6 moves (U, U', R, R', F, F')", "Vocabulary of next-token choices"),
        ("One-hot state encoding", "Tokenisation + positional encoding"),
        ("Curriculum: depth 1 → 11", "Progressive RLHF: easy → hard preferences"),
    ]))

    if capture_mode:
        capture_data["phase1"] = {
            "solved_state": list(PocketCube().state),
            "scrambled_state": list(scrambled.state),
            "scramble_moves": scramble_moves,
            "solution": solution,
        }

    # ── Training mode selection ──
    if capture_mode:
        training_mode = "full"
    else:
        console.print()
        console.print("[bold bright_cyan]Choose training mode:[/bold bright_cyan]")
        console.print("  [bold]1.[/bold] Quick  — curriculum depth 1-7,  ~2.5 min on GPU, ~5 min on CPU")
        console.print("  [bold]2.[/bold] Full   — curriculum depth 1-11, ~12 min on GPU, ~25 min on CPU")
        console.print()
        choice = Prompt.ask(
            "[dim]Enter 1 or 2[/dim]",
            choices=["1", "2"],
            default="1",
        )
        training_mode = "quick" if choice == "1" else "full"

    if training_mode == "full":
        max_depth = 11
        max_eps = 6000
        solve_depths = [1, 3, 5, 7, 9]
        stress_range = range(1, 15)
        compare_range = range(1, 12)
    else:
        max_depth = 7
        max_eps = 3000
        solve_depths = [1, 3, 5]
        stress_range = range(1, 11)
        compare_range = range(1, 8)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: PPO Training with Curriculum
    # ══════════════════════════════════════════════════════════════════════

    console.print(render_phase_header(
        2, "PPO Training with Curriculum",
        f"Train using PPO with curriculum learning — depth 1→{max_depth}, advance when solve rate ≥ 80%.",
    ))

    config = TrainConfig(
        max_depth=max_depth,
        advance_threshold=0.80,
        eval_episodes=100,
        max_episodes_per_depth=max_eps,
        episodes_per_rollout=128,
        ppo_epochs=4,
        mini_batch_size=64,
    )

    trainer = CurriculumTrainer(config, device=device, seed=seed)

    # Live training display
    def training_callback(depth, ds, stats):
        if not capture_mode:
            live.update(render_training_progress(depth, ds, stats))

    if capture_mode:
        stats = trainer.train()
    else:
        with Live(
            render_training_progress(1, trainer.stats.depth_stats[0] if trainer.stats.depth_stats else
                                     type('DS', (), {'episodes_trained': 0, 'solve_rate': 0, 'avg_return': 0,
                                                     'avg_episode_length': 0, 'policy_loss': 0, 'value_loss': 0,
                                                     'entropy': 0})(),
                                     trainer.stats),
            console=console, refresh_per_second=2,
        ) as live:
            stats = trainer.train(callback=training_callback)

    console.print(render_curriculum_summary(stats))

    if capture_mode:
        capture_data["phase2"] = {
            "depth_stats": [
                {
                    "depth": ds.depth,
                    "episodes_trained": ds.episodes_trained,
                    "solve_rate": ds.solve_rate,
                    "avg_return": ds.avg_return,
                    "avg_episode_length": ds.avg_episode_length,
                    "policy_loss": ds.policy_loss,
                    "value_loss": ds.value_loss,
                    "entropy": ds.entropy,
                    "advanced": ds.advanced,
                    "time_seconds": ds.time_seconds,
                }
                for ds in stats.depth_stats
            ],
            "total_episodes": stats.total_episodes,
            "total_updates": stats.total_updates,
            "total_time": stats.total_time,
            "solve_rate_history": stats.solve_rate_history,
        }
    else:
        Prompt.ask("\n[dim]Press Enter to see live solving[/dim]")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 3: Live Solving
    # ══════════════════════════════════════════════════════════════════════

    console.print(render_phase_header(
        3, "Live Solving",
        "Watch the trained agent solve cubes at increasing difficulty.",
    ))

    solve_results = []

    for depth in solve_depths:
        result = evaluate_policy(
            trainer.policy, depth, 1, config, device,
            random.Random(seed + depth), record_moves=True,
        )
        sr = result.results[0]
        solve_results.append(sr)
        console.print(render_solve_attempt(sr))
        if sr.states:
            # Show initial and final cube
            initial_cube = PocketCube()
            initial_cube.state = list(sr.states[0])
            console.print(render_cube(initial_cube, f"Scrambled (depth {depth})"))
            final_cube = PocketCube()
            final_cube.state = list(sr.states[-1])
            label = "Solved ✓" if sr.solved else "Final state"
            console.print(render_cube(final_cube, label))

    if capture_mode:
        capture_data["phase3"] = {
            "solve_results": [
                {
                    "depth": sr.depth,
                    "solved": sr.solved,
                    "steps": sr.steps,
                    "moves": sr.moves,
                    "states": sr.states,
                }
                for sr in solve_results
            ],
        }
    else:
        Prompt.ask("\n[dim]Press Enter for stress test[/dim]")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 4: Stress Test
    # ══════════════════════════════════════════════════════════════════════

    console.print(render_phase_header(
        4, "Stress Test",
        "Test the agent across scramble depths 1-{} (100 cubes each).".format(len(list(stress_range))),
    ))

    test_depths = list(stress_range)
    stress_results = []
    for depth in test_depths:
        tr = evaluate_policy(
            trainer.policy, depth, 100, config, device,
            random.Random(seed + depth * 100),
        )
        stress_results.append(tr)

    console.print(render_difficulty_test(stress_results))

    if capture_mode:
        capture_data["phase4"] = {
            "stress_results": _serialize_test_results(stress_results),
        }
    else:
        Prompt.ask("\n[dim]Press Enter for comparison[/dim]")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 5: Comparison — Random vs Trained
    # ══════════════════════════════════════════════════════════════════════

    console.print(render_phase_header(
        5, "Random vs Trained Agent",
        "Compare the trained PPO agent against a uniformly random agent.",
    ))

    compare_depths = list(compare_range)
    random_results = random_agent_test(compare_depths, 100, config, random.Random(seed))

    trained_compare = []
    for depth in compare_depths:
        tr = evaluate_policy(
            trainer.policy, depth, 100, config, device,
            random.Random(seed + depth * 200),
        )
        trained_compare.append(tr)

    console.print(render_comparison(random_results, trained_compare))
    console.print(render_summary(stats, stress_results))

    # LLM parallel conclusion
    console.print(render_llm_parallel("Takeaway: Cube RL ↔ LLM Alignment", [
        ("Curriculum depth 1→7", "Train on easy tasks first, scale up"),
        ("PPO clipped objective", "Same optimizer used in ChatGPT/Claude RLHF"),
        ("Dense reward shaping", "Reward model signal guiding token choices"),
        ("Solve rate plateau at deep scrambles", "Alignment tax — harder alignment = lower capability"),
    ]))

    if capture_mode:
        capture_data["phase5"] = {
            "random_results": _serialize_test_results(random_results),
            "trained_results": _serialize_test_results(trained_compare),
        }
        # Write capture data
        with open(args.capture, "w") as f:
            json.dump(capture_data, f, indent=2)
        console.print(f"\n[bold green]Screenshot data saved to {args.capture}[/bold green]")
    else:
        console.print("\n[bold bright_cyan]Adventure 03 complete! 🎲[/bold bright_cyan]\n")


if __name__ == "__main__":
    main()
