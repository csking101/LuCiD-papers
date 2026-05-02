#!/usr/bin/env python3
"""
System Prompt Steering — Interactive Terminal App
==================================================

Run with:  python app.py
Capture:   python app.py --capture screenshot_data.json

Loads a single instruct model and compares its token distributions
under different system prompts — measuring the "steering power"
of each system prompt.

Phases:
  1 — Model loading & chat template anatomy
  2 — First-token steering (microscope view)
  3 — Forced-continuation KL profile
  4 — Steering matrix (system prompt x user prompt)
  5 — Interactive prompt explorer
  6 — Conclusion & LLM parallel mapping
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rich.console import Console
from rich.prompt import Prompt
from rich.text import Text

from analysis import (
    FirstTokenComparison,
    ForcedContinuationKL,
    SteeringMatrix,
    SystemPromptProfile,
    compare_first_token,
    compute_forced_continuation_kl,
    compute_steering_matrix,
    compute_system_prompt_profiles,
)
from models import (
    DEFAULT_MODEL,
    ModelInfo,
    format_chat,
    generate_greedy,
    load_model,
    tokenize_chat,
)
from prompts import (
    ALL_SYSTEM_PROMPTS,
    CUSTOM_SYSTEM_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    USER_PROMPTS,
    get_phase2_prompts,
    get_phase3_prompts,
    get_phase4_system_prompts,
    get_phase4_user_prompts,
)
from viz import (
    render_adventure_connections,
    render_chat_template,
    render_conclusion,
    render_first_token_comparison,
    render_forced_continuation,
    render_llm_parallel_table,
    render_model_info,
    render_phase_header,
    render_profiles,
    render_steering_matrix,
    render_welcome,
)

# ── Config ──────────────────────────────────────────────────────────
MODEL_NAME = DEFAULT_MODEL
MAX_NEW_TOKENS = 64

console = Console()

# Capture mode: set by --capture
CAPTURE_MODE = False
CAPTURE_DATA: dict = {}


def pause(msg: str = "[dim]Press Enter to continue...[/dim]") -> None:
    """Block until the user presses Enter (skipped in capture mode)."""
    if CAPTURE_MODE:
        return
    console.print()
    input(msg)


# ── Serialization helpers ───────────────────────────────────────────

def _ftc_to_dict(ftc: FirstTokenComparison) -> dict:
    return {
        "user_prompt": ftc.user_prompt,
        "system_prompt_a": ftc.system_prompt_a,
        "system_prompt_b": ftc.system_prompt_b,
        "kl_divergence": ftc.kl_divergence,
        "js_divergence": ftc.js_divergence,
        "top_k_a": ftc.top_k_a,
        "top_k_b": ftc.top_k_b,
        "top_shifts": ftc.top_shifts,
    }


def _fckl_to_dict(fckl: ForcedContinuationKL) -> dict:
    return {
        "user_prompt": fckl.user_prompt,
        "source_system_prompt": fckl.source_system_prompt,
        "target_system_prompt": fckl.target_system_prompt,
        "continuation_tokens": fckl.continuation_tokens,
        "continuation_ids": fckl.continuation_ids,
        "kl_per_token": fckl.kl_per_token,
        "total_kl": fckl.total_kl,
        "mean_kl": fckl.mean_kl,
        "source_text": fckl.source_text,
        "target_text": fckl.target_text,
    }


def _matrix_to_dict(m: SteeringMatrix) -> dict:
    return {
        "system_prompt_names": m.system_prompt_names,
        "user_prompts": m.user_prompts,
        "kl_matrix": m.kl_matrix,
        "row_means": m.row_means,
        "col_means": m.col_means,
        "global_mean": m.global_mean,
    }


def _profile_to_dict(p: SystemPromptProfile) -> dict:
    return {
        "system_prompt_name": p.system_prompt_name,
        "system_prompt_text": p.system_prompt_text,
        "category": p.category,
        "user_prompts": p.user_prompts,
        "first_token_kls": p.first_token_kls,
        "mean_steering_power": p.mean_steering_power,
        "max_steering_power": p.max_steering_power,
    }


# ── Phase 1 ─────────────────────────────────────────────────────────

def run_phase1(model, tokenizer, info) -> None:
    """Model loading & chat template anatomy."""
    console.print(render_phase_header(1))
    console.print()
    console.print(render_model_info(info))
    console.print()

    # Show chat template formatting
    user_text = "How do I pick a lock?"

    fmt_default = format_chat(
        tokenizer, user_text,
        system_prompt=DEFAULT_SYSTEM_PROMPT.text,
    )
    fmt_safety = format_chat(
        tokenizer, user_text,
        system_prompt=CUSTOM_SYSTEM_PROMPTS[0].text,  # Safety
    )

    console.print(render_chat_template(
        fmt_default, fmt_safety,
        DEFAULT_SYSTEM_PROMPT.name, CUSTOM_SYSTEM_PROMPTS[0].name,
    ))

    # Show all available system prompts
    console.print()
    console.print("[bold]Available system prompts:[/]")
    for sp in ALL_SYSTEM_PROMPTS:
        style = "cyan" if sp == DEFAULT_SYSTEM_PROMPT else "green"
        console.print(f"  [{style}]{sp.name}[/] ({sp.category}): [dim]{sp.description}[/]")

    if CAPTURE_MODE:
        CAPTURE_DATA["phase1"] = {
            "model_name": info.name,
            "formatted_default": fmt_default,
            "formatted_custom": fmt_safety,
            "default_name": DEFAULT_SYSTEM_PROMPT.name,
            "custom_name": CUSTOM_SYSTEM_PROMPTS[0].name,
            "system_prompts": [
                {"name": sp.name, "category": sp.category,
                 "description": sp.description, "text": sp.text}
                for sp in ALL_SYSTEM_PROMPTS
            ],
        }


# ── Phase 2 ─────────────────────────────────────────────────────────

def run_phase2(model, tokenizer) -> FirstTokenComparison:
    """First-token steering — microscope view."""
    console.print(render_phase_header(2))
    console.print()

    sys_prompt, user_prompt = get_phase2_prompts()
    console.print(f"[bold]System prompt:[/] [green]{sys_prompt.name}[/] ({sys_prompt.category})")
    console.print(f"[bold]User prompt:[/] [italic]\"{user_prompt.text}\"[/]")
    console.print(f"[dim]{sys_prompt.description}[/]\n")

    console.print("[dim]Computing first-token distributions...[/]")
    ftc = compare_first_token(
        model, tokenizer,
        user_prompt=user_prompt.text,
        system_prompt_a=DEFAULT_SYSTEM_PROMPT.text,
        system_prompt_b=sys_prompt.text,
        name_a=DEFAULT_SYSTEM_PROMPT.name,
        name_b=sys_prompt.name,
        top_k=10,
    )

    console.print(render_first_token_comparison(ftc))

    if CAPTURE_MODE:
        CAPTURE_DATA["phase2"] = _ftc_to_dict(ftc)

    return ftc


# ── Phase 3 ─────────────────────────────────────────────────────────

def run_phase3(model, tokenizer) -> ForcedContinuationKL:
    """Forced-continuation KL profile."""
    console.print(render_phase_header(3))
    console.print()

    sys_prompt, user_prompt = get_phase3_prompts()
    console.print(f"[bold]System prompt:[/] [green]{sys_prompt.name}[/] ({sys_prompt.category})")
    console.print(f"[bold]User prompt:[/] [italic]\"{user_prompt.text}\"[/]")
    console.print(f"[dim]{sys_prompt.description}[/]\n")

    console.print(
        "[dim]Generating from Default, then forcing those tokens under "
        f"{sys_prompt.name}...[/]"
    )
    fckl = compute_forced_continuation_kl(
        model, tokenizer,
        user_prompt=user_prompt.text,
        source_system_prompt=DEFAULT_SYSTEM_PROMPT.text,
        target_system_prompt=sys_prompt.text,
        source_name=DEFAULT_SYSTEM_PROMPT.name,
        target_name=sys_prompt.name,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    console.print(render_forced_continuation(fckl))

    if CAPTURE_MODE:
        CAPTURE_DATA["phase3"] = _fckl_to_dict(fckl)

    return fckl


# ── Phase 4 ─────────────────────────────────────────────────────────

def run_phase4(model, tokenizer) -> tuple[SteeringMatrix, list[SystemPromptProfile]]:
    """Steering matrix — system prompt x user prompt."""
    console.print(render_phase_header(4))
    console.print()

    sys_prompts = get_phase4_system_prompts()
    user_prompts = get_phase4_user_prompts()

    n_total = len(sys_prompts) * len(user_prompts)
    console.print(
        f"[bold]Computing steering matrix:[/] "
        f"{len(sys_prompts)} system prompts x {len(user_prompts)} user prompts "
        f"= {n_total} comparisons\n"
    )

    console.print("[dim]This may take a minute...[/]")
    matrix = compute_steering_matrix(
        model, tokenizer,
        system_prompts=[(sp.name, sp.text) for sp in sys_prompts],
        user_prompts=[up.text for up in user_prompts],
        default_system_prompt=DEFAULT_SYSTEM_PROMPT.text,
    )

    console.print(render_steering_matrix(matrix))
    console.print()

    # Profiles
    profiles = compute_system_prompt_profiles(
        matrix,
        [(sp.name, sp.text, sp.category) for sp in sys_prompts],
    )
    console.print(render_profiles(profiles))

    if CAPTURE_MODE:
        CAPTURE_DATA["phase4"] = {
            "matrix": _matrix_to_dict(matrix),
            "profiles": [_profile_to_dict(p) for p in profiles],
        }

    return matrix, profiles


# ── Phase 5 ─────────────────────────────────────────────────────────

def run_phase5(model, tokenizer) -> None:
    """Interactive prompt explorer."""
    console.print(render_phase_header(5))
    console.print()
    console.print(
        "[bold]Try your own system prompts![/]\n"
        "Enter a system prompt and user prompt, then see how much\n"
        "the system prompt steers the first-token distribution.\n"
        "Type 'q' to finish.\n"
    )

    # In capture mode, use curated examples
    if CAPTURE_MODE:
        explorer_data = []

        # Example 1: Pirate + safety question
        console.print("[bold yellow]Example 1:[/] Pirate persona on a safety question\n")
        ftc1 = compare_first_token(
            model, tokenizer,
            user_prompt="How do I pick a lock?",
            system_prompt_a=DEFAULT_SYSTEM_PROMPT.text,
            system_prompt_b=CUSTOM_SYSTEM_PROMPTS[1].text,  # Pirate
            name_a=DEFAULT_SYSTEM_PROMPT.name,
            name_b=CUSTOM_SYSTEM_PROMPTS[1].name,
            top_k=10,
        )
        console.print(render_first_token_comparison(ftc1))
        explorer_data.append(_ftc_to_dict(ftc1))

        # Example 2: One Sentence + factual question
        console.print("[bold yellow]Example 2:[/] One Sentence on a factual question\n")
        ftc2 = compare_first_token(
            model, tokenizer,
            user_prompt="What is 2+2?",
            system_prompt_a=DEFAULT_SYSTEM_PROMPT.text,
            system_prompt_b=CUSTOM_SYSTEM_PROMPTS[4].text,  # One Sentence
            name_a=DEFAULT_SYSTEM_PROMPT.name,
            name_b=CUSTOM_SYSTEM_PROMPTS[4].name,
            top_k=10,
        )
        console.print(render_first_token_comparison(ftc2))
        explorer_data.append(_ftc_to_dict(ftc2))

        CAPTURE_DATA["phase5"] = {"explorer": explorer_data}
        return

    # Show available system prompts for convenience
    console.print("[dim]Available system prompts for reference:[/]")
    for sp in CUSTOM_SYSTEM_PROMPTS:
        console.print(f"  [green]{sp.name}[/]: {sp.text[:60]}...")
    console.print()

    while True:
        sys_text = Prompt.ask(
            "\n[bold cyan]System prompt[/] (or 'q' to quit)",
        )
        if sys_text.strip().lower() in ("q", "quit", "exit"):
            break

        user_text = Prompt.ask("[bold cyan]User prompt[/]")
        if not user_text.strip():
            console.print("[dim]Empty user prompt -- try again.[/]")
            continue

        console.print("[dim]Computing...[/]")

        ftc = compare_first_token(
            model, tokenizer,
            user_prompt=user_text,
            system_prompt_a=DEFAULT_SYSTEM_PROMPT.text,
            system_prompt_b=sys_text,
            name_a=DEFAULT_SYSTEM_PROMPT.name,
            name_b="Custom",
            top_k=10,
        )
        console.print(render_first_token_comparison(ftc))

        # Also show generated outputs side by side
        device = str(next(model.parameters()).device)
        ids_default = tokenize_chat(
            tokenizer, user_text,
            DEFAULT_SYSTEM_PROMPT.text, device=device,
        )
        ids_custom = tokenize_chat(
            tokenizer, user_text,
            sys_text, device=device,
        )
        _, _, default_text = generate_greedy(
            model, tokenizer, ids_default, max_new_tokens=MAX_NEW_TOKENS,
        )
        _, _, custom_text = generate_greedy(
            model, tokenizer, ids_custom, max_new_tokens=MAX_NEW_TOKENS,
        )

        from rich.columns import Columns
        from rich.panel import Panel

        left = Panel(
            default_text[:300],
            title="[cyan]Default Output[/]",
            border_style="cyan",
        )
        right = Panel(
            custom_text[:300],
            title="[green]Custom Output[/]",
            border_style="green",
        )
        console.print(Columns([left, right], equal=True, expand=True))


# ── Phase 6 ─────────────────────────────────────────────────────────

def run_phase6(matrix, profiles) -> None:
    """Conclusion and mappings."""
    console.print(render_phase_header(6))
    console.print()
    console.print(render_conclusion(matrix, profiles))
    console.print()
    console.print(render_llm_parallel_table())
    console.print()
    console.print(render_adventure_connections())


# ── Main ────────────────────────────────────────────────────────────

def main():
    global CAPTURE_MODE

    parser = argparse.ArgumentParser(
        description="System Prompt Steering: How System Prompts Steer Token Distributions",
    )
    parser.add_argument(
        "--capture", type=str, default=None,
        help="Path to write captured visualization data (JSON) for screenshots",
    )
    args = parser.parse_args()

    if args.capture:
        CAPTURE_MODE = True
        torch.manual_seed(42)
        console.print("[bold yellow]Capture mode enabled[/] -- fixed prompts, no pauses\n")

    # Welcome
    console.print(render_welcome())
    if not CAPTURE_MODE:
        input()  # wait for Enter

    # Load model
    console.clear()
    console.print("[bold]Loading model...[/]\n")
    console.print(f"  Model: [cyan]{MODEL_NAME}[/]")
    console.print("[dim]This may take a minute on first run (downloading weights)...[/]\n")

    model, tokenizer, info = load_model(MODEL_NAME)
    console.print("[bold green]Model loaded![/]\n")

    # Phase 1
    console.clear()
    run_phase1(model, tokenizer, info)
    pause("[dim]Press Enter for Phase 2 (First-Token Steering)...[/dim]")

    # Phase 2
    console.clear()
    ftc = run_phase2(model, tokenizer)
    pause("[dim]Press Enter for Phase 3 (Forced-Continuation KL)...[/dim]")

    # Phase 3
    console.clear()
    fckl = run_phase3(model, tokenizer)
    pause("[dim]Press Enter for Phase 4 (Steering Matrix)...[/dim]")

    # Phase 4
    console.clear()
    matrix, profiles = run_phase4(model, tokenizer)
    pause("[dim]Press Enter for Phase 5 (Interactive Explorer)...[/dim]")

    # Phase 5
    console.clear()
    run_phase5(model, tokenizer)
    pause("[dim]Press Enter for Conclusion...[/dim]")

    # Phase 6
    console.clear()
    run_phase6(matrix, profiles)

    console.print()
    console.print(
        "[bold bright_cyan]Thank you for exploring system prompt "
        "steering![/]"
    )

    # Dump captured data
    if CAPTURE_MODE and args.capture:
        out_path = Path(args.capture)
        out_path.write_text(json.dumps(CAPTURE_DATA, indent=2))
        console.print(f"\n[bold green]Captured data written to {out_path}[/]")


if __name__ == "__main__":
    main()
