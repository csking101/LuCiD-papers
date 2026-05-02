#!/usr/bin/env python3
"""
KL Divergence: Implication on LLM Outputs — Interactive Terminal App
=====================================================================

Run with:  python app.py

Loads a base LLM and its RLHF-aligned variant side-by-side, then
walks through how KL divergence manifests in real token distributions.

Phases:
  1 — Model loading & global KL overview
  2 — Token-level KL anatomy (detailed per-position analysis)
  3 — Where models diverge (curated prompts across categories)
  4 — KL-constrained generation (simulate different beta values)
  5 — Interactive prompt explorer (your own prompts)
  6 — Conclusion & LLM parallel mapping
"""

from __future__ import annotations

import time

from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from kl import (
    compute_category_summaries,
    compute_global_kl,
    compute_sequence_kl,
    compute_token_kl,
    generate_interpolated,
)
from models import (
    DEFAULT_BASE_MODEL,
    DEFAULT_INSTRUCT_MODEL,
    ModelPair,
    get_logits_pair,
    load_model_pair,
)
from prompts import (
    ALL_PROMPTS,
    CATEGORY_DESCRIPTIONS,
    PROMPTS_BY_CATEGORY,
    get_phase2_prompt,
    get_phase3_prompts,
)
from viz import (
    render_adventure_connection,
    render_category_summaries,
    render_conclusion,
    render_global_kl,
    render_interpolated_outputs,
    render_kl_heatmap,
    render_llm_parallel_table,
    render_metrics_table,
    render_model_info,
    render_phase_header,
    render_sequence_comparison,
    render_token_kl_table,
    render_welcome,
)

# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL = DEFAULT_BASE_MODEL
INSTRUCT_MODEL = DEFAULT_INSTRUCT_MODEL
MAX_NEW_TOKENS = 64
INTERPOLATION_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

console = Console()


def pause(msg: str = "[dim]Press Enter to continue...[/dim]") -> None:
    """Block until the user presses Enter."""
    console.print()
    input(msg)


# ── Phase 1 ─────────────────────────────────────────────────────────

def run_phase1(pair: ModelPair) -> tuple[float, list[float]]:
    """Load models and show global KL overview."""
    console.print(render_phase_header(1))
    console.print()
    console.print(render_model_info(pair))
    console.print()

    # Compute global KL across all curated prompts
    all_texts = [p.text for p in ALL_PROMPTS]
    console.print("[dim]Computing KL across curated prompts...[/]")
    global_mean, per_prompt = compute_global_kl(pair, all_texts)

    console.print(render_global_kl(global_mean, per_prompt, all_texts))

    return global_mean, per_prompt


# ── Phase 2 ─────────────────────────────────────────────────────────

def run_phase2(pair: ModelPair) -> None:
    """Token-level KL anatomy for a single prompt."""
    console.print(render_phase_header(2))
    console.print()

    prompt_obj = get_phase2_prompt()
    console.print(f"[bold]Prompt:[/] [italic]\"{prompt_obj.text}\"[/]")
    console.print(f"[dim]{prompt_obj.description}[/]\n")

    # Get logits from both models
    console.print("[dim]Running forward pass on both models...[/]")
    base_tl, inst_tl = get_logits_pair(pair, prompt_obj.text)
    token_kl = compute_token_kl(base_tl, inst_tl, pair.tokenizer, top_k=5)

    # Token-level table
    console.print(render_token_kl_table(token_kl))
    console.print()

    # KL heatmap
    console.print(render_kl_heatmap(token_kl))
    console.print()

    # Generate and compare outputs
    console.print("[dim]Generating continuations from both models...[/]")
    seq_kl = compute_sequence_kl(pair, prompt_obj.text, max_new_tokens=MAX_NEW_TOKENS)
    console.print(render_sequence_comparison(seq_kl))


# ── Phase 3 ─────────────────────────────────────────────────────────

def run_phase3(pair: ModelPair) -> list:
    """Where models diverge -- curated prompts across categories."""
    console.print(render_phase_header(3))
    console.print()

    prompts = get_phase3_prompts()

    for prompt_obj in prompts:
        console.print(f"\n[bold]{prompt_obj.category.upper()}:[/] "
                       f"[italic]\"{prompt_obj.text}\"[/]")
        console.print(f"[dim]{prompt_obj.description}[/]")

        seq_kl = compute_sequence_kl(
            pair, prompt_obj.text, max_new_tokens=MAX_NEW_TOKENS,
        )
        console.print(render_sequence_comparison(seq_kl))

    # Category summaries
    console.print("\n[dim]Computing category-level statistics...[/]")
    summaries = compute_category_summaries(pair, PROMPTS_BY_CATEGORY)
    console.print(render_category_summaries(summaries))

    return summaries


# ── Phase 4 ─────────────────────────────────────────────────────────

def run_phase4(pair: ModelPair) -> None:
    """KL-constrained generation at different alpha values."""
    console.print(render_phase_header(4))
    console.print()

    console.print(
        "[bold]How does the KL budget affect generation?[/]\n\n"
        "We interpolate between the base and instruct model's\n"
        "log-probabilities at each token position:\n\n"
        "  [cyan]log p_mixed = (1-alpha) * log p_base + alpha * log p_instruct[/]\n\n"
        "  alpha=0.0 -> pure base model (no alignment)\n"
        "  alpha=1.0 -> pure instruct model (full RLHF)\n",
    )

    # Let user choose a prompt or use default
    default_prompt = "What are some tips for learning a new programming language?"
    console.print(f"[dim]Default prompt: \"{default_prompt}\"[/]")
    user_input = Prompt.ask(
        "Enter a prompt (or press Enter for default)",
        default=default_prompt,
    )

    console.print(f"\n[dim]Generating at {len(INTERPOLATION_ALPHAS)} alpha values...[/]")

    outputs = []
    for alpha in INTERPOLATION_ALPHAS:
        console.print(f"  [dim]alpha={alpha:.2f}...[/]")
        out = generate_interpolated(
            pair, user_input, alpha=alpha,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        outputs.append(out)

    console.print()
    console.print(render_interpolated_outputs(outputs, user_input))


# ── Phase 5 ─────────────────────────────────────────────────────────

def run_phase5(pair: ModelPair) -> None:
    """Interactive prompt explorer."""
    console.print(render_phase_header(5))
    console.print()
    console.print(
        "[bold]Try your own prompts![/]\n"
        "Type a prompt and see KL analysis. Type 'q' to finish.\n"
        "\n[dim]Suggestions: try a safety-sensitive prompt, a factual question,\n"
        "a creative writing task, or an opinion question to see how\n"
        "KL divergence varies across domains.[/]\n"
    )

    while True:
        user_input = Prompt.ask("\n[bold cyan]Prompt[/]")
        if user_input.strip().lower() in ("q", "quit", "exit"):
            break

        if not user_input.strip():
            console.print("[dim]Empty prompt -- try again or type 'q' to quit.[/]")
            continue

        # Token-level KL
        console.print("[dim]Analysing...[/]")
        base_tl, inst_tl = get_logits_pair(pair, user_input)
        token_kl = compute_token_kl(base_tl, inst_tl, pair.tokenizer, top_k=5)
        console.print(render_token_kl_table(token_kl, title=f"KL for: \"{user_input}\""))
        console.print(render_kl_heatmap(token_kl))

        # Generate and compare
        console.print("[dim]Generating continuations...[/]")
        seq_kl = compute_sequence_kl(pair, user_input, max_new_tokens=MAX_NEW_TOKENS)
        console.print(render_sequence_comparison(seq_kl))


# ── Phase 6 ─────────────────────────────────────────────────────────

def run_phase6(
    global_mean_kl: float,
    summaries: list,
) -> None:
    """Conclusion and mappings."""
    console.print(render_phase_header(6))
    console.print()
    console.print(render_conclusion(global_mean_kl, summaries))
    console.print()
    console.print(render_llm_parallel_table())
    console.print()
    console.print(render_adventure_connection())


# ── Main ────────────────────────────────────────────────────────────

def main():
    # Welcome
    console.print(render_welcome())
    input()  # wait for Enter

    # Load models
    console.clear()
    console.print("[bold]Loading models...[/]\n")
    console.print(f"  Base:     [cyan]{BASE_MODEL}[/]")
    console.print(f"  Instruct: [green]{INSTRUCT_MODEL}[/]")
    console.print("[dim]This may take a minute on first run (downloading weights)...[/]\n")

    pair = load_model_pair(BASE_MODEL, INSTRUCT_MODEL)
    console.print("[bold green]Models loaded![/]\n")

    # Phase 1
    console.clear()
    global_mean_kl, per_prompt_kl = run_phase1(pair)
    pause("[dim]Press Enter for Phase 2 (Token-Level KL Anatomy)...[/dim]")

    # Phase 2
    console.clear()
    run_phase2(pair)
    pause("[dim]Press Enter for Phase 3 (Where Models Diverge)...[/dim]")

    # Phase 3
    console.clear()
    summaries = run_phase3(pair)
    pause("[dim]Press Enter for Phase 4 (KL-Constrained Generation)...[/dim]")

    # Phase 4
    console.clear()
    run_phase4(pair)
    pause("[dim]Press Enter for Phase 5 (Interactive Prompt Explorer)...[/dim]")

    # Phase 5
    console.clear()
    run_phase5(pair)
    pause("[dim]Press Enter for Conclusion...[/dim]")

    # Phase 6
    console.clear()
    run_phase6(global_mean_kl, summaries)

    console.print()
    console.print(
        "[bold bright_cyan]Thank you for exploring KL divergence "
        "in LLM outputs![/]"
    )


if __name__ == "__main__":
    main()
