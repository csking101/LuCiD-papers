"""
Rich terminal rendering for system prompt steering analysis.
=============================================================

All visualisations are Rich renderables -- no external display needed.
"""

from __future__ import annotations

from typing import Optional

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analysis import (
    FirstTokenComparison,
    ForcedContinuationKL,
    SteeringMatrix,
    SystemPromptProfile,
)
from models import ModelInfo


# ── Helpers ─────────────────────────────────────────────────────────

def sparkline(values: list[float], width: int = 20) -> str:
    """Tiny Unicode sparkline for a list of floats."""
    if not values:
        return ""
    blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    recent = values[-width:]
    return "".join(
        blocks[min(int((v - mn) / rng * 7) + 1, 8)] for v in recent
    )


def _kl_color(kl: float) -> str:
    """Return a Rich color name based on KL magnitude."""
    if kl < 0.1:
        return "dim"
    if kl < 0.5:
        return "green"
    if kl < 1.0:
        return "yellow"
    if kl < 2.0:
        return "bright_red"
    return "bold bright_red"


def _kl_bar(kl: float, max_kl: float = 3.0, width: int = 20) -> str:
    """Simple horizontal bar for a KL value."""
    filled = int(min(kl / max(max_kl, 0.001), 1.0) * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def _fmt_prob(p: float) -> str:
    """Format a probability as a compact string."""
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


# ── Welcome ─────────────────────────────────────────────────────────

def render_welcome() -> Panel:
    """Adventure title screen."""
    text = Text()
    text.append("System Prompt Steering\n", style="bold bright_cyan")
    text.append("How System Prompts Steer Token Distributions\n", style="italic")
    text.append("=" * 48 + "\n\n", style="dim")
    text.append(
        "How much control does a system prompt have over\n"
        "what an LLM says next?\n\n",
        style="italic",
    )
    text.append(
        "This demo loads a single instruct model and compares\n"
        "its token distributions under different system prompts.\n"
        "You'll see exactly how much each system prompt 'steers'\n"
        "the model's first-token probabilities.\n\n",
    )
    text.append("You'll explore:\n", style="bold")
    text.append("  1. Chat template anatomy -- how system prompts are formatted\n")
    text.append("  2. First-token steering -- microscope view of probability shifts\n")
    text.append("  3. Forced continuation -- where the system prompt fights hardest\n")
    text.append("  4. Steering matrix -- which prompts steer which questions most\n")
    text.append("  5. Free exploration -- try your own system prompts\n")
    text.append(
        "\nPress Enter to begin (model will be downloaded on first run)...",
        style="dim",
    )
    return Panel(text, border_style="bright_cyan", title="Adventure 04")


# ── Phase headers ───────────────────────────────────────────────────

PHASE_TITLES = {
    1: ("Model & Chat Template", "bright_cyan"),
    2: ("First-Token Steering", "green"),
    3: ("Forced-Continuation KL Profile", "yellow"),
    4: ("Steering Matrix", "bright_magenta"),
    5: ("Interactive Explorer", "bright_blue"),
    6: ("Conclusion", "bright_green"),
}


def render_phase_header(phase: int) -> Panel:
    """Render a phase title banner."""
    title, color = PHASE_TITLES.get(phase, (f"Phase {phase}", "white"))
    return Panel(
        f"[bold {color}]Phase {phase}: {title}[/]",
        border_style=color,
        expand=True,
    )


# ── Model info ──────────────────────────────────────────────────────

def render_model_info(info: ModelInfo) -> Panel:
    """Model architecture summary."""
    table = Table(show_header=False, expand=True, border_style="dim")
    table.add_column("", style="bold", width=14)
    table.add_column("", style="cyan")

    table.add_row("Name", info.name)
    table.add_row("Parameters", f"{info.num_params / 1e9:.2f}B")
    table.add_row("Layers", str(info.num_layers))
    table.add_row("Hidden size", str(info.hidden_size))
    table.add_row("Vocab size", f"{info.vocab_size:,}")
    table.add_row("Dtype", info.dtype)
    table.add_row("Device", info.device)

    return Panel(table, title="Model Info", border_style="bright_cyan")


# ── Chat template display ──────────────────────────────────────────

def render_chat_template(
    formatted_default: str,
    formatted_custom: str,
    default_name: str = "Default",
    custom_name: str = "Custom",
) -> Panel:
    """Show side-by-side chat template formatting."""
    left = Panel(
        formatted_default,
        title=f"[cyan]{default_name}[/]",
        border_style="cyan",
    )
    right = Panel(
        formatted_custom,
        title=f"[green]{custom_name}[/]",
        border_style="green",
    )

    note = Text()
    note.append(
        "The chat template wraps system + user messages with special tokens.\n",
        style="dim",
    )
    note.append(
        "Different system prompts change the prefix tokens, which shifts\n"
        "the model's hidden state before it generates the first response token.",
        style="dim",
    )

    return Panel(
        Group(Columns([left, right], equal=True, expand=True), Text(), note),
        title="Chat Template Comparison",
        border_style="bright_cyan",
    )


# ── First-token comparison ─────────────────────────────────────────

def render_first_token_comparison(ftc: FirstTokenComparison) -> Panel:
    """Detailed first-token distribution comparison."""
    # Header
    header = Text()
    header.append("User prompt: ", style="bold")
    header.append(f'"{ftc.user_prompt}"\n', style="italic")
    header.append(f"Comparing: ", style="bold")
    header.append(f"{ftc.system_prompt_a}", style="cyan")
    header.append(" vs ", style="dim")
    header.append(f"{ftc.system_prompt_b}\n", style="green")

    # Metrics
    kl_color = _kl_color(ftc.kl_divergence)
    metrics = Text()
    metrics.append(f"\nKL(Custom || Default): ", style="bold")
    metrics.append(f"{ftc.kl_divergence:.4f} nats", style=kl_color)
    metrics.append(f"   JS divergence: ", style="bold")
    metrics.append(f"{ftc.js_divergence:.4f} nats\n", style="dim")

    # Top-K tables side by side
    table_a = Table(
        title=f"[cyan]{ftc.system_prompt_a} Top Tokens[/]",
        show_header=True, border_style="cyan",
    )
    table_a.add_column("Token", width=14)
    table_a.add_column("Prob", justify="right", width=8)
    for tok, prob in ftc.top_k_a[:8]:
        table_a.add_row(repr(tok), _fmt_prob(prob))

    table_b = Table(
        title=f"[green]{ftc.system_prompt_b} Top Tokens[/]",
        show_header=True, border_style="green",
    )
    table_b.add_column("Token", width=14)
    table_b.add_column("Prob", justify="right", width=8)
    for tok, prob in ftc.top_k_b[:8]:
        table_b.add_row(repr(tok), _fmt_prob(prob))

    # Top shifts table
    shift_table = Table(
        title="Biggest Probability Shifts",
        show_header=True, border_style="yellow",
    )
    shift_table.add_column("Token", width=14)
    shift_table.add_column(ftc.system_prompt_a, justify="right", width=10)
    shift_table.add_column(ftc.system_prompt_b, justify="right", width=10)
    shift_table.add_column("Delta", justify="right", width=10)

    for tok, p_a, p_b in ftc.top_shifts[:8]:
        delta = p_b - p_a
        delta_color = "green" if delta > 0 else "red"
        shift_table.add_row(
            repr(tok),
            _fmt_prob(p_a),
            _fmt_prob(p_b),
            f"[{delta_color}]{delta:+.4f}[/]",
        )

    return Panel(
        Group(
            header,
            metrics,
            Text(),
            Columns([table_a, table_b], equal=True, expand=True),
            Text(),
            shift_table,
        ),
        title="First-Token Distribution Comparison",
        border_style="green",
    )


# ── Forced continuation KL ─────────────────────────────────────────

def render_forced_continuation(fckl: ForcedContinuationKL) -> Panel:
    """Show per-token KL profile for forced continuation."""
    header = Text()
    header.append("User prompt: ", style="bold")
    header.append(f'"{fckl.user_prompt}"\n', style="italic")
    header.append("Source: ", style="bold")
    header.append(f"{fckl.source_system_prompt}", style="cyan")
    header.append("  Target: ", style="bold")
    header.append(f"{fckl.target_system_prompt}\n", style="green")

    # Side-by-side generated text
    source_panel = Panel(
        fckl.source_text[:300] + ("..." if len(fckl.source_text) > 300 else ""),
        title=f"[cyan]{fckl.source_system_prompt} Output[/]",
        border_style="cyan",
    )
    target_panel = Panel(
        fckl.target_text[:300] + ("..." if len(fckl.target_text) > 300 else ""),
        title=f"[green]{fckl.target_system_prompt} Output[/]",
        border_style="green",
    )

    # Per-token KL table
    if fckl.continuation_tokens:
        table = Table(show_header=True, expand=True, border_style="dim")
        table.add_column("Pos", style="dim", width=4, justify="right")
        table.add_column("Token", style="bold white", width=14)
        table.add_column("KL", justify="right", width=8)
        table.add_column("", width=22)

        max_kl = max(fckl.kl_per_token) if fckl.kl_per_token else 1.0
        n = min(30, len(fckl.continuation_tokens))
        for i in range(n):
            kl = fckl.kl_per_token[i]
            color = _kl_color(kl)
            tok = repr(fckl.continuation_tokens[i])
            if len(tok) > 12:
                tok = tok[:12] + "..."
            table.add_row(
                str(i),
                tok,
                f"[{color}]{kl:.4f}[/]",
                f"[{color}]{_kl_bar(kl, max_kl=max(max_kl, 0.01))}[/]",
            )
        if len(fckl.continuation_tokens) > 30:
            table.add_row("...", f"+{len(fckl.continuation_tokens) - 30} more", "", "")
    else:
        table = Text("[dim]No continuation tokens generated[/]")

    # Summary
    summary = Text()
    summary.append(f"\nTotal KL: {fckl.total_kl:.4f}", style="bold")
    summary.append(f"  |  Mean KL: {fckl.mean_kl:.4f}", style="bold")
    if fckl.kl_per_token:
        summary.append(f"  |  Max KL: {max(fckl.kl_per_token):.4f}", style="yellow")
        summary.append(
            f"\n\nSparkline: {sparkline(fckl.kl_per_token, width=40)}",
            style="dim",
        )
        # High-KL positions
        high = [i for i, kl in enumerate(fckl.kl_per_token) if kl > 0.5]
        if high:
            summary.append(f"\nHigh-KL positions (>0.5): {len(high)}/{len(fckl.kl_per_token)}", style="yellow")

    return Panel(
        Group(
            header,
            Columns([source_panel, target_panel], equal=True, expand=True),
            Text(),
            table,
            summary,
        ),
        title="Forced-Continuation KL Profile",
        border_style="yellow",
    )


# ── Steering matrix ────────────────────────────────────────────────

def render_steering_matrix(matrix: SteeringMatrix) -> Panel:
    """Heatmap-style table of system prompt x user prompt KL values."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("System Prompt", style="bold", width=18)

    # Column headers: truncated user prompts
    for up in matrix.user_prompts:
        truncated = up[:20] + ("..." if len(up) > 20 else "")
        table.add_column(truncated, justify="right", width=12)
    table.add_column("Mean", justify="right", width=8, style="bold")

    # Rows: one per system prompt
    for i, sys_name in enumerate(matrix.system_prompt_names):
        row: list[str] = [sys_name]
        for j in range(len(matrix.user_prompts)):
            kl = matrix.kl_matrix[i][j]
            color = _kl_color(kl)
            row.append(f"[{color}]{kl:.3f}[/]")
        row_mean = matrix.row_means[i]
        row_color = _kl_color(row_mean)
        row.append(f"[{row_color}]{row_mean:.3f}[/]")
        table.add_row(*row)

    # Column means footer
    footer: list[str] = ["[bold]Mean[/]"]
    for cm in matrix.col_means:
        color = _kl_color(cm)
        footer.append(f"[{color}]{cm:.3f}[/]")
    color = _kl_color(matrix.global_mean)
    footer.append(f"[{color}]{matrix.global_mean:.3f}[/]")
    table.add_row(*footer)

    # Summary
    summary = Text()
    summary.append(f"\nGlobal mean steering KL: ", style="bold")
    summary.append(f"{matrix.global_mean:.4f} nats", style="bright_cyan")

    if matrix.row_means:
        top_sys_idx = max(range(len(matrix.row_means)), key=lambda i: matrix.row_means[i])
        bot_sys_idx = min(range(len(matrix.row_means)), key=lambda i: matrix.row_means[i])
        summary.append(f"\nStrongest steerer: {matrix.system_prompt_names[top_sys_idx]}", style="green")
        summary.append(f" ({matrix.row_means[top_sys_idx]:.4f})")
        summary.append(f"\nWeakest steerer: {matrix.system_prompt_names[bot_sys_idx]}", style="dim")
        summary.append(f" ({matrix.row_means[bot_sys_idx]:.4f})")

    if matrix.col_means:
        top_user_idx = max(range(len(matrix.col_means)), key=lambda i: matrix.col_means[i])
        bot_user_idx = min(range(len(matrix.col_means)), key=lambda i: matrix.col_means[i])
        top_prompt = matrix.user_prompts[top_user_idx][:30]
        bot_prompt = matrix.user_prompts[bot_user_idx][:30]
        summary.append(f"\nMost steerable prompt: \"{top_prompt}\"", style="yellow")
        summary.append(f" ({matrix.col_means[top_user_idx]:.4f})")
        summary.append(f"\nLeast steerable: \"{bot_prompt}\"", style="dim")
        summary.append(f" ({matrix.col_means[bot_user_idx]:.4f})")

    return Panel(
        Group(table, summary),
        title="Steering Matrix: KL(Custom || Default)",
        border_style="bright_magenta",
    )


# ── System prompt profiles ─────────────────────────────────────────

def render_profiles(profiles: list[SystemPromptProfile]) -> Panel:
    """Summary table of system prompt steering power."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("System Prompt", style="bold", width=18)
    table.add_column("Category", width=12)
    table.add_column("Mean KL", justify="right", width=10)
    table.add_column("Max KL", justify="right", width=10)
    table.add_column("", width=22)

    max_kl = max((p.max_steering_power for p in profiles), default=1.0)
    for p in sorted(profiles, key=lambda x: -x.mean_steering_power):
        color = _kl_color(p.mean_steering_power)
        table.add_row(
            p.system_prompt_name,
            p.category,
            f"[{color}]{p.mean_steering_power:.4f}[/]",
            f"{p.max_steering_power:.4f}",
            f"[{color}]{_kl_bar(p.mean_steering_power, max_kl=max(max_kl, 0.01))}[/]",
        )

    return Panel(table, title="System Prompt Steering Power", border_style="bright_magenta")


# ── LLM parallel mapping ───────────────────────────────────────────

def render_llm_parallel_table() -> Panel:
    """Mapping between this demo and real LLM RLHF."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("What you see here", style="cyan", ratio=1)
    table.add_column("What happens in LLM RLHF", style="green", ratio=1)

    rows = [
        ("System prompt = soft constraint", "KL penalty = hard constraint in PPO objective"),
        ("Different system prompts shift distributions", "Different beta values shift the KL budget"),
        ("First-token KL measures steering power", "Per-token KL penalty constrains policy updates"),
        ("Persona prompts force vocabulary shifts", "RLHF trains new response patterns"),
        ("Safety prompts add refusal probability", "Safety RLHF teaches refusal behavior"),
        ("Forced continuation shows 'tension'", "KL penalty resists large policy changes"),
        ("Steering matrix = prompt x question", "Reward model scores vary by context"),
        ("Global mean KL = average steering cost", "Total KL budget = alignment tax"),
    ]
    for left, right in rows:
        table.add_row(left, right)

    return Panel(table, title="LLM Parallel Mapping", border_style="bright_green")


# ── Adventure connections ───────────────────────────────────────────

def render_adventure_connections() -> Panel:
    """Connection to Adventures 01 and 02."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("Adventure 01 (Grid World)", style="cyan", ratio=1)
    table.add_column("Adventure 02 (Base vs Instruct)", style="yellow", ratio=1)
    table.add_column("Adventure 04 (System Prompts)", style="green", ratio=1)

    rows = [
        (
            "KL(new_policy || old_policy)",
            "KL(instruct || base) per token",
            "KL(custom_prompt || default_prompt)",
        ),
        (
            "beta controls alignment strength",
            "alpha interpolates base & instruct",
            "System prompt text controls steering",
        ),
        (
            "Path changes at high-KL cells",
            "Output changes at high-KL tokens",
            "First token shifts under new prompt",
        ),
        (
            "Hard constraint (objective term)",
            "Hard constraint (model weights)",
            "Soft constraint (input conditioning)",
        ),
        (
            "Reward hacking without KL",
            "Unaligned base model output",
            "Default prompt = minimal steering",
        ),
    ]
    for left, mid, right in rows:
        table.add_row(left, mid, right)

    return Panel(
        table,
        title="Connections Across Adventures",
        border_style="bright_yellow",
    )


# ── Conclusion ──────────────────────────────────────────────────────

def render_conclusion(
    matrix: SteeringMatrix,
    profiles: list[SystemPromptProfile],
) -> Panel:
    """Final summary panel."""
    text = Text()
    text.append("Key Findings\n", style="bold bright_green underline")
    text.append(
        f"\n  Global mean steering KL: {matrix.global_mean:.4f} nats",
        style="bright_cyan",
    )

    if profiles:
        top = max(profiles, key=lambda p: p.mean_steering_power)
        bot = min(profiles, key=lambda p: p.mean_steering_power)
        text.append(f"\n  Strongest steerer: {top.system_prompt_name} ({top.mean_steering_power:.4f})")
        text.append(f"\n  Weakest steerer: {bot.system_prompt_name} ({bot.mean_steering_power:.4f})")
        if bot.mean_steering_power > 0:
            ratio = top.mean_steering_power / bot.mean_steering_power
            text.append(f"\n  Ratio: {ratio:.1f}x")

    text.append(
        "\n\n  System prompts are 'soft' constraints -- they change the model's\n"
        "  input context rather than its weights. Yet they can significantly\n"
        "  steer token distributions, especially for persona and formatting.\n"
        "  Compare this to RLHF (Adventure 02), which permanently reshapes\n"
        "  the model's weights -- a 'hard' constraint.\n",
        style="dim",
    )

    return Panel(text, border_style="bright_green", title="Conclusion")
