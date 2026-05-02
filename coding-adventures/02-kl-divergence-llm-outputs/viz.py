"""
Rich terminal rendering for KL divergence analysis.
=====================================================

All visualisations are Rich renderables -- no external display needed.
"""

from __future__ import annotations

from typing import Optional

from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from kl import (
    CategoryKLSummary,
    InterpolatedOutput,
    SequenceKL,
    TokenKL,
)
from models import ModelInfo, ModelPair


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
    filled = int(min(kl / max_kl, 1.0) * width)
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
    text.append("KL Divergence: Implication on LLM Outputs\n", style="bold bright_cyan")
    text.append("=" * 45 + "\n\n", style="dim")
    text.append(
        "How does RLHF change what an LLM says?\n\n",
        style="italic",
    )
    text.append(
        "This demo loads a base model and its RLHF-aligned variant\n"
        "side-by-side, then lets you see exactly where and how\n"
        "alignment shifts the token probability distributions.\n\n",
    )
    text.append(
        "You'll explore:\n",
        style="bold",
    )
    text.append("  1. Global KL overview -- how much did RLHF shift the model?\n")
    text.append("  2. Token-level anatomy -- which tokens changed most?\n")
    text.append("  3. Category comparison -- safety vs style vs helpfulness\n")
    text.append("  4. KL-constrained generation -- simulate different beta values\n")
    text.append("  5. Free exploration -- try your own prompts\n")
    text.append("\nPress Enter to begin (models will be downloaded on first run)...", style="dim")
    return Panel(text, border_style="bright_cyan", title="Adventure 02")


# ── Phase headers ───────────────────────────────────────────────────

PHASE_TITLES = {
    1: ("Model Loading & Global KL Overview", "bright_cyan"),
    2: ("Token-Level KL Anatomy", "green"),
    3: ("Where Models Diverge", "yellow"),
    4: ("KL-Constrained Generation", "bright_magenta"),
    5: ("Interactive Prompt Explorer", "bright_blue"),
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

def render_model_info(pair: ModelPair) -> Panel:
    """Side-by-side model architecture comparison."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("", style="bold", width=14)
    table.add_column("Base Model", style="cyan")
    table.add_column("Instruct Model", style="green")

    b, i = pair.base_info, pair.instruct_info
    table.add_row("Name", b.name, i.name)
    table.add_row("Parameters", f"{b.num_params / 1e9:.2f}B", f"{i.num_params / 1e9:.2f}B")
    table.add_row("Layers", str(b.num_layers), str(i.num_layers))
    table.add_row("Hidden size", str(b.hidden_size), str(i.hidden_size))
    table.add_row("Vocab size", f"{b.vocab_size:,}", f"{i.vocab_size:,}")
    table.add_row("Dtype", b.dtype, i.dtype)
    table.add_row("Device", b.device, i.device)

    return Panel(table, title="Model Comparison", border_style="bright_cyan")


# ── Global KL ───────────────────────────────────────────────────────

def render_global_kl(
    global_mean: float,
    per_prompt_kl: list[float],
    prompt_texts: list[str],
) -> Panel:
    """Summary of global KL divergence across prompts."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("#", style="dim", width=3)
    table.add_column("Prompt", style="white", ratio=3)
    table.add_column("Mean KL", justify="right", width=10)
    table.add_column("", width=22)

    max_kl = max(per_prompt_kl) if per_prompt_kl else 1.0
    for idx, (kl, prompt) in enumerate(zip(per_prompt_kl, prompt_texts)):
        color = _kl_color(kl)
        truncated = prompt[:50] + ("..." if len(prompt) > 50 else "")
        table.add_row(
            str(idx + 1),
            truncated,
            f"[{color}]{kl:.4f}[/]",
            f"[{color}]{_kl_bar(kl, max_kl=max(max_kl, 0.01))}[/]",
        )

    summary = Text()
    summary.append(f"\nGlobal mean KL(instruct || base): ", style="bold")
    summary.append(f"{global_mean:.4f} nats/token", style="bright_cyan")
    summary.append(
        f"\n\nThis means the instruct model's token distribution "
        f"differs from the base model\nby an average of {global_mean:.4f} "
        f"nats at each position -- the 'alignment tax'.",
        style="dim",
    )

    return Panel(
        Group(table, summary),
        title="Global KL Overview",
        border_style="bright_cyan",
    )


# ── Token-level KL table ───────────────────────────────────────────

def render_token_kl_table(
    token_kl: TokenKL,
    title: str = "Token-Level KL Divergence",
    max_rows: Optional[int] = 30,
) -> Panel:
    """Detailed per-token KL table with top-K distributions."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("Pos", style="dim", width=4, justify="right")
    table.add_column("Token", style="bold white", width=14)
    table.add_column("KL", justify="right", width=8)
    table.add_column("", width=22)
    table.add_column("Base Top-3", width=28)
    table.add_column("Instruct Top-3", width=28)

    n = min(max_rows, token_kl.num_tokens) if max_rows else token_kl.num_tokens
    max_kl = max(token_kl.kl_per_token[:n]) if token_kl.kl_per_token else 1.0

    for i in range(n):
        kl = token_kl.kl_per_token[i]
        color = _kl_color(kl)
        tok = repr(token_kl.tokens[i])
        if len(tok) > 12:
            tok = tok[:12] + "..."

        base_top3 = " ".join(
            f"{t}({_fmt_prob(p)})" for t, p in token_kl.base_top_k[i][:3]
        )
        inst_top3 = " ".join(
            f"{t}({_fmt_prob(p)})" for t, p in token_kl.instruct_top_k[i][:3]
        )

        table.add_row(
            str(i),
            tok,
            f"[{color}]{kl:.4f}[/]",
            f"[{color}]{_kl_bar(kl, max_kl=max(max_kl, 0.01))}[/]",
            base_top3,
            inst_top3,
        )

    summary = Text()
    summary.append(f"\nTotal KL: {token_kl.total_kl:.4f}", style="bold")
    summary.append(f"  |  Mean KL: {token_kl.mean_kl:.4f}", style="bold")
    high = token_kl.high_kl_positions(threshold=0.5)
    if high:
        summary.append(f"  |  High-KL positions: {len(high)}/{token_kl.num_tokens}", style="yellow")

    return Panel(Group(table, summary), title=title, border_style="green")


# ── Sequence comparison ─────────────────────────────────────────────

def render_sequence_comparison(seq_kl: SequenceKL) -> Panel:
    """Side-by-side base vs instruct output with KL summary."""
    base_panel = Panel(
        seq_kl.base_text[:300] + ("..." if len(seq_kl.base_text) > 300 else ""),
        title="[cyan]Base Model Output[/]",
        border_style="cyan",
    )
    inst_panel = Panel(
        seq_kl.instruct_text[:300] + ("..." if len(seq_kl.instruct_text) > 300 else ""),
        title="[green]Instruct Model Output[/]",
        border_style="green",
    )

    kl_text = Text()
    kl_text.append(f"Prompt: ", style="bold")
    kl_text.append(f'"{seq_kl.prompt}"', style="italic")
    kl_text.append(f"\nMean KL: {seq_kl.mean_kl:.4f} nats/token", style="bright_cyan")
    kl_text.append(f"  |  Total KL: {seq_kl.total_kl:.4f}", style="dim")

    return Panel(
        Group(kl_text, Text(), Columns([base_panel, inst_panel], equal=True, expand=True)),
        border_style="yellow",
    )


# ── Category comparison ─────────────────────────────────────────────

def render_category_summaries(summaries: list[CategoryKLSummary]) -> Panel:
    """Table comparing KL across prompt categories."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("Category", style="bold", width=14)
    table.add_column("Prompts", justify="center", width=8)
    table.add_column("Mean KL", justify="right", width=10)
    table.add_column("Min KL", justify="right", width=10)
    table.add_column("Max KL", justify="right", width=10)
    table.add_column("", width=22)

    max_kl = max((s.max_kl for s in summaries), default=1.0)
    for s in sorted(summaries, key=lambda x: -x.mean_kl):
        color = _kl_color(s.mean_kl)
        table.add_row(
            s.category.title(),
            str(s.num_prompts),
            f"[{color}]{s.mean_kl:.4f}[/]",
            f"{s.min_kl:.4f}",
            f"{s.max_kl:.4f}",
            f"[{color}]{_kl_bar(s.mean_kl, max_kl=max(max_kl, 0.01))}[/]",
        )

    note = Text()
    if summaries:
        top = max(summaries, key=lambda s: s.mean_kl)
        bot = min(summaries, key=lambda s: s.mean_kl)
        ratio = top.mean_kl / bot.mean_kl if bot.mean_kl > 0 else float("inf")
        note.append(
            f"\n{top.category.title()} prompts have {ratio:.1f}x higher KL "
            f"than {bot.category.title()} prompts.",
            style="italic dim",
        )

    return Panel(
        Group(table, note),
        title="KL Divergence by Category",
        border_style="yellow",
    )


# ── Interpolated generation ─────────────────────────────────────────

def render_interpolated_outputs(
    outputs: list[InterpolatedOutput],
    prompt: str,
) -> Panel:
    """Show generation at multiple alpha values."""
    header = Text()
    header.append("Prompt: ", style="bold")
    header.append(f'"{prompt}"\n', style="italic")
    header.append(
        "alpha=0.0 is pure base model, alpha=1.0 is pure instruct model.\n",
        style="dim",
    )

    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("alpha", style="bold", width=7, justify="center")
    table.add_column("Total KL", justify="right", width=10)
    table.add_column("Generated Text", ratio=3)

    for out in outputs:
        if out.alpha == 0.0:
            label = "[cyan]0.0[/]"
            style = "cyan"
        elif out.alpha == 1.0:
            label = "[green]1.0[/]"
            style = "green"
        else:
            label = f"{out.alpha:.2f}"
            style = "white"

        text_preview = out.text[:200] + ("..." if len(out.text) > 200 else "")
        table.add_row(
            label,
            f"{out.total_kl:.3f}",
            f"[{style}]{text_preview}[/]",
        )

    note = Text()
    if len(outputs) >= 2:
        kl_range = outputs[-1].total_kl - outputs[0].total_kl
        note.append(
            f"\nKL budget ranges from {outputs[0].total_kl:.3f} "
            f"(alpha={outputs[0].alpha}) to {outputs[-1].total_kl:.3f} "
            f"(alpha={outputs[-1].alpha})",
            style="dim",
        )

    return Panel(
        Group(header, table, note),
        title="KL-Constrained Generation",
        border_style="bright_magenta",
    )


# ── KL heatmap (inline sparkline-style) ────────────────────────────

def render_kl_heatmap(token_kl: TokenKL, width: int = 60) -> Panel:
    """Compact KL heatmap showing position-level divergence."""
    text = Text()
    text.append("KL per position: ", style="bold")
    text.append(sparkline(token_kl.kl_per_token, width=width))
    text.append(f"\n\nTokens: {token_kl.num_tokens}")
    text.append(f"  |  Mean: {token_kl.mean_kl:.4f}")
    text.append(f"  |  Total: {token_kl.total_kl:.4f}")

    high = token_kl.high_kl_positions(threshold=0.5)
    if high:
        text.append(f"\nHigh-KL tokens (>0.5): ", style="yellow")
        preview = high[:8]
        for idx in preview:
            tok = token_kl.tokens[idx]
            kl = token_kl.kl_per_token[idx]
            text.append(f" {repr(tok)}", style="bold")
            text.append(f"({kl:.2f})", style="dim")
        if len(high) > 8:
            text.append(f" ... +{len(high) - 8} more", style="dim")

    return Panel(text, title="KL Heatmap", border_style="green")


# ── Metrics table (generic) ────────────────────────────────────────

def render_metrics_table(
    metrics: dict[str, tuple[str, str]],
    title: str = "Metrics",
) -> Panel:
    """Key-value metrics table with optional sparklines."""
    table = Table(show_header=False, expand=True, border_style="dim", padding=(0, 1))
    table.add_column("", style="bold", ratio=1)
    table.add_column("", ratio=1)
    table.add_column("", ratio=1, style="dim")

    for key, (value, spark) in metrics.items():
        table.add_row(key, value, spark)

    return Panel(table, title=title, border_style="dim")


# ── LLM parallel mapping ───────────────────────────────────────────

def render_llm_parallel_table() -> Panel:
    """Mapping between this demo and real LLM RLHF."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("What you see here", style="cyan", ratio=1)
    table.add_column("What happens in LLM RLHF", style="green", ratio=1)

    rows = [
        ("Base model = pre-trained Qwen2.5", "Base LLM before alignment"),
        ("Instruct model = RLHF'd Qwen2.5", "LLM after RLHF fine-tuning"),
        ("KL(instruct || base) per token", "KL penalty in PPO objective"),
        ("High KL at safety tokens", "Model learns to refuse harmful requests"),
        ("High KL at formatting tokens", "Model learns structured output style"),
        ("alpha=0 → base output", "beta=inf → no alignment (pure pre-trained)"),
        ("alpha=1 → instruct output", "beta=0 → full alignment (risk of reward hacking)"),
        ("Interpolated generation", "KL budget controls alignment strength"),
    ]
    for left, right in rows:
        table.add_row(left, right)

    return Panel(table, title="LLM Parallel Mapping", border_style="bright_green")


# ── Adventure 01 connection ─────────────────────────────────────────

def render_adventure_connection() -> Panel:
    """Show the connection to Adventure 01."""
    table = Table(show_header=True, expand=True, border_style="dim")
    table.add_column("Grid World (Adventure 01)", style="cyan", ratio=1)
    table.add_column("Token Space (Adventure 02)", style="green", ratio=1)

    rows = [
        ("Policy = action probs at each cell", "Policy = token probs at each position"),
        ("KL(pi_RLHF || pi_pretrained) over cells", "KL(P_instruct || P_base) over vocab"),
        ("beta=0 → reward hacking, bad paths", "alpha=0 → unaligned base model output"),
        ("High KL where path changed", "High KL where RLHF changed behavior"),
        ("Alignment tax = total KL cost", "Alignment tax = distributional shift"),
    ]
    for left, right in rows:
        table.add_row(left, right)

    return Panel(
        table,
        title="Connection to Adventure 01: Path-Finding Preference Game",
        border_style="bright_yellow",
    )


# ── Conclusion ──────────────────────────────────────────────────────

def render_conclusion(
    global_mean_kl: float,
    summaries: list[CategoryKLSummary],
) -> Panel:
    """Final summary panel."""
    text = Text()
    text.append("Key Findings\n", style="bold bright_green underline")
    text.append(f"\n  Global mean KL: {global_mean_kl:.4f} nats/token\n", style="bright_cyan")

    if summaries:
        top = max(summaries, key=lambda s: s.mean_kl)
        bot = min(summaries, key=lambda s: s.mean_kl)
        text.append(f"\n  Highest divergence: {top.category.title()} ({top.mean_kl:.4f})")
        text.append(f"\n  Lowest divergence:  {bot.category.title()} ({bot.mean_kl:.4f})")
        if bot.mean_kl > 0:
            text.append(f"\n  Ratio: {top.mean_kl / bot.mean_kl:.1f}x")

    text.append(
        "\n\n  The KL divergence is not uniform -- RLHF reshapes the model's\n"
        "  distributions most aggressively at safety-critical and formatting\n"
        "  tokens, while leaving factual knowledge relatively untouched.\n"
        "  This is the 'surgical' nature of alignment.\n",
        style="dim",
    )

    return Panel(text, border_style="bright_green", title="Conclusion")
