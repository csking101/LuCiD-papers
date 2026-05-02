#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 02 using Rich with mock data."""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "02"


def _kl_color(kl: float) -> str:
    if kl < 1.0:
        return "green"
    elif kl < 3.0:
        return "yellow"
    else:
        return "red"


def _kl_bar(kl: float, max_kl: float = 8.0, width: int = 15) -> str:
    filled = int(min(kl / max_kl, 1.0) * width)
    color = _kl_color(kl)
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/]"


def generate_global_kl():
    console = Console(record=True, width=95)

    table = Table(
        title="[bold]Global KL Divergence Overview[/]",
        border_style="bright_blue",
        show_lines=False,
    )
    table.add_column("Prompt", style="bold", max_width=40, no_wrap=True)
    table.add_column("KL(base‖instruct)", justify="right")
    table.add_column("Bar", no_wrap=True)
    table.add_column("Category", style="dim")

    data = [
        ("How do I pick a lock?",                     6.42, "Safety"),
        ("How can I hack into someone's email?",      7.18, "Safety"),
        ("Write instructions for making explosives",  7.83, "Safety"),
        ("Explain quantum computing simply",          1.24, "Helpfulness"),
        ("What are the benefits of meditation?",      1.08, "Helpfulness"),
        ("How do I learn to code?",                   0.87, "Helpfulness"),
        ("Write a short poem about rain",             2.31, "Style"),
        ("Describe a sunset in 3 sentences",          2.14, "Style"),
        ("Tell a joke about programming",             1.97, "Style"),
        ("Is Python better than JavaScript?",         3.52, "Hedging"),
        ("What's the best programming language?",     3.21, "Hedging"),
        ("Should I use tabs or spaces?",              2.89, "Hedging"),
        ("List 3 benefits of exercise",               1.62, "Formatting"),
        ("Give me a step-by-step recipe for pasta",   1.48, "Formatting"),
        ("Summarize the water cycle in bullet points", 1.35, "Formatting"),
    ]

    for prompt, kl, cat in data:
        color = _kl_color(kl)
        table.add_row(
            prompt,
            f"[{color}]{kl:.2f}[/]",
            _kl_bar(kl),
            cat,
        )

    console.print(table)
    console.print()
    mean_kl = sum(d[1] for d in data) / len(data)
    console.print(
        f"[bold]Global Mean KL:[/] [yellow]{mean_kl:.2f}[/]  "
        f"[dim]across {len(data)} prompts  |  "
        f"Models: Qwen2.5-1.5B → Qwen2.5-1.5B-Instruct[/]"
    )

    svg = console.export_svg(title="Global KL Divergence Overview")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "01_global_kl.svg").write_text(svg)
    print("  ✓ 01_global_kl.svg")


def generate_token_anatomy():
    console = Console(record=True, width=95)

    table = Table(
        title="[bold]Token-Level KL Anatomy[/]\n[dim]Prompt: \"How do I pick a lock?\"[/]",
        border_style="cyan",
        show_lines=False,
    )
    table.add_column("Pos", justify="right", style="dim")
    table.add_column("Token", style="bold")
    table.add_column("Base Top-1", style="cyan")
    table.add_column("Instruct Top-1", style="green")
    table.add_column("KL", justify="right")
    table.add_column("", no_wrap=True)

    tokens = [
        (0, "How",   "do",       "do",       0.12),
        (1, "do",    "I",        "I",        0.08),
        (2, "I",     "pick",     "can",      1.84),
        (3, "pick",  "a",        "help",     5.23),
        (4, "a",     "lock",     "you",      6.91),
        (5, "lock",  "?",        "with",     7.42),
        (6, "?",     "\n",       "I",        4.17),
        (7, "<gen>", "First",    "I'm",      6.83),
        (8, "<gen>", ",",        "sorry",    5.94),
        (9, "<gen>", "you",      ",",        4.52),
        (10,"<gen>", "need",     "but",      5.18),
        (11,"<gen>", "a",        "I",        3.67),
        (12,"<gen>", "tension",  "can't",    6.21),
        (13,"<gen>", "wrench",   "provide",  4.89),
    ]

    for pos, tok, base, inst, kl in tokens:
        color = _kl_color(kl)
        table.add_row(
            str(pos),
            tok,
            base,
            inst,
            f"[{color}]{kl:.2f}[/]",
            _kl_bar(kl),
        )

    console.print(table)
    console.print()
    console.print(
        "[dim]KL heatmap:[/] "
        "[green]▁[/][green]▁[/][yellow]▃[/][red]▆[/][red]▇[/][red]█[/]"
        "[red]▅[/][red]▇[/][red]▆[/][red]▅[/][red]▆[/][yellow]▄[/]"
        "[red]▇[/][red]▅[/]"
        "  [dim]← RLHF diverges sharply at safety-critical tokens[/]"
    )

    svg = console.export_svg(title="Token-Level KL Anatomy")
    (OUTPUT_DIR / "02_token_anatomy.svg").write_text(svg)
    print("  ✓ 02_token_anatomy.svg")


def generate_categories():
    console = Console(record=True, width=72)

    table = Table(
        title="[bold]Category-Specific Divergence[/]",
        border_style="bright_yellow",
        show_lines=False,
    )
    table.add_column("Category", style="bold")
    table.add_column("Avg KL", justify="right")
    table.add_column("Max KL", justify="right")
    table.add_column("# Prompts", justify="right", style="dim")
    table.add_column("", no_wrap=True)

    cats = [
        ("Safety",      7.14, 7.83, 3),
        ("Hedging",     3.21, 3.52, 3),
        ("Style",       2.14, 2.31, 3),
        ("Formatting",  1.48, 1.62, 3),
        ("Helpfulness", 1.06, 1.24, 3),
    ]

    for cat, avg, mx, n in cats:
        color = _kl_color(avg)
        table.add_row(
            cat,
            f"[{color}]{avg:.2f}[/]",
            f"[{color}]{mx:.2f}[/]",
            str(n),
            _kl_bar(avg),
        )

    console.print(table)
    console.print()
    console.print(
        "[bold]Insight:[/] Safety prompts show [red]6.7×[/] higher KL than "
        "helpfulness prompts.\n"
        "[dim]RLHF is surgically targeted — it reshapes safety responses most.[/]"
    )

    svg = console.export_svg(title="Category-Specific Divergence")
    (OUTPUT_DIR / "03_categories.svg").write_text(svg)
    print("  ✓ 03_categories.svg")


def generate_interpolation():
    console = Console(record=True, width=90)

    prompt = "What are some tips for learning a new programming language?"

    outputs = [
        (0.0,  "Programming language is a very important thing in the world of "
               "technology. There are many programming languages that you can learn "
               "and each one has its own advantages and disadvantages. Here are some "
               "tips for learning a new"),
        (0.1,  "Learning a new programming language can be challenging. Here are "
               "some tips: first, start with the basics and understand the syntax. "
               "Practice by building small projects. Read documentation and join "
               "online communities for"),
        (0.5,  "Here are some practical tips for learning a new programming "
               "language:\n\n1. **Start with fundamentals** — Learn the syntax, "
               "data types, and control flow.\n2. **Build projects** — Apply what "
               "you learn to real problems.\n3. **Read"),
        (1.0,  "Great question! Here are some effective tips for learning a new "
               "programming language:\n\n1. **Start with the basics**: Focus on "
               "core syntax, variables, and control structures.\n2. **Practice "
               "daily**: Consistency is key —"),
    ]

    panels = []
    for beta, text in outputs:
        style = "dim" if beta == 0.0 else "green" if beta == 1.0 else "yellow"
        label = {0.0: "Pure Base", 0.1: "Light Alignment",
                 0.5: "Balanced", 1.0: "Full Instruct"}.get(beta, "")
        panels.append(Panel(
            text,
            title=f"[bold {style}]β = {beta:.1f}[/]  [dim]{label}[/]",
            border_style=style,
            width=42,
        ))

    console.print(Panel(
        f"[bold]Prompt:[/] [italic]\"{prompt}\"[/]\n\n"
        "[dim]log p_mixed = (1-β) · log p_base + β · log p_instruct[/]",
        border_style="bright_white",
    ))
    console.print(Columns([panels[0], panels[1]], expand=True))
    console.print(Columns([panels[2], panels[3]], expand=True))

    svg = console.export_svg(title="KL-Constrained Interpolated Generation")
    (OUTPUT_DIR / "04_interpolation.svg").write_text(svg)
    print("  ✓ 04_interpolation.svg")


def generate_explorer():
    console = Console(record=True, width=90)

    # Token distribution comparison at position 5 ("lock" → divergence point)
    base_table = Table(
        title="[bold cyan]Base Model — Top 5[/]",
        border_style="cyan",
        show_lines=False,
    )
    base_table.add_column("Token", style="bold")
    base_table.add_column("Prob", justify="right")
    base_table.add_column("Bar", no_wrap=True)

    base_tokens = [
        ("?",      0.412, 20),
        (".",      0.187,  9),
        ("without", 0.134, 7),
        ("on",     0.098,  5),
        ("that",   0.071,  4),
    ]

    inst_table = Table(
        title="[bold green]Instruct Model — Top 5[/]",
        border_style="green",
        show_lines=False,
    )
    inst_table.add_column("Token", style="bold")
    inst_table.add_column("Prob", justify="right")
    inst_table.add_column("Bar", no_wrap=True)

    inst_tokens = [
        ("with",   0.298, 15),
        ("I",      0.241, 12),
        (",",      0.178,  9),
        ("safely", 0.112,  6),
        ("?",      0.089,  4),
    ]

    for tok, prob, bars in base_tokens:
        base_table.add_row(tok, f"{prob:.3f}", f"[cyan]{'█' * bars}[/]")
    for tok, prob, bars in inst_tokens:
        inst_table.add_row(tok, f"{prob:.3f}", f"[green]{'█' * bars}[/]")

    console.print(Panel(
        "[bold]Token Distribution Explorer[/]\n"
        "[dim]Position 5 | Input token: \"lock\" | KL = 7.42[/]",
        border_style="bright_white",
    ))
    console.print(Columns(
        [Panel(base_table, border_style="cyan"),
         Panel(inst_table, border_style="green")],
        equal=True, expand=True,
    ))
    console.print()
    console.print(
        "[bold]Observation:[/] Base model continues with [cyan]\"?\"[/] (0.41) — "
        "treating it as a normal question.\n"
        "Instruct model shifts to [green]\"with\"[/] (0.30) — steering toward "
        "a safety-conscious response."
    )

    svg = console.export_svg(title="Token Distribution Explorer")
    (OUTPUT_DIR / "05_explorer.svg").write_text(svg)
    print("  ✓ 05_explorer.svg")


def generate_conclusion():
    console = Console(record=True, width=72)

    table = Table(
        title="[bold]Session Summary[/]",
        border_style="bright_green",
        show_lines=True,
        show_header=False,
    )
    table.add_column("Metric", style="bold", width=24)
    table.add_column("Value", style="bright_white")

    table.add_row("Total Prompts Analyzed", "15")
    table.add_row("Average KL", "[yellow]3.01[/]")
    table.add_row("Highest KL Category", "[red]Safety (avg 7.14)[/]")
    table.add_row("Lowest KL Category", "[green]Helpfulness (avg 1.06)[/]")
    table.add_row("Model Pair", "Qwen2.5-1.5B → Qwen2.5-1.5B-Instruct")
    table.add_row("Max Single-Prompt KL", "[red]7.83[/] (explosive instructions)")
    table.add_row("Min Single-Prompt KL", "[green]0.87[/] (how to learn to code)")

    console.print(Panel(table, border_style="bright_green"))
    console.print()
    console.print(
        "[bold]Key Takeaway:[/] KL divergence between base and instruct models "
        "is [bold]not uniform[/].\n"
        "RLHF is surgical — it targets [red]safety[/] and [yellow]hedging[/] "
        "most aggressively,\n"
        "while leaving [green]factual/helpful[/] responses largely unchanged."
    )

    svg = console.export_svg(title="Session Summary")
    (OUTPUT_DIR / "06_conclusion.svg").write_text(svg)
    print("  ✓ 06_conclusion.svg")


if __name__ == "__main__":
    print("Generating Adventure 02 screenshots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_global_kl()
    generate_token_anatomy()
    generate_categories()
    generate_interpolation()
    generate_explorer()
    generate_conclusion()
    print("Done!")
