#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 01 using Rich with mock data."""

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "01"


def generate_grid_world():
    console = Console(record=True, width=80)

    # 8x8 grid layout
    grid = [
        ["S", ".", ".", "#", ".", "$", ".", "."],
        [".", "$", ".", "#", ".", ".", ".", "."],
        [".", ".", ".", ".", ".", "#", "*", "."],
        ["#", "#", ".", "#", ".", "#", ".", "."],
        [".", ".", "$", ".", ".", ".", ".", "$"],
        [".", "#", ".", "#", "#", ".", "*", "."],
        [".", ".", ".", ".", "$", ".", ".", "."],
        ["T", ".", "$", ".", ".", "*", ".", "G"],
    ]

    cell_styles = {
        "S": ("S", "bold white on blue"),
        "G": ("G", "bold white on green"),
        "T": ("T", "bold white on red"),
        "#": ("#", "white on rgb(60,60,60)"),
        "$": ("$", "bold yellow on rgb(40,40,40)"),
        "*": ("*", "bold magenta on rgb(40,40,40)"),
        ".": (" ", "on rgb(30,30,30)"),
    }

    table = Table(
        title="[bold]Grid World Environment[/]",
        show_header=False,
        show_lines=True,
        border_style="bright_blue",
        padding=(0, 1),
    )
    for _ in range(8):
        table.add_column(width=3, justify="center")

    for row in grid:
        cells = []
        for cell in row:
            char, style = cell_styles[cell]
            cells.append(f"[{style}] {char} [/]")
        table.add_row(*cells)

    legend = Text.from_markup(
        "[bold]Legend:[/]  "
        "[bold white on blue] S [/] Start  "
        "[bold white on green] G [/] Goal  "
        "[bold white on red] T [/] Trap  "
        "[white on rgb(60,60,60)] # [/] Wall  "
        "[bold yellow] $ [/] Coin (+0.5)  "
        "[bold magenta] * [/] Gem (+2.0)"
    )

    console.print(Panel(table, border_style="bright_blue", subtitle="8×8 with 3-corridor layout"))
    console.print(legend)

    svg = console.export_svg(title="Grid World Environment")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "01_grid_world.svg").write_text(svg)
    print("  ✓ 01_grid_world.svg")


def generate_preferences():
    console = Console(record=True, width=90)

    # Segment A
    seg_a = Table(title="[bold cyan]Segment A[/]", border_style="cyan", show_lines=True, show_header=False)
    seg_a.add_column(width=3, justify="center")
    for _ in range(7):
        seg_a.add_column(width=3, justify="center")

    path_a = {(0,0),(0,1),(0,2),(1,2),(2,2),(2,3),(2,4),(3,4),(4,4),(4,5),(4,6),(4,7),
              (5,7),(6,7),(7,7)}
    grid_a_rows = []
    for r in range(8):
        row = []
        for c in range(8):
            if (r,c) in path_a:
                row.append("[bold cyan]•[/]")
            elif r == 0 and c == 0:
                row.append("[bold blue]S[/]")
            elif r == 7 and c == 7:
                row.append("[bold green]G[/]")
            else:
                row.append("[dim]·[/]")
        grid_a_rows.append(row)
    for row in grid_a_rows:
        seg_a.add_row(*row)

    panel_a = Panel(
        seg_a,
        title="[bold cyan]Segment A[/]",
        subtitle="Steps: 14 | Coins: 2 | Turns: 5",
        border_style="cyan",
    )

    # Segment B
    seg_b = Table(title="[bold yellow]Segment B[/]", border_style="yellow", show_lines=True, show_header=False)
    seg_b.add_column(width=3, justify="center")
    for _ in range(7):
        seg_b.add_column(width=3, justify="center")

    path_b = {(0,0),(1,0),(1,1),(2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(3,6),(4,6),(5,6),
              (6,6),(6,7),(7,7)}
    grid_b_rows = []
    for r in range(8):
        row = []
        for c in range(8):
            if (r,c) in path_b:
                row.append("[bold yellow]•[/]")
            elif r == 0 and c == 0:
                row.append("[bold blue]S[/]")
            elif r == 7 and c == 7:
                row.append("[bold green]G[/]")
            else:
                row.append("[dim]·[/]")
        grid_b_rows.append(row)
    for row in grid_b_rows:
        seg_b.add_row(*row)

    panel_b = Panel(
        seg_b,
        title="[bold yellow]Segment B[/]",
        subtitle="Steps: 16 | Gems: 1 | Turns: 4",
        border_style="yellow",
    )

    console.print(Panel(
        "[bold]Preference Pair 12 / 30[/]",
        border_style="bright_white",
    ))
    console.print(Columns([panel_a, panel_b], equal=True, expand=True))
    console.print()
    console.print(
        "[bold bright_white]Which is better?[/]  "
        "[cyan][1][/] Segment A    "
        "[yellow][2][/] Segment B    "
        "[dim][s] Skip[/]"
    )

    svg = console.export_svg(title="Human Preference Collection")
    (OUTPUT_DIR / "02_preferences.svg").write_text(svg)
    print("  ✓ 02_preferences.svg")


def generate_reward_model():
    console = Console(record=True, width=72)

    table = Table(
        title="[bold]Reward Model Training[/]",
        border_style="cyan",
        show_lines=False,
    )
    table.add_column("Epoch", justify="right", style="bold")
    table.add_column("Train Loss", justify="right", style="red")
    table.add_column("Val Loss", justify="right", style="yellow")
    table.add_column("Accuracy", justify="right", style="green")

    data = [
        (1,   0.6931, 0.6928, "50.0%"),
        (10,  0.5842, 0.5901, "62.3%"),
        (20,  0.4713, 0.4890, "71.7%"),
        (30,  0.3821, 0.4102, "78.3%"),
        (50,  0.2647, 0.3105, "84.0%"),
        (70,  0.1893, 0.2541, "88.7%"),
        (90,  0.1342, 0.2190, "91.3%"),
        (100, 0.1108, 0.2053, "93.3%"),
    ]

    for epoch, tl, vl, acc in data:
        table.add_row(str(epoch), f"{tl:.4f}", f"{vl:.4f}", acc)

    console.print(table)
    console.print()
    console.print(
        "[dim]Loss:[/]  ████████████████░░░░ → ███░░░░░░░░░░░░░░░░░  "
        "[green]↓ converging[/]"
    )
    console.print(
        "[dim]Acc: [/]  ██░░░░░░░░░░░░░░░░░░ → ██████████████████░░  "
        "[green]↑ 93.3%[/]"
    )

    svg = console.export_svg(title="Reward Model Training")
    (OUTPUT_DIR / "03_reward_model.svg").write_text(svg)
    print("  ✓ 03_reward_model.svg")


def generate_ppo_training():
    console = Console(record=True, width=88)

    table = Table(
        title="[bold]PPO Fine-Tuning with KL Penalty[/]",
        border_style="magenta",
        show_lines=False,
    )
    table.add_column("Iter", justify="right", style="bold")
    table.add_column("Goal Rate", justify="right", style="green")
    table.add_column("Trap Rate", justify="right", style="red")
    table.add_column("Avg Reward", justify="right", style="cyan")
    table.add_column("Entropy", justify="right", style="yellow")
    table.add_column("KL Div", justify="right", style="magenta")

    data = [
        (1,   "42%", "18%",  "1.23",  "1.386", "0.000"),
        (10,  "48%", "14%",  "2.87",  "1.301", "0.012"),
        (20,  "56%", "12%",  "4.15",  "1.214", "0.031"),
        (30,  "63%", "10%",  "5.42",  "1.148", "0.058"),
        (40,  "70%",  "8%",  "6.71",  "1.095", "0.089"),
        (50,  "76%",  "6%",  "7.83",  "1.052", "0.124"),
        (60,  "81%",  "4%",  "8.56",  "1.018", "0.157"),
        (70,  "85%",  "4%",  "9.12",  "0.991", "0.183"),
        (80,  "88%",  "2%",  "9.47",  "0.972", "0.201"),
        (100, "92%",  "2%", "10.03",  "0.958", "0.218"),
    ]

    for it, gr, tr, ar, ent, kl in data:
        table.add_row(str(it), gr, tr, ar, ent, kl)

    console.print(table)
    console.print()
    console.print(
        "[dim]β (KL coeff) = 0.20  |  "
        "RM Score trend:[/] ▁▂▃▄▅▆▇██  "
        "[dim]KL trend:[/] ▁▁▂▂▃▃▄▅▆▇"
    )

    svg = console.export_svg(title="PPO Fine-Tuning with KL Penalty")
    (OUTPUT_DIR / "04_ppo_training.svg").write_text(svg)
    print("  ✓ 04_ppo_training.svg")


def generate_comparison():
    console = Console(record=True, width=90)

    # Arrow grids showing policy direction at each cell
    arrows_before = [
        ["→", "→", "↓", " ", "↓", "→", "↓", "↓"],
        ["↓", "→", "↓", " ", "↓", "←", "↓", "↓"],
        ["↓", "→", "→", "→", "↓", " ", "↓", "↓"],
        [" ", " ", "↓", " ", "↓", " ", "↓", "↓"],
        ["→", "→", "→", "→", "→", "→", "↓", "→"],
        ["↓", " ", "↓", " ", " ", "↓", "→", "↓"],
        ["→", "→", "→", "→", "→", "→", "→", "↓"],
        ["→", "→", "→", "→", "→", "→", "→", "★"],
    ]
    arrows_after = [
        ["→", "→", "↓", " ", "↓", "↓", "↓", "↓"],
        ["↓", "↓", "↓", " ", "↓", "↓", "↓", "↓"],
        ["↓", "→", "→", "→", "→", " ", "↓", "↓"],
        [" ", " ", "↓", " ", "↓", " ", "↓", "↓"],
        ["→", "→", "→", "↓", "↓", "→", "→", "↓"],
        ["↓", " ", "↓", " ", " ", "↓", "↓", "↓"],
        ["→", "→", "→", "→", "→", "→", "→", "↓"],
        ["→", "→", "→", "→", "→", "→", "→", "★"],
    ]

    def make_arrow_table(arrows, style):
        t = Table(show_header=False, show_lines=True, border_style=style, padding=(0, 1))
        for _ in range(8):
            t.add_column(width=2, justify="center")
        for row in arrows:
            cells = []
            for a in row:
                if a == " ":
                    cells.append("[dim]#[/]")
                elif a == "★":
                    cells.append("[bold green]★[/]")
                else:
                    cells.append(f"[{style}]{a}[/]")
            t.add_row(*cells)
        return t

    before_table = make_arrow_table(arrows_before, "cyan")
    after_table = make_arrow_table(arrows_after, "bright_green")

    panel_before = Panel(
        before_table,
        title="[bold cyan]Before RLHF[/]",
        subtitle="Goal: 42% | Trap: 18% | Avg Steps: 24",
        border_style="cyan",
    )
    panel_after = Panel(
        after_table,
        title="[bold bright_green]After RLHF[/]",
        subtitle="Goal: 92% | Trap: 2% | Avg Steps: 16",
        border_style="bright_green",
    )

    console.print(Panel(
        "[bold]Pre-training vs RLHF Policy[/]\n"
        "[dim]Arrows show greedy action at each cell. Walls shown as #.[/]",
        border_style="bright_white",
    ))
    console.print(Columns([panel_before, panel_after], equal=True, expand=True))
    console.print()
    console.print(
        "[bold]Alignment Tax:[/]  KL(π_RLHF ‖ π_pretrained) = [yellow]0.2180[/]  "
        "[dim]— how far the policy shifted to match your preferences[/]"
    )

    svg = console.export_svg(title="Pre-training vs RLHF Policy")
    (OUTPUT_DIR / "05_comparison.svg").write_text(svg)
    print("  ✓ 05_comparison.svg")


if __name__ == "__main__":
    print("Generating Adventure 01 screenshots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_grid_world()
    generate_preferences()
    generate_reward_model()
    generate_ppo_training()
    generate_comparison()
    print("Done!")
