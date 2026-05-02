"""Rich terminal rendering for Adventure 05 — Intent Hooks.

All visualisation is done via the Rich library for terminal output.
Each function returns a Rich renderable (Panel, Table, etc.) that can
be printed with ``console.print()``.
"""

from __future__ import annotations

from rich.align import Align
from rich.columns import Columns
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from classifier import ConfusionMatrix, LayerSweepResult, ProbeResult
from hooks import ModelInfo
from pipeline import ComparisonResult, GuardrailResult, StressTestResult


# ─── Welcome & headers ──────────────────────────────────────────────────────

def render_welcome() -> Panel:
    """Render the welcome banner."""
    text = Text()
    text.append("Intent Hooks\n", style="bold bright_white")
    text.append("Classify and Deny Harmful Prompts\n", style="dim")
    text.append("=" * 48 + "\n\n", style="dim")
    text.append("Can we detect harmful intent from a model's\n")
    text.append("hidden states — before it even starts generating?\n\n")
    text.append("This demo hooks into a transformer's layers,\n")
    text.append("trains a probe on hidden-state representations,\n")
    text.append("and builds a guardrail that blocks harmful\n")
    text.append("requests at the hidden-state level.\n\n")
    text.append("You'll explore:\n")
    text.append("  1. Hook anatomy — how PyTorch hooks capture hidden states\n")
    text.append("  2. Intent dataset — curated benign/harmful/jailbreak prompts\n")
    text.append("  3. Layer-wise probing — which layers encode intent?\n")
    text.append("  4. Guardrail pipeline — hook → classify → deny\n")
    text.append("  5. Jailbreak stress test — does the hook catch bypasses?\n\n")
    text.append("Press Enter to begin (model will be downloaded on first run)...", style="dim italic")
    return Panel(text, title="Adventure 05", border_style="bright_blue")


def render_phase_header(phase: int, title: str, description: str = "") -> Panel:
    """Render a phase header."""
    text = Text()
    text.append(f"Phase {phase}: ", style="bold bright_yellow")
    text.append(title, style="bold bright_white")
    if description:
        text.append(f"\n{description}", style="dim")
    return Panel(text, border_style="bright_yellow", padding=(0, 1))


# ─── Phase 1: Hook anatomy ──────────────────────────────────────────────────

def render_model_info(info: ModelInfo) -> Panel:
    """Render model architecture info."""
    table = Table(show_header=False, border_style="dim", padding=(0, 2))
    table.add_column("Key", style="bright_cyan")
    table.add_column("Value", style="bright_white")

    table.add_row("Model", info.name)
    table.add_row("Layers", str(info.num_layers))
    table.add_row("Hidden size", str(info.hidden_size))
    table.add_row("Vocab size", f"{info.vocab_size:,}")
    table.add_row("Parameters", f"{info.num_params:,}")
    table.add_row("Device", info.device)
    table.add_row("Dtype", info.dtype)

    return Panel(table, title="Model Architecture", border_style="bright_blue")


def render_module_hierarchy(paths: list[str], title: str = "Module Hierarchy") -> Panel:
    """Render a tree-like view of model modules."""
    text = Text()
    for path in paths:
        depth = path.count(".")
        indent = "  " * depth
        name = path.split(".")[-1]
        prefix = "├── " if depth > 0 else ""
        text.append(f"{indent}{prefix}", style="dim")
        text.append(f"{name}\n", style="bright_green" if depth == 0 else "white")
    return Panel(text, title=title, border_style="dim")


def render_hook_demo(
    layer: int,
    hidden_shape: tuple[int, ...],
    prompt_text: str,
) -> Panel:
    """Render the result of a single hook capture demo."""
    text = Text()
    text.append("Prompt: ", style="dim")
    text.append(f'"{prompt_text}"\n\n', style="bright_white")
    text.append(f"Hooked layer: ", style="dim")
    text.append(f"{layer}\n", style="bright_yellow")
    text.append(f"Hidden state shape: ", style="dim")
    text.append(f"{list(hidden_shape)}\n", style="bright_cyan")
    text.append(f"  → batch={hidden_shape[0]}, seq_len={hidden_shape[1]}, hidden={hidden_shape[2]}\n", style="dim")
    text.append("\nThe hook captured the hidden state ", style="dim")
    text.append("without modifying the forward pass.", style="bright_green")
    return Panel(text, title="Hook Capture Demo", border_style="bright_green")


# ─── Phase 2: Intent dataset ────────────────────────────────────────────────

def render_dataset_summary(summary: dict[str, int]) -> Panel:
    """Render dataset statistics."""
    table = Table(title="Dataset Summary", border_style="dim")
    table.add_column("Label", style="bright_cyan")
    table.add_column("Count", style="bright_white", justify="right")
    table.add_column("", justify="left")

    colors = {"benign": "green", "harmful": "red", "ambiguous": "yellow", "jailbreak": "magenta"}
    total = sum(summary.values())

    for label, count in sorted(summary.items()):
        bar_len = int(count / total * 30) if total > 0 else 0
        bar = "█" * bar_len
        color = colors.get(label, "white")
        table.add_row(label, str(count), Text(bar, style=color))

    table.add_section()
    table.add_row("Total", str(total), "", style="bold")

    return Panel(table, border_style="bright_blue")


def render_prompt_samples(
    prompts: list[tuple[str, str, str]],
    title: str = "Sample Prompts",
) -> Panel:
    """Render a sample of prompts.

    Args:
        prompts: list of (text, label, category) tuples.
    """
    table = Table(border_style="dim", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Label", width=10)
    table.add_column("Category", width=12)
    table.add_column("Prompt", ratio=1)

    label_colors = {"benign": "green", "harmful": "red", "ambiguous": "yellow", "jailbreak": "magenta"}

    for i, (text, label, category) in enumerate(prompts, 1):
        color = label_colors.get(label, "white")
        # Truncate long prompts
        display_text = text if len(text) <= 80 else text[:77] + "..."
        table.add_row(
            str(i),
            Text(label, style=f"bold {color}"),
            category,
            display_text,
        )

    return Panel(table, title=title, border_style="bright_blue")


# ─── Phase 3: Layer-wise probing ────────────────────────────────────────────

def render_layer_sweep(sweep: LayerSweepResult) -> Panel:
    """Render the layer-wise accuracy curve as a bar chart + table."""
    table = Table(border_style="dim")
    table.add_column("Layer", style="bright_cyan", justify="right", width=6)
    table.add_column("Accuracy", style="bright_white", justify="right", width=10)
    table.add_column("F1", style="dim", justify="right", width=8)
    table.add_column("Loss", style="dim", justify="right", width=10)
    table.add_column("", width=30)

    max_acc = max(r.accuracy for r in sweep.results) if sweep.results else 1.0

    for r in sorted(sweep.results, key=lambda x: x.layer):
        bar_len = int(r.accuracy / max(max_acc, 0.01) * 20)
        is_best = r.layer == sweep.best_layer
        bar_char = "█" if is_best else "▓"
        bar_color = "bright_green" if is_best else "green"
        bar = Text(bar_char * bar_len, style=bar_color)

        acc_style = "bold bright_green" if is_best else "bright_white"
        layer_str = f"→ {r.layer}" if is_best else str(r.layer)

        table.add_row(
            layer_str,
            f"{r.accuracy:.1%}",
            f"{r.f1:.3f}",
            f"{r.train_loss:.4f}",
            bar,
        )

    # Summary
    summary = Text()
    summary.append(f"\nBest layer: ", style="dim")
    summary.append(f"{sweep.best_layer}", style="bold bright_yellow")
    summary.append(f"  Accuracy: ", style="dim")
    summary.append(f"{sweep.best_accuracy:.1%}", style="bold bright_green")

    return Panel(
        Group(table, summary),
        title="Layer-wise Probe Accuracy",
        border_style="bright_blue",
    )


def render_probe_detail(result: ProbeResult) -> Panel:
    """Render detailed metrics for a single probe."""
    table = Table(show_header=False, border_style="dim", padding=(0, 2))
    table.add_column("Metric", style="bright_cyan")
    table.add_column("Value", style="bright_white")

    table.add_row("Layer", str(result.layer))
    table.add_row("Accuracy", f"{result.accuracy:.1%}")
    table.add_row("Precision", f"{result.precision:.3f}")
    table.add_row("Recall", f"{result.recall:.3f}")
    table.add_row("F1 Score", f"{result.f1:.3f}")
    table.add_row("Train Loss", f"{result.train_loss:.4f}")
    table.add_row("Train / Test", f"{result.num_train} / {result.num_test}")

    return Panel(table, title=f"Probe at Layer {result.layer}", border_style="bright_green")


# ─── Phase 4: Guardrail pipeline ────────────────────────────────────────────

def render_guardrail_result(result: GuardrailResult) -> Panel:
    """Render the result of processing one prompt through the guardrail."""
    if result.was_denied:
        decision_text = Text("DENIED", style="bold bright_red")
        border = "bright_red"
    else:
        decision_text = Text("ALLOWED", style="bold bright_green")
        border = "bright_green"

    text = Text()
    text.append("Prompt: ", style="dim")
    text.append(f'"{result.prompt}"\n\n', style="bright_white")
    text.append("Decision: ")
    text.append_text(decision_text)
    text.append(f"\nP(harmful): ", style="dim")

    prob_color = "bright_red" if result.prob_harmful >= 0.5 else "bright_green"
    text.append(f"{result.prob_harmful:.3f}", style=prob_color)
    text.append(f"\nLayer: {result.layer}", style="dim")

    text.append(f"\n\nResponse:\n", style="dim")
    response_preview = result.response[:300] + "..." if len(result.response) > 300 else result.response
    text.append(response_preview, style="italic")

    text.append(f"\n\nLatency: ", style="dim")
    text.append(f"hook={result.latency_hook_ms:.1f}ms  ", style="dim")
    text.append(f"classify={result.latency_classify_ms:.2f}ms  ", style="dim")
    text.append(f"generate={result.latency_generate_ms:.1f}ms  ", style="dim")
    text.append(f"total={result.latency_total_ms:.1f}ms", style="bright_cyan")

    return Panel(text, title="Guardrail Result", border_style=border)


# ─── Phase 5: Stress test & jailbreak ───────────────────────────────────────

def render_confusion_matrix(cm: ConfusionMatrix, title: str = "Confusion Matrix") -> Panel:
    """Render a confusion matrix."""
    table = Table(border_style="dim", show_lines=True)
    table.add_column("", style="bold", width=14)
    table.add_column("Pred: Benign", justify="center", width=14)
    table.add_column("Pred: Harmful", justify="center", width=14)

    table.add_row(
        "True: Benign",
        Text(f"TN = {cm.tn}", style="bright_green"),
        Text(f"FP = {cm.fp}", style="bright_red" if cm.fp > 0 else "dim"),
    )
    table.add_row(
        "True: Harmful",
        Text(f"FN = {cm.fn}", style="bright_red" if cm.fn > 0 else "dim"),
        Text(f"TP = {cm.tp}", style="bright_green"),
    )

    summary = Text()
    summary.append(f"\nAccuracy: {cm.accuracy:.1%}  ", style="bright_white")
    summary.append(f"Precision: {cm.precision:.3f}  ", style="dim")
    summary.append(f"Recall: {cm.recall:.3f}  ", style="dim")
    summary.append(f"F1: {cm.f1:.3f}", style="dim")

    return Panel(
        Group(table, summary),
        title=title,
        border_style="bright_blue",
    )


def render_stress_test(stress: StressTestResult) -> Panel:
    """Render stress test results."""
    table = Table(border_style="dim", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Prompt", ratio=1)
    table.add_column("True", width=8)
    table.add_column("Decision", width=10)
    table.add_column("P(harm)", justify="right", width=8)
    table.add_column("", width=3)

    for i, (result, true_label) in enumerate(
        zip(stress.results, stress.true_labels), 1
    ):
        true_str = "harmful" if true_label == 1 else "benign"
        true_color = "red" if true_label == 1 else "green"

        dec_str = "DENY" if result.was_denied else "ALLOW"
        dec_color = "bright_red" if result.was_denied else "bright_green"

        # Correctness indicator
        predicted_harmful = result.was_denied
        actually_harmful = true_label == 1
        correct = predicted_harmful == actually_harmful
        icon = "✓" if correct else "✗"
        icon_color = "bright_green" if correct else "bright_red"

        prompt_text = result.prompt if len(result.prompt) <= 60 else result.prompt[:57] + "..."

        table.add_row(
            str(i),
            prompt_text,
            Text(true_str, style=true_color),
            Text(dec_str, style=dec_color),
            f"{result.prob_harmful:.3f}",
            Text(icon, style=icon_color),
        )

    summary = Text()
    summary.append(f"\nTotal: {stress.num_total}  ", style="dim")
    summary.append(f"Denied: {stress.num_denied}  ", style="bright_red")
    summary.append(f"Allowed: {stress.num_allowed}  ", style="bright_green")
    summary.append(f"Mean latency: {stress.mean_latency_ms:.1f}ms", style="dim")

    return Panel(
        Group(table, summary),
        title="Stress Test Results",
        border_style="bright_blue",
    )


def render_comparison(comp: ComparisonResult) -> Panel:
    """Render side-by-side comparison of model vs hook."""
    # Model response
    model_text = Text()
    model_text.append("Model's response (no guardrail):\n", style="dim")
    model_preview = comp.model_response[:200] + "..." if len(comp.model_response) > 200 else comp.model_response
    model_text.append(model_preview, style="italic")
    if comp.model_refused:
        model_text.append("\n[Model refused]", style="bright_yellow")

    model_panel = Panel(model_text, title="Built-in Safety", border_style="bright_yellow")

    # Hook response
    hook_text = Text()
    hook_text.append("Hook-based guardrail:\n", style="dim")
    hook_text.append(f"Decision: ", style="dim")
    if comp.hook_denied:
        hook_text.append("DENIED", style="bold bright_red")
    else:
        hook_text.append("ALLOWED", style="bold bright_green")
    hook_text.append(f"\nP(harmful): {comp.guardrail_result.prob_harmful:.3f}", style="dim")

    hook_border = "bright_red" if comp.hook_denied else "bright_green"
    hook_panel = Panel(hook_text, title="Intent Hook", border_style=hook_border)

    # Agreement
    agreement_map = {
        "both_refuse": ("Both Refuse", "bright_green"),
        "both_allow": ("Both Allow", "bright_green"),
        "hook_catches": ("Hook Catches (model missed!)", "bright_yellow"),
        "model_refuses": ("Model Refuses Only", "dim"),
    }
    agree_label, agree_color = agreement_map.get(comp.agreement, ("Unknown", "dim"))

    header = Text()
    header.append(f'Prompt: "{comp.prompt}"\n', style="bright_white")
    header.append(f"Agreement: ", style="dim")
    header.append(agree_label, style=agree_color)

    return Panel(
        Group(header, Text(), Columns([model_panel, hook_panel], equal=True, expand=True)),
        title="Model vs Hook Comparison",
        border_style="bright_blue",
    )


# ─── Phase 6: Conclusion ────────────────────────────────────────────────────

def render_conclusion(
    best_layer: int,
    best_accuracy: float,
    num_layers: int,
    cm: ConfusionMatrix,
    jailbreak_catch_rate: float,
) -> Panel:
    """Render the conclusion summary."""
    text = Text()
    text.append("Key Findings\n\n", style="bold bright_white")
    text.append(f"  Best probing layer: ", style="dim")
    text.append(f"{best_layer}", style="bold bright_yellow")
    text.append(f" / {num_layers - 1}\n", style="dim")
    text.append(f"  Probe accuracy: ", style="dim")
    text.append(f"{best_accuracy:.1%}\n", style="bold bright_green")
    text.append(f"  Guardrail precision: ", style="dim")
    text.append(f"{cm.precision:.3f}\n", style="bright_white")
    text.append(f"  Guardrail recall: ", style="dim")
    text.append(f"{cm.recall:.3f}\n", style="bright_white")
    text.append(f"  Jailbreak catch rate: ", style="dim")
    text.append(f"{jailbreak_catch_rate:.1%}\n\n", style="bold bright_cyan")

    text.append("  System prompts are 'soft' constraints (Adventure 04).\n", style="dim")
    text.append("  RLHF is a 'hard' constraint on weights (Adventure 02).\n", style="dim")
    text.append("  Intent hooks are a 'gate' constraint — external to the model.\n", style="dim")
    text.append("  A linear probe detects what the model already 'knows'\n", style="dim")
    text.append("  about harmful intent, before generating any output.\n", style="dim")

    return Panel(text, title="Conclusion", border_style="bright_blue")


def render_llm_parallel_table() -> Panel:
    """Render the LLM parallel mapping table."""
    table = Table(border_style="dim", show_lines=True)
    table.add_column("What you see here", style="bright_white", ratio=1)
    table.add_column("What happens in LLM RLHF", style="bright_white", ratio=1)

    table.add_row("Forward hook = monitoring", "RLHF = training (weight changes)")
    table.add_row("Hidden state probe = external classifier", "Reward model = learned preference signal")
    table.add_row("Hook-based denial = hard gate before output", "Model refusal = soft learned response")
    table.add_row("Layer-wise accuracy = info emergence curve", "Training stages = capability building")
    table.add_row("Jailbreak detection via hidden states", "Toxicity paradox (InstructGPT limitation)")
    table.add_row("Linear probe = what model 'knows' internally", "RLHF = what model 'does' externally")
    table.add_row("Classification threshold = sensitivity dial", "KL penalty beta = alignment strength")
    table.add_row("False positives = over-refusal", "Over-alignment = alignment tax")

    return Panel(table, title="LLM Parallel Mapping", border_style="bright_blue")


def render_adventure_connections() -> Panel:
    """Render the cross-adventure connections table."""
    table = Table(border_style="dim", show_lines=True)
    table.add_column("Adventure 01 (Grid World)", ratio=1)
    table.add_column("Adventure 02/04\n(KL / Prompts)", ratio=1)
    table.add_column("Adventure 05\n(Intent Hooks)", ratio=1)

    table.add_row(
        "Reward model trained on preferences",
        "KL divergence between distributions",
        "Linear probe on hidden states",
    )
    table.add_row(
        "KL penalty constrains policy",
        "System prompt shifts token probs",
        "Hook captures what model 'knows'",
    )
    table.add_row(
        "Agent learns to avoid bad paths",
        "Model learns to refuse (RLHF)",
        "External gate blocks harmful inputs",
    )
    table.add_row(
        "Reward hacking without KL",
        "Jailbreaks bypass soft safety",
        "Hidden states may catch bypasses",
    )

    return Panel(table, title="Connections Across Adventures", border_style="bright_blue")
