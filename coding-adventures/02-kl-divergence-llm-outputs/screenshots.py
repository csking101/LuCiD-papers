#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 02 from real captured data.

Loads ``screenshot_data.json`` (produced by ``app.py --capture``) and
reconstructs the dataclass instances needed by ``viz.py``, so the
screenshots are pixel-perfect renderings of actual model outputs — not
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

from kl import (
    CategoryKLSummary,
    InterpolatedOutput,
    SequenceKL,
    TokenKL,
)
from viz import (
    render_adventure_connection,
    render_category_summaries,
    render_conclusion,
    render_global_kl,
    render_interpolated_outputs,
    render_kl_heatmap,
    render_llm_parallel_table,
    render_sequence_comparison,
    render_token_kl_table,
)

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "02"


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _dict_to_token_kl(d: dict) -> TokenKL:
    """Reconstruct a TokenKL from a JSON dict."""
    # base_top_k / instruct_top_k: list of list of [token_str, prob]
    # Need to convert inner lists to tuples
    base_top_k = [
        [(tok, prob) for tok, prob in entries]
        for entries in d["base_top_k"]
    ]
    instruct_top_k = [
        [(tok, prob) for tok, prob in entries]
        for entries in d["instruct_top_k"]
    ]
    return TokenKL(
        tokens=d["tokens"],
        token_ids=d["token_ids"],
        kl_per_token=d["kl_per_token"],
        base_top_k=base_top_k,
        instruct_top_k=instruct_top_k,
        total_kl=d["total_kl"],
        mean_kl=d["mean_kl"],
    )


def _dict_to_seq_kl(d: dict) -> SequenceKL:
    """Reconstruct a SequenceKL from a JSON dict."""
    return SequenceKL(
        prompt=d["prompt"],
        base_text=d["base_text"],
        instruct_text=d["instruct_text"],
        prompt_token_kl=_dict_to_token_kl(d["prompt_token_kl"]),
        total_kl=d["total_kl"],
        mean_kl=d["mean_kl"],
    )


def _dict_to_interp(d: dict) -> InterpolatedOutput:
    """Reconstruct an InterpolatedOutput from a JSON dict."""
    return InterpolatedOutput(
        alpha=d["alpha"],
        text=d["text"],
        token_ids=d["token_ids"],
        tokens=d["tokens"],
        kl_per_token=d["kl_per_token"],
        total_kl=d["total_kl"],
    )


def _dict_to_cat_summary(d: dict) -> CategoryKLSummary:
    """Reconstruct a CategoryKLSummary from a JSON dict."""
    return CategoryKLSummary(
        category=d["category"],
        num_prompts=d["num_prompts"],
        mean_kl=d["mean_kl"],
        max_kl=d["max_kl"],
        min_kl=d["min_kl"],
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
# Screenshot 1: Global KL Overview
# ---------------------------------------------------------------------------

def generate_global_kl(data: dict):
    console = Console(record=True, width=95)

    p1 = data["phase1"]
    global_mean = p1["global_mean"]
    per_prompt_kl = p1["per_prompt_kl"]
    prompt_texts = p1["prompt_texts"]

    panel = render_global_kl(global_mean, per_prompt_kl, prompt_texts)
    console.print(panel)

    _save(console, "01_global_kl.svg", "Global KL Divergence Overview")


# ---------------------------------------------------------------------------
# Screenshot 2: Token-Level KL Anatomy
# ---------------------------------------------------------------------------

def generate_token_anatomy(data: dict):
    console = Console(record=True, width=100)

    p2 = data["phase2"]
    token_kl = _dict_to_token_kl(p2["token_kl"])

    prompt_text = p2["prompt_text"]
    table = render_token_kl_table(
        token_kl,
        title=f'Token-Level KL — "{prompt_text}"',
    )
    console.print(table)

    heatmap = render_kl_heatmap(token_kl)
    console.print(heatmap)

    _save(console, "02_token_anatomy.svg", "Token-Level KL Anatomy")


# ---------------------------------------------------------------------------
# Screenshot 3: Category Comparison
# ---------------------------------------------------------------------------

def generate_categories(data: dict):
    console = Console(record=True, width=80)

    p3 = data["phase3"]
    summaries = [_dict_to_cat_summary(s) for s in p3["summaries"]]

    panel = render_category_summaries(summaries)
    console.print(panel)

    _save(console, "03_categories.svg", "Category-Specific Divergence")


# ---------------------------------------------------------------------------
# Screenshot 4: KL-Constrained Interpolation
# ---------------------------------------------------------------------------

def generate_interpolation(data: dict):
    console = Console(record=True, width=95)

    p4 = data["phase4"]
    prompt = p4["prompt"]
    outputs = [_dict_to_interp(o) for o in p4["outputs"]]

    panel = render_interpolated_outputs(outputs, prompt)
    console.print(panel)

    _save(console, "04_interpolation.svg", "KL-Constrained Interpolated Generation")


# ---------------------------------------------------------------------------
# Screenshot 5: Sequence comparison (explorer-style)
# ---------------------------------------------------------------------------

def generate_explorer(data: dict):
    console = Console(record=True, width=95)

    p5 = data["phase5"]
    # Use the first explorer prompt (high-KL safety prompt)
    explorer_entry = p5["explorer"][0]

    token_kl = _dict_to_token_kl(explorer_entry["token_kl"])
    seq_kl = _dict_to_seq_kl(explorer_entry["seq_kl"])

    comparison = render_sequence_comparison(seq_kl)
    console.print(comparison)

    table = render_token_kl_table(
        token_kl,
        title=f"Token-Level KL Details",
        max_rows=14,
    )
    console.print(table)

    _save(console, "05_explorer.svg", "Token Distribution Explorer")


# ---------------------------------------------------------------------------
# Screenshot 6: Conclusion
# ---------------------------------------------------------------------------

def generate_conclusion(data: dict):
    console = Console(record=True, width=80)

    p1 = data["phase1"]
    p3 = data["phase3"]

    global_mean_kl = p1["global_mean"]
    summaries = [_dict_to_cat_summary(s) for s in p3["summaries"]]

    conclusion = render_conclusion(global_mean_kl, summaries)
    console.print(conclusion)

    parallel = render_llm_parallel_table()
    console.print(parallel)

    connection = render_adventure_connection()
    console.print(connection)

    _save(console, "06_conclusion.svg", "Session Summary")


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
    print("Generating Adventure 02 screenshots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_global_kl(data)
    generate_token_anatomy(data)
    generate_categories(data)
    generate_interpolation(data)
    generate_explorer(data)
    generate_conclusion(data)

    print("Done!")
