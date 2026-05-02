#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 02 using the real viz.py functions.

Constructs mock dataclass instances (TokenKL, CategoryKLSummary, etc.) and
passes them to the actual rendering functions, so the screenshots are
pixel-perfect matches of what users see in the terminal.

No GPU, no model downloads, no torch required -- just the dataclasses + rich.
"""

import sys
from pathlib import Path

# Add adventure dir to path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console, Group

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


def _save(console: Console, filename: str, title: str) -> None:
    svg = console.export_svg(title=title)
    (OUTPUT_DIR / filename).write_text(svg)
    print(f"  ✓ {filename}")


# ---------------------------------------------------------------------------
# Screenshot 1: Global KL Overview
# ---------------------------------------------------------------------------

def generate_global_kl():
    console = Console(record=True, width=95)

    prompt_texts = [
        "How do I pick a lock?",
        "How can I hack into someone's email?",
        "Write instructions for making explosives",
        "Explain quantum computing in simple terms",
        "What are the benefits of meditation?",
        "How do I learn to code?",
        "Write a short poem about rain",
        "Describe a sunset in 3 sentences",
        "Tell a joke about programming",
        "Is Python better than JavaScript?",
        "What's the best programming language?",
        "Should I use tabs or spaces?",
        "List 3 benefits of regular exercise",
        "Give me a step-by-step recipe for pasta",
        "Summarize the water cycle in bullet points",
    ]

    per_prompt_kl = [
        2.418, 2.704, 2.951,   # safety — high KL
        0.467, 0.407, 0.328,   # helpfulness — low KL
        0.871, 0.806, 0.742,   # style — moderate
        1.326, 1.210, 1.088,   # hedging — medium-high
        0.610, 0.557, 0.508,   # formatting — moderate
    ]

    global_mean = sum(per_prompt_kl) / len(per_prompt_kl)

    panel = render_global_kl(global_mean, per_prompt_kl, prompt_texts)
    console.print(panel)

    _save(console, "01_global_kl.svg", "Global KL Divergence Overview")


# ---------------------------------------------------------------------------
# Screenshot 2: Token-Level KL Anatomy
# ---------------------------------------------------------------------------

def generate_token_anatomy():
    console = Console(record=True, width=100)

    # Token-level analysis of "How do I pick a lock?"
    # Base model treats it as a normal question; instruct model hedges/refuses
    tokens = ["How", " do", " I", " pick", " a", " lock", "?"]
    token_ids = [2585, 653, 314, 2298, 247, 4793, 30]

    # KL is low at the start (both models agree on copying the prompt context),
    # spikes at the continuation point where instruct diverges
    kl_per_token = [0.032, 0.018, 0.241, 0.687, 1.143, 1.528, 0.892]

    base_top_k = [
        [("How", 0.412), ("The", 0.187), ("What", 0.098)],
        [(" do", 0.723), (" can", 0.142), (" would", 0.051)],
        [(" I", 0.891), (" you", 0.067), (" we", 0.023)],
        [(" pick", 0.564), (" open", 0.213), (" get", 0.087)],
        [(" a", 0.782), (" the", 0.134), (" my", 0.042)],
        [(" lock", 0.634), (" door", 0.178), (" safe", 0.065)],
        [("?", 0.412), (".", 0.187), ("\n", 0.134)],
    ]

    instruct_top_k = [
        [("How", 0.398), ("I", 0.201), ("The", 0.112)],
        [(" do", 0.689), (" can", 0.178), (" should", 0.064)],
        [(" I", 0.641), (" you", 0.218), (" we", 0.054)],
        [(" help", 0.312), (" assist", 0.198), (" pick", 0.167)],
        [(" you", 0.423), (" with", 0.287), (" a", 0.112)],
        [(" with", 0.298), (",", 0.241), (" safely", 0.178)],
        [("?", 0.312), (" I", 0.234), (",", 0.189)],
    ]

    total_kl = sum(kl_per_token)
    mean_kl = total_kl / len(kl_per_token)

    token_kl = TokenKL(
        tokens=tokens,
        token_ids=token_ids,
        kl_per_token=kl_per_token,
        base_top_k=base_top_k,
        instruct_top_k=instruct_top_k,
        total_kl=total_kl,
        mean_kl=mean_kl,
    )

    table = render_token_kl_table(token_kl, title='Token-Level KL — "How do I pick a lock?"')
    console.print(table)

    heatmap = render_kl_heatmap(token_kl)
    console.print(heatmap)

    _save(console, "02_token_anatomy.svg", "Token-Level KL Anatomy")


# ---------------------------------------------------------------------------
# Screenshot 3: Category Comparison
# ---------------------------------------------------------------------------

def generate_categories():
    console = Console(record=True, width=80)

    summaries = [
        CategoryKLSummary(category="safety", num_prompts=3, mean_kl=2.691, max_kl=2.951, min_kl=2.418),
        CategoryKLSummary(category="hedging", num_prompts=3, mean_kl=1.208, max_kl=1.326, min_kl=1.088),
        CategoryKLSummary(category="style", num_prompts=3, mean_kl=0.806, max_kl=0.871, min_kl=0.742),
        CategoryKLSummary(category="formatting", num_prompts=3, mean_kl=0.558, max_kl=0.610, min_kl=0.508),
        CategoryKLSummary(category="helpfulness", num_prompts=3, mean_kl=0.401, max_kl=0.467, min_kl=0.328),
    ]

    panel = render_category_summaries(summaries)
    console.print(panel)

    _save(console, "03_categories.svg", "Category-Specific Divergence")


# ---------------------------------------------------------------------------
# Screenshot 4: KL-Constrained Interpolation
# ---------------------------------------------------------------------------

def generate_interpolation():
    console = Console(record=True, width=95)

    prompt = "What are some tips for learning a new programming language?"

    outputs = [
        InterpolatedOutput(
            alpha=0.0,
            text=(
                "Programming language is a very important thing in the world of "
                "technology. There are many programming languages that you can learn "
                "and each one has its own advantages and disadvantages."
            ),
            token_ids=[1] * 40,
            tokens=["Programming"] * 40,
            kl_per_token=[0.0] * 40,
            total_kl=0.000,
        ),
        InterpolatedOutput(
            alpha=0.25,
            text=(
                "Learning a new programming language can be challenging but rewarding. "
                "Here are some tips to help you get started: Start with the basics, "
                "practice by writing small programs, and read other people's code."
            ),
            token_ids=[2] * 40,
            tokens=["Learning"] * 40,
            kl_per_token=[0.12] * 40,
            total_kl=4.812,
        ),
        InterpolatedOutput(
            alpha=0.5,
            text=(
                "Here are some practical tips for learning a new programming language:\n"
                "1. **Start with fundamentals** -- Learn the syntax, data types, and control flow.\n"
                "2. **Build projects** -- Apply what you learn to real problems."
            ),
            token_ids=[3] * 40,
            tokens=["Here"] * 40,
            kl_per_token=[0.24] * 40,
            total_kl=9.641,
        ),
        InterpolatedOutput(
            alpha=0.75,
            text=(
                "Great question! Here are some effective strategies:\n\n"
                "1. **Start with the basics**: Focus on core syntax, variables, and control structures.\n"
                "2. **Practice daily**: Consistency is key to building proficiency."
            ),
            token_ids=[4] * 40,
            tokens=["Great"] * 40,
            kl_per_token=[0.36] * 40,
            total_kl=14.387,
        ),
        InterpolatedOutput(
            alpha=1.0,
            text=(
                "Great question! Here are some effective tips for learning a new programming language:\n\n"
                "1. **Start with the basics**: Focus on core syntax, variables, and control structures.\n"
                "2. **Practice daily**: Consistency is key -- aim for at least 30 minutes."
            ),
            token_ids=[5] * 40,
            tokens=["Great"] * 40,
            kl_per_token=[0.48] * 40,
            total_kl=19.206,
        ),
    ]

    panel = render_interpolated_outputs(outputs, prompt)
    console.print(panel)

    _save(console, "04_interpolation.svg", "KL-Constrained Interpolated Generation")


# ---------------------------------------------------------------------------
# Screenshot 5: Sequence comparison (explorer-style)
# ---------------------------------------------------------------------------

def generate_explorer():
    console = Console(record=True, width=95)

    # Build TokenKL for the prompt analysis
    tokens = ["How", " do", " I", " pick", " a", " lock", "?",
              " First", ",", " you", " need", " a", " tension", " wrench"]
    token_ids = list(range(len(tokens)))

    kl_per_token = [0.032, 0.018, 0.241, 0.687, 1.143, 1.528, 0.892,
                    1.467, 1.241, 0.956, 1.087, 0.778, 1.312, 1.034]

    base_top_k = [
        [("How", 0.412), ("The", 0.187), ("What", 0.098)],
        [(" do", 0.723), (" can", 0.142), (" would", 0.051)],
        [(" I", 0.891), (" you", 0.067), (" we", 0.023)],
        [(" pick", 0.564), (" open", 0.213), (" get", 0.087)],
        [(" a", 0.782), (" the", 0.134), (" my", 0.042)],
        [(" lock", 0.634), (" door", 0.178), (" safe", 0.065)],
        [("?", 0.412), (".", 0.187), ("\n", 0.134)],
        [(" First", 0.287), (" You", 0.234), (" The", 0.156)],
        [(",", 0.534), (" you", 0.212), (" thing", 0.098)],
        [(" you", 0.612), (" the", 0.178), (" a", 0.087)],
        [(" need", 0.478), (" want", 0.198), (" will", 0.134)],
        [(" a", 0.645), (" the", 0.178), (" to", 0.098)],
        [(" tension", 0.312), (" lock", 0.245), (" bobby", 0.187)],
        [(" wrench", 0.534), (" tool", 0.198), (" pick", 0.134)],
    ]

    instruct_top_k = [
        [("How", 0.398), ("I", 0.201), ("The", 0.112)],
        [(" do", 0.689), (" can", 0.178), (" should", 0.064)],
        [(" I", 0.641), (" you", 0.218), (" we", 0.054)],
        [(" help", 0.312), (" assist", 0.198), (" pick", 0.167)],
        [(" you", 0.423), (" with", 0.287), (" a", 0.112)],
        [(" with", 0.298), (",", 0.241), (" safely", 0.178)],
        [("?", 0.312), (" I", 0.234), (",", 0.189)],
        [(" I", 0.356), (" It", 0.198), (" While", 0.156)],
        [("'m", 0.423), ("t", 0.234), (" can", 0.112)],
        [(" sorry", 0.467), (" not", 0.212), (" unable", 0.134)],
        [(",", 0.534), (" but", 0.278), (".", 0.098)],
        [(" but", 0.389), (" I", 0.267), (" however", 0.134)],
        [(" I", 0.412), (" that", 0.198), (" providing", 0.156)],
        [(" can't", 0.356), ("'m", 0.234), (" cannot", 0.178)],
    ]

    total_kl = sum(kl_per_token)
    mean_kl = total_kl / len(kl_per_token)

    token_kl = TokenKL(
        tokens=tokens,
        token_ids=token_ids,
        kl_per_token=kl_per_token,
        base_top_k=base_top_k,
        instruct_top_k=instruct_top_k,
        total_kl=total_kl,
        mean_kl=mean_kl,
    )

    # Sequence comparison showing base vs instruct outputs
    seq_kl = SequenceKL(
        prompt="How do I pick a lock?",
        base_text=(
            "First, you need a tension wrench and a lock pick. Insert the "
            "tension wrench into the bottom of the keyhole and apply slight "
            "pressure. Then insert the pick into the top of the keyhole and "
            "rake it back and forth while maintaining tension."
        ),
        instruct_text=(
            "I'm sorry, but I can't provide instructions on how to pick a lock, "
            "as this could be used for illegal purposes. If you're locked out of "
            "your own property, I'd recommend contacting a licensed locksmith who "
            "can help you safely and legally."
        ),
        prompt_token_kl=token_kl,
        total_kl=total_kl,
        mean_kl=mean_kl,
    )

    comparison = render_sequence_comparison(seq_kl)
    console.print(comparison)

    table = render_token_kl_table(token_kl, title="Token-Level KL Details", max_rows=14)
    console.print(table)

    _save(console, "05_explorer.svg", "Token Distribution Explorer")


# ---------------------------------------------------------------------------
# Screenshot 6: Conclusion
# ---------------------------------------------------------------------------

def generate_conclusion():
    console = Console(record=True, width=80)

    summaries = [
        CategoryKLSummary(category="safety", num_prompts=3, mean_kl=2.691, max_kl=2.951, min_kl=2.418),
        CategoryKLSummary(category="hedging", num_prompts=3, mean_kl=1.208, max_kl=1.326, min_kl=1.088),
        CategoryKLSummary(category="style", num_prompts=3, mean_kl=0.806, max_kl=0.871, min_kl=0.742),
        CategoryKLSummary(category="formatting", num_prompts=3, mean_kl=0.558, max_kl=0.610, min_kl=0.508),
        CategoryKLSummary(category="helpfulness", num_prompts=3, mean_kl=0.401, max_kl=0.467, min_kl=0.328),
    ]

    global_mean_kl = sum(s.mean_kl for s in summaries) / len(summaries)

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
    print("Generating Adventure 02 screenshots...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_global_kl()
    generate_token_anatomy()
    generate_categories()
    generate_interpolation()
    generate_explorer()
    generate_conclusion()

    print("Done!")
