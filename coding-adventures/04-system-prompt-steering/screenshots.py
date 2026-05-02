#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 04 from real captured data.

Loads ``screenshot_data.json`` (produced by ``app.py --capture``) and
reconstructs the dataclass instances needed by ``viz.py``, so the
screenshots are pixel-perfect renderings of actual model outputs.

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

from analysis import (
    FirstTokenComparison,
    ForcedContinuationKL,
    SteeringMatrix,
    SystemPromptProfile,
)
from viz import (
    render_adventure_connections,
    render_chat_template,
    render_conclusion,
    render_first_token_comparison,
    render_forced_continuation,
    render_llm_parallel_table,
    render_profiles,
    render_steering_matrix,
)

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "04"


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _dict_to_ftc(d: dict) -> FirstTokenComparison:
    return FirstTokenComparison(
        user_prompt=d["user_prompt"],
        system_prompt_a=d["system_prompt_a"],
        system_prompt_b=d["system_prompt_b"],
        kl_divergence=d["kl_divergence"],
        js_divergence=d["js_divergence"],
        top_k_a=[tuple(x) for x in d["top_k_a"]],
        top_k_b=[tuple(x) for x in d["top_k_b"]],
        top_shifts=[tuple(x) for x in d["top_shifts"]],
    )


def _dict_to_fckl(d: dict) -> ForcedContinuationKL:
    return ForcedContinuationKL(
        user_prompt=d["user_prompt"],
        source_system_prompt=d["source_system_prompt"],
        target_system_prompt=d["target_system_prompt"],
        continuation_tokens=d["continuation_tokens"],
        continuation_ids=d["continuation_ids"],
        kl_per_token=d["kl_per_token"],
        total_kl=d["total_kl"],
        mean_kl=d["mean_kl"],
        source_text=d["source_text"],
        target_text=d["target_text"],
    )


def _dict_to_matrix(d: dict) -> SteeringMatrix:
    return SteeringMatrix(
        system_prompt_names=d["system_prompt_names"],
        user_prompts=d["user_prompts"],
        kl_matrix=d["kl_matrix"],
        row_means=d["row_means"],
        col_means=d["col_means"],
        global_mean=d["global_mean"],
    )


def _dict_to_profile(d: dict) -> SystemPromptProfile:
    return SystemPromptProfile(
        system_prompt_name=d["system_prompt_name"],
        system_prompt_text=d["system_prompt_text"],
        category=d["category"],
        user_prompts=d["user_prompts"],
        first_token_kls=d["first_token_kls"],
        mean_steering_power=d["mean_steering_power"],
        max_steering_power=d["max_steering_power"],
    )


def _load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(console: Console, filename: str, title: str) -> None:
    svg = console.export_svg(title=title)
    (OUTPUT_DIR / filename).write_text(svg)
    print(f"  > {filename}")


# ---------------------------------------------------------------------------
# Screenshot 1: Chat Template Comparison
# ---------------------------------------------------------------------------

def generate_chat_template(data: dict):
    console = Console(record=True, width=95)

    p1 = data["phase1"]
    panel = render_chat_template(
        p1["formatted_default"],
        p1["formatted_custom"],
        p1["default_name"],
        p1["custom_name"],
    )
    console.print(panel)
    _save(console, "01_chat_template.svg", "Chat Template Comparison")


# ---------------------------------------------------------------------------
# Screenshot 2: First-Token Distribution Comparison
# ---------------------------------------------------------------------------

def generate_first_token(data: dict):
    console = Console(record=True, width=100)

    p2 = data["phase2"]
    ftc = _dict_to_ftc(p2)
    panel = render_first_token_comparison(ftc)
    console.print(panel)
    _save(console, "02_first_token.svg", "First-Token Distribution Comparison")


# ---------------------------------------------------------------------------
# Screenshot 3: Forced-Continuation KL Profile
# ---------------------------------------------------------------------------

def generate_forced_continuation(data: dict):
    console = Console(record=True, width=100)

    p3 = data["phase3"]
    fckl = _dict_to_fckl(p3)
    panel = render_forced_continuation(fckl)
    console.print(panel)
    _save(console, "03_forced_continuation.svg", "Forced-Continuation KL Profile")


# ---------------------------------------------------------------------------
# Screenshot 4: Steering Matrix
# ---------------------------------------------------------------------------

def generate_steering_matrix(data: dict):
    console = Console(record=True, width=110)

    p4 = data["phase4"]
    matrix = _dict_to_matrix(p4["matrix"])
    panel = render_steering_matrix(matrix)
    console.print(panel)

    profiles = [_dict_to_profile(p) for p in p4["profiles"]]
    panel2 = render_profiles(profiles)
    console.print(panel2)

    _save(console, "04_steering_matrix.svg", "Steering Matrix")


# ---------------------------------------------------------------------------
# Screenshot 5: Explorer Example
# ---------------------------------------------------------------------------

def generate_explorer(data: dict):
    console = Console(record=True, width=100)

    p5 = data["phase5"]
    # Use the first explorer example (Pirate + safety)
    ftc = _dict_to_ftc(p5["explorer"][0])
    panel = render_first_token_comparison(ftc)
    console.print(panel)
    _save(console, "05_explorer.svg", "Interactive Explorer")


# ---------------------------------------------------------------------------
# Screenshot 6: Conclusion
# ---------------------------------------------------------------------------

def generate_conclusion(data: dict):
    console = Console(record=True, width=90)

    p4 = data["phase4"]
    matrix = _dict_to_matrix(p4["matrix"])
    profiles = [_dict_to_profile(p) for p in p4["profiles"]]

    conclusion = render_conclusion(matrix, profiles)
    console.print(conclusion)

    parallel = render_llm_parallel_table()
    console.print(parallel)

    connections = render_adventure_connections()
    console.print(connections)

    _save(console, "06_conclusion.svg", "Session Summary")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).parent / "screenshot_data.json"
    )

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        print("Run 'python app.py --capture screenshot_data.json' first.")
        sys.exit(1)

    data = _load_data(data_path)
    print(f"Loaded capture data from {data_path}")
    print("Generating Adventure 04 screenshots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_chat_template(data)
    generate_first_token(data)
    generate_forced_continuation(data)
    generate_steering_matrix(data)
    generate_explorer(data)
    generate_conclusion(data)

    print("Done!")
