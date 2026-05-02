#!/usr/bin/env python3
"""Generate SVG screenshots for Adventure 05 from real captured data.

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

from classifier import ConfusionMatrix, LayerSweepResult, ProbeResult
from hooks import ModelInfo
from pipeline import ComparisonResult, GuardrailResult
from viz import (
    render_adventure_connections,
    render_comparison,
    render_conclusion,
    render_confusion_matrix,
    render_dataset_summary,
    render_guardrail_result,
    render_hook_demo,
    render_layer_sweep,
    render_llm_parallel_table,
    render_model_info,
    render_module_hierarchy,
    render_phase_header,
    render_probe_detail,
    render_prompt_samples,
    render_stress_test,
    render_welcome,
)

OUTPUT_DIR = Path(__file__).parent / ".." / ".." / "docs" / "adventures" / "05"


# ---------------------------------------------------------------------------
# Deserialization helpers
# ---------------------------------------------------------------------------

def _dict_to_model_info(d: dict) -> ModelInfo:
    return ModelInfo(
        name=d["name"],
        num_layers=d["num_layers"],
        hidden_size=d["hidden_size"],
        vocab_size=d["vocab_size"],
        device=d["device"],
        dtype=d["dtype"],
        num_params=d["num_params"],
    )


def _dict_to_probe_result(d: dict) -> ProbeResult:
    return ProbeResult(
        layer=d["layer"],
        accuracy=d["accuracy"],
        precision=d["precision"],
        recall=d["recall"],
        f1=d["f1"],
        train_loss=d["train_loss"],
        num_train=d["num_train"],
        num_test=d["num_test"],
    )


def _dicts_to_sweep(results: list[dict]) -> LayerSweepResult:
    return LayerSweepResult(
        results=[_dict_to_probe_result(d) for d in results],
    )


def _dict_to_guardrail_result(d: dict) -> GuardrailResult:
    return GuardrailResult(
        prompt=d["prompt"],
        decision=d["decision"],
        prob_harmful=d["prob_harmful"],
        response=d["response"],
        latency_hook_ms=d["latency_hook_ms"],
        latency_classify_ms=d["latency_classify_ms"],
        latency_generate_ms=d["latency_generate_ms"],
        latency_total_ms=d["latency_total_ms"],
        layer=d["layer"],
    )


def _dict_to_comparison(d: dict, guardrail_result: GuardrailResult) -> ComparisonResult:
    return ComparisonResult(
        prompt=d["prompt"],
        model_response=d["model_response"],
        guardrail_result=guardrail_result,
    )


def _dict_to_confusion_matrix(d: dict) -> ConfusionMatrix:
    return ConfusionMatrix(tp=d["tp"], fp=d["fp"], tn=d["tn"], fn=d["fn"])


def _load_data(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(console: Console, filename: str, title: str) -> None:
    svg = console.export_svg(title=title)
    (OUTPUT_DIR / filename).write_text(svg)
    print(f"  > {filename}")


# ---------------------------------------------------------------------------
# Screenshot 1: Hook Anatomy (model info + module hierarchy + hook demo)
# ---------------------------------------------------------------------------

def generate_hook_anatomy(data: dict):
    console = Console(record=True, width=100)

    p1 = data["phase1"]
    info = _dict_to_model_info(p1["model_info"])
    console.print(render_model_info(info))
    console.print(render_module_hierarchy(p1["hierarchy"], title="Module Hierarchy"))
    console.print(render_hook_demo(
        p1["hook_demo"]["layer"],
        tuple(p1["hook_demo"]["hidden_shape"]),
        p1["hook_demo"]["prompt"],
    ))
    _save(console, "01_hook_anatomy.svg", "Hook Anatomy")


# ---------------------------------------------------------------------------
# Screenshot 2: Intent Dataset (summary + samples)
# ---------------------------------------------------------------------------

def generate_dataset(data: dict):
    console = Console(record=True, width=100)

    p2 = data["phase2"]
    console.print(render_dataset_summary(p2["summary"]))
    console.print(render_prompt_samples(
        [tuple(s) for s in p2["samples"]],
        title="Sample Prompts (every 10th)",
    ))
    _save(console, "02_dataset.svg", "Intent Dataset")


# ---------------------------------------------------------------------------
# Screenshot 3: Layer Sweep + Best Probe Detail
# ---------------------------------------------------------------------------

def generate_layer_sweep(data: dict):
    console = Console(record=True, width=100)

    p3 = data["phase3"]
    sweep = _dicts_to_sweep(p3["sweep"])
    console.print(render_layer_sweep(sweep))

    best_result = sweep.get_result(sweep.best_layer)
    console.print(render_probe_detail(best_result))
    _save(console, "03_layer_sweep.svg", "Layer-wise Probing")


# ---------------------------------------------------------------------------
# Screenshot 4: Guardrail Pipeline (demo results)
# ---------------------------------------------------------------------------

def generate_guardrail(data: dict):
    console = Console(record=True, width=100)

    p4 = data["phase4"]
    for d in p4["demo_results"]:
        gr = _dict_to_guardrail_result(d)
        console.print(render_guardrail_result(gr))
    _save(console, "04_guardrail.svg", "Guardrail Pipeline")


# ---------------------------------------------------------------------------
# Screenshot 5: Stress Test + Confusion Matrix + Comparisons
# ---------------------------------------------------------------------------

def generate_stress_test(data: dict):
    console = Console(record=True, width=100)

    p5 = data["phase5"]

    # Build StressTestResult-like data for the stress test table
    from pipeline import StressTestResult
    results = [_dict_to_guardrail_result(d) for d in p5["stress_results"]]
    stress = StressTestResult(results=results, true_labels=p5["stress_labels"])
    console.print(render_stress_test(stress))

    cm = _dict_to_confusion_matrix(p5["confusion_matrix"])
    console.print(render_confusion_matrix(cm, title="Stress Test Confusion Matrix"))

    jailbreak_rate = p5["jailbreak_catch_rate"]
    console.print(f"[bright_cyan]Jailbreak catch rate: {jailbreak_rate:.0%}[/bright_cyan]")

    # Comparisons
    for comp_d in p5["comparisons"]:
        # Reconstruct the guardrail result for the comparison
        gr = GuardrailResult(
            prompt=comp_d["prompt"],
            decision="deny" if comp_d.get("hook_denied", False) else "allow",
            prob_harmful=comp_d["prob_harmful"],
            response="",
            latency_hook_ms=0, latency_classify_ms=0,
            latency_generate_ms=0, latency_total_ms=0,
            layer=0,
        )
        comp = ComparisonResult(
            prompt=comp_d["prompt"],
            model_response=comp_d["model_response"],
            guardrail_result=gr,
        )
        console.print(render_comparison(comp))

    _save(console, "05_stress_test.svg", "Jailbreak Stress Test")


# ---------------------------------------------------------------------------
# Screenshot 6: Conclusion + LLM Parallel + Adventure Connections
# ---------------------------------------------------------------------------

def generate_conclusion(data: dict):
    console = Console(record=True, width=100)

    p5 = data["phase5"]
    p6 = data["phase6"]

    cm = _dict_to_confusion_matrix(p6["cm"])
    console.print(render_conclusion(
        best_layer=p6["best_layer"],
        best_accuracy=p6["best_accuracy"],
        num_layers=p6["num_layers"],
        cm=cm,
        jailbreak_catch_rate=p6["jailbreak_catch_rate"],
    ))
    console.print(render_llm_parallel_table())
    console.print(render_adventure_connections())

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
        print("Run 'python app.py --capture' first.")
        sys.exit(1)

    data = _load_data(data_path)
    print(f"Loaded capture data from {data_path}")
    print("Generating Adventure 05 screenshots...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generate_hook_anatomy(data)
    generate_dataset(data)
    generate_layer_sweep(data)
    generate_guardrail(data)
    generate_stress_test(data)
    generate_conclusion(data)

    print("Done!")
